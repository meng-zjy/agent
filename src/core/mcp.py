"""MCP 桥（M5）：连接外部 MCP 工具服务器，把它的工具暴露成同步可调的 Tool。

MCP 是异步协议（asyncio），而我们的 Tool.func 是同步函数——所以用
「后台线程 + 常开事件循环 + run_coroutine_threadsafe」做桥接：
后台线程跑一个独立事件循环、保持 MCP 会话常开；同步 call_tool 把调用
投递给那个循环、阻塞等结果。
"""

import asyncio
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.core.tools import Tool
from src.core.utils import wrap_untrusted


def _clean_schema(schema: dict) -> dict:
    """只保留 OpenAI 函数调用需要的字段，避免 MCP 的额外字段（$schema 等）报错。"""
    return {
        "type": schema.get("type", "object"),
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


def _format_call_result(result) -> str:
    """把 MCP 的 CallToolResult 转成字符串（提取文本块）。

    结果包一层外部资料围栏（M9）：MCP 工具读到的文件内容是文件系统里的不可信文本，
    可能是提示词注入的载体。错误分支也包——错误信息同样是 MCP 服务器产出的。
    """
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "\n".join(parts) if parts else "(无文本输出)"
    if getattr(result, "isError", False):
        text = f"[错误] {text}"
    return wrap_untrusted(text)


class McpToolBridge:
    """连接一个 MCP 服务器，保持会话常开，暴露同步 call_tool。"""

    def __init__(self, command: str, args: list[str]) -> None:
        self._command = command
        self._args = args
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._tools: list[dict] = []
        self._error: Exception | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-bridge")

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """阻塞等待连接 + 发现工具完成，返回是否成功。"""
        self._ready.wait(timeout)
        return self._error is None and bool(self._tools)

    def tools(self) -> list[dict]:
        """返回已发现的工具（name/description/parameters）。"""
        return self._tools

    def call_tool(self, name: str, args: dict) -> str:
        """同步调用一个 MCP 工具，返回字符串结果。"""
        if self._loop is None or self._session is None:
            return f"工具 {name} 不可用（MCP 未连接）"
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, args), self._loop
        )
        return _format_call_result(future.result(timeout=60))

    def _run(self) -> None:
        """后台线程：建事件循环，跑 _main 直到进程退出（daemon）。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        try:
            server = StdioServerParameters(command=self._command, args=self._args)
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    self._tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": _clean_schema(getattr(t, "inputSchema", {}) or {}),
                        }
                        for t in result.tools
                    ]
                    self._session = session
                    self._ready.set()
                    # 常驻：挂起但让事件循环继续处理 run_coroutine_threadsafe 投递的调用
                    await asyncio.Event().wait()
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()


def wrap_mcp_tools(bridge: McpToolBridge) -> list[Tool]:
    """把桥里发现的 MCP 工具包装成我们的 Tool。"""

    def make_func(name: str):
        def func(**kwargs) -> str:
            return bridge.call_tool(name, kwargs)
        return func

    return [
        Tool(
            name=t["name"],
            description=t["description"],
            parameters=t["parameters"],
            func=make_func(t["name"]),
        )
        for t in bridge.tools()
    ]
