"""Spike：本机能否跑起 MCP 服务器 + mcp SDK 能否连接并调工具。"""

import asyncio
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

TARGET = Path("data/spike_mcp_root")
shutil.rmtree(TARGET, ignore_errors=True)
TARGET.mkdir(parents=True, exist_ok=True)
(TARGET / "hello.txt").write_text("你好 MCP", encoding="utf-8")


async def main():
    server = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(TARGET)],
    )
    print("[1] 启动 filesystem MCP 服务器（首次 npx 会下载，可能较慢）...")
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[2] 已连接，初始化完成\n")

            tools = await session.list_tools()
            print(f"[3] 服务器提供 {len(tools.tools)} 个工具：")
            for t in tools.tools:
                print(f"    - {t.name}  |  {t.description[:60]}")

            print(f"\n[4] 调用 list_directory({TARGET.name}) ...")
            try:
                result = await session.call_tool("list_directory", {"path": str(TARGET)})
                print("    返回类型:", type(result).__name__)
                print("    内容:", result.content)
            except Exception as exc:
                print("    调用失败:", type(exc).__name__, exc)


asyncio.run(main())
shutil.rmtree(TARGET, ignore_errors=True)
