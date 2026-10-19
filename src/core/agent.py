"""Agent 核心：M3 起为「ReAct 循环」，M6.5 起循环本身是真流式的。

流程：先把上下文交给模型 → 模型若要调工具就执行工具、把结果塞回对话 → 再问模型，
如此循环，直到模型给出最终文本答案。正文在每一轮里都是逐 token 往外吐的。

记忆由上层 ChatService 负责组装；工具由注入的 ToolRegistry 提供。
演进点（循环不变，只换工具来源）：
- M4 RAG：把「检索」注册成一个工具
- M5 MCP：把外部 MCP 工具注册进来
- 流式改造：工具循环从「非流式 complete() + 事后 `_chunked` 假打字机」换成
  `complete_stream()` —— 正文真流式，工具决定在每轮收尾时才读出来。
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

from src.core.logger import get_logger
from src.core.models import Message, Role
from src.core.provider import BaseProvider, ToolCall
from src.core.tools import ToolRegistry
from src.core.utils import scrub_text


# 工具名 → 前端展示用短语。MCP 工具名是动态的，未命中就用原名兜底。
_TOOL_STATUS_LABELS = {
    "calculator": "计算",
    "get_current_time": "查询当前时间",
    "search_knowledge_base": "检索知识库",
    "list_directory": "查看目录",
    "directory_tree": "查看目录树",
    "read_file": "读取文件",
    "read_text_file": "读取文件",
    "read_multiple_files": "读取多个文件",
    "write_file": "写入文件",
    "write_text_file": "写入文件",
    "edit_file": "修改文件",
    "create_directory": "创建目录",
    "move_file": "移动文件",
    "copy_file": "复制文件",
    "search_files": "搜索文件",
    "get_file_info": "查看文件信息",
}


@dataclass
class AgentStatus:
    """一次工具调用**开始之前**发出的进度提示。

    为什么需要它：模型在工具调用那一轮不产出任何正文，所以「首字耗时」里全是
    「想 + 调工具 + 再想」，用户在那段时间里只能看到一个静止的「思考中…」。
    这个提示就是给那段时间用的——告诉他系统没死、正在做什么。

    它**不是回复内容**：不进会话历史、不进 CLI 输出、不算空回复检测的那部分。
    历史与正文的组装在 `ChatService` 里，靠 `isinstance` 把这两种东西分开。
    """

    tool_name: str

    @property
    def message(self) -> str:
        label = _TOOL_STATUS_LABELS.get(self.tool_name)
        return f"正在{label}…" if label else f"正在调用 {self.tool_name}…"


class ChatAgent:
    """接收上下文消息列表，必要时循环调用工具，最终产出回复。

    无工具时走流式直答；有工具时用 `complete_stream()` 跑循环——
    每一轮都是真流式：正文边到边吐，本轮收尾时才拿到「还要不要调工具」
    的结论，要调就把结果塞回历史再开一轮。最终答案因此也是逐 token 的，
    不再是「算完再整段出现」（改造前的做法见 docs/10）。
    """

    def __init__(
        self,
        provider: BaseProvider,
        tools: ToolRegistry | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self._provider = provider
        self._tools = tools if tools is not None else ToolRegistry()
        self._max_tool_rounds = max_tool_rounds
        # 打点用（M9）：让一次请求里"工具花了多少"和"模型花了多少"能拆开看
        self._log = get_logger("agent")

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    def respond(self, messages: list[Message]) -> Iterator[str | AgentStatus]:
        cleaned = [Message(m.role, scrub_text(m.content), images=m.images) for m in messages]
        schemas = self._tools.schemas()
        if not schemas:
            # 无工具：走原来的流式直答
            yield from self._provider.chat_stream(cleaned)
            return

        for _ in range(self._max_tool_rounds):
            tool_calls: list[ToolCall] = []
            for chunk in self._provider.complete_stream(cleaned, tools=schemas):
                if chunk.text:
                    yield chunk.text        # 正文逐段往外吐，含工具调用前的铺垫话
                if chunk.ended:
                    tool_calls = chunk.tool_calls or []
                    break
            if not tool_calls:
                return                      # 最终答案已逐 token 吐完
            # 模型要求调工具：记录调用动作，再逐个执行、把结果塞回
            cleaned.append(self._assistant_tool_call_msg(tool_calls))
            for tc in tool_calls:
                # 工具执行前发一次进度提示——工具轮没有正文，这段等待是用户唯一能
                # 感知到的信号（docs/16 §7.1）
                yield AgentStatus(tool_name=tc.name)
                args = self._parse_args(tc.arguments)
                t0 = time.perf_counter()
                output = self._tools.execute(tc.name, args)
                self._log.info("工具完成 %s 耗时=%.2fs 结果=%d字",
                               tc.name, time.perf_counter() - t0, len(output))
                cleaned.append(
                    Message(role=Role.TOOL, content=output,
                            tool_call_id=tc.id, name=tc.name)
                )
        # 达到轮数上限仍未收尾（防御性兜底，避免死循环）。
        # 此时已经流出去的正文没法收回，只能记一条日志，让用户知道这轮没收好。
        self._log.warning("达到工具调用轮数上限 %d，强制收尾", self._max_tool_rounds)

    @staticmethod
    def _assistant_tool_call_msg(tool_calls: list[ToolCall]) -> Message:
        """把模型返回的 ToolCall 转成 assistant 消息，塞回对话历史。"""
        return Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in tool_calls
            ],
        )

    @staticmethod
    def _parse_args(arguments: str) -> dict:
        """把工具参数 JSON 字符串转成 dict；空 / 坏 JSON 都给空 dict。"""
        try:
            return json.loads(arguments) if arguments and arguments.strip() else {}
        except json.JSONDecodeError:
            return {}
