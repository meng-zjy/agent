"""LLM Provider 统一接口。

任何模型 / 服务商只要实现本接口即可接入 Agent，Agent 核心不感知具体厂商。
后续多路并发、多模型切换都通过该层完成。
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

from src.core.models import Message


@dataclass
class ToolCall:
    """模型发起的一次工具调用（M3）。"""

    id: str            # 调用 id，回填 tool 结果消息用
    name: str          # 函数名
    arguments: str     # 参数 JSON 字符串


@dataclass
class Completion:
    """一次非流式调用的结构化返回（M3）。

    finish_reason 为 "tool_calls" 时 tool_calls 非空、content 为空；
    为 "stop" 时 content 为最终文本。
    """

    finish_reason: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class StreamChunk:
    """流式跑「一轮对话」产出的一个片段（流式工具循环）。

    text 非空 → 一段该发给用户的正文，可能只有几个字。
    finish_reason 非 None → 这一轮已收尾：tool_calls 非空表示模型要求调工具，
    为空表示刚吐完的文本就是最终答案。

    为什么需要它：生成器把正文往外吐的时候，没法把「模型还要不要调工具」
    这个结论也一起带出来（`yield` 的值只能是正文）。收尾信息只能作为
    流的最后一个片段传出去。
    """

    text: str = ""
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None

    @property
    def ended(self) -> bool:
        """这一轮是否已经收尾。"""
        return self.finish_reason is not None


# 退化为非流式时的分段大小：与改造前 `ChatAgent._chunked(16)` 的节奏一致，
# 让 SSE 仍然能一段段推。16 字不是性能参数，只是别一次推一整个段落。
_STREAM_TEXT_SIZE = 16


class EmptyReplyError(RuntimeError):
    """模型成功返回 HTTP 200 但没有产出任何 content。

    根因：**reasoning 模型**的输出分 `reasoning`（思考过程，计数进 completion_tokens）
    和 `content`（最终答案）两部分。当 `max_tokens` 小于模型实际需要的 reasoning 量时，
    思考耗尽整个预算，content 为空，`finish_reason='length'`。

    这个机制跟厂商无关，换 DeepSeek 后照样复现（实测 `deepseek-chat` 之外的
    `deepseek-v4-pro` / `deepseek-flash` / `deepseek-reasoner` 都是 reasoning 模型，
    `max_tokens=120` → completion_tokens=120 且 reasoning_tokens=120 → content 为空）：

      sensenova-6.8-flash-lite  max_tokens=4096 → 4096 全被思考吃掉，content 为空
      deepseek-chat（非 reasoning）max_tokens=30  → finish='length'，但 content 有内容（被截断）

    所以修复方向有两条：留空 `max_tokens`（让服务端用默认上限），或换非 reasoning 模型。
    抛这个异常而不是静默返回空字符串，是为了让上层能区分
    "模型没话说"和"网络错误"，给用户明确的修复提示。
    """

    def __init__(self, max_tokens: int | None = None, usage=None) -> None:
        self.max_tokens = max_tokens
        self.usage = usage
        # usage 可能为 None：服务端不发 usage chunk 时就是这样（DeepSeek 默认不发）。
        # 先给默认值，否则下面引用 rt 会抛 UnboundLocalError——
        # 本该给用户修复提示的异常反而自己崩了。
        rt = 0
        if usage is not None:
            try:
                ud = usage.model_dump()
                rt = ud.get("reasoning_tokens") or \
                    (ud.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            except Exception:
                rt = 0
        super().__init__(
            "模型没有生成回复——它的「思考过程」耗尽了输出预算。"
            + (f"（MAX_TOKENS={max_tokens}，思考用了 {rt}）" if rt else "")
            + "　修复：在 .env 里留空 MAX_TOKENS，让服务端用它自己的默认上限"
            "（显式设一个较低的数值反而更容易触发空回复）；"
            "或者换一个非 reasoning 模型（如 deepseek-chat，它没有思考阶段，"
            "预算被吃光只会截断答案，不会产生空回复，见 docs/14）。"
            + "另外：本轮你发的消息会被撤回，历史里不会留下有问无答的记录。"
        )


class DeadlineExceeded(RuntimeError):
    """整条调用链的总时间预算用完了——不再重试，直接快速失败。

    为什么要它：重试次数会把总耗时线性放大。实测（read=1.0s 等效放大）：
    我们外层 max_retries=3 × openai SDK 自带 max_retries=2 = **9 次 HTTP 请求**，
    9 次 × read 超时 + 两层退避 = 19.36s。换成真实的 read=90s 就是
    **820s ≈ 13.7 分钟，一次调用占住一个线程那么久**。

    线上后果是具体的：uvicorn 的同步端点共享一个 40 槽的线程池，
    45 个并发会话就能把池占满，连 `/health` 都被拖到探活超时——
    负载均衡于是判服务挂掉去重启它，而上游其实只是慢。

    所以重试之前必须先查预算：预算不够就不再试。
    """

    def __init__(self, deadline: float, max_retries: int | None = None) -> None:
        self.deadline = deadline
        self.max_retries = max_retries
        super().__init__(
            f"调用超过总时间预算 {deadline:.0f}s，已停止重试"
            + (f"（已尝试 {max_retries} 次）" if max_retries else "")
        )


class BaseProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        """非流式：返回完整回复文本。"""

    @abstractmethod
    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        """流式：逐段产出回复文本。"""

    @abstractmethod
    def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> Completion:
        """非流式：一次调用，结构化返回（最终文本 或 工具调用）。"""

    def complete_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> Iterator[StreamChunk]:
        """流式跑一轮对话（可带工具）：正文边到边吐，收尾时给出工具调用。

        默认实现退化为非流式 `complete()`，把整段正文切成小块吐出——
        行为和流式工具循环改造之前完全一致。覆写本方法即可拿到真流式。

        刻意不声明成 abstractmethod：老的假 provider 只实现了 `complete()`，
        这个默认实现让它们不用改一行仍然能跑工具循环。
        """
        result = self.complete(messages, tools=tools)
        if result.tool_calls:
            yield StreamChunk(finish_reason="tool_calls", tool_calls=result.tool_calls)
            return
        for i in range(0, len(result.content), _STREAM_TEXT_SIZE):
            yield StreamChunk(text=result.content[i:i + _STREAM_TEXT_SIZE])
        yield StreamChunk(finish_reason=result.finish_reason)

    def breaker_state(self) -> dict:
        """上游熔断状态：`{"state": "closed"|"open"|"half_open", "failures", "retry_after"}`。

        默认 provider 不带熔断器，报 unknown。`/health` 从这取——
        出事先分清是「我们自己的服务」还是「上游」的问题，别瞎猜。
        """
        return {"state": "unknown", "failures": 0, "retry_after": 0.0}
