"""OpenAI 协议兼容 Provider（通用实现，厂商无关）。

DeepSeek、商汤日日新等都提供 OpenAI 兼容接口，差别只在 base_url / 模型名 /
个别默认行为。所以实现只写这一份，厂商子类（`providers/deepseek.py` 等）只声明
「我是谁」，见 docs/14。

健壮性处理（沿用 M8 结论，见 docs/11 §7）：
- 瞬时限流 / 断网 / 5xx 自动重试（指数退避）；4xx（鉴权失败等）不重试
- **总时间预算**（P0）：一次调用含所有重试被 `deadline` 秒封顶，read 超时按
  「剩余预算」收窄；预算用完抛 DeadlineExceeded 停止重试，不再线性放大耗时
- **熔断器**（P1）：连续 N 次失败后打开 T 秒，期间秒级失败，不重复烧预算
- 流式跑完但 content 为空 → 抛 EmptyReplyError（带诊断），不静默留「空气泡」
- 流式解码遇「多字节字符被网络切分」的编码异常 → 回退非流式，保证对话可用
- 流式工具循环（`complete_stream`，docs/16）：正文逐 token 吐，工具调用按 index 分段拼接
  参数、在收尾片段给出；重试只在「一个字都没吐」时做
- 输出前清洗孤立代理项，防止后续编码崩溃
- 只转发 `delta.content`：reasoning 模型的思考放在 `delta.reasoning` /
  `delta.reasoning_content` 里，一律忽略（用户不该看到思考过程）
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from src.core.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.core.logger import get_logger
from src.core.models import Message
from src.core.provider import (
    BaseProvider,
    Completion,
    DeadlineExceeded,
    EmptyReplyError,
    StreamChunk,
    ToolCall,
)
from src.core.utils import scrub_text


def _is_retryable(exc: Exception) -> bool:
    """限流、断网、5xx 服务端错误可以重试；4xx（如鉴权失败）不重试。"""
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


def _is_unhealthy(exc: BaseException) -> bool:
    """这个失败该算「上游不健康」吗？熔断器只统计这类。

    限流 / 断网 / 5xx / 总预算耗尽 → 上游的事，计入，该熔断。
    鉴权失败(4xx) / 空回复 → 上游明明活着，是我们配置的问题，不计入。
    把配置错误算成「上游挂了」，会把一个能立刻看懂的报错（API Key 错）
    换成一个看不懂的报错（已熔断），排查时反而更难。
    """
    return _is_retryable(exc) or isinstance(exc, DeadlineExceeded)


class OpenAICompatProvider(BaseProvider):
    """任意 OpenAI 协议兼容服务商的通用实现。

    子类只需要给出 `PROVIDER_NAME` 和默认 base_url / 模型名。
    """

    PROVIDER_NAME = "openai-compat"   # 用于日志与界面展示：deepseek/deepseek-chat

    # 流式请求是否要求服务端在末尾补一个带 usage 的 chunk。
    # DeepSeek 默认不发 usage，拿不到它就没有 reasoning_tokens 诊断，
    # 空回复会变成「只知道是空的，不知道为什么空」。两家都已实测支持。
    STREAM_USAGE = True

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        timeout: float = 90.0,
        deadline: float = 30.0,
        breaker_failure_threshold: int = 3,
        breaker_open_seconds: float = 30.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._timeout = timeout
        self._deadline = deadline
        # 显式超时（M9）：openai SDK 默认 read=600（10 分钟），本地也会卡那么久才报错。
        # 90s 比实测最慢的真实响应（78s）多 15% 余量，见 docs/11 §7 发现四。
        #
        # ⚠ max_retries=0 是必须的（P0）：SDK 自己也会重试（默认 max_retries=2），
        # 不关掉就是两层重试叠着跑、退避节奏互不知道——实测 3×3=9 次 HTTP 请求，
        # read=90 时单次调用最坏占住线程 820s ≈ 13.7 分钟。重试策略只留我们这一份。
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=httpx.Timeout(connect=5.0, read=timeout, write=timeout, pool=timeout),
        )
        self._log = get_logger("provider")
        # 熔断器（P1）：provider 是多会话共享的单例，所以一台进程里每家上游各一个。
        # 它决定「上游是不是该继续问了」，和上面的重试/预算决定「这一次要不要再试」
        # 是两件事——重试治偶发抖动，熔断治上游真的挂了。
        self._breaker = CircuitBreaker(
            name=self.label,
            failure_threshold=breaker_failure_threshold,
            open_seconds=breaker_open_seconds,
            is_unhealthy=_is_unhealthy,
            log=self._log,
        )

    @property
    def label(self) -> str:
        """展示用名字，如 `deepseek/deepseek-chat`。"""
        return f"{self.PROVIDER_NAME}/{self._model}"

    @property
    def breaker(self) -> CircuitBreaker:
        """上游熔断器。暴露出来给 `/health` 报状态、给测试断言用。"""
        return self._breaker

    def breaker_state(self) -> dict:
        return self._breaker.describe()

    def _build_kwargs(self, messages: list[Message], *, stream: bool) -> dict:
        """组装请求参数；未配置的生成参数不下发，使用模型默认值。"""
        kwargs: dict = {
            "model": self._model,
            "messages": [m.to_openai_dict() for m in messages],
            "stream": stream,
        }
        if stream and self.STREAM_USAGE:
            kwargs["stream_options"] = {"include_usage": True}
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return kwargs

    @staticmethod
    def _reasoning_tokens(usage) -> int:
        """从 usage 里提取 reasoning_tokens（可能在顶层或 completion_tokens_details 里）。"""
        if usage is None:
            return 0
        try:
            ud = usage.model_dump()
        except Exception:
            return 0
        rt = ud.get("reasoning_tokens")
        if rt:
            return rt
        details = ud.get("completion_tokens_details") or {}
        return details.get("reasoning_tokens") or 0

    def _reject_empty(self, max_tokens: int | None, usage) -> None:
        """流式/非流式跑完但 content 为空时抛 EmptyReplyError，并写日志带诊断信息。

        根因见 EmptyReplyError 的文档字符串：reasoning 模型耗尽了 token 预算。
        """
        rt = self._reasoning_tokens(usage)
        ct = getattr(usage, "completion_tokens", "?") if usage else "?"
        self._log.warning(
            "%s 空回复: max_tokens=%s reasoning_tokens=%s completion_tokens=%s"
            " → reasoning 耗尽 token 预算",
            self.label, max_tokens, rt, ct,
        )
        raise EmptyReplyError(max_tokens=max_tokens, usage=usage)

    # ---------- 时间预算（P0）：总耗时封顶，不随重试次数线性放大 ----------
    #
    # 诚实说明预算的边界：它约束的是「从调用进入到第一次尝试失败为止」的墙钟。
    # 流式一旦开始往外吐字，逐 chunk 的间隔受 read timeout 约束，**不受 deadline 约束**
    # （服务端每隔几秒挤一个字，流可以拖很久）。要管住那一段得上看门狗线程，
    # 这个项目的对话场景下不值得，先记在这里。

    def _remaining(self, deadline_start: float) -> float:
        """距预算终点还剩多少秒。"""
        return self._deadline - (time.monotonic() - deadline_start)

    def _timeout_for(self, deadline_start: float) -> httpx.Timeout:
        """本次尝试该用的超时：read 按**剩余预算**收窄。

        这是 P0 的核心。不收窄的话，第 2 次尝试照样等到 read=90 才超时，
        总耗时就被重试次数线性放大（9 次 × 90s）。收窄后整条调用链被 `deadline` 封顶。

        ⚠ 按**单次请求**传 timeout，不去改 `self._client.timeout`——
        provider 是多会话共享的，改 client 属性会让并发会话互相踩。
        """
        t = min(self._timeout, max(self._remaining(deadline_start), 1.0))
        return httpx.Timeout(connect=5.0, read=t, write=t, pool=t)

    def _backoff(self, attempt: int, deadline_start: float) -> float | None:
        """该睡多久；预算不够睡完就返回 None，调用方放弃重试。

        退避本身也受预算约束：一次长退避不该把剩下的预算全吃光。
        """
        remaining = self._remaining(deadline_start)
        sleep_for = min(2.0 ** attempt, remaining)
        if remaining - sleep_for < 1.0:
            return None
        return sleep_for

    def _retry(self, exc: Exception, attempt: int, start: float) -> int | None:
        """判定这条失败怎么办：睡完退避重试，还是放弃。

        返回新的 attempt 编号；抛异常表示「这条调用到此为止」。
        三个方法（chat / complete / chat_stream）共用这一段，
        保证重试与熔断的记账口径只有一份。

        ⚠ 每一次失败都记账，不只记最终的：否则熔断要等一整条调用把预算烧完
        才可能打开，那「上游真挂时秒级失败」就变成「上游挂时先陪它烧 30 秒」。
        单次失败的过滤（4xx / 空回复不算）交给 record_failure 内部判断。
        """
        self._breaker.record_failure(exc)
        if not _is_retryable(exc) or attempt >= self._max_retries - 1:
            raise exc
        attempt += 1
        wait = self._backoff(attempt, start)
        if wait is None:
            # 预算不够再试一次：停止重试，抛可辨识的错误让上层能区分
            raise DeadlineExceeded(self._deadline, attempt) from exc
        time.sleep(wait)
        return attempt

    def chat(self, messages: list[Message]) -> str:
        """非流式：返回完整回复文本。总耗时被 `deadline` 封顶。"""
        start = time.monotonic()
        kwargs = self._build_kwargs(messages, stream=False)
        attempt = 0
        while True:
            self._breaker.check()                  # 熔断打开 → 秒级失败，不再烧预算
            try:
                resp = self._client.chat.completions.create(
                    **kwargs, timeout=self._timeout_for(start)
                )
                content = resp.choices[0].message.content or ""
                if not content:
                    self._reject_empty(kwargs.get("max_tokens"), resp.usage)
                self._breaker.record_success()
                return content
            except EmptyReplyError:
                raise                       # 不重试、不计熔断：空回复重试大概率还是空
            except Exception as exc:       # noqa: BLE001  统一交给 _retry 判定
                attempt = self._retry(exc, attempt, start)

    def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> Completion:
        """非流式：一次调用，结构化返回（最终文本 或 工具调用）。"""
        kwargs = self._build_kwargs(messages, stream=False)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        start = time.monotonic()
        attempt = 0
        while True:
            self._breaker.check()
            try:
                resp = self._client.chat.completions.create(
                    **kwargs, timeout=self._timeout_for(start)
                )
                choice = resp.choices[0]
                if choice.message.tool_calls:
                    self._breaker.record_success()
                    return Completion(
                        finish_reason="tool_calls",
                        tool_calls=[
                            ToolCall(tc.id, tc.function.name, tc.function.arguments)
                            for tc in choice.message.tool_calls
                        ],
                    )
                content = choice.message.content or ""
                if not content:
                    self._reject_empty(kwargs.get("max_tokens"), resp.usage)
                self._breaker.record_success()
                return Completion(
                    finish_reason=choice.finish_reason,
                    content=content,
                )
            except EmptyReplyError:
                raise
            except Exception as exc:       # noqa: BLE001
                attempt = self._retry(exc, attempt, start)

    def complete_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> Iterator[StreamChunk]:
        """流式跑一轮对话（可带工具）：正文逐 token 吐出，收尾时给出工具调用。

        这是工具循环从「非流式 + 事后假打字机」换成真流式的关键。工具决定
        仍然要等整轮流完才能执行——参数是分段的 JSON 字符串片段，必须攒齐
        才解析得出（`{"express` + `ion": "1+1"}`）；但模型在调工具之前的
        正文、以及最终答案那一轮，都能边到边吐。

        ⚠ 重试只在**一个字符都还没吐出去**时做。已经往外吐过正文之后再
        重试，会把同一段话原样再吐一遍——那时宁可直接失败，不能重来。
        失败仍然记进熔断（每次都记，不只记最终那条，见 `_retry` 的注释）。
        """
        kwargs = self._build_kwargs(messages, stream=True)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        start = time.monotonic()
        attempt = 0
        while True:
            self._breaker.check()
            emitted = False
            pending: dict[int, dict] = {}   # index -> {id, name, arguments}
            total = ""
            usage = None
            finish_reason = ""
            try:
                stream = self._client.chat.completions.create(
                    **kwargs, timeout=self._timeout_for(start)
                )
                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    for choice in chunk.choices:
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                        delta = choice.delta
                        if delta is None:
                            continue
                        if delta.content:
                            text = scrub_text(delta.content)
                            if text:
                                total += text
                                emitted = True
                                yield StreamChunk(text=text)
                        # 工具调用按 index 分段到达，参数是 JSON 片段，必须按段拼接
                        for tc in getattr(delta, "tool_calls", None) or []:
                            slot = pending.setdefault(
                                tc.index, {"id": "", "name": "", "arguments": ""}
                            )
                            if tc.id:
                                slot["id"] = tc.id
                            fn = tc.function
                            if fn is not None:
                                if fn.name:
                                    slot["name"] += fn.name
                                if fn.arguments:
                                    slot["arguments"] += fn.arguments
                tool_calls = [
                    ToolCall(s["id"], s["name"], s["arguments"])
                    for _, s in sorted(pending.items())
                ]
                if not total and not tool_calls:
                    self._reject_empty(kwargs.get("max_tokens"), usage)
                self._breaker.record_success()
                yield StreamChunk(
                    finish_reason=finish_reason or "stop",
                    tool_calls=tool_calls or None,
                )
                return
            except EmptyReplyError:
                raise                       # 不重试、不计熔断：重试大概率还是空
            except UnicodeError:
                # 多字节字符被网络切分 → 回退非流式。回退那次调用由 complete()
                # 自己做熔断检查与记账，这里不再记一次，否则同一个失败算两遍。
                comp = self.complete(messages, tools=tools)
                if comp.tool_calls:
                    yield StreamChunk(
                        finish_reason="tool_calls", tool_calls=comp.tool_calls
                    )
                else:
                    yield StreamChunk(
                        text=comp.content, finish_reason=comp.finish_reason
                    )
                return
            except Exception as exc:       # noqa: BLE001
                if emitted:
                    self._breaker.record_failure(exc)
                    raise                  # 已经吐过字了，重试会重复输出
                attempt = self._retry(exc, attempt, start)

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        """流式：逐段产出（无工具）。

        ⚠ 这是生成器，body 要到被迭代时才执行——所以预算起点 `start` 和熔断检查
        都发生在第一次 `next()`，而不是调用 `chat_stream()` 的那一刻。
        """
        kwargs = self._build_kwargs(messages, stream=True)
        start = time.monotonic()
        attempt = 0
        while True:
            self._breaker.check()
            try:
                stream = self._client.chat.completions.create(
                    **kwargs, timeout=self._timeout_for(start)
                )
                total = ""
                usage = None
                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    for choice in chunk.choices:
                        if choice.delta and choice.delta.content:
                            text = scrub_text(choice.delta.content)
                            total += text
                            yield text
                if not total:
                    self._reject_empty(kwargs.get("max_tokens"), usage)
                self._breaker.record_success()
                return
            except EmptyReplyError:
                raise
            except UnicodeError:
                # 流式解码多字节字符被网络切分时偶发崩溃 → 回退非流式，保证可用。
                # 回退那次调用由 chat() 自己做熔断检查与记账，这里不再记一次，
                # 否则同一个失败会被算两遍。
                yield from self.chat(messages)
                return
            except Exception as exc:       # noqa: BLE001
                attempt = self._retry(exc, attempt, start)
