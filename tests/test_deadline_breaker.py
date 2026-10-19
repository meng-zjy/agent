"""P0 总时间预算 + P1 熔断器测试。

测两件事：
1. **时间预算**：一次调用（含所有重试）的墙钟被 `deadline` 封顶，不再线性放大。
   其中一条用**真实 socket** 验证（自定义 httpx transport 里的 sleep 触发不了
   read timeout，假 transport 测不出超时——见 data 里那条踩坑记录）。
2. **熔断器**：连续失败 N 次后打开，之后的请求**不再碰上游**，直接秒级失败。

其余全是内存态断言，不联网、不花额度。
"""

import socketserver
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError

from src.api import create_app
from src.core.agent import ChatAgent
from src.core.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.core.memory import MemoryStore
from src.core.provider import DeadlineExceeded, EmptyReplyError
from src.core.service import ChatService
from src.providers.deepseek import DeepSeekProvider


# ---------------------------------------------------------------------------
# 工具：假时钟 / 假 sleep / 可构造的 openai 异常
# ---------------------------------------------------------------------------


class Clock:
    """手动推进的假时钟，测熔断的「满 T 秒」不用真等。"""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _conn_error() -> APIConnectionError:
    """断网类错误，属于「可重试 + 上游不健康」。"""
    return APIConnectionError(request=httpx.Request("POST", "http://fake/v1/chat/completions"))


def _status_error(status: int) -> APIStatusError:
    return APIStatusError(
        "boom", response=httpx.Response(status, request=httpx.Request("POST", "http://fake/v1")),
        body=None,
    )


@pytest.fixture
def no_sleep(monkeypatch):
    """把 time.sleep 换成记账实现：不真等，但能断言退避节奏是否被调用。"""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


def _provider(**kw) -> DeepSeekProvider:
    """构造一个不联网的 provider：只 mock 掉 openai client。"""
    kw.setdefault("base_url", "http://fake")
    kw.setdefault("api_key", "fake")
    kw.setdefault("model", "m")
    p = DeepSeekProvider(**kw)
    p._client = MagicMock()
    return p


# ---------------------------------------------------------------------------
# 熔断器本身
# ---------------------------------------------------------------------------


def test_breaker_starts_closed():
    b = CircuitBreaker()
    assert b.state == CircuitBreaker.CLOSED
    b.check()                       # 不该抛


def test_breaker_opens_after_n_consecutive_failures():
    b = CircuitBreaker(failure_threshold=3)
    b.record_failure(ValueError("x"))
    b.record_failure(ValueError("x"))
    assert b.state == CircuitBreaker.CLOSED      # 2 次还没到阈值
    assert b.failures == 2
    b.record_failure(ValueError("x"))
    assert b.state == CircuitBreaker.OPEN


def test_success_resets_the_consecutive_counter():
    """熔断数的是「连续」失败：中间成功一次就重新计数。"""
    b = CircuitBreaker(failure_threshold=3)
    for _ in range(4):
        b.record_failure(ValueError("x"))
        b.record_success()
    assert b.state == CircuitBreaker.CLOSED
    assert b.failures == 0


def test_open_breaker_rejects_fast_without_waiting():
    """OPEN 时 check() 必须立刻抛——花时间在熔断期等于白花时间。"""
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, open_seconds=30.0, clock=clock)
    b.record_failure(ValueError("x"))
    clock.advance(10)
    start = time.perf_counter()
    with pytest.raises(CircuitOpenError) as exc_info:
        b.check()
    assert time.perf_counter() - start < 0.05
    assert exc_info.value.failures == 1
    assert exc_info.value.retry_after == pytest.approx(20.0)


class _Auth(Exception):
    """测试用的「4xx 类」异常：上游明明活着，不该计入熔断。"""


def test_non_unhealthy_failure_does_not_count():
    """鉴权失败不熔断：那种情况下上游活着，是我们配置的问题。

    把配置错误算成「上游挂了」，会把一个能立刻看懂的报错（API Key 错）
    换成一个看不懂的报错（已熔断），排查时反而更难。
    """
    b = CircuitBreaker(failure_threshold=1, is_unhealthy=lambda exc: not isinstance(exc, _Auth))
    for _ in range(5):
        b.record_failure(_Auth())
    assert b.state == CircuitBreaker.CLOSED
    assert b.failures == 0


def test_half_open_allows_exactly_one_probe():
    """OPEN 满 T 秒后只放行一个探测请求，其余仍快速失败。"""
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, open_seconds=30.0, clock=clock)
    b.record_failure(ValueError("x"))
    clock.advance(29.9)
    with pytest.raises(CircuitOpenError):
        b.check()                      # 还差一点没到
    clock.advance(0.2)
    b.check()                          # 第一个：放行，进入 HALF_OPEN
    assert b.state == CircuitBreaker.HALF_OPEN
    with pytest.raises(CircuitOpenError):
        b.check()                      # 第二个：不许跟过去


def test_half_open_probe_success_closes():
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, open_seconds=30.0, clock=clock)
    b.record_failure(ValueError("x"))
    clock.advance(31)
    b.check()
    b.record_success()
    assert b.state == CircuitBreaker.CLOSED
    assert b.failures == 0
    b.check()                          # 恢复正常放行


def test_half_open_probe_failure_reopens():
    """探测失败 → 重新 OPEN 并重新计时，不会无限试探。"""
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, open_seconds=10.0, clock=clock)
    b.record_failure(ValueError("x"))
    clock.advance(11)
    b.check()
    b.record_failure(ValueError("still down"))
    assert b.state == CircuitBreaker.OPEN
    with pytest.raises(CircuitOpenError) as exc_info:
        b.check()
    assert exc_info.value.retry_after == pytest.approx(10.0, abs=0.01)


def test_failure_while_open_does_not_extend_the_window():
    """OPEN 期间的失败不累加、不刷新时间，否则熔断会被自己的快速失败无限延长。"""
    clock = Clock()
    b = CircuitBreaker(failure_threshold=1, open_seconds=10.0, clock=clock)
    b.record_failure(ValueError("x"))
    clock.advance(5)
    b.record_failure(ValueError("x"))
    clock.advance(5)
    b.check()                          # 第 10 秒到点，该放行探测了
    assert b.state == CircuitBreaker.HALF_OPEN


def test_breaker_can_be_disabled():
    b = CircuitBreaker(failure_threshold=0)
    for _ in range(50):
        b.record_failure(ValueError("x"))
    assert b.state == CircuitBreaker.CLOSED


def test_describe_shape():
    b = CircuitBreaker()
    assert b.describe() == {"state": "closed", "failures": 0, "retry_after": 0.0}


# ---------------------------------------------------------------------------
# 时间预算（P0）：数学部分
# ---------------------------------------------------------------------------


def test_timeout_for_narrows_read_to_remaining_budget(monkeypatch):
    """read 超时按剩余预算收窄——这是 P0 生效的直接证据。"""
    p = _provider(timeout=90.0, deadline=30.0)
    start = time.monotonic()
    # 用 approx：两次 monotonic() 之间会真实流逝几微秒，remaining 不会是精确的 30.0
    assert p._timeout_for(start).read == pytest.approx(30.0, abs=0.05)
    monkeypatch.setattr(time, "monotonic", lambda: start + 25)
    assert p._timeout_for(start).read == pytest.approx(5.0, abs=0.05)      # 只剩 5 秒
    monkeypatch.setattr(time, "monotonic", lambda: start + 31)
    assert p._timeout_for(start).read == 1.0            # 预算耗尽也不给负数


def test_timeout_for_never_goes_below_one_second(monkeypatch):
    """预算为 0 时仍要给一个最小值，否则 httpx 会立刻拒绝而不是干净报错。"""
    p = _provider(timeout=90.0, deadline=0.0)
    monkeypatch.setattr(time, "monotonic", lambda: 1e9)
    assert p._timeout_for(time.monotonic()).read == 1.0


def test_backoff_gives_up_when_budget_is_too_small():
    p = _provider(deadline=0.0)
    assert p._backoff(1, time.monotonic()) is None      # 睡完就没预算了


def test_timeout_is_passed_per_request_not_on_the_client(monkeypatch):
    """按单次请求传 timeout，绝不改 client 属性——provider 是多会话共享的。"""
    p = _provider(timeout=90.0, deadline=30.0, max_retries=1)
    original = p._client.timeout
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(APIConnectionError):
        p.chat([])
    assert p._client.timeout is original                 # client 没被动过
    got = p._client.chat.completions.create.call_args.kwargs["timeout"]
    assert got.read == pytest.approx(30.0, abs=0.05)     # min(90, deadline=30)


# ---------------------------------------------------------------------------
# 时间预算（P0）：重试循环的行为
# ---------------------------------------------------------------------------


def test_retry_exhausts_then_raises_the_original_error(no_sleep):
    p = _provider(max_retries=3, deadline=600.0)
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(APIConnectionError):
        p.chat([])
    assert p._client.chat.completions.create.call_count == 3
    assert no_sleep == [2.0, 4.0]                       # 指数退避，且真被调用了


def test_budget_exhausted_raises_deadline_exceeded(no_sleep):
    """预算用光 → 抛可辨识的 DeadlineExceeded，不再多试一次、也不白睡。"""
    p = _provider(max_retries=3, deadline=0.0)
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(DeadlineExceeded) as exc_info:
        p.chat([])
    assert p._client.chat.completions.create.call_count == 1
    assert no_sleep == []
    assert exc_info.value.deadline == 0.0               # 报的是预算值
    assert exc_info.value.max_retries == 1              # 和已经试过的次数


def test_complete_also_honors_the_deadline(no_sleep):
    p = _provider(max_retries=3, deadline=0.0)
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(DeadlineExceeded):
        p.complete([])
    assert p._client.chat.completions.create.call_count == 1


def test_chat_stream_honors_the_deadline(no_sleep):
    p = _provider(max_retries=3, deadline=0.0)
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(DeadlineExceeded):
        list(p.chat_stream([]))
    assert p._client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# 不该重试 / 不该计数的情况
# ---------------------------------------------------------------------------


def test_auth_error_not_retried_and_not_counted(no_sleep):
    """4xx 立刻抛、不重试，也不让熔断器替配置错误买单。"""
    p = _provider(max_retries=3, breaker_failure_threshold=1)
    p._client.chat.completions.create.side_effect = _status_error(401)
    with pytest.raises(APIStatusError):
        p.chat([])
    assert p._client.chat.completions.create.call_count == 1
    assert no_sleep == []
    assert p._breaker.state == CircuitBreaker.CLOSED
    assert p._breaker.failures == 0


def test_retryable_5xx_does_count_toward_the_breaker(no_sleep):
    p = _provider(max_retries=1, breaker_failure_threshold=1)
    p._client.chat.completions.create.side_effect = _status_error(503)
    with pytest.raises(APIStatusError):
        p.chat([])
    assert p._breaker.state == CircuitBreaker.OPEN


def test_empty_reply_is_not_counted_by_breaker():
    """空回复是 max_tokens 配置问题，不是上游挂了——不能让熔断器背这个锅。"""
    p = _provider(max_retries=1, breaker_failure_threshold=1)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
        usage=None,
    )
    p._client.chat.completions.create.return_value = resp
    with pytest.raises(EmptyReplyError):
        p.chat([])
    assert p._breaker.state == CircuitBreaker.CLOSED
    assert p._breaker.failures == 0


def test_success_after_failure_resets_the_counter(no_sleep):
    """熔断数的是「连续」失败：抖一次、又成功，不该打开。"""
    p = _provider(max_retries=1, breaker_failure_threshold=2)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="你好"))])
    calls = {"n": 0}

    def fake_create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _conn_error()
        return resp

    p._client.chat.completions.create.side_effect = fake_create

    with pytest.raises(APIConnectionError):
        p.chat([])
    assert p._breaker.failures == 1
    assert p._breaker.state == CircuitBreaker.CLOSED

    assert p.chat([]) == "你好"
    assert p._breaker.failures == 0
    assert p._breaker.state == CircuitBreaker.CLOSED


# ---------------------------------------------------------------------------
# 熔断器（P1）：和调用链路接在一起
# ---------------------------------------------------------------------------


def test_breaker_opens_then_new_calls_fail_fast_without_touching_the_upstream(no_sleep):
    """P1 的核心：熔断打开后，新请求一个 HTTP 请求都不发。"""
    p = _provider(max_retries=1, breaker_failure_threshold=1)
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(APIConnectionError):
        p.chat([])                                     # 第 1 次失败 → 熔断打开
    assert p._breaker.state == CircuitBreaker.OPEN

    start = time.perf_counter()
    with pytest.raises(CircuitOpenError):
        p.chat([])                                     # 第 2 次：秒级失败
    assert time.perf_counter() - start < 0.05

    assert p._client.chat.completions.create.call_count == 1   # 没有第 2 个 HTTP 请求
    assert no_sleep == []                                     # 也没有退避


def test_retry_stops_when_breaker_opens_midway(no_sleep):
    """重试到一半熔断被打开 → 立刻停手，剩下的重试次数不白试。"""
    p = _provider(max_retries=3, breaker_failure_threshold=1)
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(CircuitOpenError):
        p.chat([])
    assert p._client.chat.completions.create.call_count == 1
    assert no_sleep == [2.0]                               # 睡过一次，然后看到熔断就停


def test_chat_stream_fails_fast_when_open(no_sleep):
    p = _provider(max_retries=3, breaker_failure_threshold=1)
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(CircuitOpenError):
        list(p.chat_stream([]))
    assert p._client.chat.completions.create.call_count == 1


def test_breaker_state_is_exposed_for_health_check():
    p = _provider(breaker_failure_threshold=1)
    assert p.breaker_state()["state"] == "closed"
    p._breaker.record_failure(_conn_error())
    assert p.breaker_state()["state"] == "open"


# ---------------------------------------------------------------------------
# /health：必须是 async，并报告上游状态
# ---------------------------------------------------------------------------


def _service_with(provider, tmp_path) -> ChatService:
    return ChatService(ChatAgent(provider), MemoryStore(tmp_path),
                       system_prompt="你是助手", summary_keep_turns=0)


def test_health_endpoint_is_async(tmp_path):
    """同步端点要占 uvicorn 线程池槽位；探活必须跑在事件循环上，永远不被排队。"""
    import inspect

    app = create_app(_service_with(_provider(), tmp_path))
    route = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    assert inspect.iscoroutinefunction(route.endpoint), (
        "/health 必须是 async def——线程池被上游占满时它会跟着一起超时"
    )


def test_health_reports_upstream_state(tmp_path):
    p = _provider(breaker_failure_threshold=1)
    body = TestClient(create_app(_service_with(p, tmp_path))).get("/health").json()
    assert body["ok"] is True
    assert body["max_input_chars"] > 0
    assert body["upstream"]["state"] == "closed"

    p._breaker.record_failure(_conn_error())
    body = TestClient(create_app(_service_with(p, tmp_path))).get("/health").json()
    assert body["upstream"]["state"] == "open"
    assert body["upstream"]["failures"] == 1


def test_health_still_works_with_a_provider_without_a_breaker(tmp_path):
    """假 provider 没有熔断器时不能崩，只能报 unknown。"""
    from src.core.provider import BaseProvider, Completion
    from src.core.models import Message

    class Plain(BaseProvider):
        def chat(self, messages):
            return "x"

        def chat_stream(self, messages):
            yield "x"

        def complete(self, messages, tools=None):
            return Completion(finish_reason="stop", content="x")

    body = TestClient(create_app(_service_with(Plain(), tmp_path))).get("/health").json()
    assert body["upstream"]["state"] == "unknown"


# ---------------------------------------------------------------------------
# 端到端：墙钟真的被 deadline 封顶（真实 socket，不联网）
# ---------------------------------------------------------------------------


def test_wall_clock_is_bounded_by_deadline_against_a_hanging_server():
    """P0 的整体验收：上游只接不回（真实挂掉的样子），一次调用不超过 deadline。

    改造成前，同样的场景下是「重试次数 × 单次 read 超时」线性放大——
    9 次 × read 90s ≈ 820s。这里用 deadline=1s 断言它被压回 1 秒量级。

    必须起**真实 TCP socket**：自定义 httpx transport 里的 sleep 触发不了
    read timeout，假 transport 会把这个回归漏过去。
    """
    class HangingServer(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                self.request.recv(4096)
            except OSError:
                pass
            time.sleep(5.0)             # 读完请求后一个字节都不回
            self.request.close()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), HangingServer)
    server.daemon_threads = True
    server.allow_reuse_address = True
    server.timeout = 0.1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        p = DeepSeekProvider(
            base_url=f"http://127.0.0.1:{port}/v1", api_key="fake",
            max_retries=3, timeout=90.0, deadline=1.0,
            breaker_failure_threshold=1,
        )
        start = time.perf_counter()
        with pytest.raises(DeadlineExceeded):
            p.chat([])
        elapsed = time.perf_counter() - start
    finally:
        server.shutdown()
        server.server_close()

    assert elapsed < 3.0, f"deadline=1.0s 却跑了 {elapsed:.2f}s（重试在放大耗时）"
    assert elapsed >= 0.5, f"中止得太早，不像真的等过超时：{elapsed:.2f}s"
    assert p.breaker_state()["state"] == "open"      # 挂掉的上游顺手被熔断了
