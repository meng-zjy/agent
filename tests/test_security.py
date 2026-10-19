"""输入侧防线 + 访问控制测试（M9）。

覆盖：
- 输入长度上限：服务端校验、**历史里不留这条消息**、边界值
- `/health`：200 + ok；认证 / 限流开启时仍不要求认证、不被限
- 认证中间件（默认关闭）：关闭 = 与未安装完全一致；开启 = 401
- 限流中间件（默认关闭）：关闭 = 打爆也不 429；开启 = 429
- HTTP 超时：显式传进 httpx，且低超时的 provider 调用真的能中止

全部用假 provider，不花额度、不联网。
"""

import time
import types

import httpx
import pytest

from fastapi.testclient import TestClient

from src.api import _rl_buckets, create_app
from src.config import settings
from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.models import Message
from src.core.provider import BaseProvider, Completion
from src.core.service import ChatService, InputTooLongError


class FakeProvider(BaseProvider):
    """每次返回固定回复。"""

    def __init__(self, reply: str = "你好呀") -> None:
        self._reply = reply

    def chat(self, messages: list[Message]) -> str:
        return self._reply

    def chat_stream(self, messages):
        yield self._reply

    def complete(self, messages, tools=None):
        return Completion(finish_reason="stop", content=self._reply)


def make_service(tmp_path, **kw) -> ChatService:
    return ChatService(
        ChatAgent(FakeProvider()),
        MemoryStore(tmp_path),
        system_prompt="你是助手",
        summary_keep_turns=0,
        **kw,
    )


def _boot_settings(**overrides):
    d = dict(
        llm_provider="deepseek",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_api_key="sk-d",
        deepseek_model="deepseek-chat",
        max_tokens=None,
        temperature=None,
        request_timeout=90.0,
        max_retries=3,
        deadline=30.0,
        breaker_failure_threshold=3,
        breaker_open_seconds=30.0,
    )
    d.update(overrides)
    return types.SimpleNamespace(**d)


@pytest.fixture(autouse=True)
def _reset_guard_state(monkeypatch):
    """限流计数是模块级内存态，必须每个用例都清空，否则用例会互相影响。"""
    _rl_buckets.clear()
    monkeypatch.setattr(settings, "auth_token", "")
    monkeypatch.setattr(settings, "rate_limit_rpm", 0)
    yield
    _rl_buckets.clear()


# ---------------------------------------------------------------------------
# 输入长度上限
# ---------------------------------------------------------------------------


def test_input_at_limit_is_accepted(tmp_path):
    client = TestClient(create_app(make_service(tmp_path, max_input_chars=10)))
    sid = client.post("/api/sessions").json()["id"]
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "x" * 10}).status_code == 200


def test_input_over_limit_returns_400(tmp_path):
    """超长输入拿干净的 400，而不是流里的一条错误事件。"""
    client = TestClient(create_app(make_service(tmp_path, max_input_chars=10)))
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/messages", json={"text": "x" * 11})
    assert r.status_code == 400
    assert "输入过长" in r.json()["detail"]
    assert "11" in r.json()["detail"] and "上限 10" in r.json()["detail"]


def test_input_one_below_limit_is_accepted(tmp_path):
    client = TestClient(create_app(make_service(tmp_path, max_input_chars=10)))
    sid = client.post("/api/sessions").json()["id"]
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "x" * 9}).status_code == 200


def test_rejected_input_leaves_no_message_in_history(tmp_path):
    """被拒的请求什么都没发生——历史里不能留下有问无答的记录。"""
    client = TestClient(create_app(make_service(tmp_path, max_input_chars=10)))
    sid = client.post("/api/sessions").json()["id"]
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "x" * 500}).status_code == 400
    assert client.get(f"/api/sessions/{sid}").json()["messages"] == []


def test_rejected_input_still_allows_next_normal_message(tmp_path):
    """被拒是一次性的，不影响后续正常请求。"""
    client = TestClient(create_app(make_service(tmp_path, max_input_chars=10)))
    sid = client.post("/api/sessions").json()["id"]
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "x" * 500}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "你好"}).status_code == 200
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "你好"


def test_validate_input_raises_with_length_and_limit(tmp_path):
    """CLI 走这条路：拿得到明确异常，而不是被流式吞掉。"""
    svc = make_service(tmp_path, max_input_chars=10)
    with pytest.raises(InputTooLongError) as err:
        svc.validate_input("x" * 100)
    assert err.value.length == 100
    assert err.value.limit == 10
    # 提示要报真实上限，也要指出去哪条正路
    assert "上限 10" in str(err.value)
    assert "data/knowledge" in str(err.value)


def test_send_validates_before_touching_history(tmp_path):
    """send() 内的校验发生在把用户消息加进历史之前。"""
    store = MemoryStore(tmp_path / "s2")
    svc = ChatService(ChatAgent(FakeProvider()), store, max_input_chars=10, summary_keep_turns=0)
    sid = store.create().id
    with pytest.raises(InputTooLongError):
        list(svc.send(sid, "x" * 200))
    assert store.load(sid).conversation.messages == []


def test_default_limit_is_32000(tmp_path):
    svc = make_service(tmp_path)
    svc.validate_input("x" * 32000)                       # 正好等于上限：放行
    with pytest.raises(InputTooLongError):
        svc.validate_input("x" * 32001)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(tmp_path):
    r = TestClient(create_app(make_service(tmp_path, max_input_chars=123))).get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "model" in r.json()
    # 前端从 /health 读输入上限，避免前后端各写一份
    assert r.json()["max_input_chars"] == 123


def test_health_passes_even_when_auth_is_on(tmp_path, monkeypatch):
    """探活请求不能因为缺 token 被挡——否则负载均衡会误判服务挂掉。"""
    monkeypatch.setattr(settings, "auth_token", "secret-token")
    client = TestClient(create_app(make_service(tmp_path)))
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["ok"] is True


def test_health_not_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_rpm", 1)
    client = TestClient(create_app(make_service(tmp_path)))
    for _ in range(10):
        assert client.get("/health").status_code == 200


def test_health_requires_no_session(tmp_path):
    """/health 不该依赖任何会话状态——空库、无会话也要 200。"""
    client = TestClient(create_app(make_service(tmp_path)))
    assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# 认证中间件（默认关闭）
# ---------------------------------------------------------------------------


def test_auth_off_allows_without_token(tmp_path):
    """AUTH_TOKEN 留空 = 完全关闭，行为必须与未安装完全一致。"""
    client = TestClient(create_app(make_service(tmp_path)))
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/messages", json={"text": "你好"})
    assert r.status_code == 200
    assert "你好呀" in r.text


def test_auth_on_rejects_missing_token(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auth_token", "secret-token")
    client = TestClient(create_app(make_service(tmp_path)))
    r = client.post("/api/sessions")
    assert r.status_code == 401
    assert "X-Agent-Token" in r.json()["detail"]


def test_auth_on_rejects_wrong_token(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auth_token", "secret-token")
    client = TestClient(create_app(make_service(tmp_path)))
    assert client.get("/api/sessions", headers={"X-Agent-Token": "wrong-token"}).status_code == 401


def test_auth_on_accepts_correct_token(tmp_path, monkeypatch):
    """开启后功能本身要正确，不只是挡得住。"""
    monkeypatch.setattr(settings, "auth_token", "secret-token")
    client = TestClient(create_app(make_service(tmp_path)))
    h = {"X-Agent-Token": "secret-token"}
    sid = client.post("/api/sessions", headers=h).json()["id"]
    r = client.post(f"/api/sessions/{sid}/messages", json={"text": "你好"}, headers=h)
    assert r.status_code == 200
    assert "你好呀" in r.text


def test_auth_on_blocks_reading_sessions_too(tmp_path, monkeypatch):
    """不只是发消息要认证，读会话也要——否则认证形同虚设。"""
    monkeypatch.setattr(settings, "auth_token", "secret-token")
    client = TestClient(create_app(make_service(tmp_path)))
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions/anything").status_code == 401


def test_auth_on_blocks_delete_too(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "auth_token", "secret-token")
    client = TestClient(create_app(make_service(tmp_path)))
    assert client.delete("/api/sessions/anything").status_code == 401


def test_auth_header_name_is_case_insensitive(tmp_path, monkeypatch):
    """HTTP 头名大小写不敏感，这是协议要求。"""
    monkeypatch.setattr(settings, "auth_token", "secret-token")
    client = TestClient(create_app(make_service(tmp_path)))
    sid = client.post("/api/sessions", headers={"X-Agent-Token": "secret-token"}).json()["id"]
    assert client.get(f"/api/sessions/{sid}", headers={"x-agent-token": "secret-token"}).status_code == 200


def test_auth_off_works_with_any_junk_header(tmp_path):
    """关闭态下带个错的 header 也不能报错——那就不等于没装了。"""
    client = TestClient(create_app(make_service(tmp_path)))
    assert client.get("/api/sessions", headers={"X-Agent-Token": "junk"}).status_code == 200


# ---------------------------------------------------------------------------
# 限流中间件（默认关闭）
# ---------------------------------------------------------------------------


def test_rate_limit_off_never_429(tmp_path):
    """RATE_LIMIT_RPM=0 = 完全关闭，打爆也不该 429。"""
    client = TestClient(create_app(make_service(tmp_path)))
    sid = client.post("/api/sessions").json()["id"]
    for _ in range(50):
        assert client.post(f"/api/sessions/{sid}/messages", json={"text": "hi"}).status_code == 200


def test_rate_limit_on_blocks_after_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_rpm", 3)
    client = TestClient(create_app(make_service(tmp_path)))
    sid = client.post("/api/sessions").json()["id"]          # 第 1 次
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "1"}).status_code == 200  # 2
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "2"}).status_code == 200  # 3
    r = client.post(f"/api/sessions/{sid}/messages", json={"text": "3"})                        # 4
    assert r.status_code == 429
    assert "频繁" in r.json()["detail"]


def test_rate_limited_request_leaves_no_message(tmp_path, monkeypatch):
    """被限流的请求不能留下消息。"""
    monkeypatch.setattr(settings, "rate_limit_rpm", 1)
    store = MemoryStore(tmp_path / "s4")
    svc = ChatService(ChatAgent(FakeProvider()), store, summary_keep_turns=0)
    sid = store.create().id

    client = TestClient(create_app(svc))
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "放行"}).status_code == 200
    assert client.post(f"/api/sessions/{sid}/messages", json={"text": "被拒"}).status_code == 429

    contents = [m.content for m in store.load(sid).conversation.messages if m.role.value == "user"]
    assert contents == ["放行"]


def test_rate_limit_counts_per_ip(tmp_path, monkeypatch):
    """不同来源各算各的额度。"""
    monkeypatch.setattr(settings, "rate_limit_rpm", 1)
    client = TestClient(create_app(make_service(tmp_path)))
    assert client.post("/api/sessions").status_code == 200
    assert client.post("/api/sessions").status_code == 429


# ---------------------------------------------------------------------------
# HTTP 超时：显式传入 httpx，且真的能中止
# ---------------------------------------------------------------------------


def test_provider_passes_timeout_to_httpx(tmp_path):
    from src.providers.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider(
        base_url="https://example.invalid/v1", api_key="sk-test",
        model="test-model", timeout=30.5,
    )
    assert p._client.timeout == httpx.Timeout(connect=5.0, read=30.5, write=30.5, pool=30.5)
    assert p._client.timeout.read == 30.5


def test_default_timeout_is_90s_not_sdk_default(tmp_path):
    """关键：read 不能再是 openai SDK 默认的 600——那意味着本地也能挂 10 分钟。"""
    from src.providers.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider(
        base_url="https://example.invalid/v1", api_key="sk-test", model="test-model",
    )
    assert p._client.timeout.read == 90.0
    assert p._client.timeout.read < 600


def test_bootstrap_passes_configured_timeout(tmp_path, monkeypatch):
    """配置项真能改到 httpx，不是写死的。"""
    import src.bootstrap as boot

    monkeypatch.setattr(boot, "settings", _boot_settings(request_timeout=12.0))
    assert boot.build_provider()._client.timeout.read == 12.0


def test_low_timeout_actually_aborts_a_hang(tmp_path):
    """超时真的生效：服务端迟迟不回字节时，请求在**配置的超时点**被中止。

    对照 openai SDK 的默认 read=600——那意味着本地也要等 10 分钟才发现。
    这里起一个本地 socket 当"假服务端"（只接收请求、然后不回任何字节），
    不联网，只走 127.0.0.1。
    """
    import socket
    import socketserver
    import threading

    class SlowServer(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                self.request.recv(4096)
            except OSError:
                pass
            time.sleep(1.0)              # 读完请求后一个字节都不回
            self.request.close()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), SlowServer)
    server.daemon_threads = True
    server.allow_reuse_address = True
    server.timeout = 0.1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=0.3, write=5.0, pool=5.0))
        start = time.perf_counter()
        with pytest.raises(httpx.ReadTimeout):
            client.get(f"http://127.0.0.1:{port}/v1/chat/completions")
        elapsed = time.perf_counter() - start
    finally:
        server.shutdown()
        server.server_close()

    assert elapsed < 2.0, f"超时没有按时中止，实际 {elapsed:.2f}s"
    assert elapsed >= 0.25, f"中止得太早，实际 {elapsed:.2f}s"
