"""API 层测试（Web 化）：用假 provider，不花额度、不加载 BGE 模型。"""

import base64
from collections.abc import Iterator

from fastapi.testclient import TestClient

from src.config import settings

from src.api import create_app
from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.models import Message
from src.core.provider import BaseProvider, Completion, ToolCall
from src.core.service import ChatService


class FakeProvider(BaseProvider):
    """每次返回固定回复。"""

    def __init__(self, reply: str = "你好呀") -> None:
        self._reply = reply

    def chat(self, messages: list[Message]) -> str:
        return self._reply

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        yield self._reply

    def complete(self, messages: list[Message], tools=None) -> Completion:
        return Completion(finish_reason="stop", content=self._reply)


def make_client(tmp_path, **service_options) -> TestClient:
    service = ChatService(
        ChatAgent(FakeProvider()),
        MemoryStore(tmp_path),
        system_prompt="你是助手",
        **service_options,
    )
    return TestClient(create_app(service))


def test_index_serves_html(tmp_path):
    r = make_client(tmp_path).get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_list_sessions_empty(tmp_path):
    r = make_client(tmp_path).get("/api/sessions")
    assert r.status_code == 200
    assert r.json() == []


def test_create_and_get_session(tmp_path):
    client = make_client(tmp_path)
    sid = client.post("/api/sessions").json()["id"]
    r = client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["id"] == sid


def test_get_session_404(tmp_path):
    assert make_client(tmp_path).get("/api/sessions/ghost").status_code == 404


def test_send_message_streams_reply(tmp_path):
    client = make_client(tmp_path)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/messages", json={"text": "你好"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert "你好呀" in body          # 假 provider 的回复以 delta 事件出现
    assert "[DONE]" in body


def test_send_message_emits_status_event_before_tool_round(tmp_path):
    """工具调用会先发一条 status 事件，前端用它替换静止的「思考中…」。"""
    from src.core.tools import build_default_tools

    class ToolProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        def complete(self, messages, tools=None):
            # 按调用计数判定，不要数 messages：service 会先塞一条 system prompt
            self._calls += 1
            if self._calls == 1:
                return Completion(finish_reason="tool_calls",
                                  tool_calls=[ToolCall("c1", "calculator",
                                                       '{"expression": "2+3"}')])
            return Completion(finish_reason="stop", content="结果是 5")

    service = ChatService(
        ChatAgent(ToolProvider(), tools=build_default_tools()),
        MemoryStore(tmp_path),
        system_prompt="你是助手",
        summary_keep_turns=0,
    )
    client = TestClient(create_app(service))
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/messages", json={"text": "2+3"})
    body = r.text

    assert '"status"' in body              # 有独立的 status 事件，不是伪装成 delta
    assert '"delta"' in body
    assert "正在计算" in body              # 中文未转义（ensure_ascii=False）
    # status 必须排在第一个 delta 之前——工具执行前才发，顺序反了就没意义
    assert body.index('"status"') < body.index('"delta"')
    # 而且不能落进历史：否则下次把历史发给模型时就是一段假的「正在计算…」
    assert client.get(f"/api/sessions/{sid}").json()["messages"][-1]["content"] == "结果是 5"


def test_send_message_404(tmp_path):
    r = make_client(tmp_path).post("/api/sessions/ghost/messages", json={"text": "hi"})
    assert r.status_code == 404


def test_send_message_persists(tmp_path):
    client = make_client(tmp_path)
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/messages", json={"text": "你好"})
    messages = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_send_message_rejects_unsupported_image_type(tmp_path):
    """服务端不接受 SVG 等可嵌入脚本的图片类型，不能只靠浏览器 accept 属性。"""
    client = make_client(tmp_path)
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(
        f"/api/sessions/{sid}/messages",
        json={"text": "看图", "images": ["data:image/svg+xml;base64,PHN2Zz4="]},
    )
    assert r.status_code == 400
    assert "仅支持" in r.json()["detail"]


def test_send_message_rejects_oversized_decoded_image(tmp_path):
    """限制按解码后字节算，不能被 base64 长度或伪造 MIME 绕过。"""
    client = make_client(tmp_path, max_image_bytes=8)
    sid = client.post("/api/sessions").json()["id"]
    encoded = base64.b64encode(b"too-large").decode()
    r = client.post(
        f"/api/sessions/{sid}/messages",
        json={"text": "看图", "images": [f"data:image/png;base64,{encoded}"]},
    )
    assert r.status_code == 400
    assert "不能超过" in r.json()["detail"]


def test_send_message_rejects_forged_image_mime_type(tmp_path):
    """data URI 自称 PNG 但内容不是 PNG 时，不能进入模型请求。"""
    client = make_client(tmp_path)
    sid = client.post("/api/sessions").json()["id"]
    encoded = base64.b64encode(b"not a png").decode()
    r = client.post(
        f"/api/sessions/{sid}/messages",
        json={"text": "看图", "images": [f"data:image/png;base64,{encoded}"]},
    )
    assert r.status_code == 400
    assert "不匹配" in r.json()["detail"]


def test_health_reports_image_limits_from_service(tmp_path):
    client = make_client(tmp_path, max_images_per_message=2, max_image_bytes=1234)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["max_images_per_message"] == 2
    assert health.json()["max_image_bytes"] == 1234


def test_clear_session(tmp_path):
    client = make_client(tmp_path)
    sid = client.post("/api/sessions").json()["id"]
    client.post(f"/api/sessions/{sid}/messages", json={"text": "你好"})
    r = client.post(f"/api/sessions/{sid}/clear")
    assert r.json()["messages"] == []


def test_delete_session(tmp_path):
    client = make_client(tmp_path)
    sid = client.post("/api/sessions").json()["id"]
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_accounts_require_invite_and_isolate_sessions(tmp_path, monkeypatch):
    """所有权必须在服务端检查，第二个用户不能探测或读取第一个用户的会话。"""
    monkeypatch.setattr(settings, "account_enabled", True)
    monkeypatch.setattr(settings, "registration_invite_code", "test-invite-code")
    monkeypatch.setattr(settings, "auth_database", str(tmp_path / "auth.db"))
    service = ChatService(ChatAgent(FakeProvider()), MemoryStore(tmp_path / "sessions"))
    app = create_app(service)
    anonymous = TestClient(app, base_url="https://testserver")
    assert anonymous.get("/api/sessions").status_code == 401

    alice = TestClient(app, base_url="https://testserver")
    wrong = alice.post("/api/auth/register", json={
        "username": "alice", "password": "password-123", "invite_code": "wrong",
    })
    assert wrong.status_code == 400
    assert alice.post("/api/auth/register", json={
        "username": "alice", "password": "password-123", "invite_code": "test-invite-code",
    }).status_code == 200
    session_id = alice.post("/api/sessions").json()["id"]

    bob = TestClient(app, base_url="https://testserver")
    assert bob.post("/api/auth/register", json={
        "username": "bob", "password": "password-456", "invite_code": "test-invite-code",
    }).status_code == 200
    assert bob.get("/api/sessions").json() == []
    assert bob.get(f"/api/sessions/{session_id}").status_code == 404
    assert bob.delete(f"/api/sessions/{session_id}").status_code == 404


# ---------------------------------------------------------------------------
# 并发：会话忙时，前端要收到 error 事件而不是写坏历史
# ---------------------------------------------------------------------------


def test_send_busy_streams_error_event(tmp_path):
    """会话被占用且等不及：SSE 流里给出 error 事件，且没留下任何消息。

    前端靠这个事件显示"该会话正在生成回复，请稍候再发"。
    """
    import threading
    from src.core.service import ChatService

    service = ChatService(
        ChatAgent(FakeProvider()),
        MemoryStore(tmp_path),
        system_prompt="你是助手",
        busy_timeout=0,
        housekeeping_timeout=0,
    )
    client = TestClient(create_app(service))

    sid = client.post("/api/sessions").json()["id"]
    lock = service._lock_for(sid)
    released = threading.Event()

    def hold():
        assert lock.acquire(timeout=5)
        released.wait(10)
        lock.release()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()

    try:
        r = client.post(f"/api/sessions/{sid}/messages", json={"text": "抢进来"})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = r.text
        assert "正在生成回复" in body
        assert "error" in body
        assert "[DONE]" in body
    finally:
        released.set()
        holder.join()

    assert client.get(f"/api/sessions/{sid}").json()["messages"] == []


def test_clear_busy_returns_conflict(tmp_path):
    """会话忙时点清空：返回 409 + 可读原因，而不是静默卡住或 500。"""
    import threading
    from src.core.service import ChatService

    service = ChatService(
        ChatAgent(FakeProvider()),
        MemoryStore(tmp_path),
        system_prompt="你是助手",
        housekeeping_timeout=0,
    )
    client = TestClient(create_app(service))

    sid = client.post("/api/sessions").json()["id"]
    lock = service._lock_for(sid)
    released = threading.Event()

    def hold():
        assert lock.acquire(timeout=5)
        released.wait(10)
        lock.release()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    try:
        r = client.post(f"/api/sessions/{sid}/clear")
        assert r.status_code == 409
        assert "正在生成回复" in r.json()["detail"]
    finally:
        released.set()
        holder.join()

