"""记忆系统测试（M2）。

覆盖：
- Message.estimate_tokens()     token 估算
- MemoryStore                   会话增删改查、磁盘持久化
- ChatService.send()            多轮上下文传递
- ChatService._trim()           历史截断（轮数 + 字符）
- ChatService.clear_session()   清空历史
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.models import Message, Role
from src.core.provider import BaseProvider, Completion
from src.core.service import ChatService


# ---------------------------------------------------------------------------
# 共享 Fixture
# ---------------------------------------------------------------------------


class FakeProvider(BaseProvider):
    """记录收到的所有消息列表，每次调用返回预设文本。"""

    def __init__(self, reply: str = "OK") -> None:
        self.calls: list[list[Message]] = []
        self._reply = reply

    def chat(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        return self._reply

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        self.calls.append(list(messages))
        yield self._reply

    def complete(self, messages: list[Message], tools=None) -> Completion:
        self.calls.append(list(messages))
        return Completion(finish_reason="stop", content=self._reply)


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    """每个测试独立的临时会话目录，避免测试间互相污染。"""
    return tmp_path / "sessions"


@pytest.fixture
def store(session_dir: Path) -> MemoryStore:
    return MemoryStore(session_dir)


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider(reply="我知道了")


@pytest.fixture
def service(store: MemoryStore, provider: FakeProvider) -> ChatService:
    return ChatService(
        ChatAgent(provider),
        store,
        system_prompt="你是助手",
        max_turns=5,
        max_chars=500,
        summary_keep_turns=0,    # 本文件只测 M2；摘要见 test_summary.py
        inject_now=False,        # 当前时间注入见 test_now.py
    )


# ---------------------------------------------------------------------------
# 1. Message.estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_cjk():
    """纯中文：每个汉字约 1 token。"""
    msg = Message(role=Role.USER, content="你好世界")   # 4 个 CJK 字符
    assert msg.estimate_tokens() == 4


def test_estimate_tokens_english():
    """纯英文：约 4 字符 1 token，向下取整后至少 1。"""
    msg = Message(role=Role.USER, content="hello")      # 5 字符 -> 5//4 = 1
    assert msg.estimate_tokens() == 1


def test_estimate_tokens_mixed():
    """中英混合：CJK 按 1 计，其余按 4 计。"""
    msg = Message(role=Role.USER, content="你好 world")  # 2 CJK + 6 其它 -> 2 + 6//4 = 3
    assert msg.estimate_tokens() == 3


def test_estimate_tokens_uses_cache():
    """token_count 有缓存时直接返回，不重新计算。"""
    msg = Message(role=Role.USER, content="hello world", token_count=99)
    assert msg.estimate_tokens() == 99


def test_estimate_tokens_empty_content():
    """空消息至少返回 1，避免截断逻辑出现零值。"""
    msg = Message(role=Role.USER, content="")
    assert msg.estimate_tokens() >= 1


# ---------------------------------------------------------------------------
# 2. MemoryStore：会话 CRUD + 持久化
# ---------------------------------------------------------------------------


def test_store_create_returns_session(store: MemoryStore):
    session = store.create()
    assert session.id
    assert session.conversation.messages == []


def test_store_save_and_load(store: MemoryStore):
    """落盘后重新加载，消息内容保持一致。"""
    session = store.create()
    session.conversation.add(Role.USER, "你好")
    session.conversation.add(Role.ASSISTANT, "你好！有什么需要帮助的吗？")
    store.save(session)

    loaded = store.load(session.id)
    assert loaded is not None
    assert len(loaded.conversation.messages) == 2
    assert loaded.conversation.messages[0].content == "你好"
    assert loaded.conversation.messages[1].role == Role.ASSISTANT


def test_store_load_nonexistent_returns_none(store: MemoryStore):
    assert store.load("nonexistent-id") is None


def test_store_list_multiple_sessions(store: MemoryStore):
    """创建多个会话，list() 应全部返回。"""
    ids = {store.create().id for _ in range(3)}
    listed_ids = {s.id for s in store.list()}
    assert ids <= listed_ids


def test_store_list_order(store: MemoryStore):
    """list() 按 updated_at 倒序，最新的在第一位。"""
    s1 = store.create()
    s2 = store.create()
    # 重新 save s1，使其 updated_at 更新
    import time; time.sleep(0.01)
    store.save(s1)

    sessions = store.list()
    assert sessions[0].id == s1.id


def test_store_delete(store: MemoryStore):
    session = store.create()
    store.delete(session.id)
    assert store.load(session.id) is None


def test_store_json_on_disk(store: MemoryStore, session_dir: Path):
    """写盘的文件应是合法 JSON，且包含 id / messages / created_at / updated_at。"""
    session = store.create()
    session.conversation.add(Role.USER, "test")
    store.save(session)

    json_file = session_dir / f"{session.id}.json"
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["id"] == session.id
    assert "messages" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_store_latest(store: MemoryStore):
    """latest() 返回更新时间最新的会话。"""
    s1 = store.create()
    import time; time.sleep(0.01)
    s2 = store.create()
    assert store.latest().id == s2.id


def test_store_corrupt_file_skipped(session_dir: Path):
    """损坏的 JSON 文件不应阻塞 list()，直接跳过。"""
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "bad.json").write_text("not json at all", encoding="utf-8")
    store = MemoryStore(session_dir)
    sessions = store.list()  # 不应抛异常
    assert all(s.id != "bad" for s in sessions)


# ---------------------------------------------------------------------------
# 3. ChatService：多轮对话 + 截断 + 清空
# ---------------------------------------------------------------------------


def test_service_send_single_turn(
    service: ChatService, store: MemoryStore, provider: FakeProvider
):
    """第一轮对话：模型回复被追加入历史。"""
    session = store.create()
    result = "".join(service.send(session.id, "你好"))

    assert result == "我知道了"
    loaded = store.load(session.id)
    # system prompt 不写入历史，只写用户问题 + 模型回复
    assert len(loaded.conversation.messages) == 2
    assert loaded.conversation.messages[0].role == Role.USER
    assert loaded.conversation.messages[1].role == Role.ASSISTANT


def test_service_send_multi_turn_context(
    service: ChatService, store: MemoryStore, provider: FakeProvider
):
    """多轮对话时，模型收到的上下文包含完整历史（含 system prompt）。"""
    session = store.create()
    list(service.send(session.id, "第一轮"))
    list(service.send(session.id, "第二轮"))

    # 第二轮：provider 收到 [system, user1, assistant1, user2]
    second_call = provider.calls[1]
    roles = [m.role for m in second_call]
    assert roles[0] == Role.SYSTEM
    assert roles[1] == Role.USER
    assert roles[2] == Role.ASSISTANT
    assert roles[3] == Role.USER


def test_service_remembers_history_across_reload(
    service: ChatService,
    store: MemoryStore,
    provider: FakeProvider,
    session_dir: Path,
):
    """模拟重启：重新创建 ChatService，加载同一会话，历史仍然保留。"""
    session = store.create()
    list(service.send(session.id, "我叫小明"))

    # 重新创建 Service（模拟程序重启）
    new_store = MemoryStore(session_dir)
    new_provider = FakeProvider(reply="你叫小明")
    new_service = ChatService(
        ChatAgent(new_provider), new_store, system_prompt="你是助手"
    )

    list(new_service.send(session.id, "我叫什么"))
    context = new_provider.calls[0]
    contents = [m.content for m in context]
    assert "我叫小明" in contents


def test_service_trim_by_turns(session_dir: Path):
    """超出 max_turns 时，最旧的轮次被截断。"""
    store = MemoryStore(session_dir)
    provider = FakeProvider(reply="r")
    service = ChatService(
        ChatAgent(provider), store,
        max_turns=2, max_chars=99999, summary_keep_turns=0,
    )

    session = store.create()
    for msg in ["第1轮", "第2轮", "第3轮"]:
        list(service.send(session.id, msg))

    last_call = provider.calls[-1]
    user_messages = [m for m in last_call if m.role == Role.USER]
    assert len(user_messages) <= 2


def test_service_trim_by_chars(session_dir: Path):
    """总字符数超出 max_chars 时，最旧消息对被截断。"""
    store = MemoryStore(session_dir)
    provider = FakeProvider(reply="r" * 100)
    service = ChatService(
        ChatAgent(provider), store,
        max_turns=999, max_chars=50, summary_keep_turns=0,
    )

    session = store.create()
    for msg in ["A" * 30, "B" * 30]:
        list(service.send(session.id, msg))

    last_call = provider.calls[-1]
    user_messages = [m for m in last_call if m.role == Role.USER]
    assert len(user_messages) >= 1


def test_service_clear_session(service: ChatService, store: MemoryStore):
    """clear_session 后历史清空为空列表。"""
    session = store.create()
    list(service.send(session.id, "你好"))

    service.clear_session(session.id)
    loaded = store.load(session.id)
    assert loaded.conversation.messages == []


def test_service_send_nonexistent_session(service: ChatService):
    """向不存在的会话 id 发消息，应抛出 ValueError。"""
    with pytest.raises(ValueError, match="会话不存在"):
        list(service.send("ghost-id-000", "hello"))


def test_service_new_session(service: ChatService, store: MemoryStore):
    """new_session() 创建新会话并落盘。"""
    session = service.new_session()
    assert store.load(session.id) is not None


def test_service_list_sessions(service: ChatService, store: MemoryStore):
    """list_sessions() 返回所有持久化会话。"""
    store.create()
    store.create()
    sessions = service.list_sessions()
    assert len(sessions) >= 2


def test_service_resume_or_create_returns_non_empty(
    service: ChatService, store: MemoryStore
):
    """resume_or_create：有非空会话时，返回最近的那个。"""
    session = store.create()
    session.conversation.add(Role.USER, "hi")
    store.save(session)

    resumed = service.resume_or_create()
    assert resumed.id == session.id
