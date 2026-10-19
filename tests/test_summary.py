"""长对话摘要测试（09 Summary Memory）。

用假 provider，不花额度。核心验证：
- 摘要字段能持久化（重启还在），旧版 JSON 无该字段也能读
- 超过 summary_keep_turns 轮才触发，没超不动
- 滚动摘要：新摘要 = 总结「旧摘要 + 新溢出的旧消息」
- 注入位置：system prompt 之后、对话历史之前
- 摘要失败不影响主对话（旧消息保留、下轮再试）
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.core.agent import ChatAgent
from src.core.memory import MemoryStore, Session
from src.core.models import Message, Role
from src.core.provider import BaseProvider, Completion
from src.core.service import ChatService


class RecordingProvider(BaseProvider):
    """分开记录「回复调用」和「摘要调用」，方便断言两者各自的输入。"""

    def __init__(self, reply: str = "我知道了", summary: str = "【摘要】",
                 fail_chat: bool = False, empty_chat: bool = False) -> None:
        self._reply = reply
        self._summary = summary
        self._fail_chat = fail_chat
        self._empty_chat = empty_chat
        self.replies: list[list[Message]] = []    # chat_stream：真正的回复
        self.summaries: list[list[Message]] = []  # chat：摘要调用

    def chat(self, messages: list[Message]) -> str:
        self.summaries.append(list(messages))
        if self._fail_chat:
            raise RuntimeError("LLM 挂了")
        if self._empty_chat:
            return ""
        return self._summary

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        self.replies.append(list(messages))
        yield self._reply

    def complete(self, messages: list[Message], tools=None) -> Completion:
        self.replies.append(list(messages))
        return Completion(finish_reason="stop", content=self._reply)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "sessions")


def make_service(store: MemoryStore, provider: RecordingProvider, **kw) -> ChatService:
    """构造一个「关掉摘要和当前时间注入」的服务，需要的测试再覆盖参数。"""
    return ChatService(
        ChatAgent(provider), store,
        system_prompt="你是助手",
        summary_keep_turns=kw.pop("summary_keep_turns", 0),
        summary_max_chars=kw.pop("summary_max_chars", 100),
        inject_now=kw.pop("inject_now", False),
        **kw,
    )


def chat_rounds(service: ChatService, sid: str, n: int, prefix: str = "第") -> None:
    """连聊 n 轮。"""
    for i in range(1, n + 1):
        list(service.send(sid, f"{prefix}{i}轮"))


# ---------------------------------------------------------------------------
# 1. 数据模型：summary 字段的存取
# ---------------------------------------------------------------------------


def test_session_summary_roundtrip(tmp_path: Path):
    """摘要能落盘再读回。"""
    store = MemoryStore(tmp_path)
    session = store.create()
    session.summary = "用户叫小明，喜欢吃辣。"
    store.save(session)

    loaded = store.load(session.id)
    assert loaded is not None
    assert loaded.summary == "用户叫小明，喜欢吃辣。"


def test_session_json_contains_summary(tmp_path: Path):
    """磁盘 JSON 里确实多了 summary 字段。"""
    store = MemoryStore(tmp_path)
    session = store.create()
    session.summary = "摘要内容"
    store.save(session)

    data = json.loads((tmp_path / f"{session.id}.json").read_text(encoding="utf-8"))
    assert data["summary"] == "摘要内容"


def test_old_json_without_summary_is_loadable(tmp_path: Path):
    """旧版本写的 JSON（没有 summary 字段）仍能加载，按"无摘要"处理。"""
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "id": "legacy",
        "created_at": "2026-01-01 00:00:00.000000",
        "updated_at": "2026-01-01 00:00:00.000000",
        "messages": [{"role": "user", "content": "你好"}],
    }), encoding="utf-8")

    loaded = Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.summary == ""
    assert len(loaded.conversation.messages) == 1


def test_empty_summary_means_never_overflowed(store: MemoryStore):
    """新建会话默认无摘要。"""
    assert store.create().summary == ""


# ---------------------------------------------------------------------------
# 2. 触发条件
# ---------------------------------------------------------------------------


def test_no_summary_under_threshold(store: MemoryStore):
    """没超过阈值时不触发摘要，也不额外调用 LLM。"""
    provider = RecordingProvider()
    service = make_service(store, provider, summary_keep_turns=3)

    session = store.create()
    chat_rounds(service, session.id, 3)

    assert provider.summaries == []          # 摘要调用一次都没发生
    assert store.load(session.id).summary == ""
    assert len(store.load(session.id).conversation.messages) == 6   # 3 轮全保留


def test_summary_triggers_over_threshold(store: MemoryStore):
    """超过阈值后：旧消息被压成摘要，最近 N 轮保留。"""
    provider = RecordingProvider(summary="【摘要】第1轮的要点")
    service = make_service(store, provider, summary_keep_turns=2)

    session = store.create()
    chat_rounds(service, session.id, 3)

    loaded = store.load(session.id)
    assert len(provider.summaries) == 1
    assert loaded.summary == "【摘要】第1轮的要点"
    # 第 1 轮被压缩掉了，只剩最近 2 轮（4 条消息）
    assert len(loaded.conversation.messages) == 4
    user_msgs = [m.content for m in loaded.conversation.messages if m.role is Role.USER]
    assert user_msgs == ["第2轮", "第3轮"]


def test_summary_rolling_uses_existing_summary(store: MemoryStore):
    """滚动摘要：第二次触发时，prompt 同时含旧摘要和新溢出的消息。"""
    provider = RecordingProvider(summary="【旧摘要】")
    service = make_service(store, provider, summary_keep_turns=1)

    session = store.create()
    chat_rounds(service, session.id, 3)

    assert len(provider.summaries) == 2
    second_prompt = provider.summaries[1][0].content
    assert "【旧摘要】" in second_prompt        # 带上一次的结果
    assert "第2轮" in second_prompt              # 加这次新溢出的消息
    assert "第1轮" not in second_prompt          # 第1轮已经在旧摘要里，不重复


def test_short_conversation_unchanged(store: MemoryStore):
    """短对话行为完全不变：上下文里没有摘要消息。"""
    provider = RecordingProvider()
    service = make_service(store, provider, summary_keep_turns=20)

    session = store.create()
    list(service.send(session.id, "你好"))

    context = provider.replies[-1]
    assert [m.role for m in context] == [Role.SYSTEM, Role.USER]
    assert not any("[对话历史摘要]" in m.content for m in context)


# ---------------------------------------------------------------------------
# 3. 注入位置
# ---------------------------------------------------------------------------


def test_summary_injected_after_system_prompt(store: MemoryStore):
    """摘要作为 system 消息，插在系统提示词之后、对话历史之前。"""
    provider = RecordingProvider()
    service = make_service(store, provider)

    session = store.create()
    session.summary = "用户叫小明。"
    session.conversation.add(Role.USER, "我叫什么")

    context = service._build_context(session)
    roles = [m.role for m in context]
    assert roles == [Role.SYSTEM, Role.SYSTEM, Role.USER]
    assert context[0].content == "你是助手"
    assert "[对话历史摘要]" in context[1].content
    assert "用户叫小明" in context[1].content


def test_summary_not_persisted_as_history(store: MemoryStore):
    """摘要只进上下文，不落进会话历史（历史里不应出现摘要消息）。"""
    provider = RecordingProvider(summary="【摘要】")
    service = make_service(store, provider, summary_keep_turns=1)

    session = store.create()
    chat_rounds(service, session.id, 3)

    loaded = store.load(session.id)
    assert loaded.summary == "【摘要】"
    assert not any(m.role is Role.SYSTEM for m in loaded.conversation.messages)


def test_summary_persists_across_reload(tmp_path: Path):
    """重启后摘要仍在（模拟程序重启）。"""
    store = MemoryStore(tmp_path)
    provider = RecordingProvider(summary="用户叫小明，喜欢吃辣")
    service = make_service(store, provider, summary_keep_turns=2)

    session = store.create()
    chat_rounds(service, session.id, 4)

    fresh_store = MemoryStore(tmp_path)
    assert fresh_store.load(session.id).summary == "用户叫小明，喜欢吃辣"


def test_summary_max_chars_truncates(store: MemoryStore):
    """LLM 输出过长的摘要会被截断到上限。"""
    provider = RecordingProvider(summary="摘" * 500)
    service = make_service(store, provider, summary_keep_turns=1, summary_max_chars=100)

    session = store.create()
    chat_rounds(service, session.id, 3)

    loaded = store.load(session.id)
    assert len(loaded.summary) == 100


# ---------------------------------------------------------------------------
# 4. 失败隔离：摘要失败绝不影响主对话
# ---------------------------------------------------------------------------


def test_summary_failure_keeps_turn_intact(store: MemoryStore):
    """摘要 LLM 调用失败：回复照常落盘，旧消息不丢，下次再试。"""
    provider = RecordingProvider(reply="回复正常", fail_chat=True)
    service = make_service(store, provider, summary_keep_turns=1)

    session = store.create()
    result = "".join(service.send(session.id, "第1轮"))

    assert result == "回复正常"                  # 主回复不受影响
    loaded = store.load(session.id)
    assert len(loaded.conversation.messages) == 2   # 没有因为失败而误删
    assert loaded.summary == ""


def test_summary_failure_then_retry(store: MemoryStore):
    """这一轮摘要失败，下一轮还能重试并成功。"""
    provider = RecordingProvider(reply="OK", fail_chat=True)
    service = make_service(store, provider, summary_keep_turns=1)
    session = store.create()

    chat_rounds(service, session.id, 2)
    assert store.load(session.id).summary == ""     # 第一次失败

    provider._fail_chat = False
    list(service.send(session.id, "第3轮"))
    assert store.load(session.id).summary == "【摘要】"   # 重试成功


def test_empty_summary_result_skips_compression(store: MemoryStore):
    """LLM 返回空摘要：不压缩历史，避免"压了个寂寞"造成信息丢失。"""
    provider = RecordingProvider(reply="OK", empty_chat=True)
    service = make_service(store, provider, summary_keep_turns=1)

    session = store.create()
    chat_rounds(service, session.id, 3)

    loaded = store.load(session.id)
    assert loaded.summary == ""
    assert len(loaded.conversation.messages) == 6     # 3 轮原样保留


# ---------------------------------------------------------------------------
# 5. 与其他能力共存
# ---------------------------------------------------------------------------


def test_summary_coexists_with_tools(store: MemoryStore):
    """摘要 + 工具调用：最终回复走 complete()，摘要走 chat()，互不干扰。"""
    from src.core.tools import Tool, ToolRegistry

    registry = ToolRegistry()
    registry.register(Tool(
        name="calculator",
        description="计算",
        parameters={"type": "object", "properties": {}, "required": []},
        func=lambda: "42",
    ))
    provider = RecordingProvider(summary="【摘要】")
    service = ChatService(
        ChatAgent(provider, tools=registry), store,
        system_prompt="你是助手",
        summary_keep_turns=1,
    )

    session = store.create()
    chat_rounds(service, session.id, 3)

    loaded = store.load(session.id)
    assert loaded.summary == "【摘要】"
    # 回复走 complete（工具路径），摘要走 chat，两者分开记
    assert len(provider.replies) == 3
    assert len(provider.summaries) == 2


def test_multi_image_messages_do_not_break_summary(store: MemoryStore):
    """带图片的历史消息进入摘要时，只取文字，不报错。"""
    provider = RecordingProvider(reply="看到了", summary="【摘要】用户发过一张图")
    service = make_service(store, provider, summary_keep_turns=1)

    session = store.create()
    list(service.send(session.id, "这是什么", images=["data:image/png;base64,AAA"]))
    list(service.send(session.id, "第2轮"))
    list(service.send(session.id, "第3轮"))

    assert store.load(session.id).summary == "【摘要】用户发过一张图"
