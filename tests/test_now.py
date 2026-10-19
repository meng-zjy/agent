"""当前时间注入测试。

被测的是一个真实 bug 的修法：

    历史：[2] user「你好 现在的时间是几点」
          [3] assistant「现在是 2026 年 9 月 16 日，晚上 21 点 14 分。」
    ……过了 7 天……
          [16] user「你好 现在是几点」
          [17] assistant「现在是 2026年9月16日 星期三，晚上 21:41。」  ← 复读了旧答案

时间回答会"烂在"会话历史里。模型再被问时，复读历史里那条比重新调工具"便宜"，
于是给出 7 天前的时间。修法：每轮把实时时间注入上下文，让模型面前总有一份新鲜参照。
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.models import Message, Role
from src.core.provider import BaseProvider, Completion
from src.core.service import ChatService, NOW_HINT, format_now


class FakeProvider(BaseProvider):
    def chat(self, messages: list[Message]) -> str:
        return "OK"

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        yield "OK"

    def complete(self, messages: list[Message], tools=None) -> Completion:
        return Completion(finish_reason="stop", content="OK")


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "sessions")


def make_service(store: MemoryStore, **kw) -> ChatService:
    return ChatService(
        ChatAgent(FakeProvider()), store,
        system_prompt="你是助手",
        summary_keep_turns=0,          # 本文件不测摘要
        **kw,
    )


def now_block(ctx: list[Message]) -> Message | None:
    """找出上下文里那块「当前时间」system 消息。"""
    for m in ctx:
        if "[当前时间]" in m.content:
            return m
    return None


# ---------------------------------------------------------------------------
# 1. 注入位置与格式
# ---------------------------------------------------------------------------


def test_now_injected_after_system_prompt(store: MemoryStore):
    """实时时间作为 system 消息，紧跟系统提示词。"""
    service = make_service(store)
    session = store.create()
    session.conversation.add(Role.USER, "你好")

    ctx = service._build_context(session)
    roles = [m.role for m in ctx]
    assert roles == [Role.SYSTEM, Role.SYSTEM, Role.USER]

    block = now_block(ctx)
    assert block is not None
    assert ctx[0].content == "你是助手"
    assert ctx.index(block) == 1
    assert "[当前时间]" in block.content


def test_now_block_is_regenerated_each_turn(monkeypatch, store: MemoryStore):
    """关键：时间是**每轮现算**的，不是启动时定死再复用。

    写死在 system_prompt 里的那份跑一晚上就过期了——所以必须在 _build_context 里按轮生成。
    """
    ticks = iter(["2026-09-16 星期三 21:14:00", "2026-09-23 星期三 10:26:00"])
    monkeypatch.setattr("src.core.service.format_now", lambda: next(ticks))

    service = make_service(store)
    session = store.create()

    block1 = now_block(service._build_context(session))
    block2 = now_block(service._build_context(session))

    assert "2026-09-16" in block1.content
    assert "2026-09-23" in block2.content
    assert block1.content != block2.content


def test_format_now_shape():
    """format_now 输出中文可读的一行：日期 + 星期 + 时间。"""
    date_s, weekday_s, time_s = format_now().split(" ")
    assert len(date_s) == 10                    # YYYY-MM-DD
    assert date_s.count("-") == 2
    assert weekday_s.startswith("星期")
    assert weekday_s[2:] in "一二三四五六日"
    assert len(time_s) == 8                     # HH:MM:SS
    assert time_s.count(":") == 2


def test_now_injection_can_be_disabled(store: MemoryStore):
    """inject_now=False 时不注入，上下文回到只有系统提示词 + 历史。"""
    service = make_service(store, inject_now=False)
    session = store.create()
    session.conversation.add(Role.USER, "你好")

    ctx = service._build_context(session)
    assert [m.role for m in ctx] == [Role.SYSTEM, Role.USER]
    assert now_block(ctx) is None


def test_now_not_persisted_to_history(store: MemoryStore):
    """时间块只进上下文，不落进会话历史（否则它自己也会变成"旧时间"污染源）。"""
    service = make_service(store)
    session = store.create()

    list(service.send(session.id, "现在几点"))

    loaded = store.load(session.id)
    assert not any("[当前时间]" in m.content for m in loaded.conversation.messages)
    assert [m.role for m in loaded.conversation.messages] == [Role.USER, Role.ASSISTANT]


# ---------------------------------------------------------------------------
# 2. 与摘要共存
# ---------------------------------------------------------------------------


def test_now_then_summary_then_history(store: MemoryStore):
    """完整顺序：系统提示词 → 当前时间 → 对话摘要 → 对话历史。"""
    service = ChatService(
        ChatAgent(FakeProvider()), store, system_prompt="你是助手", inject_now=True,
    )
    session = store.create()
    session.summary = "用户叫小明。"
    session.conversation.add(Role.USER, "我叫什么")

    ctx = service._build_context(session)
    roles = [m.role for m in ctx]
    assert roles == [Role.SYSTEM, Role.SYSTEM, Role.SYSTEM, Role.USER]

    assert ctx[0].content == "你是助手"
    assert "[当前时间]" in ctx[1].content
    assert "[对话历史摘要]" in ctx[2].content
    assert ctx[3].role is Role.USER


def test_now_does_not_leak_into_summary_prompt(store: MemoryStore):
    """摘要调用走 provider.chat 的独立 prompt，不会被时间块污染。"""
    from tests.test_summary import RecordingProvider

    provider = RecordingProvider(summary="【摘要】")
    service = ChatService(
        ChatAgent(provider), store,
        system_prompt="你是助手",
        summary_keep_turns=1,          # 开启摘要
        inject_now=True,
    )
    session = store.create()
    for i in range(1, 4):
        list(service.send(session.id, f"第{i}轮"))

    assert len(provider.summaries) >= 1
    prompt = provider.summaries[0][0].content
    assert "[当前时间]" not in prompt
    assert "[对话历史摘要]" not in prompt


# ---------------------------------------------------------------------------
# 3. 回归：历史里躺着旧时间答案时，仍有一份新鲜参照
# ---------------------------------------------------------------------------


def test_stale_time_in_history_gets_fresh_counterpart(store: MemoryStore):
    """回归：历史里有几天前的时间回答时，上下文里仍有一份现算的实时时间。

    真实 bug 复现（会话 20260914-223126-555088）：
      [3]  assistant「现在是 2026 年 9 月 16 日，晚上 21 点 14 分。」
      …… 7 天后 ……
      [16] user「你好 现在是几点」
      [17] assistant「现在是 2026年9月16日 星期三，晚上 21:41。」← 复读了旧答案

    我们不删历史（删了会丢别的记忆），而是让模型面前**总有一份更新的参照**，
    并在提示词里明说"历史里的时间可能是旧答案，不要照抄"。
    """
    service = make_service(store)
    session = store.create()

    session.conversation.add(Role.USER, "你好 现在的时间是几点")
    session.conversation.add(
        Role.ASSISTANT, "现在是 2026 年 9 月 16 日，晚上 21 点 14 分。"
    )
    session.conversation.add(Role.USER, "你好 现在是几点")

    ctx = service._build_context(session)

    block = now_block(ctx)
    assert block is not None, "应该有实时时间块"

    # ① 必须在历史之前——否则模型先看到旧答案，时间块就白注入了
    idx_now = next(i for i, m in enumerate(ctx) if "[当前时间]" in m.content)
    idx_history = next(i for i, m in enumerate(ctx) if m.role is Role.USER)
    assert idx_now < idx_history

    # ② 提示词里明确禁止照抄历史里的时间
    assert "不要照抄" in block.content

    # ③ 历史里的旧答案仍保留（我们没有删历史），模型有能力对照二者差异
    assert any("9 月 16 日" in m.content for m in ctx)


def test_now_block_beats_stale_even_when_trimmed(store: MemoryStore):
    """即使历史被 _trim 截断，时间块仍在（它不在历史里，不受截断影响）。"""
    service = ChatService(
        ChatAgent(FakeProvider()), store,
        system_prompt="你是助手",
        max_turns=1, max_chars=99999,
        summary_keep_turns=0, inject_now=True,
    )
    session = store.create()
    for i in range(1, 6):
        session.conversation.add(Role.USER, f"第{i}轮问题")
        session.conversation.add(Role.ASSISTANT, "回答")

    ctx = service._build_context(session)
    assert now_block(ctx) is not None
    # 历史被截断到最近 1 轮，但时间块安然无恙
    user_msgs = [m for m in ctx if m.role is Role.USER]
    assert len(user_msgs) == 1
