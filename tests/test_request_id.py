"""全链路 request_id 测试（M9 第一位）。

验收点（docs/13 §8）：
- 一次请求的**全部**日志行共享同一个 req_id，`grep <id>` 就是完整轨迹
- **provider 层的空回复日志带上 req_id**——改之前它完全无 id，这是关键验收点
- 工具调用耗时可拆（能看出是检索慢还是模型慢）
- 跨请求、跨会话不串；请求结束后上下文被清掉
- service 的消息里不再自带 `id=`（交给 Formatter）
- file / console 两个 handler 都挂了 filter（漏一个那个出口就全是 `-`）

全部不花额度、不联网。
"""

import logging
import re

import pytest

import src.core.logger as logger_mod
from src.config import settings
from src.core.logger import (
    RequestContextFilter,
    get_logger,
    reset,
    reset_request_context,
    set_request_context,
)
from src.core.provider import Completion, StreamChunk, ToolCall


# 约定格式：2026-10-09 14:02:11 INFO  3f9a2b1c8d4e [20260908-A] agent.service: 请求开始 输入=32字
REQ_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (\w+)\s+(\S+) \[([^\]]*)\] (\S+): (.*)$"
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """独立日志目录 + 独立初始化状态，测完清理。"""
    monkeypatch.setattr(settings, "log_dir", str(tmp_path / "logs"))
    reset()
    yield
    reset()


def _log_text(tmp_path) -> str:
    p = tmp_path / "logs" / "agent.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _flush() -> None:
    for h in logging.getLogger().handlers:
        h.flush()


def _entries(tmp_path) -> list[dict]:
    """把日志文件解析成 {level, req_id, sess_id, logger, message} 列表。"""
    out = []
    for line in _log_text(tmp_path).splitlines():
        m = REQ_RE.match(line)
        assert m, f"这行不符合约定格式，说明有出口漏挂了 filter：{line!r}"
        out.append({
            "level": m[1], "req_id": m[2], "sess_id": m[3],
            "logger": m[4], "message": m[5],
        })
    return out


# ---------------------------------------------------------------------------
# 机制本身：格式 / 默认值
# ---------------------------------------------------------------------------


def test_filter_injects_ids_into_record():
    """Filter 直接往记录上写 req_id / sess_id——这是整个方案的核心机制。"""
    get_logger("ctx")                     # 触发 setup
    log = get_logger("ctx")

    assert logger_mod._request_ctx.get() is None       # 默认没处在请求里

    token = set_request_context({"req": "abcdef123456", "sess": "sess-A"})
    try:
        rec = log.makeRecord("agent.ctx", logging.INFO, "file.py", 1, "任意消息", (), None)
        RequestContextFilter().filter(rec)
        assert rec.req_id == "abcdef123456"
        assert rec.sess_id == "sess-A"
    finally:
        reset_request_context(token)


def test_outside_request_ids_are_dash(tmp_path):
    """不在请求里时填 `-`，让格式对齐而不是空着。"""
    get_logger("dash")
    get_logger("dash").info("启动阶段日志")
    _flush()

    entries = _entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["req_id"] == "-"
    assert entries[0]["sess_id"] == "-"


def test_message_body_no_longer_carries_id(tmp_path):
    """id 由 Formatter 注入，service 的消息里不该再写一遍 `id=`。

    历史上 service.py 的日志是 `请求开始 id=20260908-A 输入=32字`——那个 `id=`
    传的是 session_id，是这次要修掉的洞的来源。
    """
    store, svc = _make_service(tmp_path)
    sid = store.create().id
    list(svc.send(sid, "你好"))
    _flush()

    msgs = [e["message"] for e in _entries(tmp_path)]
    assert any(m.startswith("请求开始") for m in msgs)
    assert not any("id=" in m for m in msgs), "id 不该再写进消息体里"


# ---------------------------------------------------------------------------
# 真正的验收：一次请求跨四层日志，同一个 id
# ---------------------------------------------------------------------------


class _LoggingProvider:
    """在 provider 层打一行日志，但它自己**不写任何 id**。

    这是本方案的核心收益点：调用方一行不用改，id 自动带上。
    """

    def __init__(self, reply: str = "好的") -> None:
        self._reply = reply

    def chat(self, messages) -> str:
        return self._reply

    def chat_stream(self, messages):
        get_logger("provider").info("模型响应完成 回复=%d字", len(self._reply))
        yield self._reply

    def complete(self, messages, tools=None):
        return Completion(finish_reason="stop", content=self._reply)


def _make_service(tmp_path, **kw):
    from src.core.agent import ChatAgent
    from src.core.memory import MemoryStore
    from src.core.service import ChatService

    store = MemoryStore(tmp_path / "sessions")
    svc = ChatService(
        ChatAgent(_LoggingProvider(), tools=kw.pop("tools", None)),
        store,
        system_prompt="你是助手",
        summary_keep_turns=kw.pop("summary_keep_turns", 0),
        **kw,
    )
    return store, svc


def test_one_request_one_id_across_all_layers(tmp_path):
    get_logger("one")
    store, svc = _make_service(tmp_path)
    sid = store.create().id
    list(svc.send(sid, "你好"))
    _flush()

    entries = _entries(tmp_path)
    reqs = {e["req_id"] for e in entries}
    assert len(reqs) == 1, f"一次请求出现了多个 id：{reqs}"
    req_id = reqs.pop()
    assert len(req_id) == 12                      # uuid4 前 12 位 hex
    assert req_id != "-"

    assert all(e["sess_id"] == sid for e in entries)
    # service 和 provider 两层的日志必须都在
    assert "agent.service" in {e["logger"] for e in entries}
    assert "agent.provider" in {e["logger"] for e in entries}


def test_consecutive_requests_get_distinct_ids(tmp_path):
    get_logger("two")
    store, svc = _make_service(tmp_path)
    sid = store.create().id
    list(svc.send(sid, "第一条"))
    list(svc.send(sid, "第二条"))
    _flush()

    ids = [e["req_id"] for e in _entries(tmp_path) if e["req_id"] != "-"]
    assert len(ids) == 6                          # 每条请求 3 行（开始 / provider / 结束）
    assert len(set(ids)) == 2, "跨请求串了 id"


def test_two_sessions_interleaved_do_not_cross(tmp_path):
    """两个会话交错请求：各自的 id 不会串。

    这是 req_id 和 session_id 必须分开的理由——M7 让不同会话真并行之后，
    交错日志靠 session_id 已经分不清是谁了。
    """
    get_logger("cross")
    store, svc = _make_service(tmp_path)
    sid_a, sid_b = store.create().id, store.create().id

    list(svc.send(sid_a, "A 的第一条"))
    list(svc.send(sid_b, "B 的第一条"))
    list(svc.send(sid_a, "A 的第二条"))
    _flush()

    by_a = {e["req_id"] for e in _entries(tmp_path) if e["sess_id"] == sid_a} - {"-"}
    by_b = {e["req_id"] for e in _entries(tmp_path) if e["sess_id"] == sid_b} - {"-"}
    assert len(by_a) == 2 and len(by_b) == 1
    assert not (by_a & by_b), "两个会话的 request_id 串了"


def test_context_cleared_after_request(tmp_path):
    """请求结束后上下文必须被清掉，不能挂在 task 上污染下一次调用。"""
    get_logger("clear")
    store, svc = _make_service(tmp_path)
    sid = store.create().id
    list(svc.send(sid, "你好"))
    assert logger_mod._request_ctx.get() is None


def test_context_cleared_when_request_is_abandoned(tmp_path):
    """生成器被提前关掉（客户端断开）时 `finally` 也要执行。

    `send()` 是生成器，可能被挂起甚至被 GC 掉，`with.__exit__` 在那种情况下
    去向不保证——所以用 try/finally。
    """
    get_logger("abandon")
    store, svc = _make_service(tmp_path)
    sid = store.create().id

    gen = svc.send(sid, "你好")
    next(gen)                                    # 消费第一条 yield 后丢弃
    gen.close()
    assert logger_mod._request_ctx.get() is None


# ---------------------------------------------------------------------------
# 关键验收点：provider 层的空回复日志带上 req_id
# ---------------------------------------------------------------------------


def test_provider_empty_reply_log_carries_req_id(tmp_path):
    """改之前 provider 的 `空回复` 日志**完全没有 id**——归属不到谁身上。

    这里走的是**真的** OpenAICompatProvider（不联网，只调它的内部方法），
    不是假 provider 里自己写一行——后者证明不了"调用方不用改"这件事。
    """
    from src.core.provider import EmptyReplyError
    from src.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(
        base_url="https://example.invalid/v1", api_key="sk-test", model="test-model",
    )
    get_logger("prov")

    token = set_request_context({"req": "deadbeefcafe", "sess": "sess-empty"})
    try:
        with pytest.raises(EmptyReplyError):
            provider._reject_empty(max_tokens=4096, usage=None)
    finally:
        reset_request_context(token)
    _flush()

    text = _log_text(tmp_path)
    empty_lines = [ln for ln in text.splitlines() if "空回复" in ln]
    assert len(empty_lines) == 1, empty_lines
    assert "deadbeefcafe" in empty_lines[0]
    assert "[sess-empty]" in empty_lines[0]
    assert all(e["req_id"] == "deadbeefcafe" for e in _entries(tmp_path))


# ---------------------------------------------------------------------------
# 工具调用打点：能拆开"检索慢"还是"模型慢"
# ---------------------------------------------------------------------------


class _ToolCallingProvider(_LoggingProvider):
    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def complete(self, messages, tools=None):
        self._calls += 1
        if self._calls == 1:
            return Completion(
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall("c1", "search_knowledge_base", '{"query": "q"}'),
                    ToolCall("c2", "calculator", '{"expression": "1+1"}'),
                ],
            )
        return Completion(finish_reason="stop", content="最终答案")

    def complete_stream(self, messages, tools=None):
        # 只实现非流式，这里退化成一次性吐完——和 BaseProvider 的默认实现同构，
        # 所以这两条用例仍然是在验证「工具调用走 complete()」这条链路。
        result = self.complete(messages, tools=tools)
        if result.tool_calls:
            yield StreamChunk(finish_reason="tool_calls", tool_calls=result.tool_calls)
        else:
            yield StreamChunk(text=result.content, finish_reason=result.finish_reason)


def test_tool_calls_are_timed(tmp_path):
    """工具耗时可拆——否则 RAG 慢还是模型慢看不出（docs/13 §2.4）。"""
    from src.core.agent import ChatAgent
    from src.core.memory import MemoryStore
    from src.core.service import ChatService
    from src.core.tools import Tool, ToolRegistry

    reg = ToolRegistry()
    reg.register(Tool("search_knowledge_base", "x", {}, lambda query: "检索结果 " + query))
    reg.register(Tool("calculator", "x", {}, lambda expression: "2"))

    store = MemoryStore(tmp_path / "sessions")
    svc = ChatService(
        ChatAgent(_ToolCallingProvider(), tools=reg),
        store, system_prompt="你是助手", summary_keep_turns=0,
    )
    sid = store.create().id
    out = "".join(svc.send(sid, "查一下"))
    assert out == "最终答案"
    _flush()

    lines = [ln for ln in _log_text(tmp_path).splitlines() if "工具完成" in ln]
    assert len(lines) == 2, lines
    for ln in lines:
        # 必须带工具名 + 耗时 + 结果长度，三样缺一不可
        assert re.search(r"工具完成 \S+ 耗时=\d+\.\d{2}s 结果=\d+字", ln), ln

    names = {ln.split("工具完成 ")[1].split()[0] for ln in lines}
    assert names == {"search_knowledge_base", "calculator"}


def test_request_elapsed_is_separable_from_tool_elapsed(tmp_path):
    """同一轮里既能看到"工具花了多久"，也能看到"整轮花了多久"。"""
    from src.core.agent import ChatAgent
    from src.core.memory import MemoryStore
    from src.core.service import ChatService
    from src.core.tools import Tool, ToolRegistry

    reg = ToolRegistry()
    reg.register(Tool("search_knowledge_base", "x", {}, lambda query: "结果"))
    reg.register(Tool("calculator", "x", {}, lambda expression: "2"))

    store = MemoryStore(tmp_path / "sessions")
    svc = ChatService(
        ChatAgent(_ToolCallingProvider(), tools=reg),
        store, system_prompt="你是助手", summary_keep_turns=0,
    )
    sid = store.create().id
    "".join(svc.send(sid, "查"))
    _flush()

    entries = _entries(tmp_path)
    assert any(e["message"].startswith("请求开始") for e in entries)
    assert any(re.match(r"请求结束 耗时=\d+\.\d{2}s", e["message"]) for e in entries)
    assert any("工具完成" in e["message"] for e in entries)


# ---------------------------------------------------------------------------
# 两个 handler 都要挂 filter：漏一个那个出口就全是 `-`
# ---------------------------------------------------------------------------


def test_both_handlers_have_request_filter(tmp_path):
    get_logger("handlers")
    _flush()

    agent_handlers = [
        h for h in logging.getLogger().handlers if getattr(h, "_is_agent_handler", False)
    ]
    assert len(agent_handlers) == 2, "应该正好是 file + console 两个出口"
    for h in agent_handlers:
        assert any(isinstance(f, RequestContextFilter) for f in h.filters), (
            f"{type(h).__name__} 没挂 RequestContextFilter——这个出口的 id 会全是 `-`"
        )

    # 更硬一点的验法：两个出口的 filter 都能对任意记录注入 id
    token = set_request_context({"req": "ffff00001111", "sess": "sess-h"})
    try:
        for h in agent_handlers:
            rec = logging.LogRecord("agent.x", logging.WARNING, "f.py", 1, "x", (), None)
            for f in h.filters:
                if isinstance(f, RequestContextFilter):
                    f.filter(rec)
            assert rec.req_id == "ffff00001111"
            assert rec.sess_id == "sess-h"
    finally:
        reset_request_context(token)


# ---------------------------------------------------------------------------
# 会话管理类操作：没有请求，但有会话
# ---------------------------------------------------------------------------


def test_housekeeping_logs_have_dash_req_but_real_session(tmp_path):
    get_logger("house")
    store, svc = _make_service(tmp_path)
    sid = svc.new_session().id      # 会话创建也记一行日志
    list(svc.send(sid, "你好"))
    svc.clear_session(sid)
    _flush()

    entries = _entries(tmp_path)
    house = [e for e in entries if e["message"] in ("会话创建", "会话清空")]
    assert len(house) == 2, house
    assert all(e["req_id"] == "-" for e in house)          # 没有请求就不该有假 id
    assert all(e["sess_id"] == sid for e in house)         # 但能归到会话
