"""流式工具循环测试。

两个层次，**chunk 的对象形状不一样，不要混用**：
- **provider 层**（`complete_stream`）：喂的是 openai SDK 形状的 chunk
  （`chunk.choices[i].delta.content / delta.tool_calls`），mock 掉 client，
  验证「正文边到边吐 + 工具参数按 index 分段拼接 + 收尾才给出结论」，
  以及重试 / 熔断的记账口径。
- **agent 层**：假 provider 直接 yield 我们自己的 `StreamChunk`，
  验证循环如何衔接「工具轮 → 最终答案轮」。

全部内存态，不联网、不花额度。工具参数按段到达是 OpenAI 流式协议的真实行为
（`{"express` + `ion": "1+1"}`），所以必须有专测，不能只测整块到达。
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError

import src.providers.openai_compat as oc
from src.core.agent import AgentStatus, ChatAgent
from src.core.circuit_breaker import CircuitOpenError
from src.core.models import Message, Role
from src.core.provider import (
    BaseProvider,
    Completion,
    DeadlineExceeded,
    EmptyReplyError,
    StreamChunk,
    ToolCall,
)
from src.core.tools import Tool, ToolRegistry
from src.providers.deepseek import DeepSeekProvider


# ---------------------------------------------------------------------------
# openai SDK 形状的假 chunk（喂给 provider 层）
# ---------------------------------------------------------------------------


def _tc(index, tc_id=None, name=None, args=None):
    """一个分段到达的 tool_call delta。"""
    return SimpleNamespace(
        index=index, id=tc_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


def _raw(deltas, finish_reason=""):
    return SimpleNamespace(choices=[SimpleNamespace(delta=d, finish_reason=finish_reason)
                                    for d in deltas], usage=None)


def _text(piece):
    """一个只带正文片段的 chunk。"""
    return _raw([SimpleNamespace(content=piece, role=None, tool_calls=None)])


def _tchunk(index, args, tc_id=None, name=None):
    """一个只带 tool_call 片段的 chunk。"""
    return _raw([SimpleNamespace(
        content=None, role=None, tool_calls=[_tc(index, tc_id=tc_id, name=name, args=args)]
    )])


def _multi_tchunk(calls):
    """一个 chunk 里带多个 index 的 tool_call 片段。"""
    return _raw([SimpleNamespace(content=None, role=None, tool_calls=calls)])


def _finish(reason):
    return _raw([SimpleNamespace(content=None, role=None, tool_calls=None)], finish_reason=reason)


def _conn_error() -> APIConnectionError:
    return APIConnectionError(
        request=httpx.Request("POST", "http://fake/v1/chat/completions")
    )


def _usage(reasoning=0, completion=0):
    return SimpleNamespace(
        model_dump=lambda: {
            "completion_tokens": completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
        completion_tokens=completion,
    )


def _provider(**kw):
    kw.setdefault("base_url", "http://fake")
    kw.setdefault("api_key", "fake")
    kw.setdefault("model", "m")
    p = DeepSeekProvider(**kw)
    p._client = MagicMock()
    return p


@pytest.fixture
def no_sleep(monkeypatch):
    """不真等，但能断言退避节奏。"""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    return sleeps


# ---------------------------------------------------------------------------
# provider 层：正文流式 + 工具参数分段拼接
# ---------------------------------------------------------------------------


def test_complete_stream_yields_text_then_terminator():
    p = _provider()
    p._client.chat.completions.create.return_value = iter(
        [_text("你好"), _text("世界"), _finish("stop")]
    )
    chunks = list(p.complete_stream([]))

    assert [c.text for c in chunks if c.text] == ["你好", "世界"]
    assert chunks[-1].ended
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].tool_calls is None


def test_complete_stream_sends_tools_and_tool_choice():
    p = _provider()
    p._client.chat.completions.create.return_value = iter([_text("x"), _finish("stop")])
    tools = [{"type": "function", "function": {"name": "t"}}]
    list(p.complete_stream([], tools=tools))

    kw = p._client.chat.completions.create.call_args.kwargs
    assert kw["tools"] == tools
    assert kw["tool_choice"] == "auto"
    assert kw["stream"] is True


def test_complete_stream_assembles_arguments_across_chunks():
    """工具参数按段到达，必须攒齐才是合法 JSON。"""
    p = _provider()
    p._client.chat.completions.create.return_value = iter([
        _tchunk(0, args="", tc_id="c1", name="calculator"),
        _tchunk(0, args='{"expression":'),
        _tchunk(0, args='"1+1"}'),
        _finish("tool_calls"),
    ])

    terminator = list(p.complete_stream([]))[-1]
    assert terminator.tool_calls == [ToolCall("c1", "calculator", '{"expression":"1+1"}')]
    assert terminator.finish_reason == "tool_calls"


def test_complete_stream_merges_calls_by_index_in_order():
    """两个工具同时调、参数乱序到达，也要按 index 拼对、按序返回。"""
    p = _provider()
    p._client.chat.completions.create.return_value = iter([
        _multi_tchunk([
            _tc(1, tc_id="b", name="calculator", args=""),
            _tc(0, tc_id="a", name="get_current_time", args=""),
        ]),
        _multi_tchunk([
            _tc(0, args='{"zone": "sh"}'),
            _tc(1, args='{"expression": "2*3"}'),
        ]),
        _finish("tool_calls"),
    ])

    calls = list(p.complete_stream([]))[-1].tool_calls
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("a", "get_current_time", '{"zone": "sh"}'),
        ("b", "calculator", '{"expression": "2*3"}'),
    ]


def test_complete_stream_sensenova_style_empty_id_and_name():
    """商汤给的是 `id=''` / `name=''`（空字符串），不是 `None`。

    探针实测（2026-10-16，`spike_stream_tools.py`）：参数整块到达，且到达那段时
    id 和 name 是空字符串。用真值判断（`if tc.id:`）才不会把已经拼好的值覆盖成空串；
    写成 `if tc.id is not None:` 这里就崩了。分段拼接对「整块到达」也照样成立。
    """
    p = _provider()
    whole_args = _raw([SimpleNamespace(
        content=None, role=None,
        tool_calls=[SimpleNamespace(index=0, id="", name="",
                                    function=SimpleNamespace(name="", arguments='{"expression": "1"}'))],
    )])
    p._client.chat.completions.create.return_value = iter([
        _tchunk(0, args="", tc_id="c1", name="calculator"),
        whole_args,
        _finish("tool_calls"),
    ])

    calls = list(p.complete_stream([]))[-1].tool_calls
    assert calls == [ToolCall("c1", "calculator", '{"expression": "1"}')]


def test_complete_stream_ignores_reasoning_deltas():
    """reasoning_content 走别的字段，不能当正文吐出去。"""
    p = _provider()
    reasoning_chunk = _raw([SimpleNamespace(
        content=None, role=None, tool_calls=None, reasoning_content="内部思考"
    )])
    p._client.chat.completions.create.return_value = iter(
        [reasoning_chunk, _text("正式回答"), _finish("stop")]
    )
    assert [c.text for c in list(p.complete_stream([])) if c.text] == ["正式回答"]


def test_complete_stream_tool_calls_count_as_a_reply():
    """只有工具调用、没有正文，不算空回复。"""
    p = _provider()
    p._client.chat.completions.create.return_value = iter([
        _tchunk(0, args="", tc_id="c1", name="t"),
        _finish("tool_calls"),
    ])
    assert list(p.complete_stream([]))[-1].tool_calls


def test_complete_stream_raises_on_empty_reply():
    p = _provider(max_tokens=50)
    usage_chunk = SimpleNamespace(choices=[], usage=_usage(reasoning=50, completion=50))
    p._client.chat.completions.create.return_value = iter([_finish("length"), usage_chunk])
    with pytest.raises(EmptyReplyError) as exc_info:
        list(p.complete_stream([]))
    assert "MAX_TOKENS=50" in str(exc_info.value)


# ---------------------------------------------------------------------------
# provider 层：重试只在没吐过字时做
# ---------------------------------------------------------------------------


def test_complete_stream_retries_before_any_text(no_sleep):
    p = _provider(max_retries=3)
    p._client.chat.completions.create.side_effect = [
        _conn_error(),
        iter([_text("ok"), _finish("stop")]),
    ]
    chunks = list(p.complete_stream([]))

    assert [c.text for c in chunks if c.text] == ["ok"]
    assert p._client.chat.completions.create.call_count == 2
    assert no_sleep == [2.0]


def test_complete_stream_no_retry_after_text_emitted():
    """已经吐过字之后再重试，会把同一段话再吐一遍——那时宁可直接失败。"""
    def _text_then_blow_up():
        yield _text("已吐出")
        raise _conn_error()

    p = _provider(max_retries=3)
    p._client.chat.completions.create.return_value = _text_then_blow_up()

    out = []
    with pytest.raises(APIConnectionError):
        for c in p.complete_stream([]):
            out.append(c.text)

    assert out == ["已吐出"]
    assert p._client.chat.completions.create.call_count == 1
    assert p.breaker.failures == 1     # 失败仍然记账


def test_complete_stream_honors_deadline():
    p = _provider(max_retries=3, deadline=0.05)
    p._client.chat.completions.create.side_effect = _conn_error()
    with pytest.raises(DeadlineExceeded):
        list(p.complete_stream([]))


def test_complete_stream_breaker_opens_and_fails_fast():
    p = _provider(breaker_failure_threshold=1)
    p._client.chat.completions.create.side_effect = _conn_error()

    with pytest.raises(CircuitOpenError):
        list(p.complete_stream([]))
    with pytest.raises(CircuitOpenError):
        list(p.complete_stream([]))

    assert p._client.chat.completions.create.call_count == 1   # 打开后不再碰上游


def test_complete_stream_unicode_error_falls_back_to_non_stream(monkeypatch):
    def _cut(_piece):
        raise UnicodeError("多字节被网络切分")

    monkeypatch.setattr(oc, "scrub_text", _cut)
    p = _provider()
    p._client.chat.completions.create.side_effect = [
        iter([_text("你好")]),
        SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(tool_calls=None, content="回退答案"),
            finish_reason="stop",
        )], usage=None),
    ]
    chunks = list(p.complete_stream([]))
    assert [c.text for c in chunks] == ["回退答案"]
    assert p._client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# agent 层：循环衔接「工具轮 → 最终答案轮」
#
# 这里的 chunk 是我们自己的 StreamChunk，不是上面的 SDK 形状。
# ---------------------------------------------------------------------------


def _sc_text(piece):
    return StreamChunk(text=piece)


def _sc_end():
    return StreamChunk(finish_reason="stop")


class RoundProvider:
    """按轮返回预设 chunk 序列；轮数不够时停在最后一轮，用于撑满轮数上限。"""

    def __init__(self, rounds):
        self._rounds = rounds
        self.calls: list[list[Message]] = []

    def complete_stream(self, messages, tools=None):
        self.calls.append(messages)
        yield from self._rounds[min(len(self.calls) - 1, len(self._rounds) - 1)]


def _tool_round(tc_id="c1", name="calculator", args='{"expression": "1+1"}'):
    """一整轮「模型决定调工具」：参数已在 provider 层拼好，收尾片段里给出。

    与真 provider 的行为一致——正文片段在前，唯一的收尾片段在最后。
    """
    return [StreamChunk(
        finish_reason="tool_calls", tool_calls=[ToolCall(tc_id, name, args)]
    )]


def _texts(events):
    """只取正文片段，丢掉 AgentStatus。和 `ChatService.send()` 的过滤器同一个语义。"""
    return [e for e in events if isinstance(e, str)]


def test_agent_final_answer_streams_piecewise():
    """最终答案那一轮是逐 token 的，不再是一次性整段。"""
    provider = RoundProvider([_tool_round(), [_sc_text("答"), _sc_text("案"), _sc_end()]])
    reg = ToolRegistry()
    reg.register(Tool("calculator", "c", {}, lambda expression: "2"))

    out = _texts(ChatAgent(provider, tools=reg).respond(
        [Message(role=Role.USER, content="1+1")]
    ))
    assert out == ["答", "案"]      # 改造前是 ["答案"]（16 字一块）


def test_agent_executes_tool_and_feeds_result_back():
    seen = {}

    def fake_tool(query):
        seen["query"] = query
        return "R:" + query

    provider = RoundProvider([
        _tool_round(tc_id="c1", name="lookup", args='{"query": "q1"}'),
        [_sc_text("结果"), _sc_end()],
    ])
    reg = ToolRegistry()
    reg.register(Tool("lookup", "l", {}, fake_tool))

    out = _texts(ChatAgent(provider, tools=reg).respond(
        [Message(role=Role.USER, content="查 q1")]
    ))

    assert out == ["结果"]
    assert seen == {"query": "q1"}
    # 工具结果要带对 tool_call_id 塞回历史，assistant 的调用记录也要在
    tool_msgs = [m for m in provider.calls[1] if m.role is Role.TOOL]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "R:q1"
    assert tool_msgs[0].tool_call_id == "c1"
    assistant = [m for m in provider.calls[1] if m.role is Role.ASSISTANT]
    assert len(assistant) == 1
    assert assistant[0].tool_calls[0]["function"]["name"] == "lookup"


def test_agent_preamble_before_tool_call_is_streamed():
    provider = RoundProvider([
        [_sc_text("让我查一下"), *_tool_round()],
        [_sc_text("结果是 2"), _sc_end()],
    ])
    reg = ToolRegistry()
    reg.register(Tool("calculator", "c", {}, lambda expression: "2"))

    out = _texts(ChatAgent(provider, tools=reg).respond(
        [Message(role=Role.USER, content="1+1")]
    ))
    assert out == ["让我查一下", "结果是 2"]


def test_agent_max_tool_rounds_guard_terminates(caplog):
    provider = RoundProvider([_tool_round()])      # 永远要调工具
    reg = ToolRegistry()
    reg.register(Tool("calculator", "c", {}, lambda expression: "2"))

    out = _texts(ChatAgent(provider, tools=reg, max_tool_rounds=2).respond(
        [Message(role=Role.USER, content="1+1")]
    ))
    assert out == []
    assert len(provider.calls) == 2
    assert any("轮数上限" in r.message for r in caplog.records)


def test_agent_without_tools_still_uses_chat_stream():
    class StreamOnly(BaseProvider):
        def chat(self, messages):
            return "x"

        def chat_stream(self, messages):
            yield "A"
            yield "B"

        def complete(self, messages, tools=None):
            raise AssertionError("无工具不该走非流式")

    out = list(ChatAgent(StreamOnly()).respond([Message(role=Role.USER, content="hi")]))
    assert out == ["A", "B"]


def test_agent_status_message_labels():
    assert AgentStatus("calculator").message == "正在计算…"
    assert AgentStatus("search_knowledge_base").message == "正在检索知识库…"
    # 未收录的工具名（MCP 工具名是动态的）退回原名，不能崩、不能是空串
    assert AgentStatus("mcp_something_new").message == "正在调用 mcp_something_new…"


def test_agent_status_emitted_before_each_tool_execution():
    """顺序是关键：提示必须在工具真跑之前到达，用户才知道「卡住的这段」在干什么。"""
    order = []

    def make_tool(name):
        def _tool(*args, **kwargs):
            order.append(("tool", name))
            return "结果"
        return _tool

    provider = RoundProvider([
        [StreamChunk(finish_reason="tool_calls", tool_calls=[
            ToolCall("c1", "calculator", '{"expression": "1+1"}'),
            ToolCall("c2", "search_knowledge_base", '{"query": "q"}'),
        ])],
        [_sc_text("最终"), _sc_end()],
    ])
    reg = ToolRegistry()
    reg.register(Tool("calculator", "c", {}, make_tool("calculator")))
    reg.register(Tool("search_knowledge_base", "s", {}, make_tool("search_knowledge_base")))

    # 必须边迭代边记，不能用 list() 再回溯：工具的副作用发生在生成器 yield 之后、
    # 调用方恢复执行的那一刻，只有按到达顺序记录才看得出「提示在工具之前」。
    events = []
    for e in ChatAgent(provider, tools=reg).respond(
        [Message(role=Role.USER, content="1+1")]
    ):
        events.append(e)
        if isinstance(e, AgentStatus):
            order.append(("status", e.tool_name))

    assert order == [
        ("status", "calculator"), ("tool", "calculator"),
        ("status", "search_knowledge_base"), ("tool", "search_knowledge_base"),
    ]
    # 两次工具调用各发一次提示，正文只有一条最终答案
    assert len(events) == 3
    assert _texts(events) == ["最终"]
    assert [e.tool_name for e in events if isinstance(e, AgentStatus)] == \
        ["calculator", "search_knowledge_base"]


def test_base_default_complete_stream_keeps_plain_providers_working():
    """只实现 complete() 的老 provider，工具循环仍然能跑。"""
    class Plain(BaseProvider):
        def chat(self, messages):
            return "x"

        def chat_stream(self, messages):
            yield "x"

        def complete(self, messages, tools=None):
            if len(messages) == 1:
                return Completion(
                    finish_reason="tool_calls",
                    tool_calls=[ToolCall("c1", "calc", '{"expression": "1+1"}')],
                )
            return Completion(finish_reason="stop", content="最终答案")

    reg = ToolRegistry()
    reg.register(Tool("calc", "c", {}, lambda expression: "2"))

    out = "".join(_texts(ChatAgent(Plain(), tools=reg).respond(
        [Message(role=Role.USER, content="1+1")]
    )))
    assert out == "最终答案"


# ---------------------------------------------------------------------------
# service 层：进度提示不能进历史、不能混进 send()
# ---------------------------------------------------------------------------


def _svc(tmp_path):
    from src.core.memory import MemoryStore
    from src.core.service import ChatService

    reg = ToolRegistry()
    reg.register(Tool("calculator", "c", {}, lambda expression: "2"))
    provider = RoundProvider([_tool_round(), [_sc_text("答案是 2"), _sc_end()]])
    store = MemoryStore(tmp_path / "sessions")
    svc = ChatService(ChatAgent(provider, tools=reg), store, summary_keep_turns=0)
    return svc, store.create().id


def test_send_yields_only_text(tmp_path):
    svc, sid = _svc(tmp_path)
    out = list(svc.send(sid, "1+1"))
    assert out == ["答案是 2"]
    assert all(isinstance(c, str) for c in out)


def test_send_events_includes_status(tmp_path):
    svc, sid = _svc(tmp_path)
    events = list(svc.send_events(sid, "1+1"))
    assert _texts(events) == ["答案是 2"]
    assert [e.tool_name for e in events if isinstance(e, AgentStatus)] == ["calculator"]


def test_status_is_not_persisted_to_history(tmp_path):
    """最要紧的一条：进度提示不能落进会话历史。

    一旦写进去，下次把历史发给模型时它就是一段「正在计算…」的假回复，
    而且会永久留在磁盘上的会话 JSON 里。
    """
    from src.core.memory import MemoryStore

    svc, sid = _svc(tmp_path)
    "".join(svc.send(sid, "1+1"))

    loaded = MemoryStore(tmp_path / "sessions").load(sid)
    msgs = loaded.conversation.messages
    assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]
    assert msgs[-1].content == "答案是 2"
    assert "正在" not in msgs[-1].content
