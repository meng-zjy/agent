"""工具系统 + ReAct 循环测试（M3）。全部用假 provider，不花模型额度。"""

from collections.abc import Iterator

from src.core.agent import ChatAgent
from src.core.models import Message, Role
from src.core.provider import BaseProvider, Completion, ToolCall
from src.core.tools import _search_knowledge_base, build_default_tools, calculate
from src.core.utils import _UNTRUSTED_BEGIN, _UNTRUSTED_END, wrap_untrusted


class ScriptedProvider(BaseProvider):
    """按预设顺序返回 Completion，并记录每次收到的消息。"""

    def __init__(self, responses: list[Completion]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message]) -> str:
        return "unused"

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        yield "STREAM"

    def complete(self, messages: list[Message], tools=None) -> Completion:
        self.calls.append(list(messages))
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# 工具本身
# ---------------------------------------------------------------------------

def test_calculator_basic():
    assert calculate("2 + 3 * 4") == "14"
    assert calculate("123 * 45") == "5535"


def test_calculator_integer_float_normalized():
    assert calculate("10 / 2") == "5"      # 5.0 归一成 5
    assert calculate("4.0 + 1") == "5"


def test_calculator_rejects_dangerous_input():
    # 禁止函数调用 / 属性访问 / 名字引用，返回"计算失败"而非执行
    assert calculate("__import__('os').system('ls')").startswith("计算失败")
    assert calculate("(1).__class__").startswith("计算失败")


def test_get_current_time_returns_str():
    reg = build_default_tools()
    out = reg.execute("get_current_time", {})
    assert isinstance(out, str) and len(out) > 0


def test_registry_schemas_shape():
    schemas = build_default_tools().schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"get_current_time", "calculator"}


def test_registry_execute_unknown_tool():
    assert "未知工具" in build_default_tools().execute("nope", {})


# ---------------------------------------------------------------------------
# 外部资料围栏（M9）：只包 RAG，不包可信工具
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, source, index, text):
        self.source, self.index, self.text = source, index, text


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.queries = []

    def search(self, query, top_k):
        self.queries.append((query, top_k))
        return self._chunks


def test_search_knowledge_base_wraps_untrusted():
    """知识库是你放进去的文档，属不可信文本——返回前必须包围栏。"""
    retriever = _FakeRetriever([
        _FakeChunk("a.md", 0, "片段一"),
        _FakeChunk("b.md", 2, "片段二"),
    ])
    out = _search_knowledge_base(retriever, "问题", top_k=4)

    body = "[a.md#0] 片段一\n\n---\n\n[b.md#2] 片段二"
    assert out == wrap_untrusted(body)
    assert out.startswith(_UNTRUSTED_BEGIN)
    assert out.endswith(_UNTRUSTED_END)
    assert retriever.queries == [("问题", 4)]


def test_search_knowledge_base_empty_result_has_no_fence():
    """空库提示不是外部内容，包了就是噪音。"""
    out = _search_knowledge_base(_FakeRetriever([]), "问题", top_k=4)
    assert "（知识库为空或没有检索到相关内容）" == out
    assert _UNTRUSTED_BEGIN not in out


def test_trusted_tools_are_not_fenced():
    """计算器 / 查时间的返回值是可信的，包了反而噪音——这条反着验。"""
    reg = build_default_tools()
    for name in ("calculator", "get_current_time"):
        out = reg.execute(name, {} if name == "get_current_time" else {"expression": "1+1"})
        assert _UNTRUSTED_BEGIN not in out, f"{name} 不该被包围栏"
        assert "外部" not in out


def test_fenced_result_carries_original_text_intact():
    """围栏只加前后缀，不能改写正文本身。"""
    assert wrap_untrusted("原始内容") == _UNTRUSTED_BEGIN + "\n<<< 外部资料开始 >>>\n原始内容\n" \
                                         "<<< 外部资料结束 >>>\n" + _UNTRUSTED_END


# ---------------------------------------------------------------------------
# ReAct 循环
# ---------------------------------------------------------------------------

def test_react_loop_executes_tool_and_returns_final_answer():
    provider = ScriptedProvider([
        Completion(finish_reason="tool_calls",
                   tool_calls=[ToolCall("call_1", "calculator", '{"expression": "2 + 3"}')]),
        Completion(finish_reason="stop", content="结果是 5"),
    ])
    agent = ChatAgent(provider, tools=build_default_tools())
    # respond() 会夹带 AgentStatus 进度提示，这里只取正文（和 ChatService.send() 同语义）
    out = "".join(c for c in agent.respond([Message(role=Role.USER, content="2+3 等于几")])
                  if isinstance(c, str))

    assert out == "结果是 5"
    assert len(provider.calls) == 2
    # 第二轮调用时，对话里应包含 tool 结果，且 tool_call_id 回填正确
    second = provider.calls[1]
    tool_msgs = [m for m in second if m.role is Role.TOOL]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_1"
    assert tool_msgs[0].name == "calculator"
    assert tool_msgs[0].content == "5"


def test_react_loop_no_tools_uses_stream():
    """无工具时退回流式直答，不调用 complete()。"""
    provider = ScriptedProvider([Completion(finish_reason="stop", content="x")])
    agent = ChatAgent(provider)   # 不传 tools
    out = "".join(agent.respond([Message(role=Role.USER, content="hi")]))
    assert out == "STREAM"        # 走 chat_stream
    assert provider.calls == []   # complete 从未被调用


def test_react_loop_stops_at_max_rounds():
    always_tool = Completion(
        finish_reason="tool_calls",
        tool_calls=[ToolCall("c", "calculator", '{"expression": "1+1"}')],
    )
    provider = ScriptedProvider([always_tool] * 10)
    agent = ChatAgent(provider, tools=build_default_tools(), max_tool_rounds=3)
    out = "".join(c for c in agent.respond([Message(role=Role.USER, content="算一下")])
                  if isinstance(c, str))
    assert len(provider.calls) == 3   # 只跑 3 轮就停
    assert out == ""                  # 最后一轮仍是 tool_calls，content 为空
