"""MCP 桥测试（M5）：测纯逻辑与包装，不启动真实 npx 服务器。"""

from src.core.mcp import _clean_schema, _format_call_result, wrap_mcp_tools
from src.core.tools import Tool, build_default_tools
from src.core.utils import _UNTRUSTED_BEGIN, _UNTRUSTED_END, wrap_untrusted


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    def __init__(self, texts, is_error=False):
        self.content = [_FakeContent(t) for t in texts]
        self.isError = is_error


class _FakeBridge:
    """假的 MCP 桥：记录调用，返回固定结果。"""

    def __init__(self, tools):
        self._tools = tools
        self.called = []

    def tools(self):
        return self._tools

    def call_tool(self, name, args):
        self.called.append((name, args))
        return f"结果:{name}{args}"


def test_format_call_result_extracts_text():
    # MCP 结果是文件系统里的不可信文本，必须包上外部资料围栏（M9）
    assert _format_call_result(_FakeResult(["你好", "世界"])) == wrap_untrusted("你好\n世界")


def test_format_call_result_error_flag():
    """错误分支也包围栏——错误信息同样是 MCP 服务器产出的。"""
    assert _format_call_result(_FakeResult(["坏了"], is_error=True)) == wrap_untrusted("[错误] 坏了")


def test_format_call_result_empty_output_is_fenced():
    """没有任何文本块时给「(无文本输出)」，同样算 MCP 产出的内容。"""
    out = _format_call_result(_FakeResult([]))
    assert out == wrap_untrusted("(无文本输出)")
    assert out.startswith(_UNTRUSTED_BEGIN)
    assert out.endswith(_UNTRUSTED_END)


def test_clean_schema_drops_extra_fields():
    raw = {"type": "object", "properties": {"a": {"type": "string"}},
           "required": ["a"], "$schema": "http://x", "title": "T"}
    assert _clean_schema(raw) == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }


def test_wrap_mcp_tools_dispatches():
    bridge = _FakeBridge([{
        "name": "list_directory",
        "description": "列目录",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    }])
    tools = wrap_mcp_tools(bridge)
    assert len(tools) == 1
    assert tools[0].name == "list_directory"
    out = tools[0].func(path="/tmp")
    assert bridge.called == [("list_directory", {"path": "/tmp"})]
    assert out == "结果:list_directory{'path': '/tmp'}"


def test_build_default_tools_registers_mcp():
    extra = [Tool(name="mcp_thing", description="x", parameters={},
                  func=lambda **k: "ok")]
    reg = build_default_tools(mcp_tools=extra)
    names = {s["function"]["name"] for s in reg.schemas()}
    assert "mcp_thing" in names


def test_build_default_tools_no_mcp_without_mcp_tools():
    names = {s["function"]["name"] for s in build_default_tools().schemas()}
    assert "mcp_thing" not in names
