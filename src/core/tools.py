"""工具系统（M3）：Tool 数据模型 + ToolRegistry 注册表。

一个「工具」= 给模型看的 JSON Schema + 真正执行的 Python 函数。
M4 RAG 的「检索」、M5 MCP 的「外部工具」将来都只是往注册表里 register 新工具，
ReAct 循环、记忆、界面都不变。
"""

import ast
import operator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from src.core.utils import wrap_untrusted


# ---------------------------------------------------------------------------
# 安全计算器：只放行四则运算，拒绝任意代码执行
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST):
    """递归求值 AST，只允许数字 / 四则运算 / 括号 / 正负号。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def calculate(expression: str) -> str:
    """安全计算一个数学表达式，返回字符串结果。任何非法输入都不抛异常。"""
    try:
        result = _eval_node(ast.parse(expression.strip(), mode="eval"))
        # 整数值的浮点（如 4.0）显示为 4，避免 1/3 之类长小数
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as exc:  # noqa: BLE001  非法表达式统一转成可读错误
        return f"计算失败：{exc}"


def get_current_time() -> str:
    """返回当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Tool / ToolRegistry
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                  # JSON Schema，告诉模型"工具长什么样"
    func: Callable[..., str]          # 真正执行的函数，返回字符串结果


class ToolRegistry:
    """按名字登记工具，负责生成 API 的 tools 参数、并按名执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        """拼成 openai SDK 的 tools 参数。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, args: dict) -> str:
        """执行工具，返回字符串结果；未知工具 / 执行异常都转成可读错误。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"未知工具：{name}"
        try:
            return str(tool.func(**args))
        except Exception as exc:  # noqa: BLE001
            return f"工具 {name} 执行失败：{exc}"

    def __len__(self) -> int:
        return len(self._tools)


def build_default_tools(retriever=None, top_k: int = 4, mcp_tools=None) -> ToolRegistry:
    """注册默认工具：M3 的时间/计算器 + 可选的 M4 检索 + 可选的 M5 MCP 工具。

    retriever：传入一个带 search() 的知识库检索器（DocumentStore）时，
    额外注册 search_knowledge_base 工具；不传则只有基础工具。
    mcp_tools：MCP 桥包装出的一批 Tool，有则逐个注册。
    """
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_current_time",
            description="获取当前日期和时间。当用户询问时间、日期、星期几时调用。",
            parameters={"type": "object", "properties": {}, "required": []},
            func=get_current_time,
        )
    )
    registry.register(
        Tool(
            name="calculator",
            description="计算数学表达式并返回结果。当用户要求计算、算数、求值时调用。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 '123 * 45'",
                    }
                },
                "required": ["expression"],
            },
            func=calculate,
        )
    )   
    if retriever is not None:
        registry.register(
            Tool(
                name="search_knowledge_base",
                description="在本地知识库中检索与问题相关的文档片段。"
                            "当用户询问的问题可能在你掌握的资料/文档/知识库里有答案时调用。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要检索的问题或关键词",
                        }
                    },
                    "required": ["query"],
                },
                func=lambda query: _search_knowledge_base(retriever, query, top_k),
            )
        )
    if mcp_tools:
        for t in mcp_tools:
            registry.register(t)
    return registry


def _search_knowledge_base(retriever, query: str, top_k: int) -> str:
    """检索知识库，返回拼接好的 top-k 片段；空库/无结果给可读提示。

    结果包一层外部资料围栏（M9）：知识库是你放进去的文档，属不可信文本，
    文档里可能有提示词注入。空库提示不需要围栏——它不是外部内容。
    """
    chunks = retriever.search(query, top_k=top_k)
    if not chunks:
        return "（知识库为空或没有检索到相关内容）"
    body = "\n\n---\n\n".join(f"[{c.source}#{c.index}] {c.text}" for c in chunks)
    return wrap_untrusted(body)
