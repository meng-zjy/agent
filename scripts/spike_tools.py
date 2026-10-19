"""Spike：SenseNova flash-lite 是否支持原生 function calling。

跑 3 组真实调用，判断结论：
- 若返回标准 tool_calls 字段 → 支持原生 function calling（M3 走"原生"路线）
- 若返回纯文本（甚至把 <tool_call> 当文本吐）→ 不支持（M3 走"手写 ReAct 解析"路线）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from src.config import settings

client = OpenAI(base_url=settings.sensenova_base_url, api_key=settings.sensenova_api_key)
MODEL = settings.sensenova_model

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算一个数学表达式并返回数值结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
]


def show(tag, resp):
    print(f"\n=== {tag} ===")
    choice = resp.choices[0]
    m = choice.message
    print("finish_reason:", choice.finish_reason)
    print("content      :", repr(m.content))
    print("tool_calls   :", repr(m.tool_calls))
    if m.tool_calls:
        for tc in m.tool_calls:
            print(f"  -> id={tc.id} type={tc.type} name={tc.function.name}")
            print(f"     args={tc.function.arguments}")


def run(tag, **kwargs):
    try:
        show(tag, client.chat.completions.create(model=MODEL, **kwargs))
    except Exception as exc:
        print(f"\n=== {tag} ===\n  异常: {type(exc).__name__}: {exc}")


print(f"model={MODEL}")
run("① auto + 问时间", messages=[{"role": "user", "content": "现在几点了？"}],
    tools=tools, tool_choice="auto")
run("② 强制指定 calculator", messages=[{"role": "user", "content": "帮我算 123 * 45"}],
    tools=tools, tool_choice={"type": "function", "function": {"name": "calculator"}})
run("③ auto + 问计算", messages=[{"role": "user", "content": "帮我算 123 * 45 等于多少"}],
    tools=tools, tool_choice="auto")
