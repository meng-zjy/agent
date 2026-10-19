"""Spike：当前供应商是否支持「流式 + 工具调用」，以及 tool_calls 是怎么到达的。

这是流式工具循环改造（docs/16）的前置验证。要回答三个问题：

1. `stream=True` 和 `tools=` 能不能同时下发？（不能的话整个工具路径会退回非流式）
2. `tool_calls` 是分段到达还是一次性到达？——参数是 JSON 字符串，分段到达就必须
   按 index 拼接（`{"express` + `ion": "1+1"}`），我们的实现按这个假设写的。
3. 收尾 chunk 的 `finish_reason` 是什么值？（`tool_calls` / `stop`）

跑法：`python scripts/spike_stream_tools.py`
花真实额度，约 3~4 次短调用。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from src.config import settings

if settings.llm_provider == "deepseek":
    BASE_URL, API_KEY, MODEL = (
        settings.deepseek_base_url, settings.deepseek_api_key, settings.deepseek_model
    )
else:
    BASE_URL, API_KEY, MODEL = (
        settings.sensenova_base_url, settings.sensenova_api_key, settings.sensenova_model
    )

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0)

TOOLS = [
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


def run(tag, messages, tool_choice="auto"):
    """跑一次流式 + 工具调用，逐 chunk 打印到达过程。"""
    print(f"\n=== {tag} ===")
    pending: dict[int, dict] = {}
    content_pieces = 0
    usage_seen = False
    finish_reason = None

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            tools=TOOLS,
            tool_choice=tool_choice,
            stream_options={"include_usage": True},
        )
    except Exception as exc:
        # 这里报 400 就说明该供应商不支持「流式 + 工具」的组合
        print(f"  ⚠ 请求被拒：{type(exc).__name__}: {exc}")
        print("  → 该供应商不支持 stream=True 与 tools 同用，工具循环只能退回非流式")
        return

    n = 0
    for chunk in stream:
        n += 1
        if chunk.usage:
            usage_seen = True
        if not chunk.choices:
            print(f"  [{n:02d}] (usage 收尾 chunk)")
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta
        bits = []
        if delta.content:
            content_pieces += 1
            bits.append(f'content={delta.content!r}')
        for tc in getattr(delta, "tool_calls", None) or []:
            fn = tc.function
            bits.append(
                f"tool_calls[{tc.index}] id={tc.id!r} name={fn.name!r} args={fn.arguments!r}"
            )
        if choice.finish_reason:
            bits.append(f"finish_reason={choice.finish_reason!r}")
        # 顺便记一份，最后能确认参数拼接出来的结果
        for tc in getattr(delta, "tool_calls", None) or []:
            slot = pending.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function is not None:
                if tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        print(f"  [{n:02d}] " + ("  ".join(bits) if bits else "(空 delta)"))

    print(f"\n  正文 chunk 数        : {content_pieces}")
    print(f"  最终 finish_reason   : {finish_reason!r}")
    print(f"  末尾 usage chunk     : {'有' if usage_seen else '无'}")
    for idx in sorted(pending):
        s = pending[idx]
        print(f"  拼接后 tool_calls[{idx}]: id={s['id']!r} name={s['name']!r}")
        print(f"                        args={s['arguments']!r}")


print(f"model={MODEL}  base_url={BASE_URL}")
print("（工具循环真流式依赖：stream=True + tools 能同用，且 tool_calls 按 index 分段到达）")

run(
    "① 强制 calculator（看参数是不是分段到达）",
    messages=[{"role": "user", "content": "帮我算 123 * 45"}],
    tool_choice={"type": "function", "function": {"name": "calculator"}},
)
run(
    "② auto + 问时间（模型自己决定调工具）",
    messages=[{"role": "user", "content": "现在几点了？"}],
)
run(
    "③ auto + 不需要工具（看最终答案是不是逐 token）",
    messages=[{"role": "user", "content": "用一句话回答：2 加 2 等于几？"}],
)
