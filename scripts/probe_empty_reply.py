"""spike：空回复根因定位（花额度）。

回答 docs/11 §4.1 的三个问题：
1. 内容去了哪个字段？—— 打印 chunk 的完整 JSON 结构
2. 跟 max_tokens 有关吗？—— 对比 max_tokens=50 / 100 / 300 / None
3. 不传 max_tokens 时空回复率是多少？（当前 .env 默认不传）

判断依据是 docs/10 §7 发现三之前那次诊断：
  raw_chunks=12, 有内容段=0, finish_reasons=['', ... , 'length']
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap import active_provider_config  # noqa: E402
from src.core.models import Message, Role  # noqa: E402
from src.providers.openai_compat import OpenAICompatProvider  # noqa: E402

_name, _base_url, _api_key, MODEL = active_provider_config()
if not _api_key:
    raise SystemExit(f"缺 {_name.upper()}_API_KEY")

PROMPT = "介绍你擅长做的三件事，不超过 60 字"
SYSTEM = "请用中文简洁回答"
N_PER_CONFIG = 6   # 每档 max_tokens 跑 6 次


def probe(client, model, messages, max_tokens):
    """打一次流式请求，返回 (raw 结构, 统计)。"""
    kwargs = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    t0 = time.time()
    stream = client.chat.completions.create(**kwargs)
    chunks = list(stream)
    elapsed = time.time() - t0

    stats = {
        "raw_chunks": len(chunks),
        "content_chunks": 0,
        "total_len": 0,
        "finish_reasons": [],
        "has_tool_calls": False,
        "has_refusal": False,
        "first_non_content_keys": None,
    }
    dumps = []
    for c in chunks:
        d = c.model_dump()
        dumps.append(d)
        if c.choices:
            ch = c.choices[0]
            if ch.finish_reason:
                stats["finish_reasons"].append(ch.finish_reason)
            delta = ch.delta
            if getattr(delta, "content", None):
                stats["content_chunks"] += 1
                stats["total_len"] += len(delta.content)
            if getattr(delta, "tool_calls", None):
                stats["has_tool_calls"] = True
        # 空回复时找"第一个非 content 的字段"，看内容可能去了哪
        if stats["first_non_content_keys"] is None and dumps[-1]:
            keys = set()
            for c in dumps:
                keys.update(c.keys())
            keys -= {"id", "object", "created", "model", "system_fingerprint", "usage"}
            if keys:
                stats["first_non_content_keys"] = sorted(keys)
    return stats, dumps, elapsed


def main() -> None:
    p = OpenAICompatProvider(
        base_url=_base_url,
        api_key=_api_key,
        model=MODEL,
    )
    client = p._client
    model = p._model
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT},
    ]

    configs = [50, 100, 300, None]
    print(f"模型: {model}\n{'='*70}")

    empty_samples = []
    summary_rows = []

    for mt in configs:
        label = f"max_tokens={mt}" if mt is not None else "max_tokens=不传"
        print(f"\n### {label}（{N_PER_CONFIG} 次）")
        empty = 0
        for i in range(N_PER_CONFIG):
            stats, dumps, elapsed = probe(client, model, messages, mt)
            is_empty = stats["content_chunks"] == 0
            if is_empty:
                empty += 1
                empty_samples.append((mt, dumps))
            reason = stats["finish_reasons"][-1] if stats["finish_reasons"] else "?"
            extra = ""
            if is_empty:
                extra = f"  ← 空！keys={stats['first_non_content_keys']}"
            print(f"  第{i+1}次: {elapsed:5.2f}s, raw={stats['raw_chunks']:2d}, "
                  f"content段={stats['content_chunks']:2d}, 总长={stats['total_len']:3d}, "
                  f"finish={reason}{extra}")
        summary_rows.append((label, empty, N_PER_CONFIG))

    print(f"\n{'='*70}\n### 汇总")
    print(f"{'配置':<18} {'空回复':<10} {'占比':<8}")
    for label, empty, n in summary_rows:
        print(f"{label:<18} {empty}/{n:<7} {empty/n*100:.0f}%")

    # 深挖空回复的 chunk 结构（最多展示 1 个完整样本）
    print(f"\n{'='*70}\n### 空回复的 chunk 完整结构（若为空则本次没有空回复）")
    if empty_samples:
        mt, dumps = empty_samples[0]
        print(f"样本来源: max_tokens={mt}, 共 {len(dumps)} 个 chunk")
        for i, d in enumerate(dumps[:4]):
            print(f"\n  chunk[{i}]:")
            print("  " + json.dumps(d, ensure_ascii=False, indent=2).replace("\n", "\n  "))
        if len(dumps) > 4:
            print(f"\n  ...（中间省略 {len(dumps)-4} 个）...")
            print(f"\n  chunk[{len(dumps)-1}]（最后一个）:")
            print("  " + json.dumps(dumps[-1], ensure_ascii=False, indent=2)
                  .replace("\n", "\n  "))
        # 检查所有字段名，找出 content 之外有值的字段
        all_values = {}
        for d in dumps:
            for path, val in _walk(d, ""):
                if val and not isinstance(val, (dict, list)) and val != "" and val != []:
                    all_values[path] = all_values.get(path, 0) + 1
        print(f"\n  ### 所有非空字段的出现次数（content 路径会在这里暴露）")
        for path, count in sorted(all_values.items(), key=lambda x: -x[1]):
            if "content" in path.lower() or "delta" in path.lower() or "message" in path.lower():
                print(f"    {count:>3}×  {path}")
    else:
        print("  （本次 6 档全绿，没有空回复样本）")

    print(f"\n{'='*70}\n### 结论线索")
    if empty_samples:
        # 看 finish_reason 分布
        frs = []
        for mt, dumps in empty_samples:
            for d in dumps:
                if d.get("choices"):
                    fr = d["choices"][0].get("finish_reason")
                    if fr:
                        frs.append(fr)
        print(f"  空回复样本里的 finish_reason: {frs}")
    print(f"  空回复率对比：不传 max_tokens 时 {summary_rows[-1][1]}/{N_PER_CONFIG}，"
          f"max_tokens=50 时 {summary_rows[0][1]}/{N_PER_CONFIG}")


def _walk(obj, prefix):
    """递归展开 dict/list，产出 (路径, 叶子值)。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


if __name__ == "__main__":
    main()
