"""spike：空回复率在「真实上下文」下的实测（花额度）。

背景：docs/11 里两组数据互相矛盾——
  · 验收时真实多轮调用实测 14 次里 5 次空（36%）
  · 但 probe_empty_reply.py 测「不传 max_tokens」是 0/3 空

矛盾的可能原因是 **probe_empty_reply.py 用的是简单提问**（"介绍你擅长做的三件事，不超过 60 字"），
reasoning 消耗很低（实测 68~5525）。而真实使用是「长 system prompt + 10 轮历史 + RAG 检索片段」，
reasoning 消耗会高得多。关键是：**"不传 max_tokens" 不等于 "预算无限"**——服务端默认上限也是个有限值，
reasoning 超过它就照样空回复。

本脚本回答两个问题：
1. 真实上下文下，「不传 max_tokens」的空回复率到底是多少？
2. 如果会空，finish_reason 是不是 'length'（证明是撞上了服务端默认上限）？
   —— 如果是，说明「留空 MAX_TOKENS」这个建议本身不牢。

用法：
    python scripts/probe_empty_reply_real.py                 # 默认 15/10/10 次
    python scripts/probe_empty_reply_real.py --simple 20     # 只跑某一档，可指定次数
    python scripts/probe_empty_reply_real.py --real 30 --simple 10 --mt4096 0

注意：会花真实额度，且 reasoning 模型延迟不可预测（单次 4~60s）。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap import active_provider_config  # noqa: E402
from src.config import settings  # noqa: E42
from src.core.models import Message, Role  # noqa: E402

_PROVIDER_NAME, PROVIDER_BASE_URL, PROVIDER_API_KEY, MODEL = active_provider_config()
if not PROVIDER_API_KEY:
    raise SystemExit(f"缺 {_PROVIDER_NAME.upper()}_API_KEY")


# ---------------------------------------------------------------------------
# 两组上下文
# ---------------------------------------------------------------------------

SIMPLE_SYSTEM = "请用中文简洁回答"
SIMPLE_USER = "介绍你擅长做的三件事，不超过 60 字"

# 真实上下文：项目实际的 system prompt + 当前时间 + 10 轮历史 + RAG 检索片段 + 一个工具调用记录
# 目标是让 reasoning 消耗贴近真实使用，而不是用一个玩具提问把它压到最低。
REAL_SYSTEM = settings.system_prompt or SIMPLE_SYSTEM
REAL_TIME_HINT = "[当前时间] 2026-09-29 星期一 15:30:00"

# 10 轮历史：刻意写成「有实质内容、需要推理」的对话，而不是寒暄
HISTORY = [
    ("user", "我们公司最近要做知识库检索，你有什么建议？"),
    ("assistant", "建议先把文档切成 400 字左右的小块、重叠 50 字，再用本地 BGE 模型做向量化。"
                  "检索时取 top-4 片段塞进上下文，比整篇塞进去效果好得多，因为模型更容易定位到答案。"),
    ("user", "切块为什么要重叠？不重叠行不行？"),
    ("assistant", "不重叠会切坏句子。机械按字符切可能把「小明的狗叫黄豆」切成「小明的狗」+「叫黄豆」，"
                  "检索时两半都命中不了。重叠 50 字让关键信息大概率完整落在某一块里。"),
    ("user", "我们文档大概 500 篇，切完会有多少块？"),
    ("assistant", "按 400 字/块算，一篇 5000 字的文档约 15 块（步长 350），500 篇大约 7000~8000 块。"
                  "当前实现是暴力检索——问题和每一块都算一遍余弦相似度，这个量级毫秒级，够用。"
                  "上万块以上再考虑换 FAISS。"),
    ("user", "现在算相似度用的是归一化后的余弦，为什么不直接用点积？"),
    ("assistant", "直接比点积会受向量长度影响——长文本的向量天然更长、点积更大，短的永远比不赢。"
                  "除以各自的模之后比的是方向而不是长度，也就是「两个向量夹角的余弦」，值域 [-1,1]。"),
    ("user", "如果某个 chunk 是空向量怎么办？"),
    ("assistant", "模为 0 会除零崩掉，所以代码里写了 `if na and nb else 0.0`——空向量直接判为不相似。"),
    ("user", "那检索结果怎么标注出处？"),
    ("assistant", "每块存了 source（文件名）和 index（块号），检索结果拼成 `[文件名#块号] 片段原文` 的格式，"
                  "模型回答时可以反问你「根据 x.md 里的记录」。方便你核对它是不是真从文档里读的。"),
    ("user", "我们想让它在回答里直接引用文件名，可以吗？"),
    ("assistant", "可以，但要注意一件事：RAG 只是把原文递给模型，不保证它逐字照抄。"
                  "我们实测过它会把文档里的人名安到用户头上——那是编造，不是代码 bug。"
                  "严格要求引用的场景得额外加约束。"),
]

# 模拟 RAG 检索工具返回的片段（带出处前缀，和 _search_knowledge_base 的拼接格式一致）
RAG_RESULT = (
    "[公司技术架构.md#3] 知识库检索模块采用本地 BGE 模型（BAAI/bge-small-zh-v1.5）做向量化，"
    "首次运行会自动下载约 100MB 模型，走 hf-mirror.com 镜像。切块大小 400 字符、重叠 50 字符，"
    "检索返回 top-4 片段。当前实现为暴力余弦相似度检索，千级文档量下为毫秒级响应。\n\n"
    "---\n\n"
    "[公司技术架构.md#7] 并发模型：不同会话各拿各的会话级锁，可以真正并行；同一会话的请求会排队，"
    "等待 30 秒超时后抛出 SessionBusyError。存储层使用「临时文件 + os.replace」做原子写，"
    "避免并发写入产生半截 JSON。注意：并发只保证单进程内，不要同时开 CLI 和 Web 共享同一个 SESSION_DIR。\n\n"
    "---\n\n"
    "[运维手册.md#1] 模型 sensenova-6.8-flash-lite 是 reasoning 模型，输出分 reasoning（思考过程）"
    "和 content（最终答案）两部分，reasoning_tokens 计入 completion_tokens。如果 MAX_TOKENS 小于模型"
    "实际需要的 reasoning 量，思考会耗尽整个预算，content 为空，即出现空回复。建议留空 MAX_TOKENS，"
    "让服务端使用默认上限。\n\n"
    "---\n\n"
    "[运维手册.md#4] 已知问题：模型回复延迟不可预测，同一个提问延迟从 4 秒到 52 秒不等，"
    "差异来自 reasoning_tokens 的消耗量（68 到 5525）。这是服务端行为，客户端无法优化。"
)

REAL_USER = (
    "帮我总结一下：知识库检索这块现在最大的技术风险是什么？"
    "如果是你，接下来会优先解决哪个问题？给我个排序，不超过 100 字。"
)


def build_real_messages() -> list[dict]:
    """拼出真实上下文：system + 时间 + 历史 + 一次 RAG 工具往返 + 当前提问。"""
    msgs: list[dict] = [
        {"role": "system", "content": REAL_SYSTEM},
        {"role": "system", "content": REAL_TIME_HINT},
    ]
    for role, text in HISTORY:
        msgs.append({"role": role, "content": text})
    # 模拟一次工具调用往返（assistant 发起调用 + tool 返回结果）
    msgs.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_probe01",
            "type": "function",
            "function": {"name": "search_knowledge_base", "arguments": json.dumps(
                {"query": "知识库检索 技术风险"}, ensure_ascii=False)},
        }],
    })
    msgs.append({
        "role": "tool",
        "content": RAG_RESULT,
        "tool_call_id": "call_probe01",
        "name": "search_knowledge_base",
    })
    msgs.append({"role": "user", "content": REAL_USER})
    return msgs


# ---------------------------------------------------------------------------
# 单次探测
# ---------------------------------------------------------------------------

def probe(client, model: str, messages: list[dict], max_tokens: int | None) -> dict:
    """打一次流式请求，返回统计。不打印完整 chunk JSON（那部分 probe_empty_reply.py 已做过）。"""
    kwargs: dict = {"model": model, "messages": messages, "stream": True}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    t0 = time.time()
    try:
        chunks = list(client.chat.completions.create(**kwargs))
    except Exception as exc:  # 限流/断网：不算空回复，单独标出
        return {"error": f"{type(exc).__name__}: {exc}", "elapsed": time.time() - t0}
    elapsed = time.time() - t0

    content_len = 0
    content_chunks = 0
    finish_reasons: list[str] = []
    usage = None
    for c in chunks:
        if c.usage:
            usage = c.usage
        for choice in c.choices:
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)
            if getattr(choice.delta, "content", None):
                content_chunks += 1
                content_len += len(choice.delta.content)

    ud = {}
    if usage is not None:
        try:
            ud = usage.model_dump()
        except Exception:
            ud = {}
    rt = ud.get("reasoning_tokens") or (ud.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    ct = ud.get("completion_tokens") or 0

    return {
        "empty": content_chunks == 0,
        "content_chunks": content_chunks,
        "content_len": content_len,
        "finish_reason": finish_reasons[-1] if finish_reasons else "?",
        "reasoning_tokens": rt,
        "completion_tokens": ct,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_cell(client, model, tag, messages, max_tokens, n) -> list[dict]:
    print(f"\n### {tag}（{n} 次）", flush=True)
    rows: list[dict] = []
    for i in range(n):
        r = probe(client, model, messages, max_tokens)
        rows.append(r)
        if "error" in r:
            print(f"  第{i+1:>2}次: {r['elapsed']:6.2f}s  ⚠ 请求失败 {r['error']}", flush=True)
            continue
        flag = "  ← 空回复" if r["empty"] else ""
        print(f"  第{i+1:>2}次: {r['elapsed']:6.2f}s  content={r['content_len']:>4}字  "
              f"reasoning={r['reasoning_tokens']:>5}  completion={r['completion_tokens']:>5}  "
              f"finish={r['finish_reason']}{flag}", flush=True)
    return rows


def summarize(tag: str, rows: list[dict]) -> None:
    ok = [r for r in rows if "error" not in r]
    empty = [r for r in ok if r["empty"]]
    failed = len(rows) - len(ok)
    print(f"\n{'='*78}")
    print(f"### 汇总：{tag}  成功 {len(ok)}/{len(rows)}，失败 {failed}")
    if not ok:
        return
    print(f"  空回复率          : {len(empty)}/{len(ok)} = {len(empty)/len(ok)*100:.0f}%")
    print(f"  reasoning_tokens  : min={min(r['reasoning_tokens'] for r in ok)}  "
          f"中位={statistics.median(r['reasoning_tokens'] for r in ok):.0f}  "
          f"max={max(r['reasoning_tokens'] for r in ok)}")
    print(f"  content 字数      : min={min(r['content_len'] for r in ok)}  "
          f"max={max(r['content_len'] for r in ok)}")
    print(f"  耗时              : min={min(r['elapsed'] for r in ok):.2f}s  "
          f"中位={statistics.median(r['elapsed'] for r in ok):.2f}s  "
          f"max={max(r['elapsed'] for r in ok):.2f}s")
    # 空回复是不是撞上服务端上限？
    if empty:
        frs = {r["finish_reason"] for r in empty}
        same = all(r["reasoning_tokens"] == r["completion_tokens"] for r in empty)
        print(f"  空回复的 finish_reason: {sorted(frs)}")
        print(f"  reasoning == completion（预算被思考全吃完）: {same}")
    else:
        print("  （本次没有空回复样本）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simple", type=int, default=10, help="简单提问 × 不传 max_tokens 跑几次")
    ap.add_argument("--real", type=int, default=15, help="真实上下文 × 不传 max_tokens 跑几次")
    ap.add_argument("--mt4096", type=int, default=10, help="真实上下文 × max_tokens=4096 跑几次")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=PROVIDER_BASE_URL, api_key=PROVIDER_API_KEY)
    model = MODEL

    real_msgs = build_real_messages()
    simple_msgs = [
        {"role": "system", "content": SIMPLE_SYSTEM},
        {"role": "user", "content": SIMPLE_USER},
    ]
    print(f"模型: {model}")
    print(f"真实上下文 token 粗估: {sum(len(m.get('content',''))+len(json.dumps(m, ensure_ascii=False)) for m in real_msgs)} 字符")
    print(f"当前 .env: MAX_TOKENS={settings.max_tokens}（{'不传' if settings.max_tokens is None else '显式设置'}）")

    results = []
    if args.simple:
        results.append(run_cell(client, model, "① 简单提问 × 不传 max_tokens",
                                simple_msgs, None, args.simple))
    if args.real:
        results.append(run_cell(client, model, "② 真实上下文 × 不传 max_tokens（关键）",
                                real_msgs, None, args.real))
    if args.mt4096:
        results.append(run_cell(client, model, "③ 真实上下文 × max_tokens=4096",
                                real_msgs, 4096, args.mt4096))

    for (tag, rows) in zip(
        ["① 简单提问 × 不传 max_tokens", "② 真实上下文 × 不传 max_tokens（关键）",
         "③ 真实上下文 × max_tokens=4096"],
        results,
    ):
        summarize(tag, rows)

    # 最终对照
    print(f"\n{'='*78}\n### 关键对照")
    if len(results) >= 2:
        def rate(rows):
            ok = [r for r in rows if "error" not in r]
            e = sum(1 for r in ok if r["empty"])
            return f"{e}/{len(ok)} ({e/len(ok)*100:.0f}%)" if ok else "n/a"
        print(f"  简单提问、不传 max_tokens      : {rate(results[0])}")
        print(f"  真实上下文、不传 max_tokens    : {rate(results[1])}   ← 这个才是真实使用时的数字")
        if len(results) == 3:
            print(f"  真实上下文、max_tokens=4096   : {rate(results[2])}")


if __name__ == "__main__":
    main()
