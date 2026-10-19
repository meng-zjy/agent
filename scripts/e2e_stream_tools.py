"""端到端验证：工具循环现在是真流式的（真实调模型，耗少量额度）。

改造前（M3~M6）：工具有被注册，`agent.respond()` 走非流式 `complete()`，
最终答案是算完之后**一次性整段**推给前端的。
改造后（docs/16）：走 `complete_stream()`，最终答案逐 token 到达。

这个脚本量三样东西：
1. 工具已注册时的真实回复 —— chunk 数 / 单段长度 / 字与字的间隔。
2. 首字耗时 —— 工具调用那一轮没有正文，所以首字耗时里包含了「想 + 调工具 + 再想」。
3. 同一句问题在 provider 层的对照：`complete()` 返回一个整块字符串（旧路径），
   `complete_stream()` 返回一串小片段（新路径）。

跑法：`python scripts/e2e_stream_tools.py`
"""

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = Path("data/e2e_stream_tools")
shutil.rmtree(TMP, ignore_errors=True)

from src.bootstrap import build_provider
from src.config import settings
from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.service import ChatService
from src.core.tools import build_default_tools

provider = build_provider()
svc = ChatService(
    ChatAgent(provider, tools=build_default_tools()),
    MemoryStore(TMP),
    system_prompt=settings.system_prompt,
    summary_keep_turns=0,
)
sid = svc.new_session().id

print(f"model={settings.llm_provider}/{settings.deepseek_model if settings.llm_provider == 'deepseek' else settings.sensenova_model}")
print("（工具已注册 → 所有回复都走流式工具循环 complete_stream()）")


def ask_with_gaps(text):
    """问一句，记录每个 chunk 的长度与到达时刻。

    两个不同的时间，别混：
    - **首字耗时**：从发出请求到看到第一个字。工具调用那一轮没有正文，
      所以这段包含了「模型想 + 调工具 + 再想」的全部时间。
    - **chunk 间隔**：最终答案一轮里字与字之间的间隔。这才说明是不是真流式——
      一次性下发会是「一个大 chunk + 后面全 0」，真流式是一连串小 chunk。
    """
    print(f"\n你：{text}")
    t_start = time.perf_counter()
    arrivals = []      # 每个 chunk 距「发出请求」的时刻
    chunks = []
    for c in svc.send(sid, text):
        arrivals.append(time.perf_counter() - t_start)
        chunks.append(c)
    total = "".join(chunks)
    print(f"AI：{total.strip()[:160]}")

    sizes = [len(c) for c in chunks]
    intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    print(f"\n  chunk 数          : {len(chunks)}")
    print(f"  总字符数          : {len(total)}")
    print(f"  首字耗时          : {arrivals[0]:.2f}s（含工具轮，那一轮没有正文）")
    print(f"  答案轮时长        : {arrivals[-1] - arrivals[0]:.2f}s")
    print(f"  单 chunk 最大长度 : {max(sizes)} 字")
    print(f"  平均 chunk 间隔   : {sum(intervals) / len(intervals):.3f}s")
    print(f"  最大 chunk 间隔   : {max(intervals):.3f}s")
    # 判据：段数够多且每段都很短 → 真流式；一段吃掉大半 → 一次性下发
    head_share = sizes[0] / max(len(total), 1)
    verdict = "真流式" if len(chunks) > 4 and head_share < 0.5 else "接近一次性下发"
    print(f"  结论              : {verdict}（首 chunk 占全文 {head_share:.0%}）")
    return total


# ① 需要调工具：第一轮模型只给 tool_calls（0 个正文 chunk），第二轮才是最终答案
a1 = ask_with_gaps("帮我算一下 123 × 45 等于多少？")
print("\n  ✓ 算出了 5535" if "5535" in a1 else "\n  ✗ 未算出 5535")

# ② 不需要工具：第一轮直接是最终答案，最能看清逐 token
ask_with_gaps("用 20 个字介绍一下杭州。")

# ③ 同一句问题在 provider 层的对照：complete() 一整块 vs complete_stream() 一串片段
print("\n=== provider 层对照（同一句问题） ===")
from src.core.models import Message, Role

prompt = [Message(role=Role.USER, content="用 20 个字介绍一下杭州。")]

t0 = time.perf_counter()
blob = provider.complete(prompt).content
print(f"complete()        : 1 个整块，{len(blob)} 字，耗时 {time.perf_counter() - t0:.2f}s")

t0 = time.perf_counter()
pieces = [c.text for c in provider.complete_stream(prompt) if c.text]
print(f"complete_stream() : {len(pieces)} 个片段，共 {sum(len(p) for p in pieces)} 字，"
      f"耗时 {time.perf_counter() - t0:.2f}s，单片最大 {max(len(p) for p in pieces)} 字")
print(f"片段预览          : {pieces[:8]}")
