"""Spike：模型会不会「自己决定委派」，以及委派到底贵多少。

回答 4 个问题（全真机调用，不 mock）：
  ① 基准 —— 不给委派工具，同一个复杂任务直答。量：延迟、字数。
  ② 委派 —— 给委派工具，同一个任务。量：真的调了吗、用户等多久看到第一个字。
  ③ 并发 —— 一个任务派 2 个子代理，串行 vs 并发。量：墙钟时间差多少。
  ④ 反向 —— 简单问候也给委派工具。量：会不会白派一次（这是描述写错的信号）。

结论写在这里 ↓↓↓（2026-10-15 实跑，deepseek-chat，deadline=120s）
  ① 直答    首字=1.65s  总=13.15s  1 次调用（stream）  3144 字
  ② 委派    首字=38.71s 总=38.71s  3 次调用  5665 字  ← 真的调了
     拆解   父 agent 2 次（complete 4.42s + 11.58s = 16.0s）
            子代理  1 次（stream 22.71s）= 墙钟的 59%，这 22.7s 用户看到 0 个字
  ③ 串行 3.90s vs 并发 2.53s  省 1.37s（35%）← 并发是真的省时间
  ④ 反向    1 次调用，直答没派 ← 模型能分辨，描述写得对
  一句话：委派的路由判断没问题，问题全在体验——首字延迟 1.65s → 38.71s（23 倍），
          而且这 38.7s 是全程黑屏（工具循环必须非流式，见 docs/12 §5.2）。
          接进产品要先把「黑屏期给用户反馈」做出来，否则用户只看到程序死了。
          另注：项目在 .env 里 deadline=30，本例 38.7s 会被 DeadlineExceeded 掐断。

为什么值得跑：② 的「首字延迟」这个东西读文档是想不出来的，只能感受到。
它正是「多 agent 要不要接进产品」的决定性数字。

用法：
  python scripts/spike_multiagent.py              # 全跑，约 3~8 分钟，花额度
  python scripts/spike_multiagent.py --quick      # 跳过 ③，约 1~3 分钟
  python scripts/spike_multiagent.py --dry         # 只验装配，不发请求，不花额度

隔离：只 import `src.core.agent` / `tools` / `providers` / `config`。
      不 import bootstrap / service / memory —— 不碰会话、不落盘、不碰前端，
      所以这个文件怎么改都不会影响产品路径。
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import settings
from src.core.agent import ChatAgent
from src.core.models import Message, Role
from src.core.tools import ToolRegistry, Tool
from src.providers.deepseek import DeepSeekProvider
from src.providers.sensenova import SenseNovaProvider

# --dry：只验装配不发请求；--quick：跳过 ③
DRY = "--dry" in sys.argv
QUICK = "--quick" in sys.argv


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------

def build_provider():
    """按 LLM_PROVIDER 建 provider。不 import bootstrap，避免带上会话/记忆。"""
    name = (settings.llm_provider or "").strip().lower()
    if name == "sensenova":
        return SenseNovaProvider(
            base_url=settings.sensenova_base_url,
            api_key=settings.sensenova_api_key,
            model=settings.sensenova_model,
            **COMMON_KW,
        )
    if name == "deepseek":
        return DeepSeekProvider(
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            **COMMON_KW,
        )
    sys.exit(f"未知的 LLM_PROVIDER={name!r}，改成 deepseek 或 sensenova")


# deadline 故意放宽到 120：项目的 30s 会掐断实测 78s 的长尾（docs/15 §8.2），
# 那会量到「预算被掐」而不是真实延迟。这里就是要量真实延迟。
# max_retries=1：本 spike 不研究重试，少一层免得总耗时失控。
COMMON_KW = dict(max_tokens=settings.max_tokens, temperature=settings.temperature,
                 max_retries=1, timeout=180.0, deadline=120.0,
                 breaker_failure_threshold=99)


class CountingProvider:
    """包住真实 provider：只数调用次数、记每次耗时（带路径标记）。别的都原样透传。

    为什么需要它——「父 agent 跑一次委派」到底触发几次上游调用，靠读代码数不准
    （子代理自己也是一个完整 agent）。包一层直接量。

    耗时带 `complete` / `stream` 标记：父 agent 走 complete（非流式工具循环），
    子代理走 stream（tools=None → 无 schema → 流式直答）。两种路径混在一个列表里
    看不出是谁花了钱，标记一下才对得上号。
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.durations = []
        self.last_messages = None

    def complete(self, messages, tools=None):
        self.calls += 1
        self.last_messages = messages
        t = time.perf_counter()
        try:
            return self._inner.complete(messages, tools)
        finally:
            self.durations.append(f"complete {time.perf_counter() - t:.2f}s")

    def chat(self, messages):
        self.calls += 1
        self.last_messages = messages
        return self._inner.chat(messages)

    def chat_stream(self, messages):
        # 注意这里也是生成器：calls / 计时都在首次被拉取时才真正执行。
        # 子代理走的就是这条路径（tools=None → 无 schema → 流式直答），
        # 漏掉它 = 子代理那几十秒完全没被计量。
        self.calls += 1
        self.last_messages = messages
        t = time.perf_counter()
        try:
            yield from self._inner.chat_stream(messages)
        finally:
            self.durations.append(f"stream {time.perf_counter() - t:.2f}s")

    def breaker_state(self):
        return self._inner.breaker_state()

    def reset(self):
        self.calls = 0
        self.durations = []
        self.last_messages = None


# ---------------------------------------------------------------------------
# 委派工具：子代理 = 一个工具
# ---------------------------------------------------------------------------

PARENT_SYSTEM = "你是一个助手，用中文回答。"

# 这段描述是整个实验的核心变量：模型只靠它判断「这个任务值不值得委派」。
# ② 和 ④ 的差别，几乎全在这一段措辞上。改它，重跑。
DELEGATE_DESC = (
    "把一个可以独立完成、步骤较多或需要较长篇幅的子任务交给子代理处理，"
    "返回子任务的结论。"
    "适合：撰写结构化长文档、需要覆盖多个并列方面、需要多步骤整理的任务。"
    "不适合：一两句话能答完的问题、打招呼、需要向用户追问才能继续的任务。"
)

SUB_SYSTEM = "你是子代理。专注完成给定的具体任务，输出结论要点，不要客套，不要反问用户。"

# ② 用同一个任务对照 ① —— 任务难度一致，才看得出差异来自委派而非任务本身
TASK_LONG = (
    "请写一份《本地单机 AI 聊天应用技术选型报告》，结构化分节，覆盖以下五节，"
    "每节给出结论和理由：\n"
    "1. 语言与运行时\n2. 模型接入方式\n3. 对话记忆存储\n4. 工具调用机制\n5. 安全性与隐私"
)

# ④ 故意给一个最不该委派的东西
TASK_SHORT = "你好！"

# ③ 两个互不依赖、体量相近的子任务（并行才可能真的省时间）
TASKS_PARALLEL = [
    "用 200 字说明「滑动窗口注意力」的机制和用途。",
    "用 200 字说明「余弦相似度」的机制和用途。",
]


def make_delegate(provider):
    """造一个子代理执行器，返回 (执行函数, 工具定义, 子代理计数器)。"""
    # 单独包一层计数，否则分不清「这几十秒是父 agent 花的还是子代理花的」
    sub_counter = CountingProvider(provider)
    # tools=None → 无 schema → 走流式直答，且永远不会有工具 → 不会再往下委派。
    # 递归深度自动封顶为 1，不需要额外写死循环检测。
    sub = ChatAgent(provider=sub_counter, max_tool_rounds=3)

    def run(task: str) -> str:
        msgs = [Message(Role.SYSTEM, SUB_SYSTEM), Message(Role.USER, task)]
        return "".join(sub.respond(msgs))

    tool = Tool(
        name="delegate_task",
        description=DELEGATE_DESC,
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "交给子代理的具体任务描述"},
            },
            "required": ["task"],
        },
        func=run,
    )
    return run, tool, sub_counter


# ---------------------------------------------------------------------------
# 测量
# ---------------------------------------------------------------------------

def run_turn(label, provider, agent, messages):
    """跑一遍 agent.respond，量三样：首段延迟、总耗时、回复字数。

    「首段延迟」是这次最该看的数字：父 agent 走工具循环必须非流式，
    最终答案才算完才 yield 第一段 —— 所以从开始到用户看到第一个字之间，
    是全程黑屏。委派把这段黑屏乘以子代理的轮数。
    """
    provider.reset()
    t0 = time.perf_counter()
    first = None
    text = ""
    try:
        for chunk in agent.respond(messages):
            if first is None:
                first = time.perf_counter() - t0
            text += chunk
    except Exception as exc:
        print(f"\n=== {label} ===")
        print(f"  异常: {type(exc).__name__}: {exc}")
        return None
    total = time.perf_counter() - t0
    first_s = first if first is not None else total
    print(f"\n=== {label} ===")
    print(f"  上游调用次数 : {provider.calls}")
    print(f"  每次调用耗时 : {provider.durations}")
    print(f"  首字延迟     : {first_s:.2f}s   ← 用户在此之前全程无反馈")
    print(f"  总耗时       : {total:.2f}s")
    print(f"  回复字数     : {len(text)}")
    print(f"  回复预览     : {text[:160].replace(chr(10), ' / ')}")
    return {"calls": provider.calls, "total": total, "first": first_s,
            "chars": len(text), "text": text}


def main():
    print(f"provider = {settings.llm_provider}")
    print(f"deadline = 120s（项目默认 30s，见 docs/15 §8.2）")

    base = build_provider()
    provider = CountingProvider(base)
    print(f"model    = {base.label}")

    # ---- 子代理执行器 ----
    sub_run, delegate_tool, sub_counter = make_delegate(provider)

    if DRY:
        print("\n--dry：装配完成，不发请求。")
        print(f"  委派工具名      : {delegate_tool.name}")
        print(f"  委派工具描述    : {delegate_tool.description}")
        print(f"  子代理系统提示  : {SUB_SYSTEM}")
        print(f"  子代理上下文长度: 2 条消息（system + user），不含任何父会话历史")
        print(f"  并发子任务数    : {len(TASKS_PARALLEL)}")
        print("  ✓ 未发起任何网络请求")
        return

    # ---- ① 基准：不给委派工具，模型直答 ----
    # 只给委派工具以外的空工具表 → schemas() 为空 → 走流式直答路径
    bench = ChatAgent(provider=provider, tools=ToolRegistry())
    r1 = run_turn("① 基准：不给委派工具，直答同一个复杂任务",
                  provider, bench,
                  [Message(Role.SYSTEM, PARENT_SYSTEM), Message(Role.USER, TASK_LONG)])

    # ---- ② 委派：给委派工具，同一个任务 ----
    reg = ToolRegistry()
    reg.register(delegate_tool)
    parent = ChatAgent(provider=provider, tools=reg, max_tool_rounds=5)
    sub_counter.reset()
    r2 = run_turn("② 委派：给 delegate_task，同一个任务",
                  provider, parent,
                  [Message(Role.SYSTEM, PARENT_SYSTEM), Message(Role.USER, TASK_LONG)])

    if r2 is not None:
        # 拆清楚钱花在哪：父 agent 走 complete（非流式），子代理走 stream
        print(f"\n  ② 拆解：上游共 {provider.calls} 次")
        print(f"    父 agent  自身: {provider.calls - sub_counter.calls} 次  "
              f"{[d for d in provider.durations if d.startswith('complete')]}")
        print(f"    子代理      : {sub_counter.calls} 次  {sub_counter.durations}")
        if sub_counter.calls > 0:
            sub_s = sum(float(d.split()[1].rstrip('s')) for d in sub_counter.durations)
            print(f"    子代理占墙钟 : {sub_s / r2['total'] * 100:.0f}% "
                  f"（这段期间用户看到 0 个字）")

    if r1 and r2:
        print(f"\n  ①→② 总耗时   {r1['total']:.2f}s → {r2['total']:.2f}s "
              f"（{'慢' if r2['total'] > r1['total'] else '快'} "
              f"{abs(r2['total'] - r1['total']):.2f}s）")
        print(f"  ①→② 上游调用 {r1['calls']} → {r2['calls']} 次")

    # ---- ③ 并发：同一个子代理执行器，串行 vs 并发 ----
    if not QUICK:
        print("\n--- ③ 两个子代理：串行 ---")
        t0 = time.perf_counter()
        out_serial = [sub_run(t) for t in TASKS_PARALLEL]
        serial = time.perf_counter() - t0
        print(f"  串行总耗时: {serial:.2f}s  两段共 {sum(len(o) for o in out_serial)} 字")

        print("\n--- ③ 两个子代理：并发 ---")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            out_para = list(pool.map(sub_run, TASKS_PARALLEL))
        para = time.perf_counter() - t0
        print(f"  并发总耗时: {para:.2f}s  两段共 {sum(len(o) for o in out_para)} 字")
        if serial > 0:
            print(f"  省下      : {serial - para:.2f}s "
                  f"（{(serial - para) / serial * 100:.0f}%）")

    # ---- ④ 反向：简单问候也给委派工具 ----
    r4 = run_turn("④ 反向：一句问候也给 delegate_task",
                  provider, parent,
                  [Message(Role.SYSTEM, PARENT_SYSTEM), Message(Role.USER, TASK_SHORT)])
    if r4:
        print(f"  → 上游调用 {r4['calls']} 次"
              f"{'（派了 1 次子代理）' if r4['calls'] > 1 else '（直答，没派）'}")


main()
