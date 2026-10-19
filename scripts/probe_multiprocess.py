"""探针：两个进程写同一个会话，到底丢多少条消息？（不花额度）

这是 docs/10 §10.2 判掉「多进程隔离」那条结论的实测验证。判掉的推理是：

- `RLock` 是进程内的：两个进程各自加锁，互不知情
- `os.replace` 是原子的，但只是"整块替换"——两个进程各自读到旧快照、
  各自落盘，**后写的赢**，前写的那批消息就没了

用假 provider，不花额度。跑法：

    python scripts/probe_multiprocess.py            # 两个进程，各 50 条
    python scripts/probe_multiprocess.py --count 200 --step 0.05

预期：会丢消息。如果实测证实了，说明 docs/10 那个「判掉」是对的
（本地单用户不会有两个人同时开着两个进程连同一个会话）。
**如果实测发现单进程多线程路径也受影响，那就是真 bug**——见文末。
"""

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.agent import ChatAgent  # noqa: E402
from src.core.memory import MemoryStore  # noqa: E402
from src.core.models import Message, Role  # noqa: E402
from src.core.provider import BaseProvider, Completion  # noqa: E402
from src.core.service import ChatService  # noqa: E402

RESULT_FILE = Path("data/tmp/probe_multiprocess_result.txt")


def _enable_utf8_stdio() -> None:
    """Windows 控制台默认 GBK，中文会打乱——和 src/cli.py 同一个处理。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


class _EchoProvider(BaseProvider):
    """固定回复，回复内容很短，把落盘频率拉高。"""

    def chat(self, messages: list[Message]) -> str:
        return "R"

    def chat_stream(self, messages):
        yield "R"

    def complete(self, messages, tools=None):
        return Completion(finish_reason="stop", content="R")


def _send_one(store: MemoryStore, session_id: str, text: str) -> None:
    """走一遍真实的服务端路径（含锁、含落盘），不经过 HTTP。"""
    svc = ChatService(
        ChatAgent(_EchoProvider()), store,
        system_prompt="x", summary_keep_turns=0,
    )
    list(svc.send(session_id, text))


def worker(session_dir: str, session_id: str, prefix: str, count: int, step: float,
           barrier, order, out) -> None:
    """一个子进程：拿全局序号拼出可追踪的标签，然后发 count 条消息。

    用共享序号而不是本地序号，这样**丢的是哪几条**能从标签直接读出来。
    """
    store = MemoryStore(session_dir)
    barrier.wait()                       # 两个进程同时开跑，把时间窗打满
    sent: list[str] = []
    for _ in range(count):
        with order.get_lock():
            n = order.value
            order.value = n + 1
        label = f"{prefix}{n:04d}"
        _send_one(store, session_id, label)
        sent.append(label)
        time.sleep(step)
    out.put((prefix, sent))


def _control(store: MemoryStore, session_id: str, count: int) -> list[str]:
    """对照组：同一个进程里发同样多的消息，应该一条都不丢。"""
    sent = []
    for i in range(count):
        label = f"C{i:04d}"
        _send_one(store, session_id, label)
        sent.append(label)
    return sent


def _user_labels(session_id: str, store: MemoryStore) -> list[str]:
    s = store.load(session_id)
    return [m.content for m in s.conversation.messages if m.role is Role.USER]


def _count_by_prefix(labels: list[str]) -> dict[str, int]:
    """按标签前缀（P1- / P2- / C）分组统计，看是哪个进程的写入活下来了。"""
    out: dict[str, int] = {}
    for l in labels:
        p = l.split("-")[0]
        out[p] = out.get(p, 0) + 1
    return out


def run(session_dir: Path, count: int, step: float, workers: int) -> dict:
    """真正丢多少条，靠这个函数说话。"""
    ctx = multiprocessing.get_context("spawn")
    store = MemoryStore(session_dir)
    sid = store.create().id

    barrier = ctx.Barrier(workers)
    order = ctx.Value("i", 0)
    out: multiprocessing.Queue = ctx.Queue()

    procs = [
        ctx.Process(
            target=worker,
            args=(str(session_dir), sid, f"P{i + 1}-", count, step, barrier, order, out),
        )
        for i in range(workers)
    ]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    elapsed = round(time.perf_counter() - t0, 2)

    reported = {}
    while not out.empty():
        prefix, sent = out.get()
        reported[prefix] = sent

    # JSON 有没有被写坏（os.replace 是原子的，不该出现半截文件）
    corrupt = False
    try:
        json.loads((session_dir / f"{sid}.json").read_text(encoding="utf-8"))
    except Exception:
        corrupt = True

    actual = _user_labels(sid, store)
    expected = sorted(l for labels in reported.values() for l in labels)
    lost = sorted(set(expected) - set(actual))

    return {
        "elapsed": elapsed,
        "sent": sum(len(v) for v in reported.values()),
        "per_worker": {k: len(v) for k, v in sorted(reported.items())},
        "actual": len(actual),
        "lost": len(lost),
        "corrupt": corrupt,
        "actual_by_prefix": _count_by_prefix(actual),
        "lost_labels": lost,
        "json_ok": not corrupt,
    }


def report(r: dict, count: int, workers: int) -> None:
    print(f"\n{'=' * 62}")
    print(f"工作进程数 {workers} × 每进程 {count} 条，共 {r['sent']} 条用户消息")
    print(f"墙钟耗时 {r['elapsed']}s（两个进程同时跑，不是相加）")
    print(f"{'-' * 62}")
    print(f"最终会话里的用户消息数：{r['actual']} 条（期望 {r['sent']} 条）")
    print(f"丢失：{r['lost']} 条" + (f"（{r['lost'] / r['sent']:.0%}）" if r['sent'] else ""))
    print(f"按来源统计实际存活：{r['actual_by_prefix']}")
    print(f"会话 JSON 是否完整：{'是（os.replace 原子性成立）' if r['json_ok'] else '否！文件已损坏'}")
    if r["lost"]:
        shown = r["lost_labels"][:12]
        more = len(r["lost_labels"]) - len(shown)
        print(f"丢失的标签（前 {len(shown)} 个）：{' '.join(shown)}" + (f" …还有 {more} 个" if more > 0 else ""))
    print(f"{'=' * 62}")


def save_result(r: dict, control: tuple[int, int], args) -> None:
    """把关键数字落到磁盘，供 docs/13 引用——文档里的数字必须是跑出来的。"""
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(
        "probe_multiprocess.py 实测结果\n"
        f"参数：workers={args.workers} count={args.count} step={args.step}\n"
        f"单进程对照：发出 {control[0]} 条，落盘 {control[1]} 条，丢失 {control[0] - control[1]} 条\n"
        f"多进程：发出 {r['sent']} 条，落盘 {r['actual']} 条，丢失 {r['lost']} 条"
        f"（{r['lost'] / r['sent']:.0%}）\n"
        f"存活按来源：{r['actual_by_prefix']}\n"
        f"会话 JSON 完整：{r['json_ok']}\n"
        f"墙钟耗时 {r['elapsed']}s\n",
        encoding="utf-8",
    )
    print(f"\n结果已写入 {RESULT_FILE}")


def main() -> int:
    _enable_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", type=int, default=50, help="每个进程发的条数")
    ap.add_argument("--step", type=float, default=0.02, help="每条消息之间的间隔秒数")
    ap.add_argument("--workers", type=int, default=2, help="并发进程数")
    args = ap.parse_args()

    session_dir = Path("data/tmp/probe_multiprocess")
    if session_dir.exists():
        import shutil
        shutil.rmtree(session_dir)

    print("实验 0（对照组）：同一个进程发同样多的消息，确认没有丢")
    store = MemoryStore(session_dir / "control")
    sid_c = store.create().id
    control_sent = _control(store, sid_c, args.count)
    control_actual = _user_labels(sid_c, store)
    print(f"  发出 {len(control_sent)} 条，落盘 {len(control_actual)} 条，"
          f"丢失 {len(set(control_sent) - set(control_actual))} 条")

    print("\n实验 1：多个进程连同一个 session_id")
    r = run(session_dir / "multi", args.count, args.step, args.workers)
    report(r, args.count, args.workers)
    save_result(r, (len(control_sent), len(control_actual)), args)

    print("\n结论：")
    if r["lost"] == 0:
        print("  两个进程之间没有丢消息——docs/10 §10.2 的判掉结论需要修正。")
    else:
        print(f"  确实会丢（{r['lost']} 条），docs/10 §10.2 的判掉结论被实测证实：")
        print("  `RLock` 是进程内的，两个进程各自加锁互不知情；")
        print("  `os.replace` 只保证整块替换，谁后写谁赢，前写的那批消息整批消失。")
        print("  本地单用户不会有两个人同时开着两个进程连同一个会话，所以判掉是对的。")
        print("  真要跨进程写，需要文件锁（如 `portalocker` / `msvcrt.locking`）——")
        print("  这是 M10 的范围，不是本期。")

    if not r["json_ok"]:
        print("\n  ⚠ 而且 JSON 被写坏了——这超出预期，属于真 bug，要单独记一条。")
    if not control_sent or len(set(control_sent) - set(control_actual)) > 0:
        print("\n  ⚠ 单进程对照也丢了消息——这是 docs/10 时期就该修的问题，不是多进程的问题。")

    return 1 if r["lost"] else 0


if __name__ == "__main__":
    sys.exit(main())
