"""探针：实测总时间预算（P0）与熔断器（P1）的实际效果。不联网、不花额度。

起一个本地 TCP 服务端当「挂掉的上游」——接上连接后一个字节都不回，
这是线上最典型的那种失败（连得上、但不回包，只能等 read 超时）。

⚠ 这里必须用**真实 TCP socket**。自定义 httpx transport 里的 time.sleep
  触发不了 read timeout，假 transport 会把这个回归漏过去。

三个实验：
  1. 时间预算：不封顶 vs 封顶，同一台挂掉的上游，墙钟差多少倍
  2. 熔断：连续 6 次调用，有熔断 vs 无熔断，总耗时各多少
  3. 半开探测：熔断打开 → 等一个窗口 → 只放行一个探测请求

为让脚本能在几十秒内跑完，超时用的是 3 秒而不是配置的 90 秒（缩小 30 倍）。
线性放大的结论不受影响，文末会按真实配置外推。
"""

import socketserver
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers.deepseek import DeepSeekProvider  # noqa: E402


# 服务端视角数「真的进来的连接」——这就是「真的发出去的 HTTP 请求数」。
# 用 mock client 数不到这个数，因为 mock 里连"有没有真的连上"都是假的。
CONNS = {"n": 0}


class HangingServer(socketserver.BaseRequestHandler):
    """只接请求、永不回字节——「上游挂了」的最小复现。"""

    def handle(self):
        CONNS["n"] += 1
        try:
            self.request.recv(4096)
        except OSError:
            pass
        time.sleep(6.0)              # 读完请求后一个字节都不回
        self.request.close()


def start_server():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), HangingServer)
    server.daemon_threads = True
    server.allow_reuse_address = True
    server.timeout = 0.1
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def call(provider):
    """调一次并返回 (耗时, 异常类型名)。"""
    t = time.perf_counter()
    try:
        provider.chat([])
        return time.perf_counter() - t, "OK"
    except Exception as exc:                       # noqa: BLE001
        return time.perf_counter() - t, type(exc).__name__


def experiment_budget(port):
    print("=" * 68)
    print("实验 1：总时间预算——不封顶 vs 封顶")
    print(f"（read 超时 {3}s，重试 3 次；真实配置是 {90}s × 3 次）")
    print("=" * 68)

    print("\nA. 不封顶（deadline 极大，等价于改造前的行为）")
    p = DeepSeekProvider(base_url=f"http://127.0.0.1:{port}/v1", api_key="x",
                         timeout=3.0, max_retries=3, deadline=10000.0)
    CONNS["n"] = 0
    dt, err = call(p)
    print(f"   1 次调用 = {dt:.2f}s  →  {err}")
    print(f"   服务端收到的连接数 = {CONNS['n']}（3 次重试各发一次）")
    print(f"   分解：3 次尝试 × read 3s + 退避 2s + 4s = {3*3+2+4}s")

    print("\nB. 封顶（deadline=1.0）")
    p = DeepSeekProvider(base_url=f"http://127.0.0.1:{port}/v1", api_key="x",
                         timeout=3.0, max_retries=3, deadline=1.0)
    CONNS["n"] = 0
    dt, err = call(p)
    print(f"   1 次调用 = {dt:.2f}s  →  {err}")
    print(f"   服务端收到的连接数 = {CONNS['n']}（预算用完即停，不再重试）")
    print("   分解：第 1 次尝试的 read 被收窄到 1s，剩余额度不够退避+再试一次")

    print("\n外推到真实配置（read=90s，3 次重试）：")
    print(f"   A 不封顶：{3*90+2+4}s ≈ {(3*90+2+4)/60:.1f} 分钟")
    print(f"   B 封顶　：{30}s（DEADLINE 默认值）")
    print(f"   A 若叠加 openai SDK 自带的 2 次重试 = 3×3=9 次 HTTP 请求：{9*90+6+12+24+48}s")
    return dt


def experiment_breaker(port):
    print()
    print("=" * 68)
    print("实验 2：熔断——连续 6 次调用（上游一直挂着）")
    print("=" * 68)

    for label, threshold in (("无熔断（阈值 0 = 关闭）", 0), ("有熔断（连续 1 次失败即开）", 1)):
        p = DeepSeekProvider(base_url=f"http://127.0.0.1:{port}/v1", api_key="x",
                             timeout=3.0, max_retries=3, deadline=1.0,
                             breaker_failure_threshold=threshold)
        CONNS["n"] = 0
        t0 = time.perf_counter()
        parts = []
        for _ in range(6):
            dt, _ = call(p)
            parts.append(f"{dt:.3f}s")
        total = time.perf_counter() - t0
        print(f"\n{label}")
        print(f"   6 次耗时：{', '.join(parts)}")
        print(f"   总耗时 {total:.2f}s　熔断状态 {p.breaker_state()['state']}")
        print(f"   服务端收到的连接数 = {CONNS['n']}　← 真的发出去的 HTTP 请求")


def experiment_half_open(port):
    print()
    print("=" * 68)
    print("实验 3：熔断打开 → 等窗口 → 只放行一个探测请求")
    print("=" * 68)
    p = DeepSeekProvider(base_url=f"http://127.0.0.1:{port}/v1", api_key="x",
                         timeout=3.0, max_retries=1, deadline=1.0,
                         breaker_failure_threshold=1, breaker_open_seconds=2.0)
    CONNS["n"] = 0

    dt, err = call(p)
    print(f"\n第 1 次：{dt:.3f}s → {err}　熔断状态 {p.breaker_state()['state']}")

    dt, err = call(p)
    print(f"第 2 次：{dt:.4f}s → {err}　熔断状态 {p.breaker_state()['state']}"
          "　（OPEN 期间秒级失败，没有发请求）")

    print("等待 2.5s 让熔断窗口过期…")
    time.sleep(2.5)
    dt, err = call(p)
    print(f"第 3 次：{dt:.3f}s → {err}　熔断状态 {p.breaker_state()['state']}"
          "　（HALF_OPEN 探测）")

    dt, err = call(p)
    print(f"第 4 次：{dt:.4f}s → {err}　熔断状态 {p.breaker_state()['state']}"
          "　（探测失败，重新 OPEN）")

    print(f"\n4 次调用里真的发出去的 HTTP 请求 = {CONNS['n']}")


if __name__ == "__main__":
    server = start_server()
    port = server.server_address[1]
    print(f"本地「挂掉的上游」：127.0.0.1:{port}（接上连接后不回任何字节）\n")
    try:
        experiment_budget(port)
        experiment_breaker(port)
        experiment_half_open(port)
    finally:
        server.shutdown()
        server.server_close()
