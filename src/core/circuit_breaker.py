"""熔断器（circuit breaker）：上游连续失败后**秒级失败**，别再为它排队付钱。

重试解决的是「偶发抖动」，熔断解决的是「上游真的挂了」。上游挂掉时，每一次
调用都会把整份时间预算烧完才失败（连得上、但不回包，只能等 read 超时）；
几十个并发请求各烧一遍，uvicorn 的 40 槽线程池被占满，连 `/health` 都被拖到
探活超时——负载均衡于是判服务挂掉去重启，而上游其实只是慢（见 docs/11 §7）。

三态：

    CLOSED    正常放行。连续失败到 N 次 → 转 OPEN
    OPEN      拒绝一切请求（毫秒级返回 CircuitOpenError），满 T 秒 → 放行一个探测请求
    HALF_OPEN 只放行**一个**探测请求：成功 → CLOSED，失败 → 重新 OPEN（重新计时）

为什么 OPEN 满 T 秒后要「先探测一个」而不是直接全放：直接把所有请求都放过去，
万一上游还没恢复，等于重复付一次集体代价。放一个、看到结果再决定，代价最小。
同时 HALF_OPEN 期间其他请求仍然快速失败，所以并发下不会出现 N 个探测请求。

只有「上游不健康」的失败才计数：断网 / 5xx / 429 / 总预算耗尽。
鉴权失败（4xx）和空回复**不计入**——那种情况下上游明明活着，是我们配置或
请求参数的问题；让熔断器替配置错误买单，会把一个能立刻看懂的报错
（"API Key 错"）换成一个看不懂的报错（"已熔断"），反而更难排查。

线程安全：provider 是多会话共享的单例，uvicorn 线程池里会有真正的并发调用，
所以状态迁移全部持锁。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable


class CircuitOpenError(RuntimeError):
    """熔断器处于 OPEN：上游连续失败，暂时不接受新请求。

    这不是「服务坏了」，是「我们先别问了」。等 `open_until` 到了，
    熔断器会放行一个探测请求自动恢复，不需要人动手。

    ⚠ 这个错误不该被当作「可重试」：重试也是问上游，而问上游正是问题所在。
    """

    def __init__(self, retry_after: float, failures: int) -> None:
        self.retry_after = retry_after
        self.failures = failures
        super().__init__(
            f"上游连续失败 {failures} 次，已熔断，约 {retry_after:.0f}s 后自动恢复；请稍后重试"
        )


class CircuitBreaker:
    """三态熔断器。`failure_threshold` / `open_seconds` 都在构造时定好。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str = "upstream",
        failure_threshold: int = 3,
        open_seconds: float = 30.0,
        is_unhealthy: Callable[[BaseException], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        log: logging.Logger | None = None,
    ) -> None:
        self._name = name
        # <= 0 视为关闭熔断（阈值取无穷，永不打开）：留个线上临时摘掉的开关。
        self._threshold = (
            float("inf") if failure_threshold <= 0 else max(1, failure_threshold)
        )
        self._open_seconds = open_seconds
        self._is_unhealthy = is_unhealthy or (lambda exc: True)
        self._clock = clock
        self._log = log

        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._failures = 0        # 连续失败次数；一次成功清零
        self._opened_at = 0.0     # 本次进入 OPEN 的时刻

    # ---------- 读状态（供日志 / 探活 / 测试用） ----------

    @property
    def failure_threshold(self) -> float:
        """生效的失败阈值。改了 .env 后从这里确认是不是真的传进来了。"""
        return self._threshold

    @property
    def open_seconds(self) -> float:
        return self._open_seconds

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def retry_after(self) -> float:
        """OPEN 时离自动恢复还有多少秒；其他状态为 0.0。"""
        with self._lock:
            if self._state != self.OPEN:
                return 0.0
            return max(0.0, self._open_seconds - (self._clock() - self._opened_at))

    def describe(self) -> dict:
        """给 `/health` / 监控看的一行状态：出事先知道是「我们」还是「上游」。"""
        return {
            "state": self.state,
            "failures": self.failures,
            "retry_after": self.retry_after,
        }

    # ---------- 调用前检查 ----------

    def check(self) -> None:
        """调用前调它。OPEN 时抛 CircuitOpenError（不花时间等上游）。

        HALF_OPEN 表示「已经有一个探测请求在路上」，所以同样拒绝——
        放行多个探测请求等于重复付一次集体代价，正是想避免的事。
        """
        with self._lock:
            if self._state == self.CLOSED:
                return
            if self._state == self.HALF_OPEN:
                raise CircuitOpenError(1.0, self._failures)
            # state == OPEN
            elapsed = self._clock() - self._opened_at
            if elapsed < self._open_seconds:
                raise CircuitOpenError(self._open_seconds - elapsed, self._failures)
            self._state = self.HALF_OPEN          # 到点：放行这一条，它成了探测请求

    # ---------- 记账 ----------

    def record_success(self) -> None:
        """一次成功：清零连续失败计数。HALF_OPEN 的探测成功 = 恢复 CLOSED。"""
        with self._lock:
            if self._state == self.CLOSED and self._failures == 0:
                return                      # 本来就好着，没必要动状态
            if self._state == self.HALF_OPEN and self._log:
                self._log.warning("%s 熔断恢复：探测请求成功", self._name)
            self._state = self.CLOSED
            self._failures = 0

    def record_failure(self, exc: BaseException) -> None:
        """记一次失败。不是「上游的错」的异常不计入（见模块文档）；
        已经 OPEN 时也不再累加——否则熔断会被自己的快速失败无限延长。"""
        with self._lock:
            if self._state == self.OPEN:
                return                      # 谁也不该在 OPEN 期间拿到放行
            if self._state == self.HALF_OPEN:
                # 探测失败 → 还没恢复，重新开一次、重新计时
                self._state = self.OPEN
                self._opened_at = self._clock()
                if self._log:
                    self._log.warning("%s 探测失败，熔断继续打开: %s", self._name, exc)
                return
            if not self._is_unhealthy(exc):
                return
            self._failures += 1
            if self._failures >= self._threshold:
                self._state = self.OPEN
                self._opened_at = self._clock()
                if self._log:
                    self._log.error(
                        "%s 熔断打开：连续失败 %d 次，%s 内快速失败: %s",
                        self._name, self._failures, self._open_seconds, exc,
                    )
