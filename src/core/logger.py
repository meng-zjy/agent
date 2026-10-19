"""日志基础设施（M8）。

用标准库 logging，不引第三方。日志写到 `data/logs/agent.log`，轮转 1MB×3。

**只记元数据，不记消息正文**：会话历史已完整落盘到 `data/sessions/`，
日志重复记录既膨胀又没必要（也避免日志文件成为隐私泄露点）。

**全链路 request_id（M9）**：一次请求在入口设一个上下文（req_id + sess_id），
`RequestContextFilter` 把它自动注入每一条日志记录——调用方不需要改一行代码，
所以 provider / mcp / rag 层的日志即使自己没记 id，也自动带上。
`grep <req_id>` 就是这次请求的完整轨迹。

输出策略：
- 文件：所有级别
- stderr：WARNING 及以上（降级提示、错误在前台跑时仍可见；请求级 INFO 不刷屏）
"""

import contextvars
import logging
import logging.handlers
from pathlib import Path

from src.config import settings

_initialized = False
_MAX_BYTES = 1024 * 1024      # 单文件 1MB
_BACKUP_COUNT = 3             # 保留 3 个轮转文件

# 请求上下文：{"req": 本次请求的追踪 id, "sess": 所属会话 id}
# 用 contextvars 而不是全局变量：asyncio 下每个请求一个 task、各持自己的上下文，
# 并发时不会互相污染。
_request_ctx = contextvars.ContextVar("request_ctx", default=None)


def set_request_context(ctx: dict) -> contextvars.Token:
    """设置请求上下文，返回 reset 用的 Token。调用方在 finally 里 reset 掉。"""
    return _request_ctx.set(ctx)


def reset_request_context(token: contextvars.Token) -> None:
    """清掉请求上下文。必须在 finally 里调——否则这个上下文会一直挂在当前 task 上，
    下一次请求拿不到自己的 id（asyncio 下每个请求是独立 task，所以不会跨请求串，
    但同 task 内的清理仍然是卫生要求）。"""
    try:
        _request_ctx.reset(token)
    except (ValueError, LookupError):
        # Token 来自别的 Context，reset 会抛；忽略，不影响日志
        pass


class RequestContextFilter(logging.Filter):
    """把请求上下文注入每一条日志记录。

    没处在请求里时（如启动阶段的 bootstrap 日志）填 `-`，让格式对齐。
    两个 handler 都要挂这个 filter，漏一个那个出口就全是 `-`。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _request_ctx.get()
        record.req_id = ctx["req"] if ctx else "-"
        record.sess_id = ctx["sess"] if ctx else "-"
        return True


def _effective_level() -> int:
    """把 .env 里的 LOG_LEVEL 字符串映射成 logging 级别；非法值回退 INFO。"""
    name = getattr(settings, "log_level", "INFO")
    level = logging.getLevelName(str(name).strip().upper())
    if isinstance(level, int):
        return level
    return logging.INFO


def setup() -> logging.Logger:
    """幂等初始化全局日志，返回 root logger。重复调用不会重复挂 handler。

    handler 挂在 root 上（而不是 'agent' 命名空间），这样：
    - 任何模块的日志都会落到文件，不用逐处 propagate
    - pytest 的 caplog fixture 天然能捕获（caplog 也挂在 root）
    """
    global _initialized
    if _initialized:
        return logging.getLogger()

    root = logging.getLogger()
    root.setLevel(_effective_level())

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(req_id)s [%(sess_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "agent.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,               # 首次写日志才建文件
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(RequestContextFilter())
    file_handler._is_agent_handler = True      # reset() 靠这个标记区分
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)   # 只在 stderr 显示警告以上
    console_handler.setFormatter(fmt)
    console_handler.addFilter(RequestContextFilter())
    console_handler._is_agent_handler = True
    root.addHandler(console_handler)

    # 第三方库降级，避免每次请求都刷一行
    # 注意：openai 3.3.1 的请求日志用的是 `httpx2` 这个 logger（不是 `httpx`），
    # `openai/_utils/_logs.py` 里写死的是 getLogger("httpx2")；只降 httpx 不生效。
    for _name in ("httpx", "httpcore", "httpx2", "openai"):
        logging.getLogger(_name).setLevel(logging.WARNING)

    _initialized = True
    return root


def get_logger(name: str) -> logging.Logger:
    """取一个子 logger（如 `get_logger("service")` → `agent.service`）。

    首次调用时自动 setup，所以任何模块直接 import 就能用。
    """
    if not _initialized:
        setup()
    return logging.getLogger(f"agent.{name}")


def reset() -> None:
    """测试用：清掉 handler 和初始化标记，允许重新 setup。"""
    global _initialized
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, (logging.handlers.RotatingFileHandler, logging.StreamHandler)) \
                and getattr(h, "_is_agent_handler", False):
            h.close()
            root.removeHandler(h)
    _initialized = False
