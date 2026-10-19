"""并发测试（M7 前置：多窗口同时跑）。

被测的是三条硬约定：
1. 不同会话：真并行（各会话独立拿锁，互不阻塞）
2. 同一会话：串行（排队），绝不互相覆盖历史——这是修掉的真实数据丢失
3. 等不及：抛 SessionBusyError，而不是写坏数据

外加存储层：并发写 + 并发读不能产生半截 JSON。
"""

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.models import Message, Role
from src.core.provider import BaseProvider, Completion
from src.core.service import ChatService, SessionBusyError


class TimedProvider(BaseProvider):
    """每次调用睡一会儿，用来制造"两个请求时间上重叠"的条件。"""

    def __init__(self, delay: float = 0.3, reply: str = "OK") -> None:
        self._delay = delay
        self._reply = reply
        self.overlaps: list[tuple[float, float]] = []

    def chat(self, messages: list[Message]) -> str:
        time.sleep(self._delay)
        return self._reply

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        start = time.time()
        time.sleep(self._delay)
        self.overlaps.append((start, time.time()))
        yield self._reply

    def complete(self, messages: list[Message], tools=None) -> Completion:
        return Completion(finish_reason="stop", content=self._reply)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "sessions")


def make_service(store: MemoryStore, provider: TimedProvider, **kw) -> ChatService:
    return ChatService(
        ChatAgent(provider), store,
        system_prompt="你是助手",
        summary_keep_turns=kw.pop("summary_keep_turns", 0),   # 并发测试与摘要无关
        summary_max_chars=kw.pop("summary_max_chars", 100),
        **kw,
    )


def hold_lock(lock: threading.RLock):
    """在后台线程占住锁，模拟"另一个请求正在生成回复"。

    必须在别的线程占——RLock 可重入，同一线程占完再拿会直接成功，测不出排队。
    返回 (线程, release_event)：set event 就会放锁。
    """
    started = threading.Event()
    release = threading.Event()

    def runner() -> None:
        assert lock.acquire(timeout=5.0)
        started.set()
        release.wait(10)
        lock.release()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    assert started.wait(5.0), "占锁线程没能拿到锁"
    return thread, release


def send_in_thread(service: ChatService, sid: str, text: str,
                   results: dict, errors: list) -> None:
    try:
        results[text] = "".join(service.send(sid, text))
    except Exception as exc:  # noqa: BLE001
        errors.append((text, type(exc).__name__, str(exc)))


# ---------------------------------------------------------------------------
# 1. 同一会话：串行，不丢消息
# ---------------------------------------------------------------------------


def test_same_session_concurrent_no_lost_messages(store: MemoryStore):
    """回归：同一会话并发两条消息，两条都要落进历史。

    修复前这里只剩 2 条——两个请求读到同一份旧快照、各自落盘互相覆盖。
    """
    provider = TimedProvider(delay=0.3)
    service = make_service(store, provider)

    session = store.create()
    results: dict = {}
    errors: list = []
    barrier = threading.Barrier(2)

    def worker(tag: str) -> None:
        barrier.wait()
        send_in_thread(service, session.id, tag, results, errors)

    threads = [threading.Thread(target=worker, args=(t,))
               for t in ("问题一", "问题二")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert set(results) == {"问题一", "问题二"}

    loaded = store.load(session.id)
    assert len(loaded.conversation.messages) == 4
    roles = [m.role for m in loaded.conversation.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]
    users = [m.content for m in loaded.conversation.messages if m.role is Role.USER]
    assert set(users) == {"问题一", "问题二"}


def test_same_session_requests_do_not_overlap(store: MemoryStore):
    """同一会话的两轮请求在时间上互斥（真串行，不是靠运气）。"""
    provider = TimedProvider(delay=0.3)
    service = make_service(store, provider)

    session = store.create()
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        list(service.send(session.id, "并发问题"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    (s1, e1), (s2, e2) = sorted(provider.overlaps)
    assert e1 <= s2 or e2 <= s1          # 时间区间不重叠


def test_cross_session_requests_run_in_parallel(store: MemoryStore):
    """不同会话的请求时间上必须重叠——这才是"多窗口同时跑"。"""
    provider = TimedProvider(delay=0.4)
    service = make_service(store, provider)

    sid_a = store.create().id
    sid_b = store.create().id
    barrier = threading.Barrier(2)
    t0 = time.time()

    def worker(sid: str) -> None:
        barrier.wait()
        list(service.send(sid, "并行问题"))

    threads = [threading.Thread(target=worker, args=(sid,)) for sid in (sid_a, sid_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall = time.time() - t0
    assert wall < 0.7, f"两个请求被串行化了（总耗时 {wall:.2f}s，delay 各 0.4s）"

    (s1, e1), (s2, e2) = sorted(provider.overlaps)
    assert s2 < e1, "两个不同会话的请求应当同时在进行"


# ---------------------------------------------------------------------------
# 2. 等不及就报忙，不写坏数据
# ---------------------------------------------------------------------------


def test_busy_rejected_when_timeout_expires(store: MemoryStore):
    """会话被占用且等不及 → SessionBusyError，而不是悄悄写坏历史。"""
    provider = TimedProvider(delay=0.5)
    service = make_service(store, provider, busy_timeout=0)

    session = store.create()
    thread, release = hold_lock(service._lock_for(session.id))
    try:
        with pytest.raises(SessionBusyError):
            list(service.send(session.id, "抢进来"))
    finally:
        release.set()
        thread.join()

    # 被拒的请求不能留下任何痕迹
    assert store.load(session.id).conversation.messages == []


def test_busy_timeout_large_enough_waits_and_succeeds(store: MemoryStore):
    """等得久一点就能排上队：两个请求都成功，没有丢失。"""
    provider = TimedProvider(delay=0.2)
    service = make_service(store, provider, busy_timeout=5.0)

    session = store.create()
    results: dict = {}
    errors: list = []
    barrier = threading.Barrier(2)

    def worker(tag: str) -> None:
        barrier.wait()
        send_in_thread(service, session.id, tag, results, errors)

    threads = [threading.Thread(target=worker, args=(t,)) for t in ("先", "后")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(store.load(session.id).conversation.messages) == 4


def test_clear_rejects_when_session_busy(store: MemoryStore):
    """会话正在生成时点清空：快速报错，不阻塞，也不误清空。"""
    provider = TimedProvider(delay=0.5)
    service = make_service(store, provider, housekeeping_timeout=0)

    session = store.create()
    session.conversation.add(Role.USER, "别删我")
    store.save(session)

    thread, release = hold_lock(service._lock_for(session.id))
    try:
        with pytest.raises(SessionBusyError):
            service.clear_session(session.id)
    finally:
        release.set()
        thread.join()

    loaded = store.load(session.id)
    assert len(loaded.conversation.messages) == 1


def test_delete_rejects_when_session_busy(store: MemoryStore):
    """会话正在生成时删除：快速报错，文件还在（避免被在途回复写回来）。"""
    provider = TimedProvider(delay=0.5)
    service = make_service(store, provider, housekeeping_timeout=0)

    session = store.create()
    thread, release = hold_lock(service._lock_for(session.id))
    try:
        with pytest.raises(SessionBusyError):
            service.delete_session(session.id)
    finally:
        release.set()
        thread.join()

    assert store.load(session.id) is not None


def test_delete_waits_for_inflight_then_removes(store: MemoryStore):
    """空闲时可以正常删除。"""
    provider = TimedProvider(delay=0)
    service = make_service(store, provider)

    session = store.create()
    service.delete_session(session.id)
    assert store.load(session.id) is None


# ---------------------------------------------------------------------------
# 3. 存储层：并发写 + 并发读不产生半截 JSON
# ---------------------------------------------------------------------------


def test_atomic_save_survives_concurrent_reads(tmp_path: Path):
    """一边 8 线程并发写同一会话，一边并发 list()：不能读到半截 JSON。

    用 os.replace 原子替换前，写入中途的快照可以被并发读者看到，
    list() 会把它当损坏文件跳过——表现为"会话列表里会话突然消失"。
    """
    store = MemoryStore(tmp_path)
    session = store.create()

    stop = threading.Event()
    seen_missing = threading.Event()

    def writers() -> None:
        for i in range(60):
            session.conversation.add(Role.USER, f"消息{i}")
            session.conversation.add(Role.ASSISTANT, f"回复{i}")
            store.save(session)

    def readers() -> None:
        while not stop.is_set():
            ids = {s.id for s in store.list()}
            if session.id not in ids:
                seen_missing.set()

    w = threading.Thread(target=writers)
    rs = [threading.Thread(target=readers) for _ in range(3)]
    for t in rs:
        t.start()
    w.start()
    w.join()
    stop.set()
    for t in rs:
        t.join()

    assert not seen_missing.is_set(), "并发写期间会话在 list() 里短暂消失了"

    # 最终文件必须是合法 JSON，且消息完整
    raw = json.loads((tmp_path / f"{session.id}.json").read_text(encoding="utf-8"))
    assert len(raw["messages"]) == 120
    assert store.load(session.id).conversation.messages[-1].content == "回复59"


def test_no_leftover_tmp_files(tmp_path: Path):
    """原子写入的临时文件不应残留（残留会污染目录、混淆 glob）。"""
    store = MemoryStore(tmp_path)
    session = store.create()
    for i in range(20):
        session.conversation.add(Role.USER, f"消息{i}")
        store.save(session)

    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_max_turns_clamped_to_summary_keep_turns(store: MemoryStore):
    """max_turns 不能比摘要保留窗口更短——否则未摘要的历史会被 _trim 静默丢掉。"""
    provider = TimedProvider()
    service = make_service(store, provider, max_turns=2, summary_keep_turns=20)

    assert service._max_turns == 20


def test_max_turns_kept_when_larger(store: MemoryStore):
    """max_turns 本来就更大时不动它。"""
    provider = TimedProvider()
    service = make_service(store, provider, max_turns=40, summary_keep_turns=20)

    assert service._max_turns == 40
