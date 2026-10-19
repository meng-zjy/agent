"""会话记忆：Session 数据模型 + MemoryStore 文件存储。

M2 记忆系统核心之一。会话历史以 JSON 文件形式持久化到磁盘，
程序重启后仍可继续对话。后续 M4 RAG 可复用这份历史数据。
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.core.models import Conversation, Message, Role

_SESSION_FILE = "{}.json"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def _new_id() -> str:
    """可读、可排序的会话 id，形如 20260908-153000-123456。

    含微秒后缀，保证同一秒内多次创建也不会 id 碰撞。
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


@dataclass
class Session:
    """一个会话 = 元信息 + 消息历史（Conversation）+ 早期对话摘要（09）。"""

    id: str
    conversation: Conversation
    summary: str = ""          # 早期对话的滚动摘要；空字符串 = 还没溢出过
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        """序列化为可写 JSON 的字典。"""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "messages": self.conversation.to_openai_list(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        messages = [
            Message(role=Role(m["role"]), content=m["content"]) for m in data["messages"]
        ]
        return cls(
            id=data["id"],
            conversation=Conversation(messages=messages),
            summary=data.get("summary", ""),   # 旧版 JSON 无此字段，按"无摘要"处理
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class MemoryStore:
    """把 Session 存成 data/sessions/<id>.json。只负责存取，不掺对话逻辑。

    所有磁盘读写都持一个实例锁：并发多会话、多请求同时访问时，
    不会读到写了一半的文件，也不会出现「后写覆盖先写」。
    """

    def __init__(self, session_dir: str | Path) -> None:
        self._dir = Path(session_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, session_id: str) -> Path:
        return self._dir / _SESSION_FILE.format(session_id)

    def create(self) -> Session:
        session = Session(id=_new_id(), conversation=Conversation(messages=[]))
        self.save(session)
        return session

    def load(self, session_id: str) -> Session | None:
        with self._lock:
            path = self._path(session_id)
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as f:
                return Session.from_dict(json.load(f))

    def save(self, session: Session) -> None:
        """写盘；每次落盘都刷新 updated_at 为当前时间。

        先写临时文件再 os.replace 原子替换：进程中途被杀也不会留下半截 JSON。
        替换被别的进程占用时（例如同时开着 CLI 和 Web）短暂重试。
        """
        with self._lock:
            session.updated_at = _now()
            self._atomic_write(self._path(session.id), session.to_dict())

    def _atomic_write(self, path: Path, data: dict) -> None:
        tmp = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        for attempt in range(3):
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmp, path)
                return
            except PermissionError:      # 文件正被别处打开，稍等再换
                time.sleep(0.02 * (attempt + 1))
        raise

    def list(self) -> list[Session]:
        """按最近更新倒序返回所有会话。"""
        with self._lock:
            sessions = []
            for path in self._dir.glob(_SESSION_FILE.format("*")):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        sessions.append(Session.from_dict(json.load(f)))
                except (json.JSONDecodeError, KeyError, ValueError, OSError):
                    continue  # 跳过损坏文件，不阻塞启动
            sessions.sort(key=lambda s: s.updated_at, reverse=True)
            return sessions

    def latest(self) -> Session | None:
        sessions = self.list()
        return sessions[0] if sessions else None

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._path(session_id).unlink(missing_ok=True)
