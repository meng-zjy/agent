"""知识库、文档与授权的持久化边界，以及仅检索已授权索引的适配器。"""

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from src.core.rag import DocumentStore
from src.core.security import SecurityViolation


class KnowledgeError(ValueError):
    """可安全返回给浏览器的知识库输入错误。"""


class KnowledgeCatalog:
    """独立管理知识数据；通过 AuthStore 的公开方法判断管理员身份。

    写操作与审计记录同事务提交。普通用户只获得公开库和显式授权库，
    管理权限在这里再次校验，不能仅依赖页面或路由。
    """

    def __init__(self, path: str | Path, is_admin: Callable[[int], bool]):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._is_admin = is_admin
        self.lock = threading.RLock()
        with self._database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    public INTEGER NOT NULL CHECK(public IN (0, 1)),
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_by INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY, knowledge_base_id TEXT NOT NULL
                        REFERENCES knowledge_bases(id),
                    name TEXT NOT NULL, content TEXT NOT NULL,
                    created_by INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(knowledge_base_id, name)
                );
                CREATE TABLE IF NOT EXISTS knowledge_grants (
                    knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id),
                    user_id INTEGER NOT NULL, granted_by INTEGER NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(knowledge_base_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS knowledge_audit (
                    id INTEGER PRIMARY KEY, actor_id INTEGER NOT NULL,
                    action TEXT NOT NULL, target_id TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def _database(self):
        db = sqlite3.connect(self._path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def require_admin(self, actor_id: int) -> None:
        if not self._is_admin(actor_id):
            raise PermissionError("需要管理员权限")

    def _audit(self, db, actor_id: int, action: str, target_id: str) -> None:
        db.execute(
            "INSERT INTO knowledge_audit(actor_id, action, target_id, created_at) VALUES (?, ?, ?, ?)",
            (actor_id, action, target_id, _now()),
        )

    def list_bases(self, user_id: int) -> list[dict]:
        with self._database() as db:
            if self._is_admin(user_id):
                rows = db.execute("SELECT * FROM knowledge_bases ORDER BY created_at, id")
            else:
                rows = db.execute("""SELECT * FROM knowledge_bases WHERE public = 1 OR id IN (
                    SELECT knowledge_base_id FROM knowledge_grants WHERE user_id = ?)
                    ORDER BY created_at, id""", (user_id,))
            return [dict(row) for row in rows]

    def _require_read(self, user_id: int, base_id: str) -> None:
        if not any(base["id"] == base_id for base in self.list_bases(user_id)):
            # 不区分不存在和无权限，避免泄露受限库的存在。
            raise LookupError("知识库不存在")

    def create_base(self, actor_id: int, name: str, public: bool = False) -> dict:
        self.require_admin(actor_id)
        name = name.strip()
        if not name or len(name) > 80:
            raise KnowledgeError("知识库名称长度必须为 1 到 80 个字符")
        base_id = uuid.uuid4().hex
        with self.lock, self._database() as db:
            db.execute("""INSERT INTO knowledge_bases
                (id, name, public, created_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (base_id, name, int(public), actor_id, _now(), _now()))
            self._audit(db, actor_id, "create_base", base_id)
        return next(base for base in self.list_bases(actor_id) if base["id"] == base_id)

    def list_documents(self, user_id: int, base_id: str) -> list[dict]:
        with self.lock:
            self._require_read(user_id, base_id)
            with self._database() as db:
                return [dict(row) for row in db.execute("""SELECT id, name, updated_at
                    FROM knowledge_documents WHERE knowledge_base_id = ? ORDER BY name""", (base_id,))]

    def read_document(self, user_id: int, base_id: str, document_id: str) -> dict:
        with self.lock:
            self._require_read(user_id, base_id)
            with self._database() as db:
                row = db.execute("""SELECT id, name, content FROM knowledge_documents
                    WHERE knowledge_base_id = ? AND id = ?""", (base_id, document_id)).fetchone()
                if row is None:
                    raise LookupError("文档不存在")
                return dict(row)

    def put_document(self, actor_id: int, base_id: str, name: str, content: str) -> None:
        self.require_admin(actor_id)
        name = name.strip()
        if (not name or len(name) > 120 or "/" in name or "\\" in name
                or Path(name).suffix.lower() not in {".md", ".txt"}):
            raise KnowledgeError("文档名须为不含路径的 .md 或 .txt 文件名，最多 120 字")
        if not content.strip() or len(content) > 200_000:
            raise KnowledgeError("文档正文长度必须为 1 到 200000 个字符")
        with self.lock, self._database() as db:
            self._require_read(actor_id, base_id)
            db.execute("""INSERT INTO knowledge_documents
                (id, knowledge_base_id, name, content, created_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(knowledge_base_id, name) DO UPDATE SET
                    content = excluded.content, updated_at = excluded.updated_at""",
                (uuid.uuid4().hex, base_id, name, content, actor_id, _now(), _now()))
            db.execute("UPDATE knowledge_bases SET revision = revision + 1, updated_at = ? WHERE id = ?",
                       (_now(), base_id))
            self._audit(db, actor_id, "put_document", base_id)

    def delete_document(self, actor_id: int, base_id: str, document_id: str) -> None:
        self.require_admin(actor_id)
        with self.lock, self._database() as db:
            deleted = db.execute("DELETE FROM knowledge_documents WHERE knowledge_base_id = ? AND id = ?",
                                 (base_id, document_id))
            if not deleted.rowcount:
                raise LookupError("文档不存在")
            db.execute("UPDATE knowledge_bases SET revision = revision + 1, updated_at = ? WHERE id = ?",
                       (_now(), base_id))
            self._audit(db, actor_id, "delete_document", document_id)

    def grants(self, actor_id: int, base_id: str) -> list[int]:
        self.require_admin(actor_id)
        self._require_read(actor_id, base_id)
        with self._database() as db:
            return [row[0] for row in db.execute(
                "SELECT user_id FROM knowledge_grants WHERE knowledge_base_id = ? ORDER BY user_id", (base_id,))]

    def set_grant(self, actor_id: int, base_id: str, user_id: int, enabled: bool) -> None:
        """重复授权和重复撤权均安全；调用方先确认用户确实存在。"""
        self.require_admin(actor_id)
        with self.lock, self._database() as db:
            self._require_read(actor_id, base_id)
            if enabled:
                db.execute("INSERT OR IGNORE INTO knowledge_grants VALUES (?, ?, ?, ?)",
                           (base_id, user_id, actor_id, _now()))
            else:
                db.execute("DELETE FROM knowledge_grants WHERE knowledge_base_id = ? AND user_id = ?",
                           (base_id, user_id))
            self._audit(db, actor_id, "grant" if enabled else "revoke", f"{base_id}:{user_id}")

    def index_documents(self, base_id: str) -> list[dict]:
        """仅供检索适配器在授权范围内构建索引，不是公开读接口。"""
        with self._database() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, name, content FROM knowledge_documents WHERE knowledge_base_id = ? ORDER BY name",
                (base_id,))]

    def import_legacy(self, directory: Path) -> None:
        """旧目录一次性快照到受限库；不把未知来源文档默认公开。"""
        with self.lock, self._database() as db:
            if db.execute("SELECT 1 FROM knowledge_bases WHERE id = 'legacy'").fetchone():
                return
            db.execute("""INSERT INTO knowledge_bases
                (id, name, public, created_at, updated_at) VALUES ('legacy', '原有知识库', 0, ?, ?)""",
                (_now(), _now()))
            root = directory.resolve()
            for path in sorted(directory.rglob("*")):
                if (path.is_file() and path.suffix.lower() in {".txt", ".md"}
                        and not path.is_symlink() and path.resolve().is_relative_to(root)):
                    db.execute("""INSERT INTO knowledge_documents
                        (id, knowledge_base_id, name, content, created_at, updated_at)
                        VALUES (?, 'legacy', ?, ?, ?, ?)""",
                        (uuid.uuid4().hex, path.relative_to(directory).as_posix(),
                         path.read_text(encoding="utf-8"), _now(), _now()))


class AuthorizedRetriever:
    """共享索引但不共享用户身份；每次工具执行重新读取授权，无结果缓存。"""

    def __init__(self, catalog: KnowledgeCatalog, embedder, chunk_size=400, chunk_overlap=50):
        self.catalog = catalog
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._indexes: dict[str, tuple[int, DocumentStore]] = {}

    def search_for_user(self, user_id: int | None, query: str, top_k: int = 4):
        if user_id is None:
            raise SecurityViolation("知识库检索需要登录身份。")
        # 授权读取、索引和搜索使用同一锁；撤权提交后发起的检索不会读到旧授权。
        with self.catalog.lock:
            candidates = []
            for base in self.catalog.list_bases(user_id):
                cached = self._indexes.get(base["id"])
                if cached is None or cached[0] != base["revision"]:
                    store = DocumentStore(self._embedder, self._chunk_size, self._chunk_overlap)
                    for document in self.catalog.index_documents(base["id"]):
                        source = f'{base["name"]}/{document["name"]} (kb={base["id"]}, doc={document["id"]})'
                        store.ingest_text(document["content"], source)
                    self._indexes[base["id"]] = (base["revision"], store)
                store = self._indexes[base["id"]][1]
                candidates.extend(store.search_scored(query, top_k))
            candidates.sort(key=lambda item: item[0], reverse=True)
            return [chunk for _, chunk in candidates[:top_k]]


def _now() -> str:
    return datetime.now(UTC).isoformat()
