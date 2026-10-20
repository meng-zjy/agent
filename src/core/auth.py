"""账户、登录会话和 Agent 会话归属的 SQLite 存储。"""

import hashlib
import hmac
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


_SCRYPT_N = 2 ** 14
_TOKEN_BYTES = 32


@dataclass(frozen=True)
class User:
    id: int
    username: str


class AuthError(ValueError):
    """注册或登录输入不合法。错误文案可以安全地返回给浏览器。"""


class AuthStore:
    """认证数据的唯一所有者；不读取或修改 JSON 对话内容。"""

    def __init__(self, database_path: str | Path, invite_code: str, session_hours: int = 168) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._invite_code = invite_code
        self._session_hours = session_hours
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _database(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash BLOB NOT NULL,
                    password_salt BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS login_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_owners (
                    agent_session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id)
                );
            """)

    def register(self, username: str, password: str, invite_code: str) -> tuple[User, str]:
        username = username.strip()
        if not hmac.compare_digest(invite_code, self._invite_code):
            raise AuthError("邀请码错误")
        if not username or len(username) > 32 or not username.replace("_", "").isalnum():
            raise AuthError("用户名只能含中文、字母、数字和下划线，最多 32 个字符")
        if len(password) < 10 or len(password) > 128:
            raise AuthError("密码长度必须为 10 到 128 个字符")
        salt = secrets.token_bytes(16)
        password_hash = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=8, p=1)
        with self._lock, self._database() as db:
            try:
                cursor = db.execute(
                    "INSERT INTO users(username, password_hash, password_salt, created_at) VALUES (?, ?, ?, ?)",
                    (username, password_hash, salt, _utc_now()),
                )
            except sqlite3.IntegrityError as exc:
                raise AuthError("用户名已存在") from exc
            user = User(id=cursor.lastrowid, username=username)
            return user, self._create_login_session(db, user.id)

    def login(self, username: str, password: str) -> tuple[User, str]:
        with self._lock, self._database() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
            if row is None:
                raise AuthError("用户名或密码错误")
            actual = hashlib.scrypt(password.encode(), salt=row["password_salt"], n=_SCRYPT_N, r=8, p=1)
            if not hmac.compare_digest(actual, row["password_hash"]):
                raise AuthError("用户名或密码错误")
            return User(id=row["id"], username=row["username"]), self._create_login_session(db, row["id"])

    def user_for_token(self, token: str | None) -> User | None:
        if not token:
            return None
        with self._lock, self._database() as db:
            row = db.execute("""SELECT users.id, users.username, login_sessions.expires_at
                              FROM login_sessions JOIN users ON users.id = login_sessions.user_id
                              WHERE token_hash = ?""", (_hash_token(token),)).fetchone()
            if row is None or row["expires_at"] <= _utc_now():
                if row is not None:
                    db.execute("DELETE FROM login_sessions WHERE token_hash = ?", (_hash_token(token),))
                return None
            return User(id=row["id"], username=row["username"])

    def logout(self, token: str | None) -> None:
        if token:
            with self._lock, self._database() as db:
                db.execute("DELETE FROM login_sessions WHERE token_hash = ?", (_hash_token(token),))

    def own(self, user_id: int, agent_session_id: str) -> None:
        with self._lock, self._database() as db:
            db.execute("INSERT INTO session_owners(agent_session_id, user_id) VALUES (?, ?)", (agent_session_id, user_id))

    def claim_unowned(self, user_id: int, agent_session_ids: list[str]) -> None:
        """首次启用账户时认领旧会话；已有归属永远不覆盖。"""
        with self._lock, self._database() as db:
            db.executemany(
                "INSERT OR IGNORE INTO session_owners(agent_session_id, user_id) VALUES (?, ?)",
                [(session_id, user_id) for session_id in agent_session_ids],
            )

    def owns(self, user_id: int, agent_session_id: str) -> bool:
        with self._database() as db:
            return db.execute("SELECT 1 FROM session_owners WHERE agent_session_id = ? AND user_id = ?", (agent_session_id, user_id)).fetchone() is not None

    def owned_ids(self, user_id: int) -> set[str]:
        with self._connect() as db:
            return {row[0] for row in db.execute("SELECT agent_session_id FROM session_owners WHERE user_id = ?", (user_id,))}

    def user_count(self) -> int:
        with self._database() as db:
            return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def _create_login_session(self, db: sqlite3.Connection, user_id: int) -> str:
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        expires = datetime.now(UTC) + timedelta(hours=self._session_hours)
        db.execute("INSERT INTO login_sessions(token_hash, user_id, expires_at) VALUES (?, ?, ?)", (_hash_token(token), user_id, expires.isoformat()))
        return token


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
