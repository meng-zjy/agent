"""日志基础设施测试（M8）。

覆盖：幂等初始化、轮转配置 + 真触发一次轮转、级别过滤、消息正文不进日志（隐私属性）。
全部不花额度、不联网。
"""

import logging
from pathlib import Path

import pytest

import src.core.logger as logger_mod
from src.config import settings
from src.core.logger import get_logger, reset, setup


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """每个测试独立日志目录 + 独立初始化状态，测完清理。"""
    monkeypatch.setattr(settings, "log_dir", str(tmp_path / "logs"))
    reset()
    yield
    reset()


def _flush() -> None:
    for h in logging.getLogger().handlers:
        h.flush()


def _log_text(tmp_path: Path) -> str:
    p = tmp_path / "logs" / "agent.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _backup_files(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "logs").glob("agent.log.*"))


def _write_lines(n: int = 300) -> None:
    """打 n 条短日志。

    消息体**只用 ASCII**：RotatingFileHandler 的 `shouldRollover` 用
    `len(msg.encode())` 算字节数，中文会让「字符数 ≠ 字节数」，
    断言里关于 maxBytes 的算术就不好算了。
    """
    log = get_logger("rot")
    for i in range(n):
        log.info("rotation check %04d", i)
    _flush()


def test_get_logger_creates_file_and_writes(tmp_path):
    log = get_logger("smoke")
    log.info("hello 世界")
    _flush()
    assert "hello 世界" in _log_text(tmp_path)


def test_setup_is_idempotent(tmp_path):
    get_logger("a")
    n = len(logging.getLogger().handlers)
    get_logger("b")
    get_logger("c")
    assert len(logging.getLogger().handlers) == n


def test_level_from_settings_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "log_level", "WARNING")
    log = get_logger("level")
    log.info("这条不该进文件")
    log.warning("这条应该进文件")
    _flush()
    text = _log_text(tmp_path)
    assert "这条不该进文件" not in text
    assert "这条应该进文件" in text


def test_invalid_level_falls_back_to_info(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "log_level", "BOGUS_LEVEL")
    log = get_logger("level")
    log.info("非法级别时 INFO 仍应写入")
    _flush()
    assert "非法级别时 INFO 仍应写入" in _log_text(tmp_path)


def test_rotation_config(tmp_path):
    get_logger("rot")
    handler = next(
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    )
    assert handler.maxBytes == logger_mod._MAX_BYTES
    assert handler.backupCount == logger_mod._BACKUP_COUNT


def test_rotation_actually_rotates(tmp_path, monkeypatch):
    """真触发一次轮转——不只是断言配置值。

    docs/11 §7 发现八 判这条「配置正确但没跑过」，理由是「要写满 1MB 才能
    触发」。那个理由其实站不住：`RotatingFileHandler` 的轮转逻辑跟阈值
    大小无关，512 字节和 1MB 走的是同一条 `shouldRollover` / `rollOver`，
    阈值只是个参数。所以把阈值调小就能在毫秒级真跑一遍，不必等生产流量。
    """
    max_bytes = 512
    monkeypatch.setattr(logger_mod, "_MAX_BYTES", max_bytes)
    _write_lines(300)
    log_file = tmp_path / "logs" / "agent.log"
    # 核心断言：写了很多之后文件仍没超阈值，说明真在轮转而不是无限增长
    assert log_file.stat().st_size <= max_bytes
    assert (tmp_path / "logs" / "agent.log.1").exists()


def test_rotation_caps_backup_count(tmp_path, monkeypatch):
    """轮转文件最多 `backupCount` 个——最旧的会被删掉。

    `test_rotation_config` 只验了 `backupCount == 3` 这个**数字**；
    这条验的是「真的滚出第 4 份时最旧的会被清掉」，即永远不会有
    `agent.log.4`。`rollOver()` 每次先删 `agent.log.3` 再往前挪，所以
    备份索引的硬上限就是 `backupCount`。
    """
    monkeypatch.setattr(logger_mod, "_MAX_BYTES", 512)
    _write_lines(300)
    assert _backup_files(tmp_path) == ["agent.log.1", "agent.log.2", "agent.log.3"]
    assert not (tmp_path / "logs" / "agent.log.4").exists()


def test_message_body_never_logged(tmp_path):
    """隐私属性：日志只记元数据，不记消息正文。

    这是 docs/11 §4.3 的设计承诺。测试方式是构造一段明显可识别的
    消息内容，断言它不会出现在日志文件里。
    """
    from src.core.agent import ChatAgent
    from src.core.memory import MemoryStore
    from src.core.models import Message, Role
    from src.core.provider import BaseProvider, Completion
    from src.core.service import ChatService

    SECRET = "这是一段极不可能随机出现的秘密正文 XXXXZZZZ9999"

    class EmptyProvider(BaseProvider):
        def chat(self, messages):
            return ""
        def chat_stream(self, messages):
            yield ""
        def complete(self, messages, tools=None):
            return Completion(finish_reason="stop", content="")

    store = MemoryStore(tmp_path / "sessions")
    svc = ChatService(
        ChatAgent(EmptyProvider()), store, system_prompt="x",
        summary_keep_turns=0,
    )
    sid = store.create().id
    # 消费完生成器，让 _send_locked 完整跑一遍并落盘
    list(svc.send(sid, SECRET))
    _flush()
    assert SECRET not in _log_text(tmp_path)
    # 但元数据应该在
    assert "空回复" in _log_text(tmp_path)


def test_reset_allows_reinit(tmp_path):
    get_logger("before").info("第一次写入")
    _flush()
    first = _log_text(tmp_path)
    assert "第一次写入" in first

    reset()
    assert not any(
        getattr(h, "_is_agent_handler", False)
        for h in logging.getLogger().handlers
    )
    get_logger("after").info("重置后可用")
    _flush()
    assert "重置后可用" in _log_text(tmp_path)
