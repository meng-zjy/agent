"""全局 pytest 配置。

作用：把日志目录指到每个用例独立的临时目录，避免测试往项目里的
`data/logs/agent.log` 写日志——那个文件本来是给用户排查线上问题的，
混进测试噪音会让事后翻日志时分不清哪些是真实使用、哪些是测试跑出来的。

只 patch `settings.log_dir`，不动 logger 的 handler：logger 是**懒初始化**的
（`get_logger()` 首次调用时才 `setup()`），所以不写日志的用例不会创建任何文件。
写日志的用例在 teardown 时 `reset()` 掉自己挂的 handler，防止 140 个用例各挂一份
导致文件句柄泄漏。

`tests/test_logger.py` 自己有更严格的隔离（换目录 + reset 前后清理），
两者叠加不冲突：conftest 先执行，随后 test_logger 的 fixture 会覆盖 `log_dir`
并在 reset 后重新挂 handler。
"""

import pytest

import src.core.logger as logger_mod
from src.config import settings


@pytest.fixture(autouse=True)
def _isolate_log_dir(tmp_path, monkeypatch):
    """隔离本地 .env，并让每个用例使用独立日志目录。"""
    # 开发机可能启用了账户功能。测试中未显式启用账户的场景仍要覆盖默认匿名行为，
    # 账户测试会在自身用例里覆盖这些值。
    monkeypatch.setattr(settings, "account_enabled", False)
    monkeypatch.setattr(settings, "registration_invite_code", "")
    monkeypatch.setattr(settings, "auth_cookie_secure", True)
    monkeypatch.setattr(settings, "knowledge_database", str(tmp_path / "knowledge.db"))
    monkeypatch.setattr(settings, "knowledge_dir", str(tmp_path / "knowledge"))
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "embedding_cache_dir", str(tmp_path / "models"))
    monkeypatch.setattr(settings, "embedding_local_files_only", False)
    monkeypatch.setattr(settings, "log_dir", str(tmp_path / "logs"))
    yield
    logger_mod.reset()
