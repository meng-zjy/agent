"""配置模块测试。

注意：所有用例都用 `_env_file=None` 或临时 .env 文件，
**绝不读取项目根目录的真实 .env**，避免测试依赖真实 API Key。
"""

import pytest

from src.config import Settings

_ENV_KEYS = (
    "LLM_PROVIDER",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "SENSENOVA_BASE_URL",
    "SENSENOVA_API_KEY",
    "SENSENOVA_MODEL",
    "MAX_TOKENS",
    "TEMPERATURE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个用例前清空相关环境变量，保证互不影响。"""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "deepseek"          # 默认用 DeepSeek（2026-10-09 起）
    assert s.deepseek_base_url == "https://api.deepseek.com/v1"
    assert s.deepseek_api_key == ""
    assert s.deepseek_model == "deepseek-chat"
    # SenseNova 配置保留，以便 LLM_PROVIDER=sensenova 切回
    assert s.sensenova_base_url == "https://token.sensenova.cn/v1"
    assert s.sensenova_api_key == ""
    assert s.sensenova_model == "sensenova-6.8-flash-lite"
    assert s.max_tokens is None
    assert s.temperature is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test-model")
    monkeypatch.setenv("MAX_TOKENS", "512")

    s = Settings(_env_file=None)
    assert s.llm_provider == "deepseek"
    assert s.deepseek_api_key == "sk-test-key"
    assert s.deepseek_model == "deepseek-test-model"
    assert s.max_tokens == 512


def test_env_file_parsing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=deepseek\nDEEPSEEK_API_KEY=sk-from-file\n"
        "DEEPSEEK_MODEL=deepseek-file\n",
        encoding="utf-8",
    )

    s = Settings(_env_file=str(env_file))
    assert s.deepseek_api_key == "sk-from-file"
    assert s.deepseek_model == "deepseek-file"
    assert s.deepseek_base_url == "https://api.deepseek.com/v1"  # 其余用默认值


def test_llm_provider_case_insensitive(monkeypatch):
    """人工编辑 .env 时大小写写错是常态，pydantic 读环境变量本身不区分大小写。"""
    monkeypatch.setenv("llm_provider", "SENSONOVA")
    s = Settings(_env_file=None)
    assert s.llm_provider == "SENSONOVA"        # 原样保存
