"""bootstrap 组装测试：LLM_PROVIDER 选择逻辑（全部 mock，不联网）。

换供应商只改 `.env` 里的 `LLM_PROVIDER` 一个值（docs/14），
这里锁住的是「选择 + 报错」的行为，避免以后加新供应商时把这两条弄坏。
"""

import types

import pytest

import src.bootstrap as boot
from src.providers.deepseek import DeepSeekProvider
from src.providers.sensenova import SenseNovaProvider


def _settings(**overrides):
    """构造一个假的 settings，只带 build_provider 用到的字段。"""
    d = dict(
        llm_provider="deepseek",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_api_key="sk-d",
        deepseek_model="deepseek-chat",
        sensenova_base_url="https://token.sensenova.cn/v1",
        sensenova_api_key="sk-s",
        sensenova_model="sensenova-6.8-flash-lite",
        max_tokens=None,
        temperature=None,
        request_timeout=90.0,
        max_retries=3,
        deadline=30.0,
        breaker_failure_threshold=3,
        breaker_open_seconds=30.0,
        mcp_enabled=True,
    )
    d.update(overrides)
    return types.SimpleNamespace(**d)


def test_build_provider_defaults_to_deepseek(monkeypatch):
    monkeypatch.setattr(boot, "settings", _settings())
    p = boot.build_provider()
    assert isinstance(p, DeepSeekProvider)
    assert p.label == "deepseek/deepseek-chat"
    assert str(p._client.base_url).rstrip("/") == "https://api.deepseek.com/v1"


def test_build_provider_selects_sensenova(monkeypatch):
    monkeypatch.setattr(boot, "settings", _settings(llm_provider="sensenova"))
    p = boot.build_provider()
    assert isinstance(p, SenseNovaProvider)
    assert str(p._client.base_url).rstrip("/") == "https://token.sensenova.cn/v1"


def test_build_provider_ignores_case_and_whitespace(monkeypatch):
    """人工编辑 .env 时大小写和空格写错是常态。"""
    monkeypatch.setattr(boot, "settings", _settings(llm_provider=" SenSenova "))
    assert isinstance(boot.build_provider(), SenseNovaProvider)


def test_build_provider_passes_budget_and_breaker_knobs(monkeypatch):
    """P0/P1 的旋钮必须真的传到 provider 内部——否则改 .env 不生效却没人发现。"""
    monkeypatch.setattr(
        boot, "settings",
        _settings(max_retries=5, deadline=12.5,
                  breaker_failure_threshold=7, breaker_open_seconds=45.0),
    )
    p = boot.build_provider()
    assert p._max_retries == 5
    assert p._deadline == 12.5
    assert p._timeout == 90.0            # read 超时和总预算是两个旋钮，别混成一个
    assert p._breaker.failure_threshold == 7
    assert p._breaker.open_seconds == 45.0


def test_build_provider_can_disable_breaker(monkeypatch):
    """阈值配成 0 是「关闭熔断」的开关：永不打开，照常重试。"""
    monkeypatch.setattr(boot, "settings", _settings(breaker_failure_threshold=0))
    p = boot.build_provider()
    from math import inf
    assert p._breaker.failure_threshold == inf
    for _ in range(50):
        p._breaker.record_failure(ValueError("boom"))
    assert p._breaker.state == "closed"


def test_build_provider_rejects_unknown_name(monkeypatch, capsys):
    monkeypatch.setattr(boot, "settings", _settings(llm_provider="gpt"))
    with pytest.raises(SystemExit) as exc_info:
        boot.build_provider()
    err = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "无法识别" in err
    assert "deepseek" in err and "sensenova" in err      # 报错要告诉用户可选值


def test_build_provider_missing_key(monkeypatch, capsys):
    monkeypatch.setattr(boot, "settings", _settings(deepseek_api_key=""))
    with pytest.raises(SystemExit) as exc_info:
        boot.build_provider()
    err = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "DEEPSEEK_API_KEY" in err                      # 报错要点出该填哪个变量名


def test_provider_label_reads_config_only(monkeypatch):
    monkeypatch.setattr(boot, "settings", _settings(llm_provider="deepseek"))
    assert boot.provider_label() == "deepseek/deepseek-chat"


def test_provider_label_for_sensenova(monkeypatch):
    monkeypatch.setattr(boot, "settings", _settings(llm_provider="sensenova"))
    assert boot.provider_label() == "sensenova/sensenova-6.8-flash-lite"


def test_build_mcp_tools_respects_explicit_disable(monkeypatch):
    """生产配置关闭 MCP 时，不能启动 npx 或创建授权目录。"""
    monkeypatch.setattr(boot, "settings", _settings(mcp_enabled=False))
    assert boot.build_mcp_tools() == []
