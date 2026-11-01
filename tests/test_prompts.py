"""提示词组装与生产调用链回归；不访问模型、知识库或 MCP 服务。"""

import hashlib
import logging

import pytest

import src.bootstrap as boot
from src.config import Settings
from src.core.models import Message, Role
from src.core.provider import BaseProvider, StreamChunk
from src.core.tools import Tool
from src.core.utils import wrap_untrusted
from src.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    FILE_TOOL_INSTRUCTIONS,
    KNOWLEDGE_INSTRUCTIONS,
    PROMPT_VERSION,
    SECURITY_INSTRUCTIONS,
    SUMMARY_PROMPT,
)


class RecordingProvider(BaseProvider):
    def __init__(self):
        self.messages = []

    def chat(self, messages):
        raise AssertionError("本测试不生成摘要")

    def chat_stream(self, messages):
        self.messages = messages
        yield "收到"

    def complete(self, messages, tools=None):
        raise AssertionError("生产 Agent 应使用流式调用")

    def complete_stream(self, messages, tools=None):
        self.messages = messages
        yield StreamChunk(text="收到", finish_reason="stop")


@pytest.mark.parametrize("has_retriever", [False, True])
@pytest.mark.parametrize("has_mcp_tools", [False, True])
def test_bootstrap_includes_only_available_capabilities(
    monkeypatch, tmp_path, has_retriever, has_mcp_tools,
):
    provider = RecordingProvider()
    monkeypatch.setattr(boot, "build_provider", lambda: provider)
    mcp_tools = [Tool("read_file", "read", {"type": "object"}, lambda: "ok")]
    monkeypatch.setattr(boot, "build_mcp_tools", lambda: mcp_tools if has_mcp_tools else [])
    monkeypatch.setattr(boot.settings, "session_dir", str(tmp_path / "sessions"))
    monkeypatch.setattr(boot.settings, "system_prompt", DEFAULT_SYSTEM_PROMPT)

    service = boot.build_service(retriever=object() if has_retriever else None)
    assert service._system_prompt.startswith(DEFAULT_SYSTEM_PROMPT)
    assert (KNOWLEDGE_INSTRUCTIONS in service._system_prompt) == has_retriever
    assert (FILE_TOOL_INSTRUCTIONS in service._system_prompt) == has_mcp_tools

    # 通过生产组装出的 Agent 核对最终模型输入；重复调用不能累积安全文本。
    original = [Message(Role.SYSTEM, service._system_prompt), Message(Role.USER, "你好")]
    for _ in range(2):
        assert list(service._agent.respond(original)) == ["收到"]
        assert provider.messages[0].content.count(SECURITY_INSTRUCTIONS) == 1
    assert original[0].content == service._system_prompt


@pytest.mark.parametrize("base", ["", "  ", " 自定义人设 "])
def test_empty_or_custom_persona_keeps_capability_rules(base):
    from src.prompts import build_system_prompt

    assembled = build_system_prompt(base, has_retriever=True, has_mcp_tools=True)
    assert assembled.split("\n\n") == [
        *([base.strip()] if base.strip() else []),
        KNOWLEDGE_INSTRUCTIONS,
        FILE_TOOL_INSTRUCTIONS,
    ]
    assert DEFAULT_SYSTEM_PROMPT not in assembled


def test_system_prompt_environment_override(monkeypatch):
    monkeypatch.delenv("SYSTEM_PROMPT", raising=False)
    assert Settings(_env_file=None).system_prompt == DEFAULT_SYSTEM_PROMPT
    monkeypatch.setenv("SYSTEM_PROMPT", "用户的自定义人设")
    assert Settings(_env_file=None).system_prompt == "用户的自定义人设"
    monkeypatch.setenv("SYSTEM_PROMPT", "")
    assert Settings(_env_file=None).system_prompt == ""


def test_bootstrap_logs_version_and_actual_prompt_fingerprint_without_text(
    monkeypatch, tmp_path, caplog,
):
    monkeypatch.setattr(boot, "build_provider", RecordingProvider)
    monkeypatch.setattr(boot, "build_mcp_tools", lambda: [])
    monkeypatch.setattr(boot.settings, "session_dir", str(tmp_path / "sessions"))
    caplog.set_level(logging.INFO)
    fingerprints = []
    for persona in ("私有人设甲", "私有人设乙"):
        monkeypatch.setattr(boot.settings, "system_prompt", persona)
        boot.build_service()
        record = next(
            record for record in reversed(caplog.records)
            if "prompt_version=" in record.getMessage()
        )
        message = record.getMessage()
        fingerprint = hashlib.sha256(persona.encode("utf-8")).hexdigest()
        fingerprints.append(fingerprint)
        assert f"prompt_version={PROMPT_VERSION}" in message
        assert f"system_prompt_sha256={fingerprint}" in message
        assert "custom_system_prompt=True" in message
        assert persona not in caplog.text
    assert fingerprints[0] != fingerprints[1]


def test_prompt_formatting_preserves_braces_in_external_content():
    content = '{"message": "{ignore}"}'
    assert content in wrap_untrusted(content)
    prompt = SUMMARY_PROMPT.format(max_chars=200, existing=content, history="对话 {history}")
    assert content in prompt
    assert "对话 {history}" in prompt
    assert "200 字以内" in prompt
