"""RAG 启动状态和持久化缓存配置；假模型测试不依赖下载网络。"""

import sys
import types

import pytest
from fastapi.testclient import TestClient

import src.bootstrap as bootstrap
import src.core.embeddings as embeddings
from src.api import create_app
from src.config import settings
from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.service import ChatService
from tests.test_api import FakeProvider
from tests.test_rag import KeywordEmbedder


def test_embedding_adapter_passes_cache_and_offline_options(tmp_path, monkeypatch):
    calls = []

    def fake_embedding(**options):
        calls.append(options)
        return object()

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=fake_embedding))
    cache_dir = tmp_path / "persisted-models"
    embeddings.FastembedEmbedder("test-model", cache_dir=str(cache_dir), local_files_only=True)
    assert cache_dir.is_dir()
    assert calls == [{"model_name": "test-model", "cache_dir": str(cache_dir), "local_files_only": True}]


def test_bootstrap_uses_persistent_cache_and_offline_mode(tmp_path, monkeypatch):
    calls = []

    def fake_embedder(**options):
        calls.append(options)
        return KeywordEmbedder()

    monkeypatch.setattr(embeddings, "FastembedEmbedder", fake_embedder)
    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(settings, "embedding_cache_dir", str(tmp_path / "models"))
    monkeypatch.setattr(settings, "embedding_local_files_only", True)
    store = bootstrap.build_retriever()
    assert store is not None
    assert calls == [{
        "model_name": settings.embedding_model,
        "cache_dir": str(tmp_path / "models"),
        "local_files_only": True,
    }]


def test_disabled_rag_does_not_construct_model(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "rag_enabled", False)
    monkeypatch.setattr(embeddings, "FastembedEmbedder", lambda **options: calls.append(options))
    assert bootstrap.build_retriever() is None
    assert not calls


def test_failed_model_degrades_chat_and_reports_unavailable(tmp_path, monkeypatch):
    def unavailable_model(**options):
        raise OSError("private-cache-path: model unavailable")

    monkeypatch.setattr(settings, "rag_enabled", True)
    monkeypatch.setattr(embeddings, "FastembedEmbedder", unavailable_model)
    monkeypatch.setattr(bootstrap, "build_provider", FakeProvider)
    monkeypatch.setattr(bootstrap, "build_mcp_tools", lambda: [])
    monkeypatch.setattr(settings, "session_dir", str(tmp_path / "sessions"))
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["rag"] == {"state": "unavailable", "available": False}
    assert "private-cache-path" not in health.text
    session_id = client.post("/api/sessions").json()["id"]
    response = client.post(f"/api/sessions/{session_id}/messages", json={"text": "你好"})
    assert "你好呀" in response.text
    assert "[DONE]" in response.text


@pytest.mark.parametrize("enabled,available,state", [
    (True, True, "ready"),
    (True, False, "unavailable"),
    (False, False, "disabled"),
])
def test_health_reports_service_capability_without_network(tmp_path, enabled, available, state):
    service = ChatService(ChatAgent(FakeProvider()), MemoryStore(tmp_path / "sessions"),
                          retrieval_enabled=enabled, retrieval_available=available)
    client = TestClient(create_app(service))
    assert client.get("/health").json()["rag"] == {"state": state, "available": available}


def test_disabled_status_propagates_through_production_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "rag_enabled", False)
    monkeypatch.setattr(settings, "session_dir", str(tmp_path / "sessions"))
    monkeypatch.setattr(bootstrap, "build_provider", FakeProvider)
    monkeypatch.setattr(bootstrap, "build_mcp_tools", lambda: [])
    client = TestClient(create_app())
    assert client.get("/health").json()["rag"] == {"state": "disabled", "available": False}
