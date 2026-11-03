"""知识库授权回归：使用假向量模型和假聊天模型，不联网。"""

from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.core.auth import AuthStore
from src.core.knowledge import AuthorizedRetriever, KnowledgeCatalog, KnowledgeError
from src.core.security import SecurityViolation, ToolExecutionContext
from src.core.tools import build_default_tools
from src.api import create_app
from src.core.provider import Completion, ToolCall
from tests.test_api import FakeProvider
from tests.test_rag import KeywordEmbedder


@pytest.fixture
def knowledge(tmp_path):
    auth = AuthStore(tmp_path / "auth.db", "invite")
    admin, _ = auth.register("admin", "password-123", "invite")
    alice, _ = auth.register("alice", "password-123", "invite")
    bob, _ = auth.register("bob", "password-123", "invite")
    catalog = KnowledgeCatalog(tmp_path / "knowledge.db", auth.is_admin)
    return auth, catalog, admin, alice, bob


def test_first_owner_is_admin_and_survives_restart(knowledge, tmp_path):
    auth, _, admin, alice, _ = knowledge
    assert admin.is_admin
    assert not alice.is_admin
    reopened = AuthStore(tmp_path / "auth.db", "invite")
    assert reopened.login("admin", "password-123")[0].is_admin
    assert not reopened.login("alice", "password-123")[0].is_admin


def test_existing_accounts_migrate_to_first_owner_admin(tmp_path):
    import sqlite3

    auth = AuthStore(tmp_path / "auth.db", "invite")
    first, _ = auth.register("first", "password-123", "invite")
    auth.register("second", "password-123", "invite")
    with sqlite3.connect(tmp_path / "auth.db") as db:
        db.execute("DROP TABLE administrators")
    migrated = AuthStore(tmp_path / "auth.db", "invite")
    assert migrated.is_admin(first.id)
    assert not migrated.is_admin(first.id + 1)


def test_authorization_before_ranking_and_revocation(knowledge):
    _, catalog, admin, alice, bob = knowledge
    public = catalog.create_base(admin.id, "公共", True)["id"]
    secret = catalog.create_base(admin.id, "受限")["id"]
    catalog.put_document(admin.id, public, "天气.md", "今天天气不错")
    catalog.put_document(admin.id, secret, "宠物.md", "狗的秘密资料")
    retriever = AuthorizedRetriever(catalog, KeywordEmbedder())

    # 未授权的高分结果不得挤掉公共资料；不存在先全库搜索再过滤。
    assert retriever.search_for_user(alice.id, "狗", 1)[0].text == "今天天气不错"
    catalog.set_grant(admin.id, secret, alice.id, True)
    catalog.set_grant(admin.id, secret, alice.id, True)
    assert retriever.search_for_user(alice.id, "狗", 1)[0].text == "狗的秘密资料"
    assert retriever.search_for_user(bob.id, "狗", 1)[0].text == "今天天气不错"
    catalog.set_grant(admin.id, secret, alice.id, False)
    catalog.set_grant(admin.id, secret, alice.id, False)
    assert retriever.search_for_user(alice.id, "狗", 1)[0].text == "今天天气不错"
    with pytest.raises(SecurityViolation):
        retriever.search_for_user(None, "狗")


def test_tool_identity_cannot_be_supplied_by_model(knowledge):
    _, catalog, admin, alice, bob = knowledge
    base = catalog.create_base(admin.id, "私有")["id"]
    catalog.put_document(admin.id, base, "a.txt", "狗的秘密")
    catalog.set_grant(admin.id, base, alice.id, True)
    registry = build_default_tools(AuthorizedRetriever(catalog, KeywordEmbedder()))
    with pytest.raises(SecurityViolation):
        registry.execute("search_knowledge_base", {"query": "狗", "user_id": alice.id},
                         ToolExecutionContext(user_id=bob.id))
    with pytest.raises(SecurityViolation):
        registry.execute("search_knowledge_base", {"query": "狗"})
    assert "秘密" not in registry.execute("search_knowledge_base", {"query": "狗"},
                                          ToolExecutionContext(user_id=bob.id))

    # 同一个 registry 和索引并发复用，身份只来自各次调用的上下文。
    def search(user_id):
        return registry.execute("search_knowledge_base", {"query": "狗"},
                                ToolExecutionContext(user_id=user_id))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(search, [alice.id, bob.id] * 5))
    assert all("秘密" in output for output in results[::2])
    assert all("秘密" not in output for output in results[1::2])


def test_documents_update_index_and_access_is_checked(knowledge):
    _, catalog, admin, alice, _ = knowledge
    base = catalog.create_base(admin.id, "资料")["id"]
    catalog.set_grant(admin.id, base, alice.id, True)
    catalog.put_document(admin.id, base, "a.txt", "狗的旧资料")
    retriever = AuthorizedRetriever(catalog, KeywordEmbedder())
    assert "旧" in retriever.search_for_user(alice.id, "狗")[0].text
    catalog.put_document(admin.id, base, "a.txt", "狗的新资料")
    assert "新" in retriever.search_for_user(alice.id, "狗")[0].text
    document_id = catalog.list_documents(alice.id, base)[0]["id"]
    catalog.set_grant(admin.id, base, alice.id, False)
    with pytest.raises(LookupError):
        catalog.read_document(alice.id, base, document_id)
    with pytest.raises(PermissionError):
        catalog.put_document(alice.id, base, "a.txt", "覆盖")
    catalog.delete_document(admin.id, base, document_id)
    assert not retriever.search_for_user(admin.id, "狗")
    with pytest.raises(KnowledgeError):
        catalog.put_document(admin.id, base, "../escape.txt", "资料")


def test_legacy_snapshot_is_private_and_idempotent(knowledge, tmp_path):
    _, catalog, admin, alice, _ = knowledge
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "a.md").write_text("旧资料", encoding="utf-8")
    catalog.import_legacy(directory)
    catalog.import_legacy(directory)
    assert not catalog.list_bases(alice.id)
    assert len(catalog.list_documents(admin.id, "legacy")) == 1


def test_grants_documents_and_audit_survive_restart(knowledge, tmp_path):
    import sqlite3

    auth, catalog, admin, alice, _ = knowledge
    base = catalog.create_base(admin.id, "持久化")["id"]
    catalog.put_document(admin.id, base, "a.txt", "文档正文不写审计日志")
    catalog.set_grant(admin.id, base, alice.id, True)
    reopened = KnowledgeCatalog(tmp_path / "knowledge.db", auth.is_admin)
    assert reopened.list_bases(alice.id)[0]["id"] == base
    assert len(reopened.list_documents(alice.id, base)) == 1
    with sqlite3.connect(tmp_path / "knowledge.db") as db:
        records = db.execute("SELECT actor_id, action, target_id FROM knowledge_audit").fetchall()
        assert [row[1] for row in records] == ["create_base", "put_document", "grant"]
        assert "文档正文" not in repr(records)


class SearchingProvider(FakeProvider):
    """记录模型真正收到的资料，验证隔离发生在模型请求之前。"""

    def __init__(self):
        super().__init__()
        self.tool_results = []

    def complete(self, messages, tools=None):
        if messages[-1].role == "tool":
            self.tool_results.append(messages[-1].content)
            return Completion(finish_reason="stop", content="已查询")
        return Completion(finish_reason="tool_calls", tool_calls=[
            ToolCall("search", "search_knowledge_base", '{"query":"狗"}'),
        ])


def test_api_admin_grants_preview_and_chat_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "account_enabled", True)
    monkeypatch.setattr(settings, "registration_invite_code", "invite")
    monkeypatch.setattr(settings, "auth_database", str(tmp_path / "auth.db"))
    monkeypatch.setattr(settings, "session_dir", str(tmp_path / "sessions"))
    monkeypatch.setattr(settings, "summary_keep_turns", 0)
    auth = AuthStore(settings.auth_database, "invite")
    auth.register("admin", "password-123", "invite")
    auth.register("alice", "password-123", "invite")
    provider = SearchingProvider()
    import src.bootstrap as bootstrap
    import src.core.embeddings as embeddings

    monkeypatch.setattr(bootstrap, "build_provider", lambda: provider)
    monkeypatch.setattr(bootstrap, "build_mcp_tools", lambda: [])
    monkeypatch.setattr(embeddings, "FastembedEmbedder", lambda **kwargs: KeywordEmbedder())
    app = create_app()
    admin = TestClient(app, base_url="https://testserver")
    alice = TestClient(app, base_url="https://testserver")
    anonymous = TestClient(app)
    for client, username in [(admin, "admin"), (alice, "alice")]:
        assert client.post("/api/auth/login", json={"username": username, "password": "password-123"}).status_code == 200
    assert admin.get("/api/auth/me").json()["user"]["is_admin"]
    assert anonymous.get("/api/knowledge-bases").status_code == 401
    assert alice.post("/api/knowledge-bases", json={"name": "伪造"}).status_code == 403
    assert alice.get("/api/admin/users").status_code == 403
    base = admin.post("/api/knowledge-bases", json={"name": "机密"}).json()["id"]
    base_url = f"/api/knowledge-bases/{base}"
    assert admin.put(base_url + "/documents", json={"name": "a.md", "content": "狗的机密资料"}).status_code == 200
    document_id = admin.get(base_url + "/documents").json()[0]["id"]
    preview = base_url + "/documents/" + document_id
    assert alice.get(preview).status_code == 404
    assert alice.put(base_url + "/grants/2").status_code == 403
    assert admin.put(base_url + "/grants/999").status_code == 404
    session_id = alice.post("/api/sessions").json()["id"]
    message_url = f"/api/sessions/{session_id}/messages"
    assert "[DONE]" in alice.post(message_url, json={"text": "查资料"}).text
    assert "机密资料" not in provider.tool_results[-1]
    assert admin.put(base_url + "/grants/2").status_code == 200
    assert alice.get(preview).json()["content"] == "狗的机密资料"
    alice.post(message_url, json={"text": "查资料"})
    assert "机密资料" in provider.tool_results[-1]
    assert admin.delete(base_url + "/grants/2").status_code == 200
    assert alice.get(preview).status_code == 404
    alice.post(message_url, json={"text": "查资料"})
    assert "机密资料" not in provider.tool_results[-1]
    assert alice.delete(preview).status_code == 403


def test_account_mode_without_identity_does_not_build_unscoped_retriever(monkeypatch):
    import src.bootstrap as bootstrap
    import src.core.embeddings as embeddings

    monkeypatch.setattr(settings, "account_enabled", True)
    calls = []

    def forbidden_embedder(**kwargs):
        calls.append(kwargs)
        raise AssertionError("无身份时不应加载全目录索引")

    monkeypatch.setattr(embeddings, "FastembedEmbedder", forbidden_embedder)
    assert bootstrap.build_retriever() is None
    assert not calls
