"""RAG 核心测试（M4）：切块、摄入、检索、工具接入。全部用假 Embedder，不联网。"""

from pathlib import Path

from src.core.embeddings import Embedder
from src.core.rag import DocumentStore, chunk_text, cosine
from src.core.tools import build_default_tools


_VOCAB = ["狗", "天气", "苹果"]


class KeywordEmbedder(Embedder):
    """按关键词命中构造向量：向量 = 各关键词是否出现，便于控制检索结果。"""

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if w in t else 0.0 for w in _VOCAB] for t in texts]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 切块 + 余弦
# ---------------------------------------------------------------------------

def test_chunk_text_basic():
    assert chunk_text("0123456789", size=4, overlap=0) == ["0123", "4567", "89"]


def test_chunk_text_with_overlap():
    # step = size - overlap = 3
    assert chunk_text("0123456789", size=5, overlap=2) == ["01234", "34567", "6789", "9"]


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert abs(cosine([1.0, 1.0], [1.0, 1.0]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 摄入 + 检索
# ---------------------------------------------------------------------------

def test_ingest_file(tmp_path):
    store = DocumentStore(KeywordEmbedder(), chunk_size=20, chunk_overlap=0)
    n = store.ingest_file(_write(tmp_path / "a.txt", "小明养了一只狗" * 10))
    assert n == len(store) == 4   # 70 字符 / 20 = 4 块


def test_search_returns_most_relevant(tmp_path):
    store = DocumentStore(KeywordEmbedder(), chunk_size=100, chunk_overlap=0)
    store.ingest_file(_write(tmp_path / "a.txt", "小明养了一只狗"))
    store.ingest_file(_write(tmp_path / "b.txt", "今天天气不错"))

    results = store.search("他有一条宠物狗", top_k=1)
    assert len(results) == 1
    assert "狗" in results[0].text
    assert results[0].source == "a.txt"


def test_search_empty_store():
    store = DocumentStore(KeywordEmbedder())
    assert store.search("随便问", top_k=4) == []


def test_clear(tmp_path):
    store = DocumentStore(KeywordEmbedder(), chunk_size=100, chunk_overlap=0)
    store.ingest_file(_write(tmp_path / "a.txt", "小明养了一只狗"))
    assert len(store) == 1
    store.clear()
    assert len(store) == 0
    assert store.search("狗", top_k=4) == []


# ---------------------------------------------------------------------------
# 工具接入
# ---------------------------------------------------------------------------

def test_build_default_tools_registers_search_when_retriever_given():
    store = DocumentStore(KeywordEmbedder())
    names = {s["function"]["name"] for s in build_default_tools(store).schemas()}
    assert "search_knowledge_base" in names


def test_build_default_tools_no_search_without_retriever():
    names = {s["function"]["name"] for s in build_default_tools().schemas()}
    assert "search_knowledge_base" not in names


def test_search_tool_end_to_end(tmp_path):
    store = DocumentStore(KeywordEmbedder(), chunk_size=100, chunk_overlap=0)
    store.ingest_file(_write(tmp_path / "a.txt", "小明养了一只狗"))
    reg = build_default_tools(store, top_k=1)
    out = reg.execute("search_knowledge_base", {"query": "他有一条宠物狗"})
    assert "狗" in out
    assert "a.txt" in out
