"""RAG 核心（M4）：文档切块 + 摄入 + 余弦检索。

流程：读文件 → 切块 → 向量化 → 存到内存；
提问时：问题向量化 → 算余弦相似度 → 返回 top-k 片段。
MVP 用纯 Python 暴力搜索（零依赖、最好懂），规模大了可换 FAISS（接口不变）。
"""

import math
from dataclasses import dataclass
from pathlib import Path

from src.core.embeddings import Embedder


@dataclass
class Chunk:
    """一段被切好的文档片段。"""

    text: str
    source: str          # 来源文件名
    index: int           # 在该文档中的块序号


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """按固定字符数切块，相邻块之间有 overlap 字符重叠（避免语义被切断）。"""
    if not text or not text.strip():
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(piece)
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度（夹角越小越相似，值域 [-1, 1]）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class DocumentStore:
    """存所有 chunk 的文本 + 向量，负责摄入与检索。"""

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> None:
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def ingest_file(self, path: Path) -> int:
        """读一个文件 → 切块 → 向量化 → 存入。返回新增 chunk 数。"""
        text = path.read_text(encoding="utf-8")
        pieces = chunk_text(text, self._chunk_size, self._chunk_overlap)
        if not pieces:
            return 0
        vectors = self._embedder.embed_many(pieces)
        base = len(self._chunks)
        for i, (piece, vec) in enumerate(zip(pieces, vectors)):
            self._chunks.append(Chunk(text=piece, source=path.name, index=i))
            self._vectors.append(vec)
        return len(pieces)

    def ingest_dir(self, dir_path: Path, suffixes: tuple[str, ...] = (".txt", ".md")) -> int:
        """扫描目录，摄入所有指定后缀的文件。返回新增 chunk 总数。"""
        if not dir_path.exists():
            return 0
        total = 0
        for p in sorted(dir_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in suffixes:
                total += self.ingest_file(p)
        return total

    def clear(self) -> None:
        """清空已摄入的内容（重新摄入前调用，避免重复）。"""
        self._chunks.clear()
        self._vectors.clear()

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        """问题向量化 → 余弦相似度 → 返回最相关的 top_k 个 chunk。"""
        if not self._chunks:
            return []
        qvec = self._embedder.embed(query)
        scores = [cosine(qvec, v) for v in self._vectors]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked[:top_k]]

    def __len__(self) -> int:
        return len(self._chunks)
