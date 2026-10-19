"""Embedding 抽象层（M4 RAG）：把文本变成向量。

当前唯一实现是本地 BGE 中文模型（fastembed，ONNX 运行时）。
将来商汤若上线 embedding，再写一个 SenseNovaEmbedder 即可，上层零改动。

注意：必须在首次 import fastembed 之前设置好 HF 镜像 + 关闭 Xet 下载协议，
否则本机直连 huggingface.co 会证书校验失败 / Xet 下载 401（见 docs/05）。
"""

import os
from abc import ABC, abstractmethod

# 国内镜像 + 关闭 Xet 下载协议（必须在首次 import fastembed 之前生效）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class Embedder(ABC):
    """文本 → 向量。"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """单句向量化。"""

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（摄入文档时用）。"""


class FastembedEmbedder(Embedder):
    """本地 BGE 中文模型（fastembed，ONNX 运行时）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from fastembed import TextEmbedding  # 延迟导入：不装 fastembed 也能跑其它功能
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [list(vec) for vec in self._model.embed(texts)]
