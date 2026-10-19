"""Spike：验证 fastembed + BGE 中文模型在本机可用（含模型下载）。

步骤：加载 BAAI/bge-small-zh-v1.5（首次会联网下载 ~100MB）→
     嵌入 3 句中文 → 算余弦相似度（同义句应高、无关句应低）。
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "BAAI/bge-small-zh-v1.5"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


print(f"[1] 加载模型 {MODEL}（首次下载约 100MB，请稍候）...")
t0 = time.time()
from fastembed import TextEmbedding

model = TextEmbedding(model_name=MODEL)
print(f"    加载完成，耗时 {time.time() - t0:.1f}s")

sentences = [
    "小明养了一只狗",        # s1
    "他有一条宠物狗",        # s2（和 s1 同义）
    "今天天气不错",          # s3（无关）
    "我们的项目叫 SenseNova Agent，对标豆包和千问",  # s4（项目背景）
]
print(f"[2] 嵌入 {len(sentences)} 句中文...")
vecs = list(model.embed(sentences))
dims = {len(v) for v in vecs}
print(f"    得到 {len(vecs)} 个向量，维度 {dims}")

v1, v2, v3, v4 = vecs
print("\n[3] 语义相似度检查")
print(f"    同义  (s1,s2) = {cosine(v1, v2):.3f}   ← 期望 > 0.6")
print(f"    无关  (s1,s3) = {cosine(v1, v3):.3f}   ← 期望 < 同义")
print(f"    相关  (s1,s4) = {cosine(v1, v4):.3f}")

same = cosine(v1, v2)
diff = cosine(v1, v3)
if same > 0.6 and same > diff:
    print("\n✅ 通过：中文语义向量化有效（同义句明显更相似）")
else:
    print("\n⚠️ 结果不理想：同义句相似度不够高，需检查模型")
