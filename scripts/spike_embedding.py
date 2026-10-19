"""Spike：商汤 SenseNova 是否提供 embedding 接口。

步骤：
1. 列出可用模型（看有没有 embedding 模型）
2. 尝试调 /embeddings
3. 若成功，测中文语义质量（同义句应相似、无关句应不相似）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
from src.config import settings

client = OpenAI(base_url=settings.sensenova_base_url, api_key=settings.sensenova_api_key)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


print("=== ① 列出可用模型 ===")
try:
    models = client.models.list()
    ids = [m.id for m in models.data]
    print(f"  共 {len(ids)} 个模型:")
    for i in ids:
        print(f"    - {i}")
except Exception as exc:
    print(f"  models.list 失败: {type(exc).__name__}: {exc}")

print("\n=== ② 尝试调 /embeddings ===")
# 候选模型名：优先用 models.list 里带 embed 字样的；否则逐个试
candidates = []
if 'models' in dir():
    pass

candidate_names = [
    "sensenova-embedding",
    "sensenova-embedding-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "bge-m3",
    "sensenova-6.8-flash-lite",   # 兜底：用聊天模型试
]

for name in candidate_names:
    try:
        resp = client.embeddings.create(model=name, input=["你好世界"])
        vec = resp.data[0].embedding
        print(f"  ✓ 模型 {name!r}: 成功，维度={len(vec)}")
        # ③ 中文语义质量
        s1 = "小明养了一只狗"
        s2 = "他有一条宠物狗"
        s3 = "今天天气不错"
        r = client.embeddings.create(model=name, input=[s1, s2, s3])
        v1, v2, v3 = [d.embedding for d in r.data]
        print(f"      语义相似度: 同义(s1,s2)={cosine(v1, v2):.3f}  "
              f"无关(s1,s3)={cosine(v1, v3):.3f}")
        break
    except Exception as exc:
        print(f"  ✗ 模型 {name!r}: {type(exc).__name__}: {str(exc)[:120]}")
