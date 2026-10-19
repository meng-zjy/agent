"""M4 真实端到端自测：RAG 检索 + 工具调用 + 记忆共存（真实调模型 + 真实 embedding）。"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KB = Path("data/e2e_kb")
TMP = Path("data/e2e_rag_sessions")
shutil.rmtree(KB, ignore_errors=True)
shutil.rmtree(TMP, ignore_errors=True)
KB.mkdir(parents=True, exist_ok=True)

# 只有文档里才有的信息（模型不可能凭空知道）
(KB / "公司背景.md").write_text(
    "公司内部背景资料\n\n"
    "我们的项目代号是「蓝鲸计划」。\n"
    "项目负责人是王小锤，办公地点在上海张江。\n"
    "核心产品是一款面向开发者的 AI 编程助手。\n",
    encoding="utf-8",
)

from src.config import settings
from src.core.agent import ChatAgent
from src.bootstrap import build_provider
from src.core.embeddings import FastembedEmbedder
from src.core.memory import MemoryStore
from src.core.rag import DocumentStore
from src.core.service import ChatService
from src.core.tools import build_default_tools


def build(tmp, retriever):
    p = build_provider()
    agent = ChatAgent(p, tools=build_default_tools(retriever, top_k=settings.retrieval_top_k))
    return ChatService(agent, MemoryStore(tmp), system_prompt=settings.system_prompt,
                       max_turns=settings.max_history_turns, max_chars=settings.max_history_chars)


print("构建 embedding + 摄入知识库（模型已缓存，应较快）...")
embedder = FastembedEmbedder(model_name=settings.embedding_model)
retriever = DocumentStore(embedder, settings.chunk_size, settings.chunk_overlap)
n = retriever.ingest_dir(KB)
print(f"已摄入 {n} 个片段\n")

svc = build(TMP, retriever)
s = svc.new_session()


def ask(text, tag):
    out = "".join(svc.send(s.id, text))
    print(f"[{tag}] 你: {text}")
    print(f"[{tag}] AI: {out.strip()[:240]}\n")
    return out


a1 = ask("我们公司的项目代号是什么？负责人是谁？", "RAG")
print("  ✓ RAG 命中（基于文档）" if ("蓝鲸" in a1 and "王小锤" in a1) else "  ✗ 未基于文档回答")

a2 = ask("帮我算 123 × 45 等于多少", "计算")
print("  ✓ 计算工具仍工作" if "5535" in a2 else "  ✗ 计算失效")

a3 = ask("记住我的名字叫小码", "记忆")
a4 = ask("我叫什么名字？", "追问")
print("  ✓ 记忆仍工作" if "小码" in a4 else "  ✗ 记忆失效")

shutil.rmtree(KB, ignore_errors=True)
shutil.rmtree(TMP, ignore_errors=True)
print("\n完成（临时目录已清理）")
