"""Web 端到端自测：真实模型 + 真实 embedding，走 HTTP 层（TestClient）。

验证：① 首页可打开 ② 计算工具 ③ RAG 检索 ④ 记忆，全都在 Web 接口里工作。
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KB = Path("data/e2e_web_kb")
TMP = Path("data/e2e_web_sessions")
shutil.rmtree(KB, ignore_errors=True)
shutil.rmtree(TMP, ignore_errors=True)
KB.mkdir(parents=True, exist_ok=True)
(KB / "公司背景.md").write_text("项目代号是「蓝鲸计划」，负责人王小锤。\n", encoding="utf-8")

from fastapi.testclient import TestClient

from src.api import create_app
from src.bootstrap import build_retriever, build_service
from src.config import settings

# 用临时目录，避免污染真实 data/
settings.knowledge_dir = str(KB)
settings.session_dir = str(TMP)

retriever = build_retriever()
service = build_service(retriever)
client = TestClient(create_app(service))


def parse_deltas(body: str) -> str:
    """从 SSE 响应体里拼出所有 delta。"""
    parts = []
    for event in body.split("\n\n"):
        for line in event.split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                try:
                    obj = json.loads(line[6:])
                    if obj.get("delta"):
                        parts.append(obj["delta"])
                    elif obj.get("error"):
                        parts.append(f"[错误]{obj['error']}")
                except json.JSONDecodeError:
                    pass
    return "".join(parts)


def send(text, tag):
    r = client.post(f"/api/sessions/{sid}/messages", json={"text": text})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    answer = parse_deltas(r.text)
    print(f"[{tag}] 你: {text}")
    print(f"[{tag}] AI: {answer.strip()[:200]}\n")
    return answer


# ① 首页
r = client.get("/")
print(f"[首页] GET / -> {r.status_code}, {'text/html' in r.headers['content-type']}\n")

# ② 建会话
sid = client.post("/api/sessions").json()["id"]
print(f"[会话] 新建 {sid}\n")

# ③ 计算
a1 = send("帮我算 123 × 45 等于多少", "计算")
print("  ✓ 计算工具在 Web 里工作" if "5535" in a1 else "  ✗ 计算失败")

# ④ RAG
a2 = send("我们项目代号是什么？负责人是谁？", "RAG")
print("  ✓ RAG 在 Web 里工作" if ("蓝鲸" in a2 and "王小锤" in a2) else "  ✗ RAG 失败")

# ⑤ 记忆
send("记住我的名字叫小码", "记忆")
a3 = send("我叫什么名字？", "追问")
print("  ✓ 记忆在 Web 里工作" if "小码" in a3 else "  ✗ 记忆失败")

# ⑥ 会话列表 + 消息持久化
sessions = client.get("/api/sessions").json()
msg_count = len(client.get(f"/api/sessions/{sid}").json()["messages"])
print(f"\n[会话] 列表 {len(sessions)} 个，当前会话 {msg_count} 条消息（应 = 8）")

shutil.rmtree(KB, ignore_errors=True)
shutil.rmtree(TMP, ignore_errors=True)
print("\n完成（临时目录已清理）")
