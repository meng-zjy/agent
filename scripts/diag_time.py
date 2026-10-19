"""验证：用真实会话 20260914-223126-555088 的历史，问"现在几点"。

修复前：模型复读历史里 7 天前那条（2026年9月16日 星期三 21:41）。
修复后：上下文里有每轮现算的实时时间，模型应给出今天的日期。

为避免污染真实会话，把它整段历史复制到临时会话里跑。
"""

import datetime
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.bootstrap import build_retriever, build_service  # noqa: E402
from src.config import settings  # noqa: E402
from src.core.memory import Session  # noqa: E402
from src.core.models import Conversation, Message, Role  # noqa: E402
from src.core.tools import ToolRegistry  # noqa: E402

settings.session_dir = tempfile.mkdtemp(prefix="verify_time_")

service = build_service(build_retriever())

# 把你那个真实会话的历史整段搬过来
src = json.loads(
    (ROOT / "data/sessions/20260914-223126-555088.json").read_text(encoding="utf-8")
)
session = Session(
    id="verify-stale-time",
    conversation=Conversation([
        Message(role=Role(m["role"]), content=m["content"]) for m in src["messages"]
    ]),
    summary=src.get("summary", ""),
)
service._memory.save(session)

print(f"已从真实会话搬入 {len(src['messages'])} 条历史")
print("历史里已有的旧时间回答：")
for m in src["messages"]:
    if m["role"] == "assistant" and "现在" in m["content"]:
        print(f"   {m['content'][:70]!r}")

registry = service._agent._tools
original_execute = registry.execute
calls = []


def logged_execute(name, args):
    calls.append(name)
    print(f"  🔧 工具被调用: {name}")
    return original_execute(name, args)


registry.execute = logged_execute

print("\n" + "=" * 60)
print("提问：现在几点")
print("=" * 60)
answer = "".join(service.send(session.id, "现在几点"))
print(answer.strip())

today = datetime.date.today()
print("\n" + "=" * 60)
print(f"工具调用: {calls or '（未调用）'}")
print(f"今天真实日期: {today.isoformat()}（{today:%m}月{today.day}日）")
hit = today.isoformat() in answer or f"{today.month}月{today.day}日" in answer
if hit:
    print("✅ 修复生效：回答的是今天")
else:
    print("❌ 仍答的是历史里的旧时间")
