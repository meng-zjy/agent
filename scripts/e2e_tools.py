"""M3 真实端到端自测：工具调用 + 记忆共存（真实调模型，耗少量额度）。

验证：① 计算题能触发 calculator；② 问时间能触发 get_current_time；
③ 工具调用不破坏记忆（先记名字，再问计算，重启后名字仍在）。
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = Path("data/e2e_tools")
shutil.rmtree(TMP, ignore_errors=True)

from src.bootstrap import build_provider
from src.config import settings
from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.service import ChatService
from src.core.tools import build_default_tools


def build(tmp):
    p = build_provider()
    return ChatService(ChatAgent(p, tools=build_default_tools()), MemoryStore(tmp),
                       system_prompt=settings.system_prompt,
                       max_turns=settings.max_history_turns, max_chars=settings.max_history_chars)


def ask(svc, sid, text, tag):
    out = "".join(svc.send(sid, text))
    print(f"[{tag}] 你: {text}")
    print(f"[{tag}] AI: {out.strip()[:200]}\n")
    return out


svc = build(TMP)
s = svc.new_session()
print(f"会话 {s.id}\n")

a1 = ask(svc, s.id, "帮我算一下 123 × 45 等于多少？", "计算")
print("  ✓ 触发 calculator" if "5535" in a1 else "  ✗ 未算出 5535")

a2 = ask(svc, s.id, "现在几点了？", "时间")
ok_time = any(c.isdigit() for c in a2)
print("  ✓ 返回了时间" if ok_time else "  ✗ 未见时间")

a3 = ask(svc, s.id, "请记住：我的名字叫小码。", "记忆")
a4 = ask(svc, s.id, "我叫什么名字？", "追问")
print("  ✓ 记住名字" if "小码" in a4 else "  ✗ 没记住名字")

print("--- 模拟重启 ---")
svc2 = build(TMP)
s2 = svc2.resume_or_create()
assert s2.id == s.id, "重启后续接失败"
a5 = ask(svc2, s2.id, "重启了，我叫什么名字？", "重启后")
print("  ✓ 重启后仍记得" if "小码" in a5 else "  ✗ 重启后忘了")

shutil.rmtree(TMP, ignore_errors=True)
print("完成（临时目录已清理）")
