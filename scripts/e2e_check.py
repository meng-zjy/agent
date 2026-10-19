"""端到端自检：真实调模型 + 记忆持久化 + 重启续接。

用法: python scripts/e2e_check.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TMP = Path("data/sessions_e2e")
if TMP.exists():
    shutil.rmtree(TMP)

from src.bootstrap import build_provider, provider_label
from src.config import settings
from src.core.agent import ChatAgent
from src.core.memory import MemoryStore
from src.core.models import Role
from src.core.models import Message, Role
from src.core.service import ChatService

print(f"[0] 配置: {provider_label()}")


def build(tmp: Path) -> ChatService:
    """用指定会话目录构建一套完整服务（模拟一次进程启动）。"""
    provider = build_provider()
    return ChatService(
        ChatAgent(provider),
        MemoryStore(tmp),
        system_prompt=settings.system_prompt,
        max_turns=settings.max_history_turns,
        max_chars=settings.max_history_chars,
    )


def ask(svc, sid, text, tag):
    out = "".join(svc.send(sid, text))
    print(f"  {tag} 你: {text}")
    print(f"  {tag} AI: {out.strip()[:160]}")
    return out


print("[1] 第一次启动：新建会话，真实调用模型")
svc = build(TMP)
s = svc.new_session()
print(f"  会话 id = {s.id}")
ask(svc, s.id, "请记住：我的名字叫小明，我的宠物狗叫黄豆。", "R1")
ask(svc, s.id, "我叫什么名字？我的狗叫什么？", "R2")

print("[2] 查磁盘：会话文件是否真的落盘")
files = sorted(TMP.glob("*.json"))
print(f"  {TMP}/ 下 {len(files)} 个 json:")
for f in files:
    print(f"    {f.name}  ({f.stat().st_size} 字节)")

print("[3] 模拟进程重启：用新的 ChatService 实例重新加载")
svc2 = build(TMP)
s2 = svc2.resume_or_create()
print(f"  resume_or_create -> {s2.id}，历史 {len(s2.conversation.messages)} 条消息")
assert s2.id == s.id, f"续接失败: {s2.id} != {s.id}"
assert len(s2.conversation.messages) == 4, f"历史条数异常: {len(s2.conversation.messages)}"
print("  ✓ 重启后自动续接最近会话，4 条历史完整读回")
ask(svc2, s2.id, "重启之后，我还记得我叫什么吗？狗呢？", "R3")

print("[4] 多会话管理")
s3 = svc2.new_session()
svc2.clear_session(s.id)
lst = svc2.list_sessions()
print(f"  list() -> {len(lst)} 个会话，最新在前: {[x.id for x in lst]}")
assert lst[0].id == s.id, "clear 后 clear 过的会话应因 updated_at 最新而排第一"
assert len(lst[0].conversation.messages) == 0, "/clear 后应为空"
assert len(s3.conversation.messages) == 0, "new_session 应返回空历史"
print("  ✓ /clear 生效；/sessions 排序正确；新会话不受影响")

print("[5] 截断策略 _trim 单元测试（不耗模型额度）")
svc_small = ChatService(
    ChatAgent(build_provider()),
    MemoryStore(TMP),
    system_prompt="SYS",
    max_turns=2,
    max_chars=30,
)
sm = svc_small.new_session()
for i in range(6):
    sm.conversation.messages.append(Message(role=Role.USER, content=f"Q{i}"))
    sm.conversation.messages.append(Message(role=Role.ASSISTANT, content="x" * 12))
ctx = svc_small._build_context(sm)
roles = [m.role.value for m in ctx]
print(f"  12 条历史 -> _build_context 后 {len(ctx)} 条: {roles}")
assert ctx[0].role is Role.SYSTEM, "system prompt 应在最前"
assert len(ctx) == 5, "max_turns=2 应保留 system + 最后 2 组问答 = 5 条"
assert ctx[1].content == "Q4", "轮数截断应保留最旧的截断点是 Q4"
print("  ✓ 轮数截断（保留最后 N 组问答）+ system prompt 插入顺序正确")

ctx2 = svc_small._build_context(sm)  # 再验证一次幂等
assert len(ctx2) == len(ctx), "_trim 不应修改原历史"
assert len(sm.conversation.messages) == 12, "_trim 是纯函数，不能污染磁盘历史"
print("  ✓ _trim 不修改原始 Session，历史保持 12 条完整")

shutil.rmtree(TMP)
print("\n全部通过 ✓  临时目录已清理")
