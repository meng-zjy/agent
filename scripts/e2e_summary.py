"""端到端验证：长对话不丢早期记忆（09）。

花真实额度，用低阈值让"摘要"在一个很短的对话里就触发：
  1 埋下可记忆的事实
  2/3 闲聊填充（把第 1 轮挤出保留窗口，逼它进摘要）
  4 反问第 1 轮说过的事——此时第 1 轮只存在于摘要里，模型只能靠摘要回答

用法：
  python scripts/e2e_summary.py              # 默认 keep=2，聊 4 轮（省额度）
  python scripts/e2e_summary.py --keep 20    # 真实阈值，聊 22 轮（贵）
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.bootstrap import build_retriever, build_service  # noqa: E402
from src.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=2,
                        help="summary_keep_turns：保留几轮完整消息（默认 2，省额度）")
    parser.add_argument("--turns", type=int, default=4,
                        help="总共聊几轮（默认 4 = keep+2）")
    args = parser.parse_args()

    keep = args.keep
    turns = max(args.turns, keep + 2)
    print(f"阈值 summary_keep_turns={keep}，共聊 {turns} 轮\n")

    # 独立会话目录，不污染真实数据
    tmp = Path(tempfile.mkdtemp(prefix="e2e_summary_"))
    settings.session_dir = str(tmp)
    settings.summary_keep_turns = keep
    settings.knowledge_dir = "data/e2e_summary_empty"

    service = build_service(build_retriever())
    session = service.new_session()

    def ask(text: str) -> str:
        print(f"\n你：{text}")
        print("AI：", end="", flush=True)
        return "".join(service.send(session.id, text))

    fact = "我叫阿泽，我最喜欢的编程语言是 Rust"
    print(f"第 1 轮要埋的事实：{fact}")

    print(ask(fact))
    print(ask("随便聊聊，你觉得写代码最重要的是什么？"))
    print(ask("那你怎么看待代码重构？"))

    on_disk = json.loads((tmp / f"{session.id}.json").read_text(encoding="utf-8"))
    print(f"\n[此时磁盘] summary 字段：{on_disk.get('summary')!r}")
    print(f"[此时磁盘] 消息数：{len(on_disk.get('messages', []))}")

    final = ask("回到最开始：我叫什么？我最喜欢的编程语言是什么？")
    print(final)

    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    final_disk = json.loads((tmp / f"{session.id}.json").read_text(encoding="utf-8"))
    user_msgs = [m for m in final_disk.get("messages", []) if m["role"] == "user"]
    has_summary = bool(final_disk.get("summary"))
    # 按角色计数判定"第 1 轮是否被压缩"——不能搜关键词，助手回复里也会提到
    user_count_ok = len(user_msgs) == keep
    first_turn_gone = not any("我叫阿泽" in m["content"] for m in user_msgs)
    remembered = "阿泽" in final and "Rust" in final

    print(f"会话 JSON 里有摘要：{has_summary}")
    print(f"消息数：{len(final_disk.get('messages', []))}（= keep×2 = {keep * 2}）")
    print(f"用户消息条数 = {len(user_msgs)}，等于 keep={keep}：{user_count_ok}")
    print(f"第 1 轮原文已从用户消息里压缩掉：{first_turn_gone}")
    print(f"最后一轮靠摘要答出第 1 轮的信息：{remembered}")
    ok = has_summary and user_count_ok and first_turn_gone and remembered
    print(f"\n结论：{'✅ 全部通过' if ok else '❌ 未通过'}")
    print(f"临时目录：{tmp}")


if __name__ == "__main__":
    main()
