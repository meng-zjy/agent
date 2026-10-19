"""MCP 真实端到端自测：连 filesystem 服务器，验证模型能调用 MCP 工具。"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WS = Path("data/e2e_mcp_ws")
TMP = Path("data/e2e_mcp_sessions")
shutil.rmtree(WS, ignore_errors=True)
shutil.rmtree(TMP, ignore_errors=True)
WS.mkdir(parents=True, exist_ok=True)
(WS / "hello.txt").write_text("你好 MCP", encoding="utf-8")
(WS / "readme.md").write_text("# 测试文档", encoding="utf-8")

from src.bootstrap import build_service
from src.config import settings

settings.session_dir = str(TMP)
settings.mcp_allowed_dir = str(WS)
settings.knowledge_dir = "data/e2e_mcp_empty"   # 空知识库，聚焦测 MCP

print("构建服务（内部会连接 MCP filesystem 服务器）...")
service = build_service(None)
s = service.new_session()
print(f"会话 {s.id}\n")


def ask(text, tag):
    out = "".join(service.send(s.id, text))
    print(f"[{tag}] 你: {text}")
    print(f"[{tag}] AI: {out.strip()[:260]}\n")
    return out


a1 = ask(f"用工具列出目录 {WS} 里有哪些文件", "列目录")
print("  ✓ 模型调用了 MCP list_directory" if "hello.txt" in a1 or "readme" in a1
      else "  ✗ 未命中文件名")

a2 = ask("用工具把「你好世界」这行字写进 greeting.txt 文件里", "写文件")
written = (WS / "greeting.txt").exists()
print("  ✓ 模型调用了 MCP write_file，文件已创建" if written else "  ✗ 文件未创建")
if written:
    print(f"    文件内容: {(WS / 'greeting.txt').read_text(encoding='utf-8')!r}")

shutil.rmtree(WS, ignore_errors=True)
shutil.rmtree(TMP, ignore_errors=True)
print("\n完成（临时目录已清理）")
