"""M6 真实端到端：走 Web 层发一张图，验证模型看图回答。"""

import base64
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = Path("data/e2e_vision_sessions")
shutil.rmtree(TMP, ignore_errors=True)

from fastapi.testclient import TestClient

from src.api import create_app
from src.bootstrap import build_service
from src.config import settings

settings.session_dir = str(TMP)
settings.knowledge_dir = "data/e2e_vision_empty"


def make_png(rgb) -> str:
    def chunk(t, d):
        return (struct.pack(">I", len(d)) + t + d
                + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF))
    w = h = 4
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


client = TestClient(create_app(build_service(None)))
sid = client.post("/api/sessions").json()["id"]


def ask(text, images, tag):
    r = client.post(f"/api/sessions/{sid}/messages",
                    json={"text": text, "images": images})
    parts = []
    for event in r.text.split("\n\n"):
        for line in event.split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                try:
                    obj = json.loads(line[6:])
                    if obj.get("delta"):
                        parts.append(obj["delta"])
                except json.JSONDecodeError:
                    pass
    answer = "".join(parts)
    print(f"[{tag}] 你: {text}")
    print(f"[{tag}] AI: {answer.strip()[:200]}\n")
    return answer


blue = f"data:image/png;base64,{make_png((0, 0, 255, 255))}"
a1 = ask("这张图片是什么颜色？请只回答颜色", [blue], "看图")
print("  ✓ 模型看图答对" if "蓝" in a1 else "  ✗ 未答对")

# 顺带验证：图片不落盘
msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
raw = json.dumps(msgs, ensure_ascii=False)
print("  ✓ 图片未落盘" if "base64" not in raw else "  ✗ 图片泄漏进 JSON")

shutil.rmtree(TMP, ignore_errors=True)
print("\n完成（临时目录已清理）")
