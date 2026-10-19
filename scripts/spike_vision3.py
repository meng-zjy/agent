"""Spike 修正版：去掉 max_tokens，重新测视觉模型。"""

import base64
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

from src.config import settings

client = OpenAI(base_url=settings.sensenova_base_url, api_key=settings.sensenova_api_key)


def make_red_png() -> str:
    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))
    w = h = 2
    raw = b"".join(b"\x00" + b"\xff\x00\x00\xff" * w for _ in range(h))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


data_uri = f"data:image/png;base64,{make_red_png()}"

# 1. glm-5.2 纯文本（确认是能用的模型）
r = client.chat.completions.create(model="glm-5.2",
                                   messages=[{"role": "user", "content": "请只回复：收到"}])
print("glm-5.2 纯文本:", repr((r.choices[0].message.content or "")[:40]))

# 2. glm-5.2 图片（无 max_tokens）
r = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "这张图片是什么颜色？请只回答颜色。"},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]}],
)
print("glm-5.2 图片:", repr((r.choices[0].message.content or "")[:60]))

# 3. flash-lite 图片（纯文本对照，无 max_tokens）
r = client.chat.completions.create(
    model="sensenova-6.8-flash-lite",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "这张图片是什么颜色？请只回答颜色。"},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]}],
)
print("flash-lite 图片:", repr((r.choices[0].message.content or "")[:60]))
