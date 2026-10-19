"""Spike：商汤有没有视觉模型？给候选模型发一张纯红色图片，看谁认得颜色。"""

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
    """生成一张 2x2 纯红色 PNG，返回 base64。"""
    def chunk(typ: bytes, data: bytes) -> bytes:
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
question = "这张图片是什么颜色？请只回答颜色。"
print(f"测试图片：2x2 纯红色 PNG（data URI）\n")

candidates = [
    "sensenova-u1-fast",
    "sensenova-u1.5-lite",
    "glm-5.2",
    "sensenova-6.8-flash-lite",   # 纯文本模型，作对照（预期失败）
]

for model in candidates:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            max_tokens=60,
        )
        answer = (resp.choices[0].message.content or "").strip()
        print(f"✓ {model}: {answer[:60]}")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ {model}: {type(exc).__name__}: {str(exc)[:120]}")
