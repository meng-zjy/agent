"""Spike 补充：确认 glm-5.2 是不是能用的文本模型。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

from src.config import settings

client = OpenAI(base_url=settings.sensenova_base_url, api_key=settings.sensenova_api_key)

for model in ["glm-5.2", "sensenova-6.8-flash-lite"]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请只回复两个字：收到"}],
            max_tokens=20,
        )
        print(f"✓ {model} 纯文本回复: {resp.choices[0].message.content!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ {model}: {type(exc).__name__}: {str(exc)[:120]}")
