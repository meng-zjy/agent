"""探针：当前配置的 provider 到底是不是逐 token 流式下发？

如果服务端是"整段算完再一次性下发"，那"真流式"改造（重构成流式工具循环）
不会带来任何可见收益，反而要多花一倍 token。用一次真实请求验证。

读 .env 里当前启用的 LLM_PROVIDER（2026-10-09 起默认 deepseek-chat），
结论对照见 docs/14 §5：sensenova-6.8-flash-lite 是「算完再一次性下发」，
deepseek-chat 是真逐 token。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import active_provider_config  # noqa: E402
from src.core.models import Message, Role  # noqa: E402
from src.providers.openai_compat import OpenAICompatProvider  # noqa: E402

name, base_url, api_key, model = active_provider_config()
if not api_key:
    print(f"未配置 {name.upper()}_API_KEY，跳过")
    sys.exit(0)

provider = OpenAICompatProvider(base_url=base_url, api_key=api_key, model=model)
messages = [Message(role=Role.USER, content="用 80 个字描述一下春天。")]

print(f"模型：{provider._model}")
t0 = time.time()
intervals: list[float] = []
prev = t0
chunks = 0
for piece in provider.chat_stream(messages):
    now = time.time()
    intervals.append(now - prev)
    prev = now
    chunks += 1
    if chunks <= 5:
        print(f"  第 {chunks} 段 (+{intervals[-1]:.3f}s)：{piece!r}")

total = time.time() - t0
print(f"\n共 {chunks} 段，总耗时 {total:.2f}s")
print(f"段间隔：min={min(intervals):.4f}s  max={max(intervals):.4f}s  "
      f"sum={sum(intervals):.4f}s")
if max(intervals) / max(total, 1e-9) > 0.9:
    print("→ 结论：服务端一次性下发（无逐 token 流式），真流式改造无可见收益")
else:
    print("→ 结论：服务端是逐段下发的，可以支持真流式")
