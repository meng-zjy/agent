"""商汤日日新（SenseNova）Provider。

⚠ 已不再是默认供应商（2026-10-09 切到 DeepSeek，见 docs/14）。本文件保留两个用途：
  1) `.env` 里写 `LLM_PROVIDER=sensenova` 可以切回，老实测记录立刻重新适用；
  2) `docs/01`~`docs/11` 的实测数字全部基于它，切回去后那些结论仍然成立。

切走的两个原因（实测，非推测）：
- **服务端「先把整段算完再一次性下发」**：50 字提问首 token 等 62.8s，随后 62 个 chunk
  只用 4.0s——体验上是「等很久，然后整段瞬间出现」，不是打字机。
- **`sensenova-6.8-flash-lite` 是 reasoning 模型**：思考 token 计入 completion_tokens，
  `max_tokens=4096` 也会全部被思考吃光（实测 completion_tokens=4096 /
  reasoning_tokens=4096 → content 为空）。这就是 M8 查到的「空回复」根因。
"""

from __future__ import annotations

from src.providers.openai_compat import OpenAICompatProvider

DEFAULT_BASE_URL = "https://token.sensenova.cn/v1"
DEFAULT_MODEL = "sensenova-6.8-flash-lite"


class SenseNovaProvider(OpenAICompatProvider):
    """商汤日日新：OpenAI 协议兼容，无需任何额外参数。"""

    PROVIDER_NAME = "sensenova"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, model=model, **kwargs)
