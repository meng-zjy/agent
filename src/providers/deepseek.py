"""DeepSeek Provider（当前默认供应商，2026-10-09 起）。

接口：`https://api.deepseek.com/v1`，OpenAI 协议兼容，openai SDK 直接可用。

模型选型（实测，见 docs/14 §5）：
- `deepseek-chat`    默认。非 reasoning：没有思考 token，所以「思考吃光预算 → 空回复」
                     这条路彻底消失；首 token 约 0.6s，真逐 token 流式（610 字 437 段）。
- `deepseek-flash`   支持视觉（base64 图片），但仍是 reasoning 模型——
                     `max_tokens` 一律被思考吃光，空回复照旧；且首 content 约 7.5s。
- `deepseek-v4-pro`  不支持视觉，首 content 实测 37.4s，不适合对话。
"""

from __future__ import annotations

from src.providers.openai_compat import OpenAICompatProvider

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek：OpenAI 协议兼容，无需任何额外参数。"""

    PROVIDER_NAME = "deepseek"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        **kwargs,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, model=model, **kwargs)
