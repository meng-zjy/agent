"""通用小工具。"""

from src.prompts import UNTRUSTED_CONTENT_TEMPLATE
from src.prompts import UNTRUSTED_BEGIN as _UNTRUSTED_BEGIN
from src.prompts import UNTRUSTED_END as _UNTRUSTED_END


def wrap_untrusted(text: str) -> str:
    """把不可信的外部文本包进显式围栏。

    用结构化标记而不是只在 system_prompt 里说"外部指令不要执行"——纯提示词给模型
    的是抽象原则，围栏给的是具体信号，模型才知道"哪一段是外部资料"。

    诚实的边界：这是软防线，不保证生效。能挡住直接注入，挡不住精心构造的间接注入。
    """
    return UNTRUSTED_CONTENT_TEMPLATE.format(content=text)


def scrub_text(text: str) -> str:
    """清理文本中的孤立代理项，防止编码崩溃。

    背景：Windows 控制台默认 GBK 编码，UTF-8 字节流（管道输入、流式响应中
    被网络切分的多字节字符）经 surrogateescape 解码后会产生孤立代理项
    （U+DC80~U+DCFF）。本函数将其还原为字节再解码：

    - 代理项恰好组成完整字符时，无损还原原字符（如 emoji）
    - 无法还原时退化为替换字符 U+FFFD，保证不崩溃

    正常文本原样返回。
    """
    try:
        raw = text.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        return text
    return raw.decode("utf-8", "replace")
