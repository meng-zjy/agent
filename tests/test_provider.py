"""Provider 层单元测试：纯函数 + 空回复检测（mock client，不联网）。

实现只有一份 `OpenAICompatProvider`，`deepseek.py` / `sensenova.py` 只是声明身份，
所以测试参数化覆盖两家子类，同时验证「子类只是声明身份」这件事本身。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.provider import EmptyReplyError
from src.core.utils import scrub_text
from src.providers.deepseek import DEFAULT_BASE_URL as DEEPSEEK_DEFAULT_BASE_URL
from src.providers.deepseek import DEFAULT_MODEL as DEEPSEEK_DEFAULT_MODEL
from src.providers.deepseek import DeepSeekProvider
from src.providers.openai_compat import OpenAICompatProvider, _is_retryable
from src.providers.sensenova import SenseNovaProvider


PROVIDERS = [DeepSeekProvider, SenseNovaProvider]


def test_scrub_normal_text_unchanged():
    assert scrub_text("你好，世界！") == "你好，世界！"


def test_scrub_recovers_emoji_from_surrogates():
    # 模拟：多字节字符被切分后 SDK 产出的孤立代理项（surrogateescape 还原）
    broken = "\udcf0\udc9f\udc8c\udc9f"  # F0 9F 8C 9F = 🌟
    assert scrub_text(broken) == "🌟"


def test_scrub_no_crash_on_unrecoverable():
    # 单个孤立字节（0x80）无法组成完整字符 → 替换字符，不崩溃
    assert "�" in scrub_text("\udc80")


def test_scrub_windows_gbk_input_artifact():
    # Windows 管道输入被 GBK 解码后的典型产物（UTF-8 字节 → GBK surrogateescape）
    assert scrub_text("用一句中文介绍你\udcaa\udcb1") == "用一句中文介绍你��"


def test_is_retryable_unknown_error():
    # 未知/普通异常不重试
    assert not _is_retryable(ValueError("boom"))


# ---------------------------------------------------------------------------
# 子类只是「声明身份」：名字、默认地址、label
# ---------------------------------------------------------------------------


def test_deepseek_provider_defaults():
    p = DeepSeekProvider(api_key="fake")
    assert p.PROVIDER_NAME == "deepseek"
    assert str(p._client.base_url).rstrip("/") == DEEPSEEK_DEFAULT_BASE_URL.rstrip("/")
    assert p._model == DEEPSEEK_DEFAULT_MODEL
    assert p.label == f"deepseek/{DEEPSEEK_DEFAULT_MODEL}"


def test_sensenova_provider_defaults():
    p = SenseNovaProvider(api_key="fake")
    assert p.PROVIDER_NAME == "sensenova"
    assert p._model == "sensenova-6.8-flash-lite"
    assert p.label == "sensenova/sensenova-6.8-flash-lite"


def test_provider_classes_share_one_implementation():
    """两家都是 OpenAI 兼容接口，子类只声明身份，实现共用一份。"""
    assert issubclass(SenseNovaProvider, OpenAICompatProvider)
    assert issubclass(DeepSeekProvider, OpenAICompatProvider)
    assert not (_is_retryable(ValueError("boom")))


# ---------------------------------------------------------------------------
# 流式请求参数（DeepSeek 需要 stream_options 才会回 usage）
# ---------------------------------------------------------------------------


def test_stream_kwargs_include_usage():
    p = DeepSeekProvider(api_key="fake")
    kwargs = p._build_kwargs([], stream=True)
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in kwargs          # 未配置则不下发


def test_non_stream_kwargs_have_no_stream_options():
    p = DeepSeekProvider(api_key="fake")
    kwargs = p._build_kwargs([], stream=False)
    assert "stream_options" not in kwargs


def test_stream_usage_can_be_disabled():
    class NoUsageProvider(OpenAICompatProvider):
        STREAM_USAGE = False

    p = NoUsageProvider(base_url="http://fake", api_key="fake", model="m")
    assert "stream_options" not in p._build_kwargs([], stream=True)


# ---------------------------------------------------------------------------
# 空回复检测（M8，docs/11 §7 发现三 spike 结论）
# ---------------------------------------------------------------------------


def _make_provider(cls, max_tokens=None):
    """构造一个 provider，mock 掉 OpenAI client，不发网络请求。"""
    p = cls(base_url="http://fake", api_key="fake", model="test-model",
            max_tokens=max_tokens)
    p._client = MagicMock()
    return p


def _stream_chunks(content_pieces, usage=None):
    """构造流式 chunk 列表。content_pieces 是 delta.content 的列表。"""
    chunks = []
    for piece in content_pieces:
        delta = SimpleNamespace(content=piece, role=None, tool_calls=None)
        choice = SimpleNamespace(delta=delta, finish_reason="")
        chunks.append(SimpleNamespace(choices=[choice], usage=None))
    # 最后一个 chunk 带 usage
    if usage is not None:
        chunks.append(SimpleNamespace(choices=[], usage=usage))
    return chunks


def _usage(reasoning=0, completion=0):
    """构造一个带 reasoning_tokens 的 usage 对象。"""
    return SimpleNamespace(
        model_dump=lambda: {
            "completion_tokens": completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
        completion_tokens=completion,
    )


@pytest.mark.parametrize("cls", PROVIDERS)
def test_reasoning_tokens_in_details(cls):
    # reasoning_tokens 可能在 completion_tokens_details 里（DeepSeek 和商汤都是这个位置）
    assert cls._reasoning_tokens(_usage(reasoning=50)) == 50


@pytest.mark.parametrize("cls", PROVIDERS)
def test_reasoning_tokens_none_usage(cls):
    assert cls._reasoning_tokens(None) == 0


def test_empty_reply_error_message_contains_diagnostics():
    u = _usage(reasoning=50, completion=50)
    err = EmptyReplyError(max_tokens=50, usage=u)
    msg = str(err)
    assert "MAX_TOKENS=50" in msg
    assert "思考用了 50" in msg
    assert "留空 MAX_TOKENS" in msg    # 修复建议：留空，而不是设低值
    assert "非 reasoning" in msg       # 修复建议二：换非 reasoning 模型（docs/14）
    assert "撤回" in msg               # 告知用户本轮消息会被撤回


@pytest.mark.parametrize("cls", PROVIDERS)
def test_chat_stream_raises_on_empty_content(cls):
    """流式跑完但 content 全空 → 抛 EmptyReplyError（不静默返回）。"""
    p = _make_provider(cls, max_tokens=50)
    usage = _usage(reasoning=50, completion=50)
    p._client.chat.completions.create.return_value = iter(
        _stream_chunks([], usage)   # 0 个 content chunk
    )
    with pytest.raises(EmptyReplyError) as exc_info:
        list(p.chat_stream([]))
    assert "MAX_TOKENS=50" in str(exc_info.value)


@pytest.mark.parametrize("cls", PROVIDERS)
def test_chat_stream_raises_on_null_content(cls):
    """content 全是 None（reasoning 在别的字段）→ 同样抛。"""
    p = _make_provider(cls, max_tokens=50)
    usage = _usage(reasoning=50, completion=50)
    # delta.content 是 None，模拟 reasoning 在 delta.reasoning_content 里的情况
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, role=None, tool_calls=None),
                finish_reason="",
            )],
            usage=None,
        )
    ] + [SimpleNamespace(choices=[], usage=usage)]
    p._client.chat.completions.create.return_value = iter(chunks)
    with pytest.raises(EmptyReplyError):
        list(p.chat_stream([]))


@pytest.mark.parametrize("cls", PROVIDERS)
def test_chat_stream_skips_reasoning_deltas(cls):
    """DeepSeek 的流式 chunk 里思考走 delta.reasoning_content，不能当正文 yield。"""
    p = _make_provider(cls)
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(
                content=None, role=None, tool_calls=None,
                reasoning_content="这是一段内部思考",
            ),
            finish_reason="",
        )], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="正式回答", role=None, tool_calls=None),
            finish_reason="",
        )], usage=None),
    ]
    p._client.chat.completions.create.return_value = iter(chunks)
    assert list(p.chat_stream([])) == ["正式回答"]


@pytest.mark.parametrize("cls", PROVIDERS)
def test_chat_stream_no_raise_on_content(cls):
    """有 content → 正常 yield，不抛。"""
    p = _make_provider(cls)
    p._client.chat.completions.create.return_value = iter(
        _stream_chunks(["你好", "世界"])
    )
    assert list(p.chat_stream([])) == ["你好", "世界"]


@pytest.mark.parametrize("cls", PROVIDERS)
def test_chat_raises_on_empty(cls):
    """非流式返回空字符串 → 抛 EmptyReplyError。"""
    p = _make_provider(cls, max_tokens=50)
    usage = _usage(reasoning=50, completion=50)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
        usage=usage,
    )
    p._client.chat.completions.create.return_value = resp
    with pytest.raises(EmptyReplyError):
        p.chat([])


@pytest.mark.parametrize("cls", PROVIDERS)
def test_complete_raises_on_empty_content(cls):
    """complete() 返回空 content（无 tool_calls）→ 抛。"""
    p = _make_provider(cls, max_tokens=50)
    usage = _usage(reasoning=50, completion=50)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=None),
            finish_reason="length",
        )],
        usage=usage,
    )
    p._client.chat.completions.create.return_value = resp
    with pytest.raises(EmptyReplyError):
        p.complete([])


@pytest.mark.parametrize("cls", PROVIDERS)
def test_complete_no_raise_on_tool_calls(cls):
    """有 tool_calls → 正常返回，不抛（即使 content 为空）。"""
    p = _make_provider(cls)
    tc = SimpleNamespace(id="1", function=SimpleNamespace(name="calc", arguments="{}"))
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[tc]),
            finish_reason="tool_calls",
        )],
        usage=None,
    )
    p._client.chat.completions.create.return_value = resp
    result = p.complete([])
    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1


@pytest.mark.parametrize("cls", PROVIDERS)
def test_empty_reply_not_retried(cls):
    """空回复不重试：只调一次 API。"""
    p = _make_provider(cls, max_tokens=50)
    p._max_retries = 3
    usage = _usage(reasoning=50, completion=50)
    p._client.chat.completions.create.return_value = iter(
        _stream_chunks([], usage)
    )
    with pytest.raises(EmptyReplyError):
        list(p.chat_stream([]))
    assert p._client.chat.completions.create.call_count == 1
