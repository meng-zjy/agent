"""Agent 核心测试：使用 mock provider，不发起真实网络请求。"""

from collections.abc import Iterator

from src.core.agent import ChatAgent
from src.core.models import Message, Role
from src.core.provider import BaseProvider, Completion


class FakeProvider(BaseProvider):
    """记录收到的消息，并返回预设回复。"""

    def __init__(self, stream_reply: str) -> None:
        self.calls: list[list[Message]] = []
        self._reply = stream_reply

    def chat(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        return self._reply

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        self.calls.append(messages)
        yield from self._reply

    def complete(self, messages: list[Message], tools=None) -> Completion:
        self.calls.append(messages)
        return Completion(finish_reason="stop", content=self._reply)


def test_respond_yields_streamed_reply():
    provider = FakeProvider(stream_reply="你好呀！")
    agent = ChatAgent(provider)
    msgs = [Message(role=Role.USER, content="你是谁")]
    assert "".join(agent.respond(msgs)) == "你好呀！"


def test_respond_sends_single_user_message():
    provider = FakeProvider(stream_reply="ok")
    agent = ChatAgent(provider)
    msgs = [Message(role=Role.USER, content="你好")]
    list(agent.respond(msgs))

    assert len(provider.calls) == 1
    sent = provider.calls[0]
    assert len(sent) == 1
    assert sent[0].role == Role.USER
    assert sent[0].content == "你好"


def test_provider_property():
    provider = FakeProvider(stream_reply="x")
    agent = ChatAgent(provider)
    assert agent.provider is provider


def test_respond_preserves_images():
    provider = FakeProvider(stream_reply="ok")
    agent = ChatAgent(provider)
    msgs = [Message(role=Role.USER, content="看图", images=["data:image/png;base64,abc"])]
    list(agent.respond(msgs))
    assert provider.calls[0][0].images == ["data:image/png;base64,abc"]

