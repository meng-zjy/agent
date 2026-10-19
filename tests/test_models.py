"""消息模型测试。"""

from src.core.models import Conversation, Message, Role


def test_roles():
    assert Role.SYSTEM.value == "system"
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"


def test_message_to_openai_dict():
    msg = Message(role=Role.USER, content="你好")
    assert msg.to_openai_dict() == {"role": "user", "content": "你好"}


def test_message_to_openai_dict_system():
    msg = Message(role=Role.SYSTEM, content="你是助手")
    assert msg.to_openai_dict() == {"role": "system", "content": "你是助手"}


def test_conversation_holds_messages():
    conv = Conversation(messages=[Message(role=Role.USER, content="hi")])
    assert len(conv.messages) == 1
    assert conv.messages[0].role == Role.USER


def test_role_tool():
    assert Role.TOOL.value == "tool"


def test_message_to_openai_dict_with_tool_calls():
    msg = Message(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "calculator", "arguments": "{}"}}
        ],
    )
    d = msg.to_openai_dict()
    assert d["role"] == "assistant"
    assert d["tool_calls"][0]["id"] == "c1"


def test_message_to_openai_dict_tool_result():
    msg = Message(role=Role.TOOL, content="5", tool_call_id="c1", name="calculator")
    assert msg.to_openai_dict() == {
        "role": "tool",
        "content": "5",
        "tool_call_id": "c1",
        "name": "calculator",
    }


def test_message_plain_omits_tool_fields():
    # 普通消息不携带工具字段，输出仍只有 role + content（兼容旧代码）
    msg = Message(role=Role.USER, content="hi")
    assert msg.to_openai_dict() == {"role": "user", "content": "hi"}


def test_message_to_openai_dict_with_images():
    msg = Message(role=Role.USER, content="看图", images=["data:image/png;base64,abc"])
    d = msg.to_openai_dict()
    assert d["content"] == [
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_message_plain_content_stays_string():
    msg = Message(role=Role.USER, content="你好")
    assert isinstance(msg.to_openai_dict()["content"], str)


def test_conversation_to_openai_list_strips_images():
    conv = Conversation(messages=[Message(role=Role.USER, content="看图", images=["data:..."])])
    assert conv.to_openai_list() == [{"role": "user", "content": "看图"}]
