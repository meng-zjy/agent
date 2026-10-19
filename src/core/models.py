"""统一消息模型。

采用 OpenAI 消息格式作为内部统一结构，为后续扩展预留位置：
- 工具调用（M3）：消息上加 tool_calls / tool_call_id / name 字段
- 多模态（M6）：content 从纯文本演化为「文本 + 图片 URL」结构化列表
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    created_at: str | None = field(default=None)   # ISO 时间戳，记录顺序（持久化/调试用）
    token_count: int = field(default=0)             # 缓存估算结果，0 表示未缓存
    tool_calls: list[dict] | None = field(default=None)   # 模型发出的工具调用（M3）
    tool_call_id: str | None = field(default=None)        # 工具结果消息对应的调用 id（M3）
    name: str | None = field(default=None)                # 函数名（M3，可选）
    images: list[str] | None = field(default=None)        # 图片 URL/data URI（M6）

    def estimate_tokens(self) -> int:
        """粗略估算本条消息的 token 数（不调用任何外部 API）。

        规则：
        - 如果已有缓存值（token_count > 0）直接返回。
        - CJK 字符（中日韩）约 1 字符 1 token。
        - 其余字符（英文、符号、空白）约 4 字符 1 token。
        - 最少返回 1，避免空消息返回 0 导致截断逻辑除零风险。
        """
        if self.token_count > 0:
            return self.token_count
        cjk = sum(1 for ch in self.content if "\u4e00" <= ch <= "\u9fff")
        other = len(self.content) - cjk
        return max(1, cjk + other // 4)

    def to_openai_dict(self) -> dict:
        """转换为 OpenAI 协议的消息字典，供 SDK 直接消费。

        内部字段（created_at / token_count）不下发；工具字段（M3）、图片（M6）
        仅在非 None 时下发。有图片时 content 用多模态列表格式。
        """
        if self.images:
            content = [{"type": "text", "text": self.content}]
            content += [{"type": "image_url", "image_url": {"url": u}}
                        for u in self.images]
        else:
            content = self.content
        data = {"role": self.role.value, "content": content}
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data


@dataclass
class Conversation:
    """多轮对话载体（M2 记忆系统）：有序消息列表 = 会话历史。"""

    messages: list[Message]

    def add(self, role: Role, content: str, images: list[str] | None = None) -> None:
        """追加一条消息。"""
        self.messages.append(Message(role=role, content=content, images=images))

    def to_openai_list(self) -> list[dict]:
        """整段历史转成可持久化的字典列表：只保留 role + content（文字）。

        刻意不用 to_openai_dict：那是给模型用的，会带图片 base64 / 工具字段，
        写进 JSON 会撑爆、也读不回。持久化只存文字。
        """
        return [{"role": m.role.value, "content": m.content} for m in self.messages]
