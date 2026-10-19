"""ChatService：对话编排层（M2 记忆系统的"大脑"）。

职责：会话管理 + 历史组装 + 截断 + 长对话摘要（09）+ 流式转发 + 失败回滚 + 持久化。
CLI（现在）与 Web（未来）都只调这一个入口，不直接碰 Agent 和 Memory。

并发约定（M7 前置）：同一会话的请求串行（会话级锁），不同会话并行。
同一会话同时来两条消息时不会互相覆盖历史，而是排队；等不及则报 SessionBusyError。
"""

import base64
import binascii
import re
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import datetime

from src.core.agent import AgentStatus, ChatAgent
from src.core.logger import get_logger, reset_request_context, set_request_context
from src.core.memory import MemoryStore, Session
from src.core.models import Message, Role


# 当前时间提示（每轮重新生成）。
# 末尾那句是针对一个真实 bug 的补丁：时间回答会烂在会话历史里，
# 模型再被问"现在几点"时会直接复读几天前的旧答案，而不是重新查一次。
NOW_HINT = """[当前时间]
{now}

这是系统提供的实时时间，每轮都会更新。回答"现在几点 / 今天几号 / 星期几"时必须以此为准；
历史对话里出现过的时间可能是几天前的旧答案，不要照抄。"""

_WEEKDAY = "一二三四五六日"
_IMAGE_DATA_URI = re.compile(
    r"\Adata:image/(jpeg|png|webp|gif);base64,([A-Za-z0-9+/]+={0,2})\Z",
    re.IGNORECASE,
)


def format_now(now: datetime | None = None) -> str:
    """把当前时间格式化成中文可读的一行，如 `2026-09-23 星期三 10:30:00`。"""
    now = now or datetime.now()
    return f"{now:%Y-%m-%d} 星期{_WEEKDAY[now.weekday()]} {now:%H:%M:%S}"


class SessionBusyError(RuntimeError):
    """同一会话已有请求在生成回复，来不及等时抛出。

    不抛出它而直接并发，会让两个请求读到同一份旧快照、各自落盘互相覆盖——
    用户消息会被静默丢掉。宁可直接报错，也不要写坏历史。
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"会话 {session_id} 正在生成回复，请稍候再发")


class InputTooLongError(ValueError):
    """输入超过上限（M9）。

    挡两件事：把整个文档粘进聊天框烧 token，以及把上下文窗口撑爆。
    校验放在服务端而不是前端——CLI 和直接 curl 也走这条路。
    """

    def __init__(self, length: int, limit: int) -> None:
        self.length = length
        self.limit = limit
        super().__init__(
            f"输入过长：{length} 字（上限 {limit} 字）。请拆分后再发送，"
            "或放进 data/knowledge/ 走知识库检索。"
        )


class InvalidImageError(ValueError):
    """图片数量、格式或解码后大小不符合服务端边界。"""


def _matches_image_signature(image_format: str, content: bytes) -> bool:
    """核验常见图片格式的文件头，拒绝伪造 MIME 的任意 base64 内容。"""
    signatures = {
        "jpeg": content.startswith(b"\xff\xd8\xff"),
        "png": content.startswith(b"\x89PNG\r\n\x1a\n"),
        "gif": content.startswith((b"GIF87a", b"GIF89a")),
        "webp": len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP",
    }
    return signatures[image_format.lower()]


# 摘要提示词：强调"保留关键事实、丢弃寒暄、控制长度"
SUMMARY_PROMPT = """\
你是一个对话摘要器。下面给你一段已有摘要和一段新的对话历史。
请把它们合并成一段新的摘要：
- 保留关键事实：人物与称呼、事件、数字、时间、用户偏好与双方约定
- 丢弃寒暄、重复与无关细节，不要逐句复述
- 控制在 {max_chars} 字以内，用简洁的中文叙述
直接输出摘要正文，不要任何前缀或解释。

[已有摘要]
{existing}

[新对话历史]
{history}"""


class ChatService:
    def __init__(
        self,
        agent: ChatAgent,
        memory: MemoryStore,
        *,
        system_prompt: str = "",
        max_turns: int = 20,
        max_chars: int = 10000,
        max_input_chars: int = 32000,
        max_images_per_message: int = 3,
        max_image_bytes: int = 3 * 1024 * 1024,
        summary_keep_turns: int = 20,
        summary_max_chars: int = 1500,
        inject_now: bool = True,
        busy_timeout: float = 30.0,
        housekeeping_timeout: float = 5.0,
    ) -> None:
        self._agent = agent
        self._memory = memory
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._max_chars = max_chars
        self._max_input_chars = max_input_chars
        self._max_images_per_message = max_images_per_message
        self._max_image_bytes = max_image_bytes
        self._summary_keep_turns = summary_keep_turns
        self._summary_max_chars = summary_max_chars
        self._inject_now = inject_now
        self._busy_timeout = busy_timeout
        self._housekeeping_timeout = housekeeping_timeout

        # 防止「已摘要的内容又被 _trim 静默丢掉」：保留窗口不能比摘要窗口更短
        if max_turns < summary_keep_turns:
            self._max_turns = summary_keep_turns

        # 会话级锁：send / clear / delete 各会话互不阻塞，同会话互斥
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

        # 日志（M8）：只记元数据，不记消息正文
        self._log = get_logger("service")

    # ---------- 并发：会话级锁 ----------

    def _lock_for(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[session_id] = lock
            return lock

    def _acquire(self, session_id: str, timeout: float) -> bool:
        """尝试在 timeout 秒内拿到会话锁；拿不到说明该会话正在生成回复。"""
        return self._lock_for(session_id).acquire(timeout=timeout)

    def _release(self, session_id: str) -> None:
        self._lock_for(session_id).release()

    # ---------- 会话管理（薄封装，转交 MemoryStore） ----------

    def new_session(self) -> Session:
        session = self._memory.create()
        # 会话管理类操作没有 request_id，只设 sess 让日志能归到会话
        token = set_request_context({"req": "-", "sess": session.id})
        try:
            self._log.info("会话创建")
            return session
        finally:
            reset_request_context(token)

    def get_session(self, session_id: str) -> Session | None:
        return self._memory.load(session_id)

    def list_sessions(self) -> list[Session]:
        return self._memory.list()

    def clear_session(self, session_id: str) -> None:
        """清空历史。会话忙时快失败（避免点清空卡在生成中的请求后面）。"""
        token = set_request_context({"req": "-", "sess": session_id})
        try:
            if not self._acquire(session_id, timeout=self._housekeeping_timeout):
                self._log.warning("会话清空被拒（生成中，等%.0fs超时）",
                                  self._housekeeping_timeout)
                raise SessionBusyError(session_id)
            try:
                session = self._memory.load(session_id)
                if session is not None:
                    session.conversation.messages.clear()
                    self._memory.save(session)
                self._log.info("会话清空")
            finally:
                self._release(session_id)
        finally:
            reset_request_context(token)

    def delete_session(self, session_id: str) -> None:
        """删除一个会话（连同磁盘上的 JSON 文件）。

        先等该会话的在途回复结束，否则回复落盘时又会把已删的文件写回来。
        等不到就快速报错，不让用户点一下删除卡住半分钟。
        """
        token = set_request_context({"req": "-", "sess": session_id})
        try:
            if not self._acquire(session_id, timeout=self._housekeeping_timeout):
                self._log.warning("会话删除被拒（生成中，等%.0fs超时）",
                                  self._housekeeping_timeout)
                raise SessionBusyError(session_id)
            try:
                self._memory.delete(session_id)
                self._log.info("会话删除")
            finally:
                self._release(session_id)
        finally:
            reset_request_context(token)

    def resume_or_create(self) -> Session:
        """CLI 启动用：续最近一个非空会话；没有则新建。"""
        for s in self._memory.list():
            if s.conversation.messages:
                return s
        return self._memory.latest() or self._memory.create()

    # ---------- 核心：发送一条消息 ----------

    def send_events(self, session_id: str, text: str,
                    images: list[str] | None = None) -> Iterator[str | AgentStatus]:
        """把一条用户消息送进会话，流式返回**完整事件流**：正文片段 + 工具进度提示。

        成功则把回复记入历史并落盘；失败则撤回本条用户消息再落盘，
        保证磁盘上的历史始终是"有问有答"的完整对话。

        整轮持会话锁（含生成后的摘要），因此同一会话的并发请求会排队而不是
        互相覆盖；不同会话的锁互不影响，可以真正并行。

        每次调用生成一个新的 request_id 并设进日志上下文（M9），于是这一轮
        产生的所有日志——包括 provider / tools 层自己没记 id 的——都共享同一个
        id。`grep <id>` 就是这次请求的完整轨迹。

        为什么有两个入口：`AgentStatus` 不是回复内容，不该进历史、不该进 CLI 输出。
        真正产出事件流的是这里，`send()` 是它的「只要正文」视图——CLI 和绝大多数
        测试用 `send()`，只有 Web 需要进度提示、用 `send_events()`。
        """
        token = set_request_context({"req": uuid.uuid4().hex[:12], "sess": session_id})
        try:
            self._log.info("请求开始 输入=%d字", len(text))
            self.validate_input(text)
            self.validate_images(images)
            if not self._acquire(session_id, timeout=self._busy_timeout):
                self._log.warning("会话忙被拒（等%.0fs超时）", self._busy_timeout)
                raise SessionBusyError(session_id)
            start = time.perf_counter()
            try:
                yield from self._send_locked(session_id, text, images)
            finally:
                self._release(session_id)
                self._log.info("请求结束 耗时=%.2fs", time.perf_counter() - start)
        finally:
            reset_request_context(token)

    def send(self, session_id: str, text: str, images: list[str] | None = None) -> Iterator[str]:
        """只返回正文片段的 `send_events()` 视图。

        过滤掉 `AgentStatus`：它是进度提示，不是回复内容。这个过滤器是关键——
        不经过它，`reply += chunk` 会拿字符串加 dataclass 直接崩，而且提示文案
        会被写进会话历史。
        """
        return (c for c in self.send_events(session_id, text, images)
                if not isinstance(c, AgentStatus))

    def validate_input(self, text: str) -> None:
        """校验输入长度。

        单独一个方法，让 api.py 能在 StreamingResponse 返回**之前**调它拿到干净的
        400——`send()` 是生成器，函数体要到被迭代时才执行，那时候抛异常只会被
        api.py 的 `except Exception` 吞成 SSE 错误事件，拿不到 HTTP 状态码。
        校验逻辑只有一份，CLI 走 send() 内的这次调用，Web 额外预检查一次。
        """
        if len(text) > self._max_input_chars:
            self._log.warning("输入被拒 长度=%d 上限=%d", len(text), self._max_input_chars)
            raise InputTooLongError(len(text), self._max_input_chars)

    def validate_images(self, images: list[str] | None) -> None:
        """校验 Web 多模态图片的数量、data URI 类型和解码后大小。

        图片不会落盘，但会先进入应用内存、再作为请求体发送给模型。此处是权威
        边界：前端的 ``accept`` 属性和文件大小提示只为改善体验，不能当作防护。
        """
        if not images:
            return
        if len(images) > self._max_images_per_message:
            raise InvalidImageError(
                f"单条消息最多上传 {self._max_images_per_message} 张图片"
            )

        maximum_base64_chars = 4 * ((self._max_image_bytes + 2) // 3)
        for image in images:
            match = _IMAGE_DATA_URI.fullmatch(image)
            if match is None:
                raise InvalidImageError(
                    "图片格式无效：仅支持 JPEG、PNG、WebP 或 GIF 的 base64 data URI"
                )
            image_format = match.group(1)
            encoded = match.group(2)
            if len(encoded) > maximum_base64_chars:
                raise InvalidImageError(
                    f"单张图片不能超过 {self._max_image_bytes // (1024 * 1024)} MiB"
                )
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidImageError("图片 base64 数据无效") from exc
            if len(decoded) > self._max_image_bytes:
                raise InvalidImageError(
                    f"单张图片不能超过 {self._max_image_bytes // (1024 * 1024)} MiB"
                )
            if not _matches_image_signature(image_format, decoded):
                raise InvalidImageError("图片内容与声明的格式不匹配")

    @property
    def max_input_chars(self) -> int:
        """当前生效的输入上限。

        `/health` 从这取，而不是从 settings 取——真实边界是 service 里的那个值，
        settings 只是它的默认来源。直接读 settings 会在服务被单独配置过的情况下
        报出错的数，前端据此设的 maxLength 就和服务端不一致了。
        """
        return self._max_input_chars

    @property
    def max_images_per_message(self) -> int:
        """当前服务端允许的单条消息图片数，供 Web 做即时反馈。"""
        return self._max_images_per_message

    @property
    def max_image_bytes(self) -> int:
        """当前服务端允许的单张图片字节数，供 Web 做即时反馈。"""
        return self._max_image_bytes

    def upstream_state(self) -> dict:
        """上游（LLM 服务商）的熔断状态，供 `/health` 报出。

        取值只看内存字段，不碰网络、不建连接，所以探活端点可以安全地在
        事件循环上调用它。
        """
        return self._agent.provider.breaker_state()

    def _send_locked(self, session_id: str, text: str,
                     images: list[str] | None) -> Iterator[str | AgentStatus]:
        session = self._memory.load(session_id)
        if session is None:
            raise ValueError(f"会话不存在: {session_id}")

        conv = session.conversation
        conv.add(Role.USER, text, images=images)   # 1. 记录用户问题（可带图片）
        context = self._build_context(session)   # 2. 组装要发给模型的上下文
        reply = ""
        try:
            for chunk in self._agent.respond(context):
                if isinstance(chunk, AgentStatus):
                    yield chunk                  # 进度提示原样透传，不进 reply
                    continue
                reply += chunk                   # 只累积正文——历史里不能出现「正在计算…」
                yield chunk                      # 3. 流式转发给调用方
        except GeneratorExit:
            self._log.warning("请求撤回（客户端断开，本轮不落盘）")
            conv.messages.pop()                  # 用户中途打断（如 Ctrl+C）：撤回本轮消息
            raise
        except Exception:
            self._log.error("请求失败", exc_info=True)
            conv.messages.pop()                  # 4. 请求失败：撤回本轮消息，不污染历史
            raise
        else:
            # 空回复观测（docs/11 §7 发现二）：provider 层遇到空 content 会直接抛
            # EmptyReplyError，所以这里只对「不抛异常的 provider」兜观测。
            if not reply:
                self._log.warning("空回复（模型返回 0 字，仍按现状落盘）")
            conv.add(Role.ASSISTANT, reply)      # 5. 成功：记录模型回复
            self._maybe_summarize(session)       # 6. 长对话：把更早的消息压成摘要
        finally:
            self._memory.save(session)           # 7. 无论成败都落盘（save 会刷新 updated_at）

    # ---------- 内部：组装上下文（历史压缩/截断的扩展点） ----------

    def _build_context(self, session: Session) -> list[Message]:
        """历史 → 截断 → 拼上前缀 → 返回发给模型的消息。

        前缀顺序：系统提示词 → 当前时间 → 对话摘要 → 对话历史。

        「当前时间」必须是**每轮现算**的：它不能写死在 system_prompt 里（那份是进程启动
        时定死的，跑一晚上就过期了），所以在这里按轮生成。这样历史里即使留着几天前的
        时间回答，模型面前总有一份新鲜的实时时间可对照。

        摘要作为 system 消息插在系统提示词之后、对话历史之前：模型自然把它当成
        "更早聊过的大意"，而不是一段新的指令或一条用户发言。
        """
        prefix: list[Message] = []
        if self._system_prompt:
            prefix.append(Message(role=Role.SYSTEM, content=self._system_prompt))
        if self._inject_now:
            prefix.append(Message(role=Role.SYSTEM, content=NOW_HINT.format(now=format_now())))
        if session.summary:
            prefix.append(Message(role=Role.SYSTEM, content=f"[对话历史摘要]\n{session.summary}"))
        return prefix + self._trim(list(session.conversation.messages))

    def _trim(self, msgs: list[Message]) -> list[Message]:
        """双上限截断，从最旧开始丢：
        1) 轮数上限：只保留最近 max_turns 个「用户问题及其回复」
        2) 字符上限：总量超出 max_chars 时，成对丢弃最旧的「问题+回答」

        有摘要后这里通常不会触发（存储已被压缩），保留作为兜底。
        """
        # 1. 轮数限制：定位用户消息位置，只保留最后 max_turns 条用户消息起
        user_positions = [i for i, m in enumerate(msgs) if m.role is Role.USER]
        if len(user_positions) > self._max_turns:
            msgs = msgs[user_positions[-self._max_turns]:]

        # 2. 字符限制：成对丢弃最旧的一组「问题+回答」
        while len(msgs) > 2 and sum(len(m.content) for m in msgs) > self._max_chars:
            msgs = msgs[2:]
        return msgs

    # ---------- 长对话摘要（09） ----------

    def _maybe_summarize(self, session: Session) -> None:
        """历史超过 summary_keep_turns 轮时，把更早的消息压成滚动摘要。

        触发点在本轮回复成功之后——用户看到的回复不受影响，摘要在它之后悄悄做。
        摘要失败绝不影响主对话：保留旧消息、不覆盖旧摘要，下一轮再试。
        """
        msgs = session.conversation.messages
        user_positions = [i for i, m in enumerate(msgs) if m.role is Role.USER]
        if len(user_positions) <= self._summary_keep_turns:
            return                              # 没超阈值，不触发
        split = user_positions[-self._summary_keep_turns]
        if split <= 0:
            return                              # 保留窗口正好从开头起，没有旧消息可压
        old = msgs[:split]                      # 要压缩的旧消息

        try:
            new_summary = self._summarize(session.summary, old)
        except Exception:                       # noqa: BLE001  摘要失败不影响主对话
            return
        if not new_summary:
            return                              # LLM 返回空摘要：不压缩，下轮再试

        session.summary = new_summary           # 先写摘要
        del msgs[:split]                        # 再删已压缩的旧消息（同一次落盘）

    def _summarize(self, existing: str, old: list[Message]) -> str:
        """滚动摘要：已有摘要 + 新溢出的旧消息 → 新摘要。

        用 provider.chat() 而不是 agent.respond()：摘要不需要工具调用，
        非流式直答更快更省。滚动式的另一个好处是摘要长度始终有界——
        它只总结"旧摘要 + 新溢出的一段"，不会随对话长度无限膨胀。
        """
        history = "\n".join(f"{m.role.value}: {m.content}" for m in old)
        prompt = SUMMARY_PROMPT.format(
            max_chars=self._summary_max_chars,
            existing=existing or "(空)",
            history=history,
        )
        summary = self._agent.provider.chat(
            [Message(role=Role.USER, content=prompt)]
        )
        # 摘要本身也要有界：LLM 不听指令写太长时截断兜底
        return summary.strip()[: self._summary_max_chars]
