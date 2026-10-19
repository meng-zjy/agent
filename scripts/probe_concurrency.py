"""探针：实测并发行为（不花额度）。

回答三个问题：
1. 两个不同会话同时发消息，是否都能跑完、是否真并行？（后端并发能力）
2. 同一个会话同时发两条消息，历史会不会被写坏？（竞态）
3. 客户端中途断开（模拟切窗口关流），服务端是否仍会把回复落盘？
"""

import asyncio
import sys
import time
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api import create_app  # noqa: E402
from src.core.agent import ChatAgent  # noqa: E402
from src.core.memory import MemoryStore  # noqa: E402
from src.core.models import Message, Role  # noqa: E402
from src.core.provider import BaseProvider, Completion  # noqa: E402
from src.core.service import ChatService  # noqa: E402

import httpx  # noqa: E402


class SlowProvider(BaseProvider):
    """模拟"模型先把整个回复算完再下发"的服务端行为。"""

    def chat(self, messages: list[Message]) -> str:
        time.sleep(0.6)
        return "R"

    def chat_stream(self, messages: list[Message]) -> Iterator[str]:
        time.sleep(0.6)
        yield "R1"
        time.sleep(0.4)
        yield "R2"

    def complete(self, messages: list[Message], tools=None) -> Completion:
        return Completion(finish_reason="stop", content="R")


async def grab(client: httpx.AsyncClient, session_id: str, label: str,
               results: dict) -> None:
    start = time.time()
    text = ""
    async with client.stream("POST", f"/api/sessions/{session_id}/messages",
                             json={"text": label}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                text += line[6:]
    elapsed = round(time.time() - start, 3)
    results[label] = (text, elapsed)
    print(f"  [{label}] 完成，耗时 {elapsed}s")


async def main() -> None:
    tmp = Path("data/tmp/probe_concurrency")
    store = MemoryStore(tmp)
    service = ChatService(ChatAgent(SlowProvider()), store, system_prompt="你是助手")
    app = create_app(service)

    transport = httpx.ASGITransport(app=app)
    sid_a = store.create().id
    sid_b = store.create().id
    sid_c = store.create().id

    print("实验 1：两个不同会话同时发问（并行则总耗时 ~1.0s，串行则 ~2.0s）")
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        results: dict = {}
        t0 = time.time()
        await asyncio.gather(grab(c, sid_a, "会话A", results),
                             grab(c, sid_b, "会话B", results))
        print(f"  总耗时 {round(time.time() - t0, 3)}s")

    print("实验 2：同一会话同时发两条（历史是否写坏）")
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        results2: dict = {}
        await asyncio.gather(grab(c, sid_c, "会话C-1", results2),
                             grab(c, sid_c, "会话C-2", results2))
    loaded = store.load(sid_c)
    print(f"  会话C 最终消息数 = {len(loaded.conversation.messages)}（期望 4）")
    for m in loaded.conversation.messages:
        print(f"    {m.role.value:>9}: {m.content[:24]!r}")

    print("实验 3：客户端中途断开（模拟切窗口关流），服务端是否仍落盘")
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        try:
            async with c.stream("POST", f"/api/sessions/{sid_a}/messages",
                                json={"text": "断开测试"},
                                timeout=httpx.Timeout(0.2)) as r:
                async for _ in r.aiter_lines():
                    pass
        except Exception as exc:
            print(f"  客户端断开：{type(exc).__name__}")
    await asyncio.sleep(2)
    loaded = store.load(sid_a)
    print(f"  会话A 消息数 = {len(loaded.conversation.messages)}")
    for m in loaded.conversation.messages:
        print(f"    {m.role.value:>9}: {m.content[:24]!r}")


if __name__ == "__main__":
    asyncio.run(main())
