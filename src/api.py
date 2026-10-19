"""FastAPI 应用（Web 化）：给 ChatService 包一层 HTTP。

REST：会话增删改查；SSE：流式返回模型回复；GET / 返回聊天页面。
运行：python -m src.api  或  uvicorn src.api:create_app --factory
"""

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings
from src.core.agent import AgentStatus
from src.core.service import ChatService, InputTooLongError, SessionBusyError

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_STATIC_INDEX = _STATIC_DIR / "index.html"


class SendBody(BaseModel):
    text: str
    images: list[str] | None = None


# 速率限制（M9）：每 IP 一个滑动窗口计数。内存态，多进程部署各算各的（见 docs/13 §10）。
_rl_buckets: dict[str, list[float]] = {}


def _is_rate_limited(ip: str) -> bool:
    """每分钟超过 settings.rate_limit_rpm 次就返回 True。"""
    now = time.monotonic()
    window = [t for t in _rl_buckets.get(ip, []) if t > now - 60]
    _rl_buckets[ip] = window
    if len(window) >= settings.rate_limit_rpm:
        return True
    window.append(now)
    return False


def create_app(service: ChatService | None = None) -> FastAPI:
    """构建 FastAPI 应用。

    service 不传时按默认配置构建（供 uvicorn --factory / python -m src.api 用）；
    测试时传入注入假 provider 的 service，不花额度、不加载 BGE 模型。
    """
    if service is None:
        from src.bootstrap import build_retriever, build_service
        service = build_service(build_retriever())

    app = FastAPI(title="Agent")
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        """访问控制（M9，默认关闭）。

        两项配置留空/为 0 时行为与未安装完全一致——本项目跑在 127.0.0.1、
        没有攻击面，这两个中间件的价值是掌握写法，不是真防护（见 docs/13 §4.6）。
        `/health` 永远放行：探活请求不该被挡，否则负载均衡会误判服务挂掉。
        """
        if request.url.path == "/health":
            return await call_next(request)
        if settings.auth_token and request.headers.get("X-Agent-Token") != settings.auth_token:
            return JSONResponse(
                {"detail": "认证失败：缺少或错误的 X-Agent-Token"}, status_code=401,
            )
        if settings.rate_limit_rpm > 0:
            ip = request.client.host if request.client else "?"
            if _is_rate_limited(ip):
                return JSONResponse({"detail": "请求过于频繁，请稍后再试"}, status_code=429)
        return await call_next(request)

    @app.exception_handler(SessionBusyError)
    async def _busy_handler(request: Request, exc: SessionBusyError) -> JSONResponse:
        """会话正在生成回复时的占用冲突 → 409，前端据此提示"稍后再试"。"""
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(InputTooLongError)
    async def _too_long_handler(request: Request, exc: InputTooLongError) -> JSONResponse:
        """输入超长 → 400。必须在 StreamingResponse 返回**之前**抛，
        否则会被 send() 生成器里的 except 吞成 SSE 错误事件，拿不到状态码。"""
        return JSONResponse({"detail": str(exc)}, status_code=400)

    def _require(session_id: str) -> None:
        if service.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(_STATIC_INDEX.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health():
        """探活。部署后负载均衡 / 容器编排第一个会问的端点。

        报出当前模型、输入上限和熔断状态，便于事后确认"我探活探的是不是那套配置"——
        前端启动时也从这里读 `max_input_chars`，避免前后端各写一份。

        ⚠ 必须是 `async def`（P0）：同步端点跑在 uvicorn 的 40 槽线程池里，
        线程池一旦被「等上游超时」的请求占满，`/health` 也会排队、探活超时，
        负载均衡于是判服务挂掉去重启——而上游其实只是慢。探活自身不碰上游
        （provider_label 只读配置、不建连接；breaker 状态取的是内存字段），
        所以安全地跑在事件循环上，永远不需要占线程池槽位。
        """
        from src.bootstrap import provider_label
        return {
            "ok": True,
            "model": provider_label(),
            # 读 service 而不是 settings：真实边界在 service 里，
            # settings 只是它的默认来源（见 ChatService.max_input_chars）
            "max_input_chars": service.max_input_chars,
            # 熔断状态：上游挂了时这里会变 "open"，看监控的人一眼就能定位
            # 是「我们自己的服务」还是「上游」的问题
            "upstream": service.upstream_state(),
        }

    @app.get("/api/sessions")
    def list_sessions():
        return [s.to_dict() for s in service.list_sessions()]

    @app.post("/api/sessions")
    def create_session():
        return service.new_session().to_dict()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        _require(session_id)
        return service.get_session(session_id).to_dict()

    @app.post("/api/sessions/{session_id}/messages")
    def send_message(session_id: str, body: SendBody):
        _require(session_id)
        # 预检查必须在 StreamingResponse 返回之前——send() 是生成器，函数体要到
        # 被迭代时才执行，那时抛异常只会被下面的 except 吞成 SSE 错误事件。
        service.validate_input(body.text)

        def stream():
            try:
                for chunk in service.send_events(session_id, body.text, images=body.images):
                    if isinstance(chunk, AgentStatus):
                        # 进度提示单独一种事件：它不是回复内容，不能和 delta 混着发
                        yield (f"data: {json.dumps({'status': chunk.message}, ensure_ascii=False)}\n\n")
                    else:
                        yield f"data: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
            except Exception as exc:  # noqa: BLE001  请求失败也以事件形式告知前端
                yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/sessions/{session_id}/clear")
    def clear_session(session_id: str):
        _require(session_id)
        service.clear_session(session_id)
        return service.get_session(session_id).to_dict()

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        _require(session_id)
        service.delete_session(session_id)
        return {"ok": True}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:create_app", factory=True, host="127.0.0.1", port=8000)
