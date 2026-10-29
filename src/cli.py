"""命令行交互入口。运行：python -m src.cli"""

import sys
from pathlib import Path

from src.bootstrap import build_retriever, build_service, provider_label
from src.core.models import Role
from src.core.service import ChatService, InputTooLongError


def _enable_utf8_stdio() -> None:
    """Windows 下统一 stdin/stdout/stderr 为 UTF-8，避免乱码与编码崩溃。

    中文系统默认控制台编码为 GBK（cp936），errors='surrogateescape'：
    - 管道输入是 UTF-8 字节时，stdin 按 GBK 解码会产生孤立代理项，
      发送请求时序列化崩溃（UnicodeEncodeError）—— 故 stdin 也必须改 UTF-8
    - 模型回复含特殊字符时同理 —— errors='replace' 作最后兜底
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


HELP_TEXT = """\
可用命令：
  /exit         退出程序
  /help         显示本帮助
  /new          新建一个会话（旧会话保存在磁盘，可随时切回）
  /sessions     列出所有会话
  /use <id>     切换到指定会话（id 用 /sessions 查看）
  /delete <id>  删除指定会话（不可恢复）
  /clear        清空当前会话的历史消息
  /ingest       重新摄入知识库文档（data/knowledge/）
  /files <路径> 下一条消息允许读取的工作区相对路径，多个用分号分隔

其他输入都会作为对话内容发送给模型。
"""


def _print_sessions(service: ChatService) -> None:
    sessions = service.list_sessions()
    if not sessions:
        print("（暂无会话）")
        return
    for s in sessions:
        first_user = next(
            (m.content for m in s.conversation.messages if m.role is Role.USER), ""
        )
        preview = first_user if len(first_user) <= 24 else first_user[:24] + "…"
        summary_mark = " · 已摘要" if s.summary else ""
        print(
            f"  {s.id}  [{s.updated_at}]  "
            f"{len(s.conversation.messages)} 条消息{summary_mark} · {preview or '(空会话)'}"
        )


def _show_session_banner(session) -> None:
    """启动时提示当前处于哪个会话。"""
    count = len(session.conversation.messages)
    if count == 0:
        print(f"已创建新会话 {session.id} · /new 新建 · /help 帮助")
    else:
        print(f"继续会话 {session.id}（{count} 条消息）· /new 新建 · /help 帮助")


def main() -> None:
    _enable_utf8_stdio()
    retriever = build_retriever()
    service = build_service(retriever)
    session = service.resume_or_create()
    allowed_files: list[str] = []
    print(f"Agent · {provider_label()}")
    _show_session_banner(session)

    while True:
        try:
            text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not text:
            continue

        # ---- 命令处理 ----
        if text == "/exit":
            print("再见！")
            break
        if text == "/help":
            print(HELP_TEXT)
            continue
        if text.startswith("/files "):
            candidate_files = [p.strip() for p in text[7:].split(";") if p.strip()]
            try:
                service.validate_file_grants(candidate_files)
            except ValueError as exc:
                allowed_files = []
                print(f"[错误] {exc}")
            else:
                allowed_files = candidate_files
                print("已授权下一条消息读取指定文件；写入、移动和删除操作不可用。")
            continue
        if text == "/new":
            session = service.new_session()
            allowed_files = []
            print(f"已新建会话 {session.id}。")
            continue
        if text == "/sessions":
            _print_sessions(service)
            continue
        if text == "/use":
            print("用法：/use <会话id>（id 用 /sessions 查看）")
            continue
        if text.startswith("/use "):
            sid = text[5:].strip()
            target = service.get_session(sid)
            if target is None:
                print(f"会话 {sid} 不存在，用 /sessions 查看可用的会话。")
            else:
                session = target
                allowed_files = []
                print(f"已切换到会话 {session.id}（{len(session.conversation.messages)} 条消息）。")
            continue
        if text == "/delete":
            print("用法：/delete <会话id>（id 用 /sessions 查看）")
            continue
        if text.startswith("/delete "):
            sid = text[8:].strip()
            if service.get_session(sid) is None:
                print(f"会话 {sid} 不存在，用 /sessions 查看可用的会话。")
            else:
                service.delete_session(sid)
                print(f"已删除会话 {sid}。")
                if sid == session.id:
                    session = service.new_session()
                    print(f"已新建会话 {session.id}（原会话已删除）。")
            continue
        if text == "/clear":
            service.clear_session(session.id)
            print(f"已清空会话 {session.id} 的历史。")
            continue
        if text == "/ingest":
            if retriever is None:
                print("知识库不可用（fastembed 未安装或模型下载失败）。")
            else:
                retriever.clear()
                n = retriever.ingest_dir(Path(settings.knowledge_dir))
                print(f"已重新摄入 {n} 个片段。")
            continue
        if text.startswith("/"):
            print(f"未知命令：{text}（/help 查看可用命令）")
            continue

        # ---- 正常对话 ----
        # 先校验长度再打印「思考中…」，避免超长输入闪一下思考提示再报错
        try:
            service.validate_input(text)
        except InputTooLongError as exc:
            print(f"[错误] {exc}")
            continue

        try:
            print("\rAI：思考中…", end="", flush=True)
            current_files, allowed_files = allowed_files, []
            for i, chunk in enumerate(service.send(
                session.id, text, allowed_files=current_files
            )):
                if i == 0:
                    print("\r" + " " * 20, end="")
                    print("\rAI：", end="", flush=True)
                print(chunk, end="", flush=True)
            print()
        except Exception as exc:  # noqa: BLE001  MVP 统一兜底，避免程序崩溃
            print("\r" + " " * 20, end="")
            print(f"\r[错误] 请求失败：{exc}")
            print("请检查网络、API Key 是否正确，然后重试。")


if __name__ == "__main__":
    main()
