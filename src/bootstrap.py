"""应用组装（bootstrap）：CLI 与 Web 共用的"组装车间"。

把分散的配置 / provider / agent / memory / retriever 组装成可用的 ChatService。
CLI（src/cli.py）和 Web（src/api.py）都从这里取构建逻辑，保证两边行为一致。
"""

import hashlib
import sys
from pathlib import Path

from src.config import settings
from src.core.agent import ChatAgent
from src.core.logger import get_logger
from src.core.memory import MemoryStore
from src.core.provider import BaseProvider
from src.core.service import ChatService
from src.core.tools import build_default_tools
from src.prompts import DEFAULT_SYSTEM_PROMPT, PROMPT_VERSION, build_system_prompt
from src.providers.deepseek import DeepSeekProvider
from src.providers.sensenova import SenseNovaProvider

# LLM_PROVIDER 取值 → (配置字段名前缀, 需要的 KEY 环境变量名, 实现类)
# 新增一家 OpenAI 兼容服务商：写个薄子类 → 在这里加一行 → .env 里填 KEY。
PROVIDERS = {
    "deepseek": ("deepseek", "DEEPSEEK_API_KEY", DeepSeekProvider),
    "sensenova": ("sensenova", "SENSENOVA_API_KEY", SenseNovaProvider),
}


def build_provider() -> BaseProvider:
    """按 LLM_PROVIDER 构建 provider；名字写错或 KEY 缺失时给出可操作的报错。"""
    name = (settings.llm_provider or "").strip().lower()
    if name not in PROVIDERS:
        print(
            f"错误：LLM_PROVIDER={name!r} 无法识别（已忽略大小写和首尾空格）。\n"
            f"可选值：{', '.join(PROVIDERS)}（见 docs/14）",
            file=sys.stderr,
        )
        raise SystemExit(1)

    prefix, key_name, cls = PROVIDERS[name]
    if not getattr(settings, f"{prefix}_api_key"):
        print(
            f"错误：未配置 {key_name}。\n"
            f"请在项目根目录的 .env 里填入 {key_name}（参考 .env.example）后重试。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return cls(
        base_url=getattr(settings, f"{prefix}_base_url"),
        api_key=getattr(settings, f"{prefix}_api_key"),
        model=getattr(settings, f"{prefix}_model"),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        deadline=settings.deadline,
        breaker_failure_threshold=settings.breaker_failure_threshold,
        breaker_open_seconds=settings.breaker_open_seconds,
    )


def build_retriever(knowledge_catalog=None):
    """构建知识库检索器；fastembed 缺失/模型下载失败时返回 None（降级为无 RAG）。"""
    if not settings.rag_enabled:
        get_logger("bootstrap").info("知识库检索已由 RAG_ENABLED=false 关闭")
        return None
    try:
        from src.core.embeddings import FastembedEmbedder
        from src.core.rag import DocumentStore

        # 账户模式下禁止意外退回全目录检索；CLI 无身份时不加载受限资料。
        if settings.account_enabled and knowledge_catalog is None:
            return None
        embedder = FastembedEmbedder(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
        )
        if knowledge_catalog is not None:
            from src.core.knowledge import AuthorizedRetriever

            return AuthorizedRetriever(
                knowledge_catalog, embedder, settings.chunk_size, settings.chunk_overlap
            )
        store = DocumentStore(embedder, settings.chunk_size, settings.chunk_overlap)
        kdir = Path(settings.knowledge_dir)
        if kdir.exists():
            n = store.ingest_dir(kdir)
            if n:
                print(f"已从知识库摄入 {n} 个片段（{kdir}）")
        return store
    except Exception as exc:  # noqa: BLE001
        # 只提示一次，不打断启动；详情进日志方便事后排查
        get_logger("bootstrap").warning("知识库不可用，本次会话不含检索工具: %s", exc)
        return None


def build_mcp_tools():
    """连接 MCP 服务器并包装其工具；关闭、失败或超时时返回空列表。

    文件系统 MCP 含写入副作用。生产部署由 ``MCP_ENABLED=false`` 显式关闭，
    防止模型输出或外部资料间接触发服务器端文件操作。
    """
    if not settings.mcp_enabled:
        get_logger("bootstrap").info("MCP 已由 MCP_ENABLED=false 关闭")
        return []

    try:
        from src.core.mcp import McpToolBridge, wrap_mcp_tools

        allowed = Path(settings.mcp_allowed_dir).resolve()
        allowed.mkdir(parents=True, exist_ok=True)   # 授权目录必须存在，否则 filesystem 服务器起不来
        bridge = McpToolBridge(
            settings.mcp_server_command,
            settings.mcp_server_args.split() + [str(allowed)],
        )
        bridge.start()
        if not bridge.wait_ready(timeout=15):
            get_logger("bootstrap").warning("MCP 服务器连接超时，本次会话不含 MCP 工具")
            return []
        tools = wrap_mcp_tools(bridge, str(allowed))
        print(f"已连接 MCP 服务器，注册 {len(tools)} 个工具")
        return tools
    except Exception as exc:  # noqa: BLE001
        get_logger("bootstrap").warning("MCP 不可用，本次会话不含 MCP 工具: %s", exc)
        return []


def active_provider_config():
    """当前 LLM_PROVIDER 的 (名字, base_url, api_key, model)。

    探针脚本（scripts/probe_*.py）用它取配置，这样换供应商后旧探针不用改就能重跑。
    """
    name = (settings.llm_provider or "").strip().lower()
    prefix = PROVIDERS.get(name, ("deepseek",))[0]
    return (
        name or "deepseek",
        getattr(settings, f"{prefix}_base_url"),
        getattr(settings, f"{prefix}_api_key"),
        getattr(settings, f"{prefix}_model"),
    )


def provider_label() -> str:
    """当前 provider 的展示名，如 `deepseek/deepseek-chat`（只读配置，不建连接）。"""
    name, _, _, model = active_provider_config()
    return f"{name}/{model}"


def build_service(retriever=None) -> ChatService:
    """按配置构建服务；API Key 缺失时给出友好提示后退出。"""
    provider = build_provider()
    mcp_tools = build_mcp_tools()   # 失败降级返回 []
    agent = ChatAgent(
        provider,
        tools=build_default_tools(
            retriever, top_k=settings.retrieval_top_k, mcp_tools=mcp_tools
        ),
    )
    memory = MemoryStore(settings.session_dir)
    system_prompt = build_system_prompt(
        settings.system_prompt,
        has_retriever=retriever is not None,
        has_mcp_tools=bool(mcp_tools),
    )
    # 环境变量可以覆盖默认人设；记录指纹区分实际配置，禁止将提示词正文写入日志。
    get_logger("bootstrap").info(
        "prompt_version=%s system_prompt_sha256=%s custom_system_prompt=%s "
        "has_retriever=%s has_mcp_tools=%s",
        PROMPT_VERSION,
        hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        settings.system_prompt != DEFAULT_SYSTEM_PROMPT,
        retriever is not None,
        bool(mcp_tools),
    )
    return ChatService(
        agent,
        memory,
        system_prompt=system_prompt,
        max_turns=settings.max_history_turns,
        max_chars=settings.max_history_chars,
        max_input_chars=settings.max_input_chars,
        max_images_per_message=settings.max_images_per_message,
        max_image_bytes=settings.max_image_bytes,
        time_zone=settings.time_zone,
        summary_keep_turns=settings.summary_keep_turns,
        summary_max_chars=settings.summary_max_chars,
        retrieval_enabled=settings.rag_enabled,
        retrieval_available=retriever is not None,
    )
