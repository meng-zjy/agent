"""应用配置：从环境变量 / .env 文件读取。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。字段名对应环境变量（不区分大小写）。

    例如 `deepseek_api_key` <-> 环境变量 `DEEPSEEK_API_KEY`。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM 服务商：deepseek（默认） / sensenova
    # 两家都是 OpenAI 协议兼容，换服务商只改这一个值 + 对应的 KEY（见 docs/14）
    llm_provider: str = "deepseek"

    # DeepSeek
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # 商汤日日新（SenseNova）——已停用，保留以便 LLM_PROVIDER=sensenova 切回
    sensenova_base_url: str = "https://token.sensenova.cn/v1"
    sensenova_api_key: str = ""
    sensenova_model: str = "sensenova-6.8-flash-lite"

    # 可选生成参数（None 则使用模型默认值，不随请求下发）
    max_tokens: int | None = None
    temperature: float | None = None

    # 请求超时与总时间预算（M9 / P0）
    # openai SDK 默认 read=600（10 分钟），太宽松——本地也会卡那么久才报错。
    # 90 秒比实测最慢的真实响应（78s，docs/11 §7 发现四）多 15% 余量。
    request_timeout: float = 90.0

    # 一次调用的重试次数。openai SDK 自带重试已在 provider 里关掉（max_retries=0），
    # 重试策略只留我们这一份——两层重试叠着跑、退避节奏互不知道，实测 3×3=9 次
    # HTTP 请求、单次调用最坏占住线程 820s ≈ 13.7 分钟。
    max_retries: int = 3

    # 一次调用（**含所有重试**）的墙钟上限。read 超时按「剩余预算」收窄，
    # 所以整条调用链被这个值封顶，不再是「重试次数 × 单次超时」线性放大。
    #
    # ⚠ 取舍：实测中位 4.78~7.62s，但出现过 78.14s 的长尾（reasoning=8533）。
    #   设 30 意味着那种长尾请求会被掐断、抛 DeadlineExceeded。嫌短可以改大，
    #   但别回到 90 以上——那等于回到「一次调用占住线程十几分钟」的老问题。
    #   注意它同时收窄了第一次尝试的 read 超时（deadline < request_timeout 时生效）。
    deadline: float = 30.0

    # 熔断器（P1）：上游连续失败 N 次 → 打开 T 秒 → 期间的请求秒级失败。
    # 重试治「偶发抖动」，熔断治「上游真的挂了」——上游挂掉时每个请求都各烧一遍
    # 完整预算，线程池被占满，连 /health 都被拖到探活超时。
    # 0 或负数 = 关闭熔断（每次失败都照常重试）。
    breaker_failure_threshold: int = 3
    breaker_open_seconds: float = 30.0

    # 记忆 / 会话（M2）
    session_dir: str = "data/sessions"
    system_prompt: str = (
        "你是一个 AI 助手，请用中文简洁友好地回答用户。\n"
        "\n"
        "关于记忆：你运行在一个带持久化记忆的会话系统里。系统会在每一轮对话后把你的完整会话历史自动保存到本地磁盘，"
        "并在程序重启后自动重新加载、无缝续接。因此你对当前会话（包括程序重启之后）拥有连续、完整的记忆——"
        "这是本系统的既定能力，不是你的推测。请把这一点当作事实：当用户问起「重启后是否还记得」时，"
        "直接、肯定地回答「记得」，并主动说出你记得的具体内容，不要假设自己会忘记。\n"
        "\n"
        "关于时间与工具：系统每一轮都会把实时时间写进上下文（标记为「[当前时间]」）。"
        "回答「现在几点 / 今天几号 / 星期几」时以它为准，不要照抄历史对话里出现过的旧时间"
        "（那些可能是几天前的答案）。数学计算请调用 calculator 工具，拿到真实结果后再用中文回答；"
        "其他需要真实数据的操作同样优先调用对应工具，不要凭印象作答。"
        "不要输出 <tool_call>、<function=...> 之类的文本形式的工具调用标签。"
    )
    max_history_turns: int = 20
    max_history_chars: int = 10000
    # 单次输入上限（M9）：挡住"把整个文档粘进聊天框"烧 token / 撑爆上下文
    # 要处理长文档，请放进 data/knowledge/ 走 RAG 检索，不要塞进聊天框
    max_input_chars: int = 32000

    # 长对话摘要（09）
    # 保留最近几轮的完整消息，更早的压成滚动摘要
    summary_keep_turns: int = 20
    # 摘要长度上限（字符），LLM 输出过长时截断兜底
    summary_max_chars: int = 1500

    # 知识库 / RAG（M4）
    knowledge_dir: str = "data/knowledge"
    chunk_size: int = 400
    chunk_overlap: int = 50
    retrieval_top_k: int = 4
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # MCP（M5）
    mcp_server_command: str = "npx"
    mcp_server_args: str = "-y @modelcontextprotocol/server-filesystem"
    mcp_allowed_dir: str = "data/mcp_workspace"

    # 访问控制（M9，默认关闭）
    # ⚠ 留空 = 完全关闭，行为与未安装完全一致。本项目跑在 127.0.0.1，
    #    本地没有攻击面，这两项只是把中间件的写法掌握住——不是真防护。
    # 认证令牌：非空时所有请求必须带 `X-Agent-Token: <值>`；/health 除外
    auth_token: str = ""
    # 速率限制：每 IP 每分钟最多多少请求，0 = 关闭
    rate_limit_rpm: int = 0

    # 日志（M8）
    # 只记元数据（耗时/结果/错误），不记消息正文——会话历史已落盘到 data/sessions/
    log_level: str = "INFO"
    log_dir: str = "data/logs"


# 模块级单例：运行期直接使用
settings = Settings()
