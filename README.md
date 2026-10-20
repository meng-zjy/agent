# Agent

一个可上线的 AI Agent 产品（对标豆包 / 千问）——目前已完成 **M1 对话 MVP**、**M2 记忆系统**、**M3 工具调用**、**M4 RAG 知识库**、**Web 界面**、**M5 MCP 接入**、**M6 多模态**，以及 **09 长对话摘要**、**M7-a 多会话并发**、**M8 上线准备（日志 + 空回复检测）**、**M9 安全加固（request_id 全链路关联 + 输入侧边界）**、**调用可靠性（总时间预算 + 熔断器，详见 docs/15）** 和 **流式工具循环（工具参数按 index 分段拼接，详见 docs/16）**。

> 📖 **第一次看这个项目？** 先读 [docs/12_项目总览.md](docs/12_项目总览.md) —— 架构分层、一条消息的完整时序图、状态图、代码地图、17 条设计决策、扩展指南。不用翻前面 11 份文档。
>
> ⚠️ **换供应商记录**：2026-10-09 起默认供应商从商汤日日新换成 **DeepSeek**，`docs/01`~`docs/11` 里的实测数字都是换之前的。空回复和「不能真流式」这两个问题现在都已经解决了 —— 详见 [docs/14_DeepSeek接入.md](docs/14_DeepSeek接入.md)。
>
> 🚀 **服务器封闭 Beta**：已提供 Ubuntu 26.04 的 Docker + Tailscale 私有部署方案。无需域名、不开放应用端口，按 [docs/17_封闭Beta部署.md](docs/17_封闭Beta部署.md) 操作。

- **模型**：DeepSeek `deepseek-chat`（OpenAI 协议兼容；可在 `.env` 里换成 `sensenova-6.8-flash-lite` 切回商汤）
- **M1**：命令行多轮对话 + 流式输出
- **M2**：会话历史 / 多轮上下文 + 多会话管理 + **JSON 持久化（重启不丢）**
- **M3**：工具调用（ReAct 循环 + 原生 Function Calling，内置计算器 / 查时间）；精确报时由服务端确定性返回
- **M4**：RAG 知识库（文档检索增强回答，本地 BGE 中文向量化）
- **Web**：浏览器聊天界面（FastAPI + SSE，复用同一 ChatService）
- **M5**：MCP 接入（标准协议接外部工具，内置 filesystem 服务器）
- **M6**：多模态（图片理解，模型能看图回答）
- **09**：长对话摘要（聊太久不丢早期记忆，旧消息让 LLM 压成滚动摘要）
- **并发**：多会话同时跑，切会话不打断在途回复（详见 `docs/10`）
- **M8**：日志基础设施 + 空回复检测 + 一键部署（详见 `docs/11`）
- **M9**：`request_id` 全链路关联 + 输入侧边界（围栏 / 长度上限 / 显式超时 / `/health`）+ 两个可复现探针（详见 `docs/13`）
- **可靠性**：单次调用总时间封顶（`DEADLINE`）+ 上游熔断器 + `/health` 改异步不再占线程池。上游挂掉时实测 6 次调用只发 1 个 HTTP 请求（详见 `docs/15`）
- **流式工具循环**：工具有注册时最终答案也真逐 token（`complete_stream()`，工具参数按 index 分段拼接）。长回复首字 2.75s → 1.20s；工具轮那段空白从静止的「思考中…」变成「正在计算…」进度提示（详见 `docs/16`）

## 快速开始

### Windows（推荐）

```bash
# 双击 scripts\setup.bat，自动创建 venv + 安装依赖 + 复制 .env
# 然后编辑 .env 填入你的 DEEPSEEK_API_KEY

call venv\Scripts\activate.bat
python -m src.api    # Web 界面：浏览器打开 http://localhost:8000
python -m src.cli    # 命令行聊天
```

### 手动（任意平台）

```bash
python -m venv venv
# Windows: call venv\Scripts\activate.bat
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # 编辑 .env，填入真实 DEEPSEEK_API_KEY
python -m src.api       # Web 界面
python -m src.cli       # 命令行聊天
```

> `scripts/setup.bat` 仅在 **Windows 11** 上验证过。Linux / macOS 请用手动步骤。

首次启动会**自动新建一个会话**；之后每次启动都会**自动继续最近一次会话**。

## 功能演示

```
Agent · deepseek/deepseek-chat
继续会话 20260908-153000（4 条消息）· /new 新建 · /help 帮助

你：我的名字叫小码
AI：你好小码！
你：我叫什么？
AI：你叫小码。
你：/new
已新建会话 20260908-153100。
你：我叫什么？
AI：抱歉，我不知道你叫什么——这是一个新的会话。
```

## 命令

| 命令            | 作用                                             |
| --------------- | ------------------------------------------------ |
| `/exit`       | 退出程序                                         |
| `/help`       | 显示帮助                                         |
| `/new`        | 新建一个会话（旧会话保存在磁盘，可随时切回）     |
| `/sessions`   | 列出所有会话（id + 时间 + 首条消息预览 + 是否已摘要） |
| `/use <id>`   | 切换到指定会话                                   |
| `/delete <id>`| 删除指定会话（不可恢复）                         |
| `/clear`      | 清空当前会话的历史消息                           |
| `/ingest`     | 重新摄入知识库文档（`data/knowledge/`）          |

## 记忆系统说明（M2）

- **多轮上下文**：模型能记住同一会话内的对话，上下文由 `ChatService` 自动组装
- **持久化**：每个会话存为一个 JSON 文件：`data/sessions/<会话id>.json`，程序重启后自动续上
- **历史截断**：上下文超限时自动丢弃最旧对话（`MAX_HISTORY_TURNS` / `MAX_HISTORY_CHARS` 可调）
- **失败回滚**：请求失败时自动撤回本轮消息，不污染历史
- **长对话摘要**：超过阈值后由 LLM 把早期消息压成滚动摘要（09），`_build_context()` 里把摘要插在系统提示词之后、历史之前——M2 当初预留的"上下文压缩扩展点"已在 09 兑现

## 工具调用说明（M3）

- **原生 Function Calling**：`deepseek-chat` 支持原生 `tools=` 参数（已实测两轮「调工具 → 回填结果 → 再回答」），Agent 通过 ReAct 循环自动执行
- **内置工具**：`calculator`（安全计算数学表达式）、`get_current_time`（查当前时间），定义在 `src/core/tools.py`
- **安全**：计算器用 `ast` 白名单求值，拒绝任意代码执行；循环有最大轮数上限防死循环
- **扩展预留**：新增工具只需往 `ToolRegistry` 里 `register` 一个 `Tool`，M4 的检索、M5 的 MCP 工具都走这条路

## RAG 知识库说明（M4）

- **文档检索**：往 `data/knowledge/` 放 `.txt`/`.md` 文档，启动时自动摄入；模型回答资料性问题时会自动调用 `search_knowledge_base` 工具检索相关片段，再基于文档回答
- **本地向量化**：用 `fastembed` + BGE 中文模型（`bge-small-zh-v1.5`）做 embedding，完全离线免费
- **首次下载**：首次运行会自动下载 BGE 模型（约 100MB），代码里已内置走 `hf-mirror.com` 镜像
- **扩展预留**：检索被包装成一个工具接入 M3 的 ReAct 循环，`agent`/`service`/`provider` 一行没改

## Web 界面说明

- **浏览器聊天**：`python -m src.api` 启动后，打开 `http://localhost:8000`，即可聊天、切会话、新建会话
- **架构**：FastAPI 在 `ChatService` 上包了一层 HTTP，SSE 流式返回；Agent/记忆/工具/RAG 全复用
- **接口文档**：启动后可看 `http://localhost:8000/docs` 自动生成的 OpenAPI 文档
- **安全**：API Key 只存服务端（.env），前端不暴露；MVP 本机使用无认证，勿暴露公网

## MCP 接入说明（M5）

- **标准协议接工具**：通过 MCP（Model Context Protocol）接入外部工具服务器，当前内置 filesystem 服务器（14 个工具：读写文件、列目录、搜索文件等）
- **运行方式**：MCP 服务器经 `npx` 启动（需 Node.js），首次会下载；启动时自动连接并注册工具
- **安全**：filesystem 只授权 `data/mcp_workspace/` 目录，模型无法读写目录外
- **降级**：没装 Node 或 MCP 服务器启动失败时，自动降级为无 MCP 工具，聊天照常
- **部署开关**：`MCP_ENABLED=false` 可显式关闭 MCP；封闭 Beta 部署模板默认关闭，避免模型触发服务器端文件读写
- **扩展预留**：MCP 工具和计算器/检索一样，都是注册进 `ToolRegistry` 的普通工具，循环/记忆/Web 一行没改

## 多模态说明（M6）

- **图片理解**：Web 里上传图片，模型能看图回答问题（识别颜色、描述内容等）
- **模型**：直接用主力模型 `deepseek-chat`（已实测支持 base64 图片：蓝色方块图 → 答「这张图是蓝色的」，无需换模型）
- **实现**：图片以 base64 data URI 随消息发送；有图时 `content` 用多模态格式，无图时行为不变
- **持久化**：图片不落盘（base64 太大），会话 JSON 只存文字
- **范围**：仅 Web 支持图片，CLI 保持纯文字

## 长对话摘要说明（09）

- **不丢早期记忆**：聊到超过 `SUMMARY_KEEP_TURNS` 轮时，更早的消息不再被硬截断丢弃，而是**让 LLM 压成一段摘要**保存在会话里；模型看到的上下文 = 系统提示词 + 摘要 + 最近几轮原文
- **滚动摘要**：新摘要 = 总结「旧摘要 + 新溢出的旧消息」，所以摘要长度始终有界，不会随对话无限膨胀
- **持久化**：摘要存进会话 JSON 的 `summary` 字段，重启后仍记得"聊过大意"
- **失败不影响主对话**：摘要那次 LLM 调用失败（含返回空）时，旧消息原样保留，下一轮再试
- **成本**：每次溢出多一次非流式 LLM 调用（不走工具），发生在回复推完之后，**不拖慢回复本身**
- **详见**：`docs/09_SummaryMemory.md`；真实验证 `scripts/e2e_summary.py`

## 并发与多窗口说明

- **多会话同时跑**：不同会话各拿各的锁，可以同时各自生成回复（实测两个 1 秒的请求总耗时 1 秒，不是 2 秒）
- **切会话不打断**：Web 里每个会话一个独立视图，切走再切回来，在途的流式回复**还在继续输出**，侧栏亮着的绿点表示那个会话还在跑
- **同一会话串行**：同一会话的两条消息会排队（不是并发），避免两个请求读到同一份旧快照、互相覆盖导致**用户消息被静默吃掉**——这是修掉的一个真实数据丢失
- **等不及就报忙**：排队超时给出明确报错，而不是写坏历史
- **存储原子写**：会话 JSON 走「临时文件 + `os.replace`」+ 存储层锁，并发写读不会出现半截 JSON
- **已知限制**：两个窗口**看同一个会话**时历史不会自动同步（要做需加会话级推送）；只保证单进程内——详见 `docs/10_并发与多窗口.md` §10.2

## 配置项（.env）

| 环境变量               | 说明                     | 默认值                              |
| ---------------------- | ------------------------ | ----------------------------------- |
| `LLM_PROVIDER`       | 用哪家：**deepseek** / **sensenova** | `deepseek`                      |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址        | `https://api.deepseek.com/v1`     |
| `DEEPSEEK_API_KEY`  | 你的 API Key（必填）     | （无）                              |
| `DEEPSEEK_MODEL`    | 模型名（`deepseek-chat` / `deepseek-flash` / `deepseek-v4-pro`，差异见 docs/14） | `deepseek-chat` |
| `SENSENOVA_BASE_URL` | 商汤 API 地址（切回时用） | `https://token.sensenova.cn/v1`   |
| `SENSENOVA_API_KEY`  | 商汤 API Key（切回时必填） | （无）                              |
| `SENSENOVA_MODEL`    | 商汤模型名               | `sensenova-6.8-flash-lite`        |
| `MAX_TOKENS`         | 可选，单次回复最大 token。**非 reasoning 模型随意设；reasoning 模型建议留空** | 不设置则用模型默认 |
| `TEMPERATURE`        | 可选，采样温度           | 不设置则用模型默认                  |
| `SESSION_DIR`        | 会话历史存放目录         | `data/sessions`                  |
| `SYSTEM_PROMPT`      | 系统提示词（角色设定）   | 内置默认中文助手人设                |
| `TIME_ZONE`          | 精确报时使用的 IANA 时区 | `Asia/Shanghai`                    |
| `MAX_HISTORY_TURNS`  | 历史截断：最多保留几轮   | `20`                               |
| `MAX_HISTORY_CHARS`  | 历史截断：最多多少字符   | `10000`                            |
| `SUMMARY_KEEP_TURNS` | 摘要：保留最近几轮原文，更早的压成摘要（`0` = 关闭摘要） | `20` |
| `SUMMARY_MAX_CHARS`  | 摘要长度上限（字符）     | `1500`                             |
| `KNOWLEDGE_DIR`      | 知识库文档目录           | `data/knowledge`                 |
| `CHUNK_SIZE`         | 文档切块大小（字符）     | `400`                              |
| `CHUNK_OVERLAP`      | 切块重叠（字符）         | `50`                               |
| `RETRIEVAL_TOP_K`    | 检索返回片段数           | `4`                                |
| `EMBEDDING_MODEL`    | 本地向量化模型           | `BAAI/bge-small-zh-v1.5`         |
| `MCP_SERVER_COMMAND` | MCP 服务器启动命令       | `npx`                              |
| `MCP_SERVER_ARGS`    | MCP 服务器参数           | `-y @modelcontextprotocol/server-filesystem` |
| `MCP_ALLOWED_DIR`    | filesystem 授权目录      | `data/mcp_workspace`             |
| `MCP_ENABLED`        | 是否连接 MCP 工具；服务器封闭 Beta 建议关闭 | `true` |
| `MAX_IMAGES_PER_MESSAGE` | 单条消息的图片数量上限（服务端校验） | `3` |
| `MAX_IMAGE_BYTES`    | 单张图片解码后的字节上限（服务端校验） | `3145728`（3 MiB） |
| `REQUEST_TIMEOUT`    | HTTP read 超时秒数（SDK 默认 600，太宽松） | `90`                       |
| `MAX_RETRIES`        | 重试次数。SDK 自带重试已在代码里关掉，只留这一份 | `3`                         |
| `DEADLINE`           | 一次调用（**含所有重试**）的墙钟上限，read 按剩余预算收窄 | `30`                        |
| `BREAKER_FAILURE_THRESHOLD` | 连续失败 N 次后熔断打开。`0` = 关闭熔断 | `3`                         |
| `BREAKER_OPEN_SECONDS`| 熔断打开多久后自动试探恢复 | `30`                        |
| `MAX_INPUT_CHARS`    | 单次输入上限（字符），服务端校验；超长返回 400 | `32000`                  |
| `AUTH_TOKEN`         | 认证令牌。**留空 = 完全关闭**（默认），非空时请求须带 `X-Agent-Token` | （空） |
| `ACCOUNT_ENABLED`    | 启用邀请码注册、Cookie 登录与会话隔离 | `false` |
| `REGISTRATION_INVITE_CODE` | 注册邀请码（仅服务器 `.env`） | （空） |
| `RATE_LIMIT_RPM`     | 每 IP 每分钟请求上限。**`0` = 完全关闭**（默认） | `0`                    |

## 运行测试

```bash
pytest
```

## 项目结构

```
Agent_test/
├── README.md                  # 本文件
├── .env.example               # 配置模板（占位符）
├── .env.production.example    # Docker 封闭 Beta 配置模板（不含真实 Key）
├── .env                       # 真实配置（gitignore 忽略）
├── .gitignore
├── Dockerfile                 # 单 worker、非 root 的应用镜像
├── compose.yaml               # 仅绑定宿主机回环地址的容器编排
├── requirements.txt           # 依赖清单
├── pytest.ini
├── docs/
│   ├── 01_计划方案.md          # M1 计划
│   ├── 02_项目结构导读.md      # M1 代码导读
│   ├── 03_记忆系统.md          # M2 代码导读
│   ├── 04_工具调用.md          # M3 代码导读
│   ├── 05_RAG知识库.md         # M4 方案 + 设计
│   ├── 06_Web界面.md           # Web 方案 + 设计
│   ├── 07_MCP接入.md           # M5 方案 + 设计
│   ├── 08_多模态.md            # M6 方案 + 设计
│   ├── 09_SummaryMemory.md     # 09 方案 + 设计 + 实现记录
│   ├── 10_并发与多窗口.md       # 并发方案 + 诊断 + 实现记录
│   ├── 11_上线准备.md           # M8 空回复处理 + 日志 + 一键部署
│   ├── 12_项目总览.md           # ★ 全局地图：架构 / 时序图 / 状态图 / 代码地图 / 设计决策
│   ├── 13_安全加固.md           # M9 request_id 全链路关联 + 输入侧边界 + 两个探针（含负面结论）
│   ├── 14_DeepSeek接入.md       # 换供应商记录：空回复 / 流式两个问题的复测结论（2026-10-09）
│   ├── 15_调用可靠性.md         # 总时间预算 + 熔断器：上游挂掉时不再拖死线程池和探活（2026-10-11）
│   ├── 16_流式工具循环.md       # 工具循环改真流式：工具参数按 index 分段拼接 + 首字 2.75s→1.20s（2026-10-15）
│   └── 17_封闭Beta部署.md       # Ubuntu 26.04 + Docker + Tailscale 私有部署
├── data/
│   ├── sessions/              # 会话历史 JSON（含 summary，gitignore 忽略）
│   └── knowledge/             # 知识库文档（.txt/.md，gitignore 忽略）
├── static/
│   └── index.html             # Web 聊天前端（原生 HTML/JS）
├── scripts/                   # Spike 探针 + e2e 验证（花真实额度）
│   ├── e2e_summary.py         # 09 长对话不丢记忆验证（--keep 可调阈值）
│   ├── e2e_stream_tools.py    # 流式工具循环实测：首字耗时 / 分段数 / 单段长度（花额度）
│   ├── probe_concurrency.py   # 并发行为诊断（假 provider，不花额度）
│   ├── probe_streaming.py     # 验证服务端是否逐 token 流式
│   ├── probe_empty_reply.py   # 空回复根因定位（花额度）
│   ├── probe_empty_reply_real.py # 真实上下文下的空回复率 / 服务端上限（花额度）
│   ├── probe_multiprocess.py  # 多进程写同一会话丢多少条（假 provider，不花额度）
│   ├── probe_deadline_breaker.py # 总时间预算 + 熔断实测（本地假上游，不联网不花额度）
│   ├── probe_prompt_injection.py # 提示词注入实测：有围栏 vs 无围栏（花额度，跑完自动清理）
│   ├── spike_stream_tools.py  # 供应商是否支持「流式 + 工具」、参数怎么到达（花额度）
│   └── ...
├── src/
│   ├── config.py              # 配置（读取 .env）
│   ├── cli.py                 # CLI 入口（交互循环 + 会话命令）
│   ├── bootstrap.py           # 组装车间（CLI/Web 共用构建逻辑）
│   ├── api.py                 # FastAPI 应用（REST + SSE）
│   ├── core/
│   │   ├── models.py          # 统一消息模型
│   │   ├── logger.py          # 日志基础设施（标准库 logging，1MB×3 轮转）
│   │   ├── agent.py           # Agent 核心（ReAct 循环 + 流式工具循环）
│   │   ├── provider.py        # BaseProvider 抽象
│   │   ├── service.py         # ChatService（会话+历史+截断+持久化）
│   │   ├── memory.py          # Session + MemoryStore（JSON 存储）
│   │   ├── tools.py           # 工具系统（Tool + ToolRegistry）
│   │   ├── embeddings.py      # 向量化（Embedder + 本地 BGE）
│   │   ├── rag.py             # RAG（切块 + 摄入 + 检索）
│   │   ├── mcp.py             # MCP 桥（连接外部工具服务器）
│   │   └── utils.py           # 编码清洗工具
│   └── providers/
│       ├── openai_compat.py     # OpenAI 协议兼容 Provider 通用实现（重试 / 时间预算 / 熔断 / 空回复检测 / 流式工具循环 / 流式回退）
│       ├── deepseek.py          # DeepSeek 子类（当前默认，只声明 base_url + 模型名）
│       └── sensenova.py         # 商汤日日新子类（备选，只声明 base_url + 模型名）
└── tests/                     # 单元测试（全 mock，不发真实请求）
```

## 路线图

| 里程碑 | 内容 |
| ------------ | ------------------------------------------- |
| **M1** | ✅ 对话 MVP（CLI + 流式；初版用 SenseNova，2026-10-09 起默认 DeepSeek） |
| **M2** | ✅ 记忆系统（会话历史 / 多轮上下文 + 多会话 + JSON 持久化） |
| **M3** | ✅ 工具调用（Function Calling / ReAct，内置计算器 / 查时间） |
| **M4** | ✅ RAG 知识库（文档检索增强，本地 BGE 向量化） |
| **Web** | ✅ 浏览器聊天界面（FastAPI + SSE，复用 ChatService） |
| **M5** | ✅ MCP 接入（标准协议接外部工具，内置 filesystem） |
| **M6** | ✅ 多模态（图片理解；语音不做——输入不在延迟瓶颈上，且 `webkitSpeechRecognition` 不支持 Firefox） |
| **09** | ✅ 长对话摘要（滚动摘要，不丢早期记忆） |
| **M7-a** | ✅ 多会话并发（会话级锁 + 存储原子写 + 前端不销毁在途视图） |
| M7-b           | ⏳ 摘要挪后台 / 跨窗口历史同步 / 多 Agent 编排 |
| **M8** | ✅ 上线准备（日志 + 空回复检测 + 一键部署） |
| **换厂** | ✅ 默认供应商 SenseNova → DeepSeek：空回复消失、真逐 token 流式（`docs/14`） |
| **可靠性** | ✅ 总时间预算 + 熔断器 + 异步探活：上游挂掉时不再拖死线程池（`docs/15`） |
| **流式工具循环** | ✅ 工具循环改真流式：`complete_stream()` + 工具参数按 index 分段拼接，长回复首字 2.75s → 1.20s（`docs/16`） |

## 注意

- `.env` 含真实 API Key，已被 `.gitignore` 忽略，**不要提交**。
- `data/` 下是会话历史，可能含用户对话内容，**同样不要提交**。
- 对话会产生 token 消耗，注意控制使用量。
- **Windows 编码已自动处理**：程序启动时会把 stdin/stdout/stderr 统一为 UTF-8，无需手动设置 `chcp 65001` 或 `PYTHONUTF8`；模型回复含 emoji 等特殊字符也能正常显示。
- **失败处理**：限流（429）/ 断网 / 5xx 会自动退避重试，但整条调用有 `DEADLINE` 墙钟上限，超预算抛 `DeadlineExceeded` 不再重试；上游连续失败会触发**熔断**，期间秒级失败（`CircuitOpenError`），约 `BREAKER_OPEN_SECONDS` 后自动试探恢复。鉴权失败（4xx）不重试——那是配置问题，重试没意义。见 `docs/15`。
- **流式是真逐 token**：`deepseek-chat` 实测 610 字 437 段、首 token 0.74s，分段间隔可用
  `scripts/probe_streaming.py` 复现。换成 `sensenova-6.8-flash-lite` 则相反——它「先把整段算完
  再一次性下发」，50 字提问首 token 实测等 62.8s、之后 62 段只用 4.0s。这个差异决定了两件事：
  旧版 `docs/09` §9.2 里「模型不逐 token 出字，真流式改造没有可见收益」的结论**只对商汤成立**；
  换到 DeepSeek 后「流式工具循环」重新变得值得做——已在 `docs/16` 做完，最终答案现在真逐 token，
  长回复首字从 2.75s 降到 1.20s。
- **空回复只出现在 reasoning 模型上**：`deepseek-chat` 没有思考阶段，`MAX_TOKENS` 设太小只会
  把答案截断（`finish_reason='length'` 但内容有字），不会空。`sensenova-6.8-flash-lite` /
  `deepseek-v4-pro` / `deepseek-flash` / `deepseek-reasoner` 都是 reasoning 模型，
  `MAX_TOKENS` 小于思考量时思考吃光整个预算 → content 为空 → provider 抛 `EmptyReplyError`
  并撤回本轮消息（不留有问无答的脏历史）。所以想省额度：非 reasoning 模型可以设小，
  reasoning 模型请留空。见 `docs/14`。
- **会话锁按会话粒度**：不同会话真并行，同一会话串行。所以"两个窗口同时聊两个不同话题"是可以的；"同一会话被两个窗口同时问"会排队。
- **并发只保证单进程内**：Python 线程锁管不到另一个进程。别同时开 CLI 和 Web 共享同一个 `SESSION_DIR`。
- `scripts/` 下的 `e2e_*.py` 和 `probe_*.py` 会**花真实额度**，`probe_concurrency.py` 用假 provider 不花钱。
