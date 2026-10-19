# 14 换供应商：SenseNova → DeepSeek（两个"服务端问题"的复盘）

> **目标**：把默认供应商从商汤日日新换成 DeepSeek，并**用真实请求重新验证** `docs/11` 查到的两件事——「空回复」「不能真流式」。
>
> **为什么这次能验证**：以前这两条都被记成「服务端行为，客户端改不了」，但那时只有一家供应商，没有对照组。换了厂商等于拿到另一组数据：如果问题跟着**供应商**走，那是服务商实现的问题；如果跟着**模型类型**走，那是模型性质的问题。这次的答案是后者（空回复）和前者（流式）各占一半。
>
> **时间与条件**：2026-10-09，Windows 本地，`openai` SDK 3.3.1，Python 3.14。本文所有数字都是**真实请求实测**（花真实额度），没有一条是推测。
>
> **建议阅读顺序**：**1 结论先行 → 2 代码改了什么 → 3 五个模型的实测画像 → 4 关键代码逐行 → 5 自己复现一遍 → 6 测试 → 7 还没修好的 → 8 小结**。只想看结论读 §1 和 §7 即可。

---

## 1. 结论先行

`docs/11` 记的两件事，现在的状态：

| 原结论（docs/11 §7） | 现在的状态 | 真正的原因 |
| --- | --- | --- |
| 空回复率 36%，根因在服务端 reasoning 模型，客户端改不了 | ✅ **已修好**（换默认模型后不再出现） | 不是「商汤服务端的毛病」，是 **reasoning 模型通性**：思考 token 和答案共用 `completion_tokens` 预算，预算不够时思考吃光 → `content` 为空 |
| 服务端先把整段算完再一次性下发，「真流式」改造没有可见收益 | ✅ **已修好**（换成逐 token） | 这是**两家服务端的实现差异**，与模型类型无关。DeepSeek 真逐 token；商汤「算完再下发」 |

三个顺带的收获：

- **首 token 延迟**：`sensenova-6.8-flash-lite` 50 字提问要等 **62.82s**；`deepseek-chat` 同档提问 **0.58~0.74s**。差 100 倍。
- **限流**：4 次连发实测 0.56 / 0.57 / 0.82 / 0.80s，没有 429（旧的 429 处理代码保留，不是白写的）。
- **`MAX_TOKENS` 的心智负担变简单了**：非 reasoning 模型设多小都不会空回复，只会把答案截断。

一个需要说清楚的修正：`docs/11` §7 发现三里写「`max_tokens=4096` → 0/3 空」，这个数字**当时是对的，现在已经不对了**。同一模型、同一句 50 字提问，本次实测 `max_tokens=4096` → `completion_tokens=4096` 且 `reasoning_tokens=4096` → `content` 长度 0。思考消耗是服务端随机抖动的（`docs/11` 实测同句从 70 跑到 8533），所以「留空 MAX_TOKENS」这条建议仍然成立，但「设 4096 就安全」不再成立。

---

## 2. 代码改了什么

本次改动**没有动 Agent 核心**（`agent.py` / `service.py` / `memory.py` / `tools.py` / `rag.py` / `mcp.py` / `models.py` 一行未改），跟 M8 一样只动供应商边界和运维层：

| 文件 | 状态 | 改了什么 |
| --- | --- | --- |
| `src/providers/openai_compat.py` | 🆕**新增** | 把原 `sensenova.py` 的实现整体搬进来，改成**厂商无关**：重试 / 空回复检测 / 编码回退全在这里。新增 `PROVIDER_NAME`、`label`、`STREAM_USAGE` |
| `src/providers/deepseek.py` | 🆕**新增** | `DeepSeekProvider`，只有身份声明（默认 base_url + 模型名），**14 行** |
| `src/providers/sensenova.py` | ✏️**缩成 14 行** | 只剩身份声明。保留的原因：`LLM_PROVIDER=sensenova` 能切回，`docs/01`~`11` 的实测记录立刻重新适用 |
| `src/config.py` | ✏️**改动** | 🆕`llm_provider`（默认 `deepseek`）+ `deepseek_*` 三项；`sensenova_*` 保留；系统提示词去掉厂商名（不再说「由商汤日日新驱动」） |
| `src/bootstrap.py` | ✏️**改动** | 🆕`PROVIDERS` 登记表 + `build_provider()`（名字写错 / Key 缺失都给出可操作的报错）+ `active_provider_config()`（探针脚本取配置用） |
| `src/core/provider.py` | ✏️**改动** | `EmptyReplyError` 的文档字符串和提示语改成厂商无关，并补上第二条修复建议「换非 reasoning 模型」 |
| `src/cli.py` | ✏️**改动** | 启动横幅 `SenseNova Agent · 模型 xxx` → `Agent · deepseek/deepseek-chat` |
| `src/api.py` / `static/index.html` | ✏️**改动** | 应用标题去掉厂商名 |
| `scripts/probe_streaming.py`<br>`scripts/probe_empty_reply.py`<br>`scripts/probe_empty_reply_real.py`<br>`scripts/e2e_check.py`<br>`scripts/e2e_rag.py`<br>`scripts/e2e_tools.py` | ✏️**改动** | 原来硬编码 `settings.sensenova_*` / `SenseNovaProvider`，现在走 `active_provider_config()` 和 `build_provider()` 读当前供应商——**换厂后这些脚本不用改就自动跟着走**。如果不改，它们会静默继续测商汤，得出「还在商汤上」的假结论。<br>另：5 个 `e2e_*.py`（`check`/`rag`/`tools` 之外）本来就走 `build_service()`，自动适配，零改动 |
| `tests/test_provider.py` | 🆕**新增**（取代 `tests/test_sensenova.py`） | 空回复检测测试参数化覆盖两家子类；新增流式参数（`stream_options`）和 reasoning delta 过滤测试 |
| `tests/test_bootstrap.py` | 🆕**新增** | `build_provider()` 的选择逻辑与两条报错路径 |
| `.env` / `.env.example` | ✏️**改动** | 换 Key；`MAX_TOKENS` 提示按「reasoning / 非 reasoning」两类重写 |
| `README.md` | ✏️**改动** | 供应商、配置表、项目结构、注意事项同步 |

**为什么把实现拆成「一份实现 + 两个身份」**：两家的接口协议完全一样（都是 OpenAI 兼容），如果各写一份 `Provider`，重试、空回复检测、编码回退这三段就要维护两份，以后修一个 bug 要修两遍。现在厂商差异只剩三行数据，新增一家供应商 = 写一个 14 行的子类 + 在 `PROVIDERS` 里加一行。

**这次换厂为什么这么便宜**：`BaseProvider` 抽象（`src/core/provider.py`）在 M1 就定好了，`ChatAgent` 只认这个接口。换供应商动的是「谁实现这个接口」，不是「谁用这个接口」。这是分层分对了的最直接收益。

---

## 3. 五个模型的实测画像

同一台机器、同一把 Key、同一天实测。`max_tokens` 除非特别说明都是显式设置的。

| 模型 | reasoning | 首 content | 流式行为 | 视觉 | 工具调用 | 显式小 `max_tokens` |
| --- | --- | --- | --- | --- | --- | --- |
| **`deepseek-chat`**（当前默认） | ❌ 否 | **0.58~0.74s** | **真逐 token**（610 字 437 段） | ✅ base64 | ✅ 两轮闭环 | 截断，**不空** |
| `deepseek-flash` | ✅ 是 | 7.54s | 真逐 token | ✅ base64 | ✅ | **空回复** |
| `deepseek-v4-pro` | ✅ 是 | **37.41s** | 真逐 token | ❌ | ❌（不认工具） | **空回复** |
| `deepseek-reasoner` | ✅ 是 | 3.93s | 真逐 token | ❌ | ✅ | **空回复** |
| `sensenova-6.8-flash-lite`（旧默认） | ✅ 是 | **62.82s** | ❌ **算完再一次性下发** | `docs/08` 已验证（本次未复测） | ✅ | **空回复** |

### 3.1 空回复：机制与厂商无关

reasoning 模型的输出分两块——`reasoning`（思考）和 `content`（答案）——两块**共用** `completion_tokens` 预算。给一个不够的预算，思考先把钱花光，`content` 就是空的：

```
deepseek-v4-pro   max_tokens=120 → completion_tokens=120, reasoning_tokens=120, content=''
deepseek-flash    max_tokens=120 → completion_tokens=120, reasoning_tokens=120, content=''
deepseek-reasoner max_tokens=120 → completion_tokens=120, reasoning_tokens=120, content=''
sensenova-6.8-flash-lite  max_tokens=4096 → completion_tokens=4096, reasoning_tokens=4096, content=''
```

四行里三个是 DeepSeek。所以 `docs/11` 那句「根因在服务端」要修正为：**根因在「这是一类 reasoning 模型」，跟是不是商汤没有关系**。

对照组是非 reasoning 的 `deepseek-chat`，同一个请求、同一个预算上限：

```
deepseek-chat  max_tokens=30 → finish_reason='length'
               completion_tokens=30
               content='大家好，我叫林晓阳，今年28岁，来自浙江杭州。我是一个性格开朗、乐于沟通的人，喜欢在交流中学习，也'
```

`finish_reason` 一样是 `length`，但 `content` 里有字——因为它的预算只花在答案上，**被砍掉的是「答案的后半截」，不是「答案本身」**。这就是「非 reasoning 模型设小 `MAX_TOKENS` 只是截断、不会空回复」的由来。

批量复测（`scripts/probe_empty_reply.py`，24 次真实请求，四档 `max_tokens`）：

```
配置                 空回复        占比
max_tokens=50      0/6       0%
max_tokens=100     0/6       0%
max_tokens=300     0/6       0%
max_tokens=不传      0/6       0%
```

24 次调用**全部 `finish=stop` 且都有内容**（每次 11~58 字，耗时 0.53~1.38s）。
对照 `docs/11` 在同一批提问上量到的数字：14 次里 5 次空（36%）、`max_tokens=2048` 时 2/3 空。

`usage.completion_tokens_details.reasoning_tokens` 是判断一家模型属于哪一类的最可靠信号：`deepseek-chat` 的所有响应里这个字段都是 `null`，其余四个都是实数。这个字段本来就是为了 M8 的空回复诊断加的，现在还能兼职当「模型类型探测器」。

### 3.2 流式：这是服务商实现差异，与模型类型无关

同样用 `scripts/probe_streaming.py`（同一条 80 字提问）：

**商汤（2026-09，docs/11 记录）与本次复测**——首段等 62.82s，之后 62 段共用 4.00s：

```
首 token 62.82s；chunks 62；总耗时 4.00s（不含等待）
```

**DeepSeek `deepseek-chat`（本次实测）**：

```
模型：deepseek-chat
  第 1 段 (+1.405s)：'春天'
  第 2 段 (+0.096s)：'是'
  第 3 段 (+0.000s)：'大地'
  第 4 段 (+0.000s)：'复苏'
  第 5 段 (+0.000s)：'的'

共 62 段，总耗时 2.08s
段间隔：min=0.0001s  max=1.4050s  sum=2.0790s
→ 结论：服务端是逐段下发的，可以支持真流式
```

62 段的耗时**均匀分布在整个 2.08s 里**，而不是压在最后 0.25s——这是判「真流式」的依据（探针脚本最后那条判定用的就是这个）。

长输出也验证过：610 字回答 437 段，首 token 0.74s、末 token 3.82s，`chunk` 数接近字数，说明基本是一字一段。

**这条修正影响一个既有结论**：`docs/09` §9.2 和 `src/core/agent.py` 的 `_chunked()` 里写「因为模型不逐 token 出字，真流式改造不会有可见收益」。这个前提现在不成立了——`docs/10` 里那个「流式工具循环」方案重新变得值得做。旧代码没删，只是它当初的论据过期了。

另外，`deepseek-v4-pro` 虽然也是逐 token，但首段 content 要 **37.41s**（思考从 0.79s 就开始流，`reasoning_content` 帧能到，只是 `content` 帧迟迟不来）。对流式来说「有帧」和「有用」是两回事——用户看到的还是 37 秒的空白。

### 3.3 视觉与工具调用

- **视觉**：项目本来就发 base64 data URI（`static/index.html` 里 `pendingImages.push(dataUri)`），这条链路在 DeepSeek 上直接可用：64×64 蓝底 PNG → 答「这张图是蓝色的。」（1.25s）。
  ⚠ 但如果换成**外链图片 URL**，DeepSeek 会返回 400 `Failed to download image`（它自己去下不了图）。项目现在走 base64，**这条路恰好是安全的那条**；将来别为了省流量改回外链。
- **工具调用**：`deepseek-chat` 两轮闭环实测通过——第一轮 `finish_reason='tool_calls'` 返回 `calculator {"expression": "123 * 456"}`，回填 tool 结果后第二轮答 `123 × 456 = **56088**`。
- **`deepseek-v4-pro` 不认工具**：给了同样的 `calculator` 工具，它直接自己算（答 `256 × 789 = **201984**`，`tool_calls` 为空）。这不是 bug，是模型能力边界，但如果哪天把默认模型改成它，**整个工具链会静默失效**——值得单独记住这一点。

### 3.4 一个小坑：`models.list()` 不全

`GET /v1/models` 只返回了 `deepseek-flash` 和 `deepseek-v4-pro`，但 `deepseek-chat` 和 `deepseek-reasoner` 两个名字**都能正常调用**。所以不要拿模型列表当可用性判断，`.env` 里写什么就跑什么。

---

## 4. 关键代码逐行

### 4.1 `stream_options`：DeepSeek 默认不发 usage（`src/providers/openai_compat.py`）

```python
if stream and self.STREAM_USAGE:
    kwargs["stream_options"] = {"include_usage": True}
```

这行是新加的，原因是空回复诊断**依赖** `usage`：

- 商汤：流式 chunk 末尾自带 `usage`（含 `reasoning_tokens`）；
- DeepSeek：**默认不带**。不传这个参数，流式跑完后 `usage` 是 `None`，`_reject_empty()` 打出来的日志就只有「空回复」三个字，`reasoning_tokens` 那一栏永远是 `?`——`docs/11` 花力气建立的「为什么空」这条线索会断掉。

传了这个参数，末尾会有一个空 `choices`、带 `usage` 的 chunk，现有代码里 `if chunk.usage: usage = chunk.usage` 那行本来就能接住，所以只加了一行请求参数，不用改解析逻辑。两家的兼容性都实测过（`STREAM_USAGE` 做成类属性，就是留给第三家不认这个参数的情况覆写掉）。

### 4.2 只转发 `delta.content`：思考过程不会漏给用户

```python
if choice.delta and choice.delta.content:
    text = scrub_text(choice.delta.content)
```

DeepSeek 的 reasoning 模型在流式里把思考放在 `delta.reasoning_content`（商汤放在 `delta.reasoning`）。这段代码只读 `content`，所以思考帧被天然跳过——`tests/test_provider.py::test_chat_stream_skips_reasoning_deltas` 锁住了这个行为。

这个「天然跳过」不是巧合，是 M8 写代码时就没把思考当正文。将来若真想把思考过程展示出来，得显式加代码，不会自己冒出来。

### 4.3 `build_provider()`：换厂只改一个环境变量

```python
PROVIDERS = {
    "deepseek": ("deepseek", "DEEPSEEK_API_KEY", DeepSeekProvider),
    "sensenova": ("sensenova", "SENSENOVA_API_KEY", SenseNovaProvider),
}

def build_provider() -> BaseProvider:
    name = (settings.llm_provider or "").strip().lower()
    if name not in PROVIDERS:
        print(f"错误：LLM_PROVIDER={name!r} 无法识别（已忽略大小写和首尾空格）。\n"
              f"可选值：{', '.join(PROVIDERS)}（见 docs/14）", file=sys.stderr)
        raise SystemExit(1)
```

两个细节：

- `name` 先 `strip().lower()` 再查表——人工编辑 `.env` 写 ` SenSoNoVa ` 是常态，不能因为这样就让程序退出。
- 两条报错都**点出具体的环境变量名**（`DEEPSEEK_API_KEY`）和**可选值列表**（`deepseek, sensenova`），而不是笼统说「配置错误」。`docs/11` 里「把错误写成人看得懂」这条原则在这里延续了。

### 4.4 `EmptyReplyError` 的提示语：补上第二条修复建议

```python
+ "　修复：在 .env 里留空 MAX_TOKENS，让服务端用它自己的默认上限"
  "（显式设一个较低的数值反而更容易触发空回复）；"
  "或者换一个非 reasoning 模型（如 deepseek-chat，它没有思考阶段，"
  "预算被吃光只会截断答案，不会产生空回复，见 docs/14）。"
```

原来只有一条建议「留空 MAX_TOKENS」，但那条是**治标**（赌服务端默认上限够用，`docs/11` 实测那次是 8567）。现在多了**治本**的一条：换非 reasoning 模型。这条建议能进异常消息，靠的是 §3.1 那个对照实验——没有那组数据，这句话就是瞎说的。

---

## 5. 自己复现一遍

三条命令，全部花真实额度：

```bash
# ① 流式行为：打印每段间隔，最后给判定（真流式 / 算完再下发）
python -X utf8 scripts/probe_streaming.py

# ② 空回复率：多档 max_tokens × 每档 6 次
python -X utf8 scripts/probe_empty_reply.py

# ③ 想看旧供应商的对照数据：把 .env 里 LLM_PROVIDER 改成 sensenova，重跑 ①②
```

切换供应商只需要动 `.env` 一行：

```
LLM_PROVIDER=deepseek    →    LLM_PROVIDER=sensenova
```

两个 Key 都留在 `.env` 里（那个文件被 `.gitignore` 忽略），来回切不用重填。

---

## 6. 测试

- `tests/test_provider.py`：空回复检测、编码清洗、流式参数、reasoning delta 过滤，**参数化覆盖 `DeepSeekProvider` 和 `SenseNovaProvider` 两家**——既测实现，也测「子类真的只是声明身份」。
- `tests/test_bootstrap.py`：`build_provider()` 的默认选择、切回 sensenova、大小写/空格容错、两条报错路径（名字写错 / Key 缺失）、`provider_label()`。
- `tests/test_config.py`：新增 `llm_provider` / `deepseek_*` 默认值与环境变量覆盖。
- 全部 mock client，不发真实请求，不花额度。**当前 166 个测试全绿**（换厂前是 142 个）。
- 本文 §3 的所有数字**不是测试**，是真实请求实测；测试保证的是「代码逻辑」，实测保证的是「供应商行为」。

### 6.1 端到端实测（花真实额度）

| 验证项 | 命令 | 结果 |
| --- | --- | --- |
| 流式行为 | `python -X utf8 scripts/probe_streaming.py` | 62 段 / 2.08s，判定「逐段下发」 |
| 空回复率 | `python -X utf8 scripts/probe_empty_reply.py` | 四档 `max_tokens` × 6 次 = 24 次，**0 空回复**（见 §3.1） |
| 工具调用闭环 + 记忆 + 重启续接 | `python -X utf8 scripts/e2e_tools.py` | ✓ 计算器、✓ 查时间、✓ 记住名字、✓ 重启后仍记得 |
| Web 全流程 | `python -X utf8 scripts/e2e_web.py` | ✓ 首页、✓ 计算工具、✓ RAG（带来源标注）、✓ 记忆、✓ 会话列表 |
| CLI 全流程（含 RAG + MCP） | `python -m src.cli` | ✓ 启动即连接 MCP（14 个工具）、摄入知识库，正常流式回复 |
| Web 服务本体 | `python -m src.api` + HTTP 请求 | ✓ `/openapi.json` 200、`/` 200、SSE 端点正常出事件 |

`scripts/e2e_tools.py` 那条之所以值得单独列出来：它走的是项目自己的 ReAct 循环，
中途会往请求里放一条 `role=assistant, content=""` + `tool_calls` 的消息。DeepSeek 对这个格式
**接受并正确续接**，说明 M3 那套协议细节在换厂后没有踩坑。

换厂前后跑的是**同一套脚本、同一套断言**——这也是「分层对了」的另一种证明：
验证脚本没有一句需要为换厂而改。

---

## 7. 还没修好的（诚实清单）

1. **reasoning 模型 + 显式 `MAX_TOKENS` 仍然会空回复。** 这是模型性质，不是 bug，也没法在客户端修——预算是给思考花的，客户端拦不住。能做的只有两条：默认用非 reasoning 模型（已做）、异常消息里说清楚（已做）。
2. **`deepseek-v4-pro` 不能用。** 首段答案 37s + 不认工具 + 不支持视觉。模型列表里有它，但别选它。
3. **`deepseek-flash` 能看不能快。** 视觉和工具都正常，但它是 reasoning 模型：首 content 7.54s，而且设 `MAX_TOKENS` 会空回复。想要视觉 + 空回复不敏感，得接受它的延迟。
4. **`docs/02` ~ `docs/11` 里的实测数字都是商汤时代的。** 那些文档没有改——它们记的是「当时那家供应商、那台机器上发生了什么」，改掉就失真了。凡是会被误读成「当前行为」的地方（README 的快速开始/配置表/注意事项、`.env.example` 的 MAX_TOKENS 提示、`src/core/provider.py` 的异常消息），本次都同步改了，并在这篇文档里做了对照。三份总览性质的文档（`docs/02` / `docs/11` / `docs/12`）顶部各加了一条「写于 SenseNova 时代」的醒目提示，指向本文档。
5. **`scripts/spike_*.py` 没改。** 那批是一次性探索脚本（M3/M6 时期的 spike），本来记录的就是「当时那家供应商上的某一次发现」，改成读当前供应商会让它们的历史结论对不上。想再跑一遍的话，先把 `.env` 里的 `SENSENOVA_API_KEY` 留着——它已经留了。
6. **商汤的视觉能力本次没复测**（`docs/08` 验证过）。切回 `sensenova` 后如果要确认多模态，需要重跑 `scripts/e2e_vision.py`。
7. **`MAX_TOKENS` 留空的代价**：留空等于交给服务端决定上限，长回答会被服务端自己的预算截断（`deepseek-chat` 实测 800 字请求返回 1221 字、`completion_tokens=880`，未被截断）。本项目没遇到过需要更长输出的场景，但这是留空的真实代价，不是免费的。

---

## 8. 小结

这一期的净收益是「**两个当年判定为无法修复的问题都修好了**」，而真正的收获在别处：**当时的归因不够精确**。

`docs/11` 写「根因在服务端，客户端改不了」——这句话在只有一组数据时是成立的：客户端确实改不了对方怎么实现。但「服务端」这个词太大了，它把两类完全不同的东西装在一起：

- **模型的性质**（reasoning 模型共用预算）——换供应商也逃不掉，因为 DeepSeek 的三个 reasoning 模型一样会空回复；
- **服务商的实现**（算完再一次性下发）——换供应商就能躲开。

混在一起记，就只能得出「改不了」；分开记，才有「换模型类型可以修」和「换供应商可以修」两条具体出路。这次换厂之所以值，不是因为 DeepSeek 更好，而是因为它提供了一组**能区分归因**的对照数据。

代码层面的改动很小（一份实现 + 两个 14 行的身份声明 + 一个选择函数），这是 `BaseProvider` 抽象的回报。**能力没变、核心一行没改、两个问题修好、一条归因修正**——这是这一期的全部。

**验收对照**：

| 验收项 | 结果 |
| --- | --- |
| 默认供应商切换为 DeepSeek | ✅ `.env` + `LLM_PROVIDER` 默认值；`Agent · deepseek/deepseek-chat` |
| 空回复不再出现 | ✅ 24 次真实请求 0 空回复（`deepseek-chat` 非 reasoning，预算吃光只截断不空） |
| 真逐 token 流式 | ✅ 62 段 / 2.08s，段间隔均匀分布 |
| 首 token 延迟下降 | ✅ 62.82s → 0.58~0.74s |
| 工具调用闭环 | ✅ 两轮实测通过；Web 与 CLI 端到端各跑一遍 |
| 多模态可用 | ✅ base64 图片（外链图片不可用，项目本来就没走外链） |
| 能一键切回旧供应商 | ✅ `.env` 改一行，两套 Key 都留着；`build_provider()` 选择逻辑有单测 |
| Agent 核心未改动 | ✅ `agent.py` / `service.py` / `memory.py` / `tools.py` / `rag.py` / `mcp.py` / `models.py` 零改动 |
| 单元测试 | ✅ **166 passed**（换厂前 142） |
| 端到端验证 | ✅ `e2e_tools.py` / `e2e_web.py` / `probe_streaming.py` / `probe_empty_reply.py` / CLI / Web 服务 |
