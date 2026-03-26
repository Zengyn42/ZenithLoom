# BootstrapBuilder - 0号管家 基础 Builder Agent 计划

## Context

在空的 `BootstrapBuilder/` 目录里搭建"无垠智穹 0号管家"。
核心是 LangGraph 状态机（Claude 主控 + Gemini 战略顾问），通过 SQLite 持久化记忆，
支持多个接口无缝共享同一个会话状态。

**最核心的设计原则：** 所有接口共用同一个 `thread_id` + SQLite DB → 记忆自动同步，切换接口不丢失上下文。

---

## 接口优先级

| 优先级 | 接口 | 机制 | 备注 |
|--------|------|------|------|
| P0 | Discord | discord.py bot → LangGraph `astream()` | 远程唯一入口 |
| P1 | VSCode Terminal | CLI 脚本 + 固定 thread_id | VSCode 集成终端直连，看代码同时对话 |
| P1 | Tmux | `libtmux` 管理 named session `bootstrap_boss` | detach/reattach 不丢历史 |
| P2 | PixelOffice | **无需显式接入** | 监听 Claude Code JSONL transcript 自动可视化 |

> PixelOffice 是纯观测层（VSCode 扩展），只要在 VSCode 终端运行 CLI，它会自动把 agent 活动渲染成像素小人。

---

## 框架选型：LangGraph + SQLite WAL

- SqliteSaver 天然提供 thread_id → 会话映射
- `astream()` 支持 Discord 异步流式
- `PRAGMA journal_mode=WAL` 解决 Discord + CLI 并发访问锁问题

---

## 目录结构

```
BootstrapBuilder/
├── agent/
│   ├── __init__.py
│   ├── core.py          # LangGraph 状态机 + 引擎实例
│   └── tools.py         # consult_ceo_gemini 等 tool 定义
├── interfaces/
│   ├── discord_bot.py   # P0: Discord 远程接口（async）
│   └── cli.py           # P1: 本地 CLI + Tmux 集成
├── main.py              # 入口：python main.py [discord|cli|tmux]
├── requirements.txt
├── .env.example
└── PLAN.md              # 本文件
```

---

## Discord 关键设计（参考 openclaw）

1. **Draft Streaming** — 先发占位消息，边生成边 edit，节流 1200ms
2. **Fence-Aware Chunking** — 代码块感知分块，保证 ``` fence 不被截断
3. **Long Session Sliding Window** — `trim_messages()` 保留最近 40 条，System Prompt 永远保留

---

## 会话同步示意图

```
Discord Bot ──┐
              ├──→ builder_engine (LangGraph)
VSCode CLI ───┤         ↕
              │    SQLite WAL (cyber_bootstrap.db)
Tmux CLI ─────┘    thread_id: "boss_bootstrap_session_01"

PixelOffice → 自动监听 VSCode JSONL（无需集成代码）
```

---

## 验证步骤

1. `python main.py cli` → 本地对话，看到流式输出
2. Ctrl+C 重启 → 历史记忆保留
3. `python main.py tmux` → 创建 tmux session `bootstrap_boss`，detach/reattach 正常
4. `python main.py discord` → Discord 发消息，Bot 回复，与 CLI 共享历史
5. VSCode 终端运行 CLI → PixelOffice 自动显示像素小人

---

## 接口层泛化 + GChat 接入 + ExternalToolNode（已实现）

> 完整计划: `/home/kingy/.claude/plans/tidy-churning-ocean.md`

### 背景

CLI 与 Discord 之间存在大量重复代码（`invoke_agent`、session 命令、token 统计）。
本设计将其抽象为 `BaseInterface`，同时新增 GChat 接口和通用 `EXTERNAL_TOOL` 节点。

### 目标（已全部完成）

1. **BaseInterface 基类** — 提取共享逻辑，CLI/Discord/GChat 均继承
2. **GChatInterface** — 通过 `gws events +subscribe` 接收 GChat 消息，经 LangGraph 处理，`gws chat +send` 回复
3. **ExternalToolNode（EXTERNAL_TOOL）** — 通用 CLI 调用节点，`entity.json` 声明命令列表即可接入任意外部工具

### 架构原则

接口层架构不变：传输层在外部调用 `graph.astream()`，LangGraph 图不感知接口来源。

```
Discord Bot         ─┐
CLI                  ├──→ graph.astream() → LangGraph
GChat Bot           ─┘

EXTERNAL_TOOL 节点 ──→ 子进程调用任意 CLI（参数列表，无 shell）→ JSON 输出注入 messages
```

### 三层外部工具策略

| 场景 | 策略 | command 示例 |
|------|------|------|
| CLI-Anything harness 已有（blender、gimp 等） | 直接用 `cli-anything-<tool> --json` | `["cli-anything-blender", "--json", "render", "animation"]` |
| 已有结构化 CLI（gws、obsidian、git） | 直接调用，无需包装 | `["gws", "gmail", "+triage", "--json"]` |
| 无 CLI 的 GUI 应用 | 用 /cli-anything 生成 harness 后接入 | — |

判断准则：工具是否已有 `--json` flag + 稳定子命令结构 → 有则直接调用，无则用 CLI-Anything。

### gws 的双重角色

| 角色 | 说明 | 实现位置 |
|------|------|------|
| **接口层（Interface）** | `gws events +subscribe` 监听 GChat 消息作为对话入口；`gws chat +send` 回复 | `interfaces/gchat_bot.py` → `GChatInterface` |
| **工具节点（Tool Node）** | `gws gmail +triage`、`gws drive list` 等作为 EXTERNAL_TOOL 节点执行 GWS 操作 | `framework/nodes/external_tool_node.py` → `ExternalToolNode` |

两者互不干扰：GChatInterface 是图的**调用者**（图外），ExternalToolNode 是图的**被调用节点**（图内）。

### 已新建文件

| 文件 | 说明 |
|------|------|
| `framework/base_interface.py` | BaseInterface：`invoke_agent()`、`handle_command()`、`split_fence_aware()`、`extract_attachments()` |
| `interfaces/gchat_bot.py` | GChatInterface：`gws events +subscribe` NDJSON 流 → agent → `gws chat +send` |
| `framework/nodes/external_tool_node.py` | ExternalToolNode：通用 CLI 调用，`{field}` 模板注入 state，JSON pretty-print |

### 已修改文件

| 文件 | 改动 |
|------|------|
| `interfaces/cli.py` | `_CliInterface(BaseInterface)`，`!topology`/`!debug`/`!snapshots`/`!rollback` 留子类 |
| `interfaces/discord_bot.py` | `_DiscordInterface(BaseInterface)`，模块级 `bot` 保持（discord.py 约束） |
| `framework/builtins.py` | 注册 `EXTERNAL_TOOL` → `ExternalToolNode` |
| `main.py` | 新增 `gchat` 模式 |
| `framework/config.py` | AgentConfig 新增 `gchat_space`、`gchat_gcp_project`、`gchat_event_types` |

### EXTERNAL_TOOL 用法示例（entity.json）

```json
// gws：已有结构化 CLI（直接调用，无需 CLI-Anything）
{ "id": "gmail_reader",    "type": "EXTERNAL_TOOL",
  "node_config": { "command": ["gws", "gmail", "+triage", "--json"],
                   "description": "读取未读邮件摘要" } },

// obsidian：已有官方 CLI（直接调用）
{ "id": "obsidian_search", "type": "EXTERNAL_TOOL",
  "node_config": { "command": ["obsidian", "search", "--query", "{routing_context}"],
                   "description": "搜索 Obsidian vault" } },

// blender：CLI-Anything harness 已有（通过包装调用）
{ "id": "blender_render",  "type": "EXTERNAL_TOOL",
  "node_config": { "command": ["cli-anything-blender", "--json", "render", "animation"],
                   "description": "渲染 Blender 动画", "timeout": 120 } }
```

### GChat 配置（entity.json）

```json
"gchat_space": "spaces/AAAA...",
"gchat_gcp_project": "my-gcp-project",
"gchat_event_types": "google.workspace.chat.message.v1.created"
```

### GChat 事件流架构

```
GChat 用户消息
  → Google Workspace Events API
    → GCP Pub/Sub Topic (需提前配置)
      → gws events +subscribe (长进程, NDJSON poll)
        → GChatInterface._extract_chat_event()
          → invoke_agent() / handle_command()
            → gws chat +send 回复
```

### 关键约束

- `gws events +subscribe` 的实际 NDJSON 字段路径需通过 `--once` 单次拉取实测确认
- 需过滤 Bot 自身消息（`sender.type == "BOT"`）防止回声循环
- discord.py 要求 `bot` 对象在模块级注册，`DiscordInterface` 通过闭包注入 loader/controller
- 外部进程调用统一用参数列表形式（非 shell 字符串），避免注入风险
