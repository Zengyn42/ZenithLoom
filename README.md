# BootstrapBuilder — 多 Agent 框架

基于 LangGraph 的可插拔 AI Agent 框架，支持 Claude（云端）和 Llama（本地）双引擎，提供 CLI、Discord Bot 两种接入方式。

---

## 架构概览

```
BootstrapBuilder/
├── main.py                  # 统一入口（--agent / mode）
├── framework/               # 可复用框架层（LLM 无关）
│   ├── config.py            # AgentConfig dataclass
│   ├── state.py             # BaseAgentState（LangGraph 状态）
│   ├── graph.py             # 通用图构建器 + Session 管理
│   ├── agent_loader.py      # 从 agent.json 懒加载 Agent
│   ├── session_mgr.py       # 命名 Session 持久化
│   ├── heartbeat.py         # CLI/Ollama 存活探测（后台定时）
│   ├── token_tracker.py     # Claude API token 统计
│   ├── claude/node.py       # ClaudeNode — Claude Code CLI SDK 包装
│   ├── llama/node.py        # LlamaNode — Ollama HTTP 接口
│   ├── gemini/node.py       # GeminiNode — Gemini 顾问（3 轮对抗咨询）
│   └── nodes/
│       ├── agent_node.py    # AgentNode — 通用 LLM 节点（JSON 驱动）
│       ├── git_nodes.py     # Git 快照 / 回滚节点
│       ├── validate_node.py # 输出验证节点
│       └── vram_flush_node.py # GPU 显存清洗节点
├── agents/
│   ├── hani/                # Claude Agent（个人助手）
│   │   ├── agent.json       # 所有配置
│   │   ├── SOUL.md          # 价值观（不可覆盖）
│   │   ├── IDENTITY.md      # 身份认知
│   │   ├── OPERATIONAL.md   # 操作规范
│   │   └── COMMANDS.md      # 用户命令手册
│   └── asa/                 # Llama Agent（本地常驻）
│       ├── agent.json
│       └── SOUL.md
└── interfaces/
    ├── cli.py               # 本地 CLI
    └── discord_bot.py       # Discord Bot
```

---

## LangGraph 图流程

每轮对话经过以下节点：

```
用户输入
   │
   ▼
git_snapshot       ← 工程模式下自动快照（!setproject 后生效）
   │
   ▼
agent_node         ← 调用 LLM（Claude / Llama）
   │
   ▼
validate           ← 检查输出是否触发回滚 / Gemini 咨询信号
   │
   ├─ rollback_reason 非空 ──► git_rollback ──► agent_node（重试）
   │
   ├─ gemini_context 有待处理 ──► gemini_advisor ──► agent_node
   │
   └─ 正常 ──► [vram_flush（可选）] ──► END
```

---

## Agent 配置（agent.json）

每个 Agent 的行为完全由 `agents/<name>/agent.json` 驱动，无需修改框架代码。

### 完整字段说明

```jsonc
{
  // ── 基础 ──────────────────────────────────────────
  "name": "hani",                    // Agent 标识（与目录名一致）
  "llm": "claude",                   // "claude" | "llama"
  "workspace": "/path/to/project",   // 默认工作目录（可被 !setproject 覆盖）

  // ── LLM 配置 ──────────────────────────────────────
  "claude_model": null,              // null = CLI 默认模型；或 "claude-sonnet-4-6"
  "llama_model": "llama-3.3-70b",   // （llm=llama 时）
  "llama_endpoint": "http://localhost:11434",

  // ── Claude Code CLI 设置继承 ───────────────────────
  // 控制子进程加载哪些 Claude Code 设置
  "setting_sources": ["user", "project"],
  // "user"    = 继承 ~/.claude/settings.json 中已安装的全部 Skill/Plugin
  // "project" = 读取工作目录下 .claude/ 的项目级 Skill
  // 不写或 null = 不加载任何设置（SDK 默认）

  // ── 工具 ──────────────────────────────────────────
  "tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
  "permission_mode": "bypassPermissions",

  // ── 功能开关 ──────────────────────────────────────
  "vram_flush": false,    // true = 图末尾执行 GPU 显存清洗（Asa 专用）
  "heartbeat": true,      // true = 启动后台 CLI 存活探测（Asa 专用，常驻进程）
  "tombstone_enabled": true, // true = 注入历史失败案例防止重蹈覆辙

  // ── Session 持久化 ────────────────────────────────
  "db_path": "hani.db",           // LangGraph checkpoint DB（相对于 agent 目录）
  "sessions_file": "sessions.json",// 命名 Session 映射表
  "max_retries": 2,               // 最大回滚重试次数
  "max_gemini_consults": 1,       // 每轮最多 Gemini 咨询次数

  // ── Discord 频道历史 ───────────────────────────────
  "channel_history_limit": 20,    // 保存最近 N 条消息到 .discord_channel_history.txt
                                   // Agent 可用 Read 工具按需读取（不自动注入）

  // ── Persona ───────────────────────────────────────
  "persona_files": ["SOUL.md", "IDENTITY.md", "OPERATIONAL.md", "COMMANDS.md"],
  "first_turn_suffix": "Hani:",   // 首轮 prompt 末尾追加（引导角色扮演）
  "user_msg_prefix": "老板: ",    // 用户消息前缀

  // ── Gemini 咨询触发 ───────────────────────────────
  "gemini_mention_pattern": "@[Gg]emini",  // 用户消息含此正则 → 强制咨询

  // ── 动态工具规则 ──────────────────────────────────
  "tool_rules": [
    {
      "pattern": "论文|paper|arXiv",  // 用户消息正则
      "flags": ["IGNORECASE"],
      "tools": ["WebFetch", "WebSearch"]  // 动态追加到工具列表
    }
  ]
}
```

### 当前 Agent 对比

| 字段 | Hani | Asa |
|------|------|-----|
| `llm` | claude | llama |
| `vram_flush` | false | **true** |
| `heartbeat` | false | **true** |
| `setting_sources` | ["user", "project"] | — |
| 定位 | 远程助手（Discord + CLI） | 本地常驻（CLI） |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填写：
#   DISCORD_BOT_TOKEN  （Discord Bot 模式必填）
#   DISCORD_ALLOWED_USERS  （授权用户 ID，用 !whoami 获取）
```

### 3. 启动

```bash
# Hani — 本地 CLI（Claude）
python main.py cli

# Hani — Discord Bot
python main.py discord

# Asa — 本地 CLI（Llama，需要先启动 Ollama）
python main.py --agent asa cli

# 调试模式
DEBUG=1 python main.py cli
```

---

## 接口命令

### CLI 和 Discord 通用命令

| 命令 | 说明 |
|------|------|
| `!new <名称>` | 创建并切换到新命名 Session |
| `!switch <名称>` | 切换到已有 Session |
| `!session` | 显示当前 Session 名称和 thread_id |
| `!sessions` | 列出所有 Session（当前用 ◀ 标注） |
| `!memory` | 查看 Session 消息数和 DB 大小 |
| `!compact [N]` | 压缩 Session，保留最近 N 条（默认 20） |
| `!reset confirm` | 清空当前 Session 全部记忆 |
| `!clear` | 清空 Session（无需确认） |
| `!tokens` | 查看 Claude API token 消耗统计 |
| `!tokens reset` | 重置 token 计数 |
| `!setproject <路径>` | 设置工作目录（启用 Git 时间机器） |
| `!project` | 查看当前工作目录 |
| `!whoami` | 显示 Discord 用户 ID |
| `!debug` | 查看 debug 模式状态 |
| `!help` | 显示命令手册 |

### Agent 触发信号（消息内容）

| 内容 | 效果 |
|------|------|
| 消息含 `@Gemini` | 强制触发 Gemini 架构咨询 |
| Agent 输出含 `[SEND_FILE: /path]` | Discord Bot 自动发送该文件为附件 |

---

## 添加新 Agent

只需三步：

```bash
# 1. 创建目录
mkdir agents/myagent

# 2. 编写配置
cat > agents/myagent/agent.json << 'EOF'
{
  "name": "myagent",
  "llm": "claude",
  "workspace": "/path/to/workspace",
  "tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
  "permission_mode": "bypassPermissions",
  "persona_files": ["SOUL.md"],
  "setting_sources": ["user", "project"]
}
EOF

# 3. 编写人设
echo "你是一个专注于..." > agents/myagent/SOUL.md

# 启动
python main.py --agent myagent cli
```

---

## Skill 继承机制

`ClaudeNode` 支持继承 Claude Code CLI 已安装的 Skill（`setting_sources` 字段）：

- **Skill 存储位置**：`~/.claude/plugins/cache/`（本地 cache，无网络依赖）
- **首次调用**：Skill 文本注入 system prompt → Claude API 建立 Prompt Cache
- **后续调用**：命中 cache，费用约为原来的 10%
- **项目级 Skill**：在工作目录下创建 `.claude/` 目录，`"project"` 来源会自动加载

---

## 环境变量

所有字段均可通过环境变量覆盖 `agent.json`，前缀为 Agent 名称大写：

```bash
HANI_WORKSPACE=/path/to/project
HANI_CLAUDE_MODEL=claude-sonnet-4-6
HANI_SETTING_SOURCES=user,project   # 逗号分隔
HANI_MAX_RETRIES=3
```
