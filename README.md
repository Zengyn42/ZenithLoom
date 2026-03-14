# BootstrapBuilder

多 LLM Agent 编排框架，基于 LangGraph 构建。支持声明式图定义、多 Session 管理、子图嵌套、以及 Git 原子回滚。

---

## 目录结构

```
BootstrapBuilder/
├── main.py                        # 入口（CLI / Discord / Tmux 模式）
├── framework/                     # 核心框架层
│   ├── state.py                   # BaseAgentState / DebateState (TypedDict)
│   ├── config.py                  # AgentConfig dataclass（from agent.json）
│   ├── registry.py                # 节点/条件注册（装饰器驱动）
│   ├── agent_loader.py            # AgentLoader：加载 agent.json、编译图
│   ├── graph.py                   # build_agent_graph()（Priority 3 默认图）
│   ├── graph_controller.py        # GraphController：图执行 + Session 管理
│   ├── builtins.py                # 注册所有内置节点类型和条件谓词
│   ├── debug.py                   # is_debug() / set_debug()
│   ├── session_mgr.py             # SessionManager（sessions.json I/O）
│   ├── signal_parser.py           # 路由信号提取（JSON 解析）
│   ├── token_tracker.py           # Token 用量统计
│   ├── nodes/                     # 节点实现
│   │   ├── agent_node.py          # AgentNode 基类（抽象）
│   │   ├── agent_ref_node.py      # AgentRefNode：嵌入外部 Agent 子图
│   │   ├── git_nodes.py           # GitSnapshotNode / GitRollbackNode
│   │   ├── validate_node.py       # ValidateNode：输出质量检查
│   │   ├── vram_flush_node.py     # VramFlushNode：GPU 显存清理
│   │   └── subgraph_mapper.py     # SubgraphMapperNode：字段映射
│   ├── claude/
│   │   └── node.py                # ClaudeNode（Claude SDK，可 resume）
│   ├── gemini/
│   │   ├── node.py                # GeminiNode（Gemini API，独立 Session）
│   │   └── gemini_session.py      # Session 存储 & 刷新
│   └── llama/
│       └── node.py                # LlamaNode（Ollama/vLLM，stub）
├── agents/                        # 每个 Agent 一个目录
│   ├── hani/                      # 主 Agent（Claude 驱动）
│   │   ├── agent.json             # 图配置 + 工具 + 节点
│   │   ├── sessions.json          # 活跃 Session & node_sessions
│   │   ├── hani.db                # LangGraph checkpoint（SQLite）
│   │   └── *.md                   # Persona 文件（SOUL / IDENTITY / ...）
│   ├── debate_gemini_first/       # 辩论子图（Gemini 先手）
│   │   └── agent.json
│   └── debate_claude_first/       # 辩论子图（Claude 先手）
│       └── agent.json
└── interfaces/
    ├── cli.py                     # run_cli() / run_tmux()
    └── discord_bot.py             # run_discord()
```

---

## 状态 Schema

### BaseAgentState（主图）

```python
class BaseAgentState(TypedDict):
    messages:           list[BaseMessage]   # 最近 2 条（reducer: _keep_last_2）
    routing_target:     str                 # 路由目标节点 ID（空 = 无路由）
    routing_context:    str                 # 传给路由目标节点的问题/背景
    workspace:          str                 # 当前工作目录
    project_root:       str                 # !setproject 指定的项目根
    project_meta:       dict                # {"plan": "path", "tasks": "path"}
    consult_count:      int                 # 当前轮咨询次数
    last_stable_commit: str                 # git snapshot hash
    retry_count:        int                 # rollback 重试计数
    rollback_reason:    str                 # 非空 = 触发 rollback
    claude_session_id:  str                 # Claude SDK Session UUID（向后兼容）
    node_sessions:      dict                # {"claude_main": uuid, ...}
    knowledge_vault:    str                 # Obsidian vault 根路径
    project_docs:       str                 # 子项目 /docs/ 路径
    debate_conclusion:  str                 # 辩论子图最终结论
```

### DebateState（辩论子图）

与 `BaseAgentState` 字段一致，但 `messages` 使用 `add_messages` reducer（累积所有轮次消息，不截断）。

---

## Agent 配置（agent.json）

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | Agent 名称 |
| `tools` | list[str] | 允许的工具列表 |
| `permission_mode` | str | Claude SDK 权限模式（`bypassPermissions` 等） |
| `max_retries` | int | git rollback 最大重试次数（默认 2） |
| `db_path` | str | LangGraph checkpoint SQLite 路径 |
| `sessions_file` | str | sessions.json 路径 |
| `setting_sources` | list \| null | SDK 技能注入来源（注：`["user","project"]` 增加 ~14k token 系统提示） |
| `persona_files` | list[str] | Persona 文件列表，拼接为 system prompt |
| `discord_token` | str | Discord Bot Token（env `DISCORD_BOT_TOKEN` 优先） |
| `discord_allowed_users` | list[str] | 白名单用户（env `DISCORD_ALLOWED_USERS` 逗号分隔覆盖） |
| `graph` | dict | 图定义（nodes + edges） |

---

## 节点类型

| 类型 | 实现类 | 用途 | Session 存储 |
|------|--------|------|-------------|
| `CLAUDE_CLI` | ClaudeNode | Claude SDK LLM 调用（可 resume） | `~/.claude/` |
| `GEMINI_CLI` | GeminiNode | Gemini API 对话 | `~/.gemini/tmp/` |
| `LOCAL_VLLM` | LlamaNode | 本地 Ollama/vLLM（stub） | 无 |
| `GIT_SNAPSHOT` | GitSnapshotNode | 任务前自动 git commit | 无 |
| `GIT_ROLLBACK` | GitRollbackNode | 验证失败时回退到快照 | 无 |
| `VALIDATE` | ValidateNode | 输出质量检查（Python 语法、超时检测等） | 无 |
| `VRAM_FLUSH` | VramFlushNode | 杀死残留 GPU 进程 | 无 |
| `SUBGRAPH_MAPPER` | SubgraphMapperNode | 父图↔子图字段重映射 | 无 |
| `AGENT_REF` | AgentRefNode | 将外部 Agent 目录编译为子图并嵌入 | 继承父图 |

### 自定义节点注册

```python
# framework/builtins.py（或任意被 import 的模块）
from framework.registry import register_node, register_condition

@register_node("MY_NODE")
def _(config: AgentConfig, node_config: dict):
    return MyNode(config, node_config)

@register_condition("my_condition")
def _(state: dict) -> bool:
    return bool(state.get("some_field"))
```

---

## 声明式图定义

### 节点定义（agent.json → graph.nodes）

```json
{
  "id": "claude_main",
  "type": "CLAUDE_CLI",
  "model": null,
  "system_prompt": "（可选，优先级低于 persona_files）",
  "first_turn_suffix": "Hani:",
  "user_msg_prefix": "老板: ",
  "tombstone_enabled": true,
  "tool_rules": [
    {"pattern": "implement", "flags": [], "tools": ["Write", "Edit", "Bash"]}
  ]
}
```

### 边类型（agent.json → graph.edges）

| type 值 | 触发条件 | 示例 |
|---------|----------|------|
| （无） | 直接连接 | `{"from": "a", "to": "b"}` |
| `routing_to` | `state["routing_target"] == to` | `{"type": "routing_to", "from": "validate", "to": "debate_brainstorm", "max_retry": 3}` |
| `on_error` | `rollback_reason != ""` | `{"type": "on_error", "from": "validate", "to": "git_rollback"}` |
| `no_routing` | `routing_target == ""` | `{"type": "no_routing", "from": "validate", "to": "__end__"}` |
| 自定义名 | registry 中注册的条件函数 | `{"type": "my_condition", "from": "a", "to": "b"}` |

`max_retry` 字段：触发超过 N 次后条件强制返回 False（防止路由死循环）。

### AGENT_REF 节点（嵌入外部 Agent 子图）

```json
{
  "id": "debate_brainstorm",
  "type": "AGENT_REF",
  "agent_dir": "agents/debate_gemini_first",
  "state_in":  {"task": "routing_context", "knowledge_vault": "knowledge_vault"},
  "state_out": {"debate_conclusion": "last_message"}
}
```

- `state_in`: `{子图字段: 父图字段}` — 调用前注入
- `state_out`: `{父图字段: 子图字段 | "last_message"}` — 调用后写回
- `"last_message"` 特殊值：取子图最后一条消息的 `.content`
- 辩论结论自动以 `AIMessage(content="[辩论结论]\n\n...")` 注入父图 messages

---

## 图编译流程（build_graph）

三优先级系统：

```
AgentLoader.build_graph(checkpointer=_DEFAULT)
│
├─ Priority 1: agents/{name}/graph.py 存在？
│   └─ mod.build_graph(loader, checkpointer)       # 完全自定义图
│
├─ Priority 2: agent.json["graph"]["nodes"] 存在？
│   └─ _build_declarative(graph_spec)
│       ├─ 验证：节点 ID 唯一、边引用有效、BFS 可达性
│       ├─ 选择 state_schema（"base" → BaseAgentState / "debate" → DebateState）
│       ├─ StateGraph(schema).add_node() for each node
│       │   ├─ ID 含 "main" 的节点注入 system_prompt
│       │   └─ AGENT_REF 节点递归编译子图（checkpointer=None）
│       ├─ add_edge / add_conditional_edges
│       │   └─ routing_to 自动生成 target 匹配条件
│       └─ .compile(checkpointer=AsyncSqliteSaver(db_path))
│
└─ Priority 3: GraphSpec 默认图
    └─ build_agent_graph(config, agent_node, checkpointer, spec)
        # 固定拓扑：git_snapshot → claude_agent → validate
        #            validate →[on_error]→ git_rollback → claude_agent
        #            validate →[no_routing]→ __end__
```

### 图编译验证规则

1. 所有节点 ID 全局唯一（含子图内部节点）
2. 所有边引用的节点 ID 存在
3. 所有节点从 `__start__` BFS 可达
4. 若提供 system_prompt，图中恰好有 1 个 ID 含 `"main"` 的节点

---

## 主图拓扑（Hani）

```
__start__
    │
    ▼
claude_main ◄──────────────────────────────────────────────┐
    │                                                       │
    ▼                                                       │
git_snapshot                                                │
    │                                                       │
    ▼                                                       │
validate ──[routing_to:debate_brainstorm]──▶ debate_brainstorm ┘
         │                                                  │
         ├──[routing_to:debate_design]──────▶ debate_design ─┘
         │
         ├──[on_error]───────────────────────▶ git_rollback ─┘
         │
         └──[no_routing]─────────────────────▶ __end__
```

Agent 通过输出路由信号触发跳转：

```json
{"route": "debate_brainstorm", "context": "微服务 vs 单体架构选型"}
```

---

## 辩论子图拓扑

### debate_gemini_first（Gemini 先手，适合头脑风暴）

```
__start__ → gemini_propose → claude_critique_1 → gemini_revise → claude_critique_2 → gemini_conclusion → __end__
```

### debate_claude_first（Claude 先手，适合工程/设计决策）

```
__start__ → claude_propose → gemini_critique_1 → claude_revise → gemini_critique_2 → claude_conclusion → __end__
```

- 线性图，无条件边，5 轮辩论
- 每轮节点通过消息历史（`add_messages` 累积）读取所有前轮内容
- 实时流式输出，每轮标注说话节点身份
- 结论自动映射回父图 `debate_conclusion` 字段

---

## Session 架构

### sessions.json 结构

```json
{
  "default": {
    "thread_id": "hani_session_abc123",
    "node_sessions": {
      "claude_main": "claude-uuid-..."
    },
    "workspace": "/home/kingy/Projects/MyProject"
  }
}
```

### 生命周期

1. `GraphController._init_session()` — 加载 sessions.json 或创建 "default"
2. `graph.ainvoke(config={"configurable": {"thread_id": ...}})` — LangGraph 从 SQLite checkpoint 恢复 BaseAgentState
3. 每个 `AgentNode` 从 `state["node_sessions"][node_id]` 读取上次 Session UUID → resume
4. 图完成后 — GraphController 将新 `node_sessions` 写回 sessions.json

### Claude 新 Session 两阶段初始化（WSL2 Unicode 修复）

WSL2 上 Claude CLI 在创建新 Session 时，中文在 Anthropic API 请求 JSON 第 714 字节处被截断，导致 `400 invalid high surrogate` 错误。Resume 已有 Session 时不触发（CLI 跳过 system prompt 处理）。

**修复方案（ClaudeNode 内部自动执行）**：

1. Phase 1：发送 ASCII-only 消息 `"hi"` 创建 Session → 获得 `new_session_id`
2. Phase 2：立即 `resume` 该 Session，注入完整 persona + 实际 prompt

---

## AgentNode 基类

子类只需实现 `call_llm()`，框架自动处理：

- node_sessions UUID 路由（读取/写入 per-node session）
- Rollback warning 注入（失败历史）
- Gemini section 注入（跨 LLM 上下文传递）
- project_meta 注入（plan/tasks 文件内容）
- tool_rules 关键词驱动的工具选择
- 路由信号解析（`{"route": "..."}` → routing_target / routing_context）
- 资源锁（GPU 互斥，可选）

```python
class MyNode(AgentNode):
    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> tuple[str, str]:   # (response_text, new_session_id)
        ...
```

---

## 运行方式

```bash
# CLI 交互模式
python main.py cli

# CLI debug 模式（详细日志）
python main.py --debug cli

# Discord Bot
python main.py discord

# tmux 分屏模式
python main.py tmux

# E2E 测试（11 个测试）
python3 test_e2e_debate.py
```
