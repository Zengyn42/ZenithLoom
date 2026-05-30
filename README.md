# ZenithLoom

**Note:** The core engine of the project resides in this `ZenithLoom` repository. However, all the `blueprints` have been moved to a separate repository called `VoidDraft`.

A multi-LLM Agent orchestration framework built on LangGraph. It supports declarative graph definition, multi-session management, subgraph nesting, and atomic Git rollbacks.

---

## Directory Structure

```
ZenithLoom/
├── main.py                        # Entry point (CLI / Discord / Tmux modes)
├── framework/                     # Core framework layer
│   ├── state.py                   # BaseAgentState / DebateState (TypedDict)
│   ├── config.py                  # AgentConfig dataclass (from entity.json)
│   ├── registry.py                # Node/condition registration (decorator-driven)
│   ├── agent_loader.py            # AgentLoader: loads entity.json, compiles graph
│   ├── graph.py                   # build_agent_graph() (Priority 3 default graph)
│   ├── graph_controller.py        # GraphController: graph execution + session management
│   ├── builtins.py                # Registers all built-in node types and condition predicates
│   ├── debug.py                   # is_debug() / set_debug()
│   ├── session_mgr.py             # SessionManager (sessions.json I/O)
│   ├── signal_parser.py           # Routing signal extraction (JSON parsing)
│   ├── token_tracker.py           # Token usage tracking
│   ├── nodes/                     # Node implementations
│   │   ├── agent_node.py          # AgentNode base class (abstract)
│   │   ├── agent_ref_node.py      # AgentRefNode: embeds external Agent subgraphs
│   │   ├── git_nodes.py           # GitSnapshotNode / GitRollbackNode
│   │   ├── validate_node.py       # ValidateNode: output quality checks
│   │   ├── vram_flush_node.py     # VramFlushNode: GPU VRAM cleanup
│   │   └── subgraph_mapper.py     # SubgraphMapperNode: field mapping
│   ├── claude/
│   │   └── node.py                # ClaudeNode (Claude SDK, resumable)
│   ├── gemini/
│   │   ├── node.py                # GeminiNode (Gemini API, separate Session)
│   │   └── gemini_session.py      # Session storage & refresh
│   └── llama/
│       └── node.py                # LlamaNode (Ollama/vLLM, stub)
├── agents/                        # One directory per Agent
│   ├── technical_architect/       # Main Agent (Claude-driven)
│   │   ├── entity.json             # Graph config + tools + nodes
│   │   ├── sessions.json          # Active sessions & node_sessions
│   │   ├── technical_architect.db # LangGraph checkpoint (SQLite)
│   │   └── *.md                   # Persona files (SOUL / IDENTITY / ...)
│   ├── debate_gemini_first/       # Debate subgraph (Gemini starts)
│   │   └── entity.json
│   └── debate_claude_first/       # Debate subgraph (Claude starts)
│       └── entity.json
└── interfaces/
    ├── cli.py                     # run_cli() / run_tmux()
    └── discord_bot.py             # run_discord()
```

---

## State Schema

### BaseAgentState (Main Graph)

```python
class BaseAgentState(TypedDict):
    messages:           list[BaseMessage]   # Last 2 messages (reducer: _keep_last_2)
    routing_target:     str                 # Target node ID for routing (empty = no route)
    routing_context:    str                 # Question/context for the routed node
    workspace:          str                 # Current working directory
    project_root:       str                 # Project root set by !setproject
    project_meta:       dict                # {"plan": "path", "tasks": "path"}
    consult_count:      int                 # Number of consultations in the current turn
    last_stable_commit: str                 # Git snapshot hash
    retry_count:        int                 # Rollback retry counter
    rollback_reason:    str                 # Not empty = trigger rollback
    node_sessions:      dict                # {"claude_main": uuid, ...}
    knowledge_vault:    str                 # Obsidian vault root path
    project_docs:       str                 # Subproject /docs/ path
    debate_conclusion:  str                 # Final conclusion from the debate subgraph
```

### DebateState (Debate Subgraph)

Same fields as `BaseAgentState`, but `messages` uses the `add_messages` reducer (accumulates all messages, no truncation).

---

## Agent Configuration (entity.json)

### Top-level Fields

| Field                   | Type         | Description                                                              |
|-------------------------|--------------|--------------------------------------------------------------------------|
| `name`                  | str          | Agent name                                                               |
| `tools`                 | list[str]    | List of allowed tools                                                    |
| `permission_mode`       | str          | Claude SDK permission mode (`bypassPermissions`, etc.)                   |
| `max_retries`           | int          | Max git rollback retries (default 2)                                     |
| `db_path`               | str          | LangGraph checkpoint SQLite path                                         |
| `sessions_file`         | str          | sessions.json path                                                       |
| `setting_sources`       | list \| null | SDK skill injection sources (Note: `["user","project"]` adds ~14k tokens to system prompt) |
| `persona_files`         | list[str]    | List of persona files, concatenated into the system prompt               |
| `discord_token`         | str          | Discord Bot Token (env `DISCORD_BOT_TOKEN` takes precedence)             |
| `discord_allowed_users` | list[str]    | Whitelisted users (comma-separated `DISCORD_ALLOWED_USERS` env var overrides) |
| `graph`                 | dict         | Graph definition (nodes + edges)                                         |

---

## Node Types

| Type                | Implementation Class    | Purpose                                               | Session Storage     |
|---------------------|-------------------------|-------------------------------------------------------|---------------------|
| `CLAUDE_CLI`        | ClaudeCLINode           | Claude CLI subprocess (calls `claude` command directly) | `~/.claude/`        |
| `CLAUDE_SDK`        | ClaudeSDKNode           | Claude Agent SDK (via `claude_agent_sdk`)             | `~/.claude/`        |
| `GEMINI_CLI`        | GeminiNode              | Gemini API conversation                               | `~/.gemini/tmp/`    |
| `LOCAL_VLLM`        | LlamaNode               | Local Ollama/vLLM (stub)                              | None                |
| `GIT_SNAPSHOT`      | GitSnapshotNode         | Automatic git commit before a task                    | None                |
| `GIT_ROLLBACK`      | GitRollbackNode         | Reverts to snapshot on validation failure             | None                |
| `VALIDATE`          | ValidateNode            | Output quality checks (Python syntax, timeout, etc.)  | None                |
| `VRAM_FLUSH`        | VramFlushNode           | Kills residual GPU processes                          | None                |
| `SUBGRAPH_MAPPER`   | SubgraphMapperNode      | Remaps fields between parent and subgraphs            | None                |
| `AGENT_REF`         | AgentRefNode            | Compiles an external Agent directory into a subgraph and embeds it | Inherits from parent |

### Custom Node Registration

```python
# framework/builtins.py (or any imported module)
from framework.registry import register_node, register_condition

@register_node("MY_NODE")
def _(config: AgentConfig, node_config: dict):
    return MyNode(config, node_config)

@register_condition("my_condition")
def _(state: dict) -> bool:
    return bool(state.get("some_field"))
```

---

## Declarative Graph Definition

### Node Definition (entity.json → graph.nodes)

```json
{
  "id": "claude_main",
  "type": "CLAUDE_SDK",
  "model": null,
  "system_prompt": "(Optional, lower priority than persona_files)",
  "first_turn_suffix": "technical_architect:",
  "user_msg_prefix": "Boss: ",
  "tombstone_enabled": true,
  "tool_rules": [
    {"pattern": "implement", "flags": [], "tools": ["Write", "Edit", "Bash"]}
  ]
}
```

### Edge Types (entity.json → graph.edges)

| `type` value      | Trigger Condition                    | Example                                                              |
|-------------------|--------------------------------------|----------------------------------------------------------------------|
| (none)            | Direct connection                    | `{"from": "a", "to": "b"}`                                           |
| `routing_to`      | `state["routing_target"] == to`      | `{"type": "routing_to", "from": "validate", "to": "debate_brainstorm", "max_retry": 3}` |
| `on_error`        | `rollback_reason != ""`              | `{"type": "on_error", "from": "validate", "to": "git_rollback"}`      |
| `no_routing`      | `routing_target == ""`               | `{"type": "no_routing", "from": "validate", "to": "__end__"}`         |
| Custom name       | Registered condition function        | `{"type": "my_condition", "from": "a", "to": "b"}`                   |

`max_retry` field: After N triggers, the condition is forced to return `False` (prevents routing loops).

### AGENT_REF Node (Embed External Agent Subgraph)

```json
{
  "id": "debate_brainstorm",
  "type": "AGENT_REF",
  "agent_dir": "agents/debate_gemini_first",
  "state_in":  {"task": "routing_context", "knowledge_vault": "knowledge_vault"},
  "state_out": {"debate_conclusion": "last_message"}
}
```

- `state_in`: `{subgraph_field: parent_graph_field}` — Injected before call.
- `state_out`: `{parent_graph_field: subgraph_field | "last_message"}` — Written back after call.
- `"last_message"` special value: Takes the `.content` of the subgraph's last message.
- The debate conclusion is automatically injected into the parent graph's messages as `AIMessage(content="[Debate Conclusion]

...")`.

---

## Graph Compilation Flow (build_graph)

A three-priority system:

```
AgentLoader.build_graph(checkpointer=_DEFAULT)
│
├─ Priority 1: Does agents/{name}/graph.py exist?
│   └─ mod.build_graph(loader, checkpointer)       # Fully custom graph
│
├─ Priority 2: Does entity.json["graph"]["nodes"] exist?
│   └─ _build_declarative(graph_spec)
│       ├─ Validation: unique node IDs, valid edge refs, BFS reachability
│       ├─ Select state_schema ("base" → BaseAgentState / "debate" → DebateState)
│       ├─ StateGraph(schema).add_node() for each node
│       │   ├─ Node with "main" in ID gets system_prompt injected
│       │   └─ AGENT_REF nodes recursively compile subgraphs (checkpointer=None)
│       ├─ add_edge / add_conditional_edges
│       │   └─ `routing_to` automatically generates target matching condition
│       └─ .compile(checkpointer=AsyncSqliteSaver(db_path))
│
└─ Priority 3: Default GraphSpec
    └─ build_agent_graph(config, agent_node, checkpointer, spec)
        # Fixed topology: git_snapshot → claude_agent → validate
        #            validate →[on_error]→ git_rollback → claude_agent
        #            validate →[no_routing]→ __end__
```

### Graph Compilation Validation Rules

1. All node IDs are globally unique (including within subgraphs).
2. All edge-referenced node IDs exist.
3. All nodes are reachable from `__start__` via BFS.
4. If a `system_prompt` is provided, there is exactly one node with `"main"` in its ID.

---

## Main Graph Topology (technical_architect)

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

The Agent triggers transitions by outputting a routing signal:

```json
{"route": "debate_brainstorm", "context": "Microservices vs Monolith architecture choice"}
```

---

## Debate Subgraph Topology

### debate_gemini_first (Gemini starts, good for brainstorming)

```
__start__ → gemini_propose → claude_critique_1 → gemini_revise → claude_critique_2 → gemini_conclusion → __end__
```

### debate_claude_first (Claude starts, good for engineering/design decisions)

```
__start__ → claude_propose → gemini_critique_1 → claude_revise → gemini_critique_2 → claude_conclusion → __end__
```

- Linear graph, no conditional edges, 5 rounds of debate.
- Each node reads all previous rounds' content from the message history (`add_messages` accumulation).
- Real-time streaming output, with the speaker's identity marked in each round.
- The conclusion is automatically mapped back to the parent graph's `debate_conclusion` field.

---

## Session Architecture

### sessions.json Structure

```json
{
  "default": {
    "thread_id": "technical_architect_session_abc123",
    "node_sessions": {
      "claude_main": "claude-uuid-..."
    },
    "workspace": "/home/kingy/Projects/MyProject"
  }
}
```

### Lifecycle

1. `GraphController._init_session()` — Loads `sessions.json` or creates a "default" session.
2. `graph.ainvoke(config={"configurable": {"thread_id": ...}})` — LangGraph restores `BaseAgentState` from the SQLite checkpoint.
3. Each `AgentNode` resumes its session using the UUID from `state["node_sessions"][node_id]`.
4. After the graph completes, `GraphController` writes the new `node_sessions` back to `sessions.json`.

### Claude New Session Two-Phase Initialization (WSL2 Unicode Fix)

On WSL2, when creating a new session with the Claude CLI, Chinese characters can be truncated at the 714th byte of the Anthropic API request JSON, causing a `400 invalid high surrogate` error. This is not triggered when resuming an existing session because the CLI skips system prompt processing.

**Fix (handled automatically inside ClaudeNode)**:

1. Phase 1: Send an ASCII-only message `"hi"` to create the session → get `new_session_id`.
2. Phase 2: Immediately `resume` that session and inject the full persona + actual prompt.

---

## AgentNode Base Class

Subclasses only need to implement `call_llm()`, and the framework handles the rest:

- `node_sessions` UUID routing (reading/writing per-node sessions).
- Rollback warning injection (failure history).
- Gemini section injection (passing context across LLMs).
- `project_meta` injection (content of plan/tasks files).
- Keyword-driven tool selection via `tool_rules`.
- Routing signal parsing (`{"route": "..."}` → `routing_target` / `routing_context`).
- Resource locking (mutual exclusion for GPUs, optional).

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

## How to Run

```bash
# Interactive CLI mode
python main.py cli

# CLI debug mode (detailed logs)
python main.py --debug cli

# Discord Bot
python main.py discord

# tmux split-screen mode
python main.py tmux

# E2E tests (11 tests)
python3 test_e2e_debate.py
```
