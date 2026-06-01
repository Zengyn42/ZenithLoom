# ZenithLoom — Pipeline Builder Guide

> **Repo split**: The engine (this repo) handles runtime, state, and node execution. All agent blueprints (role definitions, graph configs, persona files) live in the separate **VoidDraft** repository.

---

## Overview

ZenithLoom is a declarative LangGraph orchestration engine. You define a pipeline in `entity.json`, point `awaken.py` at an `identity.json`, and the engine builds and runs the graph. No Python required for most pipelines.

```
VoidDraft/blueprints/         ← where you design pipelines
ZenithLoom/                   ← runtime engine (this repo)
EdenGateway/agents/<name>/    ← per-instance runtime data (db, sessions, tokens)
```

---

## 1. Awakening an Agent

### `identity.json` — the instance config

Every running agent instance needs an `identity.json` in its data directory:

```jsonc
// EdenGateway/agents/my-agent/identity.json
{
  "name": "my-agent",
  "blueprint": "/path/to/VoidDraft/blueprints/role_agents/my_role",
  "framework": "/path/to/ZenithLoom",
  "connector": "discord",          // "cli" | "discord" | "tmux" | "gchat"
  "discord": {
    "token": "Bot TOKEN...",
    "allowed_users": ["discord-user-id"]
  }
}
```

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Instance name — also used as the SQLite db filename |
| `blueprint` | ✅ | Path to the blueprint directory in VoidDraft |
| `framework` | optional | Path to ZenithLoom root (defaults to awaken.py's directory) |
| `connector` | optional | Interface type; defaults to `"cli"` |

### Starting the agent

```bash
# Run with a specific connector
python awaken.py --entity ~/Foundation/EdenGateway/agents/my-agent

# Override connector at runtime
python awaken.py --entity ~/Foundation/EdenGateway/agents/my-agent --connector cli

# Enable debug logging
python awaken.py --entity ~/Foundation/EdenGateway/agents/my-agent --debug

# Capture all LLM output to file
python awaken.py --entity ~/Foundation/EdenGateway/agents/my-agent --debug-output /tmp/debug.txt

# Environment variable form
ENTITY=~/Foundation/EdenGateway/agents/my-agent python awaken.py
```

### What happens at startup

```
awaken.py
  ↓ reads identity.json
  ↓ resolves blueprint_dir (VoidDraft) + data_dir (EdenGateway)
  ↓ EntityLoader(blueprint_dir, data_dir)
      ↓ reads entity.json (graph definition + persona_files)
      ↓ reads identity.json (name, tokens, allowed_users)
      ↓ starts MCP servers declared in entity.json["mcp"]
  ↓ runs connector (discord / cli / tmux / gchat)
      ↓ on each message → GraphController.run(input)
          ↓ LangGraph ainvoke() → nodes execute → state persisted to SQLite
```

---

## 2. Defining a Pipeline — `entity.json`

The blueprint directory contains `entity.json` with the graph definition:

```jsonc
// VoidDraft/blueprints/role_agents/my_role/entity.json
{
  "persona_files": ["PERSONA.md", "PROTOCOL.md"],  // injected as system prompt
  "permission_mode": "bypassPermissions",           // Claude tool permissions
  "max_retries": 2,

  "graph": {
    "entry": "main_llm",           // first node after START
    "exit": "main_llm",            // last node before END (can be same as entry)

    "nodes": [ ... ],
    "edges": [ ... ]
  },

  "mcp": [                         // optional MCP servers to auto-start
    { "name": "heartbeat", "url": "http://127.0.0.1:8100/sse", "proxy": "heartbeat" }
  ]
}
```

### Graph build priority

`EntityLoader` builds the graph in this order:

| Priority | Condition | What runs |
|---|---|---|
| 1 | `blueprint_dir/graph.py` exists with `build_graph(loader, checkpointer)` | Full custom Python graph |
| 2 | `entity.json["graph"]` has `nodes` + `edges` | **Declarative graph** (most common) |
| 3 | `entity.json["graph"]` has `GraphSpec` fields | Legacy single-LLM default graph |

---

## 3. LLM Node Types

Every LLM node in `entity.json` has a `type` field. Here is what each type does:

### `CLAUDE_SDK` — Claude via Agent SDK

```jsonc
{
  "id": "main_llm",
  "type": "CLAUDE_SDK",
  "session_key": "claude_main",     // nodes sharing a key share one LLM session
  "persona_files": ["PERSONA.md"],  // appended to system prompt for this node
  "extra_persona": true,            // also inject identity.json SOUL.md
  "tools": null,                    // null = global tool list; [] = read-only mode
  "token_limit": 100000,            // hard cap (cloud billing guard)
  "tombstone_enabled": true         // inject prior-failure warnings into prompt
}
```

**Behavior:**
- Runs `claude-agent-sdk` as a subprocess; sessions are stored in `~/.claude/`
- Resumable across turns via `session_id` stored in `node_sessions` state
- Supports all `permission_mode` values: `default`, `plan`, `acceptEdits`, `bypassPermissions`
- `tools: []` → read-only mode (blocks Write/Edit/Bash, keeps Read/Grep/Glob/Web)
- Emits routing signals: `{"route": "subgraph_id", "context": "..."}` as first line → framework auto-routes

### `GEMINI_API` — Gemini via HTTP API

```jsonc
{
  "id": "gemini_critic",
  "type": "GEMINI_API",
  "model": "gemini-2.5-pro",   // or "gemini-2.5-flash"
  "tools": []                  // Gemini API nodes are typically read-only
}
```

**Behavior:**
- Calls Google Code Assist HTTP API
- Forces a **jitter sleep** before each call (1s–20s scaled to prompt length) to avoid rate limits
- 403/429 → raises `GeminiQuotaError` immediately, no retry
- No local tool execution; `permission_mode` has no effect
- Good for: debate critiques, analysis, document generation

### `GEMINI_CLI` — Gemini via CLI subprocess

```jsonc
{
  "id": "gemini_lead",
  "type": "GEMINI_CLI",
  "model": "gemini-2.5-pro",   // or gemini-3-*-preview (CLI only)
  "session_key": "gemini_session"
}
```

**Behavior:**
- Runs `gemini` CLI as a subprocess; sessions stored in `~/.gemini/tmp/`
- Supports session resume across turns
- `permission_mode: "plan"` → passes `--no-yolo`; all other modes → `--yolo`
- Supports newer preview models not available via API (e.g. `gemini-3-flash-preview`)

### `OLLAMA` / `LOCAL_VLLM` — Local inference

```jsonc
{
  "id": "local_worker",
  "type": "OLLAMA",
  "model": "qwen3.5:27b",
  "endpoint": "http://localhost:11434",  // default Ollama endpoint
  "timeout": 120,
  "max_iterations": 10,                  // tool-call loop cap
  "options": { "temperature": 0.7 }
}
```

**Behavior:**
- Calls Ollama's OpenAI-compatible `/v1/chat/completions` endpoint
- `keep_alive=-1` keeps model resident in RAM
- Supports thinking + tool_calls + streaming simultaneously
- No billing; token limits can be set much higher

### `GROK` — Grok via browser automation

```jsonc
{
  "id": "grok_node",
  "type": "GROK"
}
```

**Behavior:**
- Uses Playwright browser automation to interact with Grok
- Requires Chrome/Chromium running with remote debugging enabled
- Good for: tasks requiring Grok-specific reasoning or web access

### `DETERMINISTIC` — Pure Python function node

```jsonc
{
  "id": "validate_output",
  "type": "DETERMINISTIC",
  "agent_dir": "blueprints/functional_graphs/my_graph"
}
```

**Behavior:**
- Loads `validators.py` from the blueprint directory
- Looks up a function matching the node `id`
- The function receives and returns `dict` state — deterministic, no LLM call
- Used for: parsing LLM output, setting routing targets, field validation, task splitting

```python
# validators.py
def validate_output(state: dict) -> dict:
    last_msg = state["messages"][-1].content
    if "DONE" in last_msg:
        return {"routing_target": ""}  # end
    return {"routing_target": "retry_node"}
```

### `SUBGRAPH_REF` — Embedded subgraph

See Section 5 below.

### `HEARTBEAT` — Background task monitor

```jsonc
{
  "id": "heartbeat_node",
  "type": "HEARTBEAT"
}
```

Connects to the Heartbeat MCP server to monitor long-running background tasks. Used by the administrative_officer role.

### `VRAM_FLUSH` — GPU VRAM cleanup

```jsonc
{ "id": "flush_vram", "type": "VRAM_FLUSH" }
```

Kills any lingering GPU processes before launching a local model node. Use before `OLLAMA` nodes in pipelines that may have residual VRAM usage.

---

## 4. Agent Runtime Conventions

> This section is especially important for agents reading this guide — it explains what you will see at runtime and how to respond correctly.

### What an agent sees in its system prompt

When a blueprint is loaded, the framework automatically appends a routing hint block to the system prompt for any node that has `SUBGRAPH_REF` siblings in the same graph:

```
<!-- [auto-generated section: routing hints collected from subgraph nodes] -->
[Available Subgraphs]
Delegate tasks to the subgraphs below when appropriate.
Routing: emit the following JSON as the FIRST AND ONLY line of your reply (no prefix, no explanation):
{"route": "<node_id>", "context": "<describe the topic and relevant background clearly>"}

Available subgraphs:
  - "debate_brainstorm": Use when the topic needs fast idea exploration and perspective collision...
  - "apex_coder": Use when the coding problem is extremely complex...

Note: routing is a heavy operation. Only use it when truly valuable — handle simple tasks directly.
```

This block is built from the `routing_hint` field in each subgraph's `entity.json`. When writing a new subgraph, always declare:

```jsonc
// VoidDraft/blueprints/functional_graphs/my_subgraph/entity.json
{
  "routing_hint": "Use when the user needs X. Triggered by: 'do X', 'help with X'.",
  ...
}
```

### How subgraph conclusions reach the parent agent

After a subgraph completes, its conclusion is stored in a state field (e.g. `debate_conclusion`). On the **parent agent's next invocation**, the framework prepends the conclusion to the user message:

```
[Debate Conclusion]
<full subgraph output here>

[ApexCoder Conclusion]
<coder output here>
```

As an agent, you should:
1. Read the injected conclusion block before responding
2. Synthesize it with the original user request
3. Deliver your final answer — do **not** route again unless needed

### How to write `output_field` for a subgraph's terminal node

The last LLM node in a subgraph must declare `output_field` to write its response into the parent state:

```jsonc
{
  "id": "conclude_node",
  "type": "CLAUDE_SDK",
  "output_field": "debate_conclusion",   // ← writes LLM response to this state field
  "channel_send_final": true             // ← suppress streaming, send one final message
}
```

Valid `output_field` values match the state fields in Section 6.

---

## 5. LangGraph Subgraphs (`SUBGRAPH_REF`)

A `SUBGRAPH_REF` node embeds another blueprint's graph as a callable sub-pipeline. It is the primary composition mechanism.

```jsonc
{
  "id": "debate_brainstorm",
  "type": "SUBGRAPH_REF",
  "agent_dir": "VoidDraft/blueprints/functional_graphs/debate_gemini_first",
  "session_mode": "fresh_per_call"
}
```

### How it works

```
Parent graph
  │
  ├── main_llm node
  │     emits: {"route": "debate_brainstorm", "context": "..."}
  │
  └── debate_brainstorm (SUBGRAPH_REF)
        ↓ _subgraph_init  — clears messages, resets node sessions
        ↓ [child graph runs: gemini_propose → claude_critique → ... → conclusion]
        ↓ _subgraph_exit   — writes conclusion to parent state field, clears messages
        → parent graph continues with debate_conclusion in state
```

### Session modes

The `session_mode` controls how state flows between parent and subgraph:

| Mode | Init clears | What carries over | Use when |
|---|---|---|---|
| `fresh_per_call` | messages + node_sessions | routing_context, workspace, project fields | **Default.** Subgraph is fully isolated each call |
| `isolated` | node_sessions only | messages from parent | Subgraph sees parent messages but starts fresh LLM sessions |
| `persistent` | nothing | everything | Subgraph resumes exactly where it left off |
| `inherit` | nothing | everything (parent sessions shared) | Subgraph shares LLM sessions with parent |

### Routing to a subgraph

Any LLM node can trigger a subgraph by emitting a routing signal as the **first line** of its response:

```json
{"route": "debate_brainstorm", "context": "Should we use microservices or monolith?"}
```

The framework:
1. Parses the signal → sets `routing_target = "debate_brainstorm"`, `routing_context = "..."`
2. Conditional edge fires → jumps to the `SUBGRAPH_REF` node
3. After subgraph completes → conclusion written to state → control returns to the declaring edge

### Subgraph output fields

Each subgraph writes its conclusion to a dedicated state field:

| Subgraph | Output field |
|---|---|
| `debate_*` | `debate_conclusion` |
| `apex_coder` | `apex_conclusion` |
| `knowledge_shelf` | `knowledge_result` |
| `tool_discovery` | `discovery_report` |

The parent's next LLM node automatically sees these injected into its prompt.

---

## 6. Edge Types

Edges connect nodes. Declare them in `entity.json["graph"]["edges"]`:

```jsonc
"edges": [
  // Direct edge
  { "from": "node_a", "to": "node_b" },

  // Conditional edge
  {
    "from": "node_a",
    "type": "conditional",
    "conditions": [
      { "target": "debate_brainstorm", "when": "routing_target == 'debate_brainstorm'" },
      { "target": "apex_coder",        "when": "routing_target == 'apex_coder'" },
      { "target": "__end__",           "when": "default" }
    ]
  }
]
```

Shorthand edge types (legacy, still supported):

| Type | Fires when |
|---|---|
| *(none)* | Always (direct connection) |
| `routing_to` | `state["routing_target"] == edge["to"]` |
| `no_routing` | `routing_target == ""` |
| `on_error` | `rollback_reason != ""` |

---

## 7. State Schema

All nodes share `BaseAgentState`. Key fields:

| Field | Type | Written by | Purpose |
|---|---|---|---|
| `messages` | `list[BaseMessage]` | LLM nodes | Conversation history (`add_messages` reducer) |
| `routing_target` | `str` | LLM nodes | Which node to route to next |
| `routing_context` | `str` | LLM nodes | Context passed to the routed node |
| `node_sessions` | `dict` | LLM nodes | `{session_key: uuid}` — per-node LLM session IDs |
| `debate_conclusion` | `str` | debate subgraphs | Debate output injected into next LLM prompt |
| `apex_conclusion` | `str` | apex_coder subgraph | Coder output |
| `previous_node_output` | `str` | LLM nodes | Last node's full response (for chaining) |
| `workspace` | `str` | GraphController | Current working directory |
| `rollback_reason` | `str` | DETERMINISTIC validators | Non-empty → triggers git rollback |

Custom state schemas (e.g. for debate subgraphs) can be registered:

```python
from framework.loader import register_state_schema
register_state_schema("debate", DebateState)
```

Then in `entity.json`: `"state_schema": "debate"`.

---

## 8. Complete Role Agent — End-to-End Walkthrough

> Step-by-step guide to build a production-grade conversational role agent with subgraph routing. This is the proven pattern used by all role agents in VoidDraft.

### The Standard Role Agent Graph

Every role agent follows this topology:

```
                    ┌─────────────────────────────────┐
                    │                                 │
__start__ ──► claude_main ──► git_snapshot ──► validate
                    ▲              │
                    │         routing_to ──► debate_brainstorm ──┐
                    │         routing_to ──► debate_design    ──┤
                    │         routing_to ──► apex_coder       ──┤
                    └─────────────────────────────────────────────┘
                               no_routing ──► __end__
                               on_error   ──► git_rollback ──► claude_main
```

**Key nodes explained:**
- `claude_main` — the primary LLM node; handles the user's message and optionally emits a routing signal
- `git_snapshot` — takes a git snapshot of the workspace before each reply (enables `!rollback`)
- `validate` — parses the routing signal from `claude_main`'s output and sets `routing_target` in state
- subgraph nodes — execute and loop back to `claude_main` which synthesizes the result
- `git_rollback` — reverts workspace to last snapshot when `rollback_reason` is set

---

### Step 1 — Create the blueprint directory in VoidDraft

```bash
mkdir -p /home/kingy/Foundation/VoidDraft/blueprints/role_agents/my_role
cd /home/kingy/Foundation/VoidDraft/blueprints/role_agents/my_role
```

Required files to create:
```
my_role/
├── entity.json    ← graph definition
├── ROLE.md        ← who the agent is, what it knows, its limits
└── PROTOCOL.md    ← how it behaves, operating rules, command reference
```

---

### Step 2 — Write the persona files

**`ROLE.md`** — defines identity and capability boundaries:

```markdown
# My Role — Role Definition

## Capabilities

**Good at:**
- [List what this agent specializes in]
- [Specific domains, tools, frameworks]

**Boundaries:**
- [What it should NOT do, or must escalate]

## Behavior Principles

1. **Transparency** — state uncertainty directly, never fabricate answers
2. **Verify before asserting** — use tools (Read/Grep/Glob) before claiming file contents
3. **[Add role-specific principles]**

## Prohibited Actions

- Never execute destructive operations without explicit user confirmation
- Never expose credentials or secrets to external services
```

**`PROTOCOL.md`** — defines runtime rules and available commands:

```markdown
# My Role — Operating Protocol

## Runtime Framework

You run inside ZenithLoom's LangGraph state machine:
- Each reply is processed by middleware before reaching the user
- Routing pipeline is active: emit JSON → system routes to subgraph → result injected into next prompt
- When you see a [Debate Conclusion] or [ApexCoder Conclusion] block, the subgraph has finished — synthesize and reply

## Operating Rules

1. Be concise. Output results or ask for approval — no preamble.
2. [Add role-specific rules]
3. Reply in [language]. Use English for code and commands.

## Available Commands

| Command | Description |
|---------|-------------|
| `!session` | Show current session info |
| `!topology` | Show agent graph |
| `!rollback N` | Roll back to snapshot N |
| `!tokens` | Show token usage |
```

---

### Step 3 — Write `entity.json`

```jsonc
// VoidDraft/blueprints/role_agents/my_role/entity.json
{
  "llm": "claude",
  "channel_history_limit": 0,
  "max_retries": 2,

  "graph": {
    "nodes": [
      {
        "id": "claude_main",
        "type": "CLAUDE_SDK",
        "session_key": "claude_main",
        "persona_files": ["./ROLE.md", "./PROTOCOL.md"],
        "extra_persona": true,
        "tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        "permission_mode": "bypassPermissions",
        "tombstone_enabled": true
      },
      { "id": "git_snapshot", "type": "GIT_SNAPSHOT" },
      { "id": "validate",     "type": "VALIDATE"     },
      { "id": "git_rollback", "type": "GIT_ROLLBACK" },

      // ── Subgraph nodes (add only the ones your role needs) ──────────
      {
        "id": "debate_brainstorm",
        "agent_dir": "/home/kingy/Foundation/VoidDraft/blueprints/functional_graphs/debate_gemini_first",
        "session_mode": "fresh_per_call"
      },
      {
        "id": "debate_design",
        "agent_dir": "/home/kingy/Foundation/VoidDraft/blueprints/functional_graphs/debate_claude_first",
        "session_mode": "fresh_per_call"
      },
      {
        "id": "apex_coder",
        "agent_dir": "/home/kingy/Foundation/VoidDraft/blueprints/functional_graphs/apex_coder",
        "session_mode": "inherit",
        "routing_hint": "Use when the coding problem is extremely complex, a bug has failed multiple attempts, or a full architecture refactor is needed."
      }
    ],

    "edges": [
      // Main loop
      { "from": "__start__",     "to": "claude_main"   },
      { "from": "claude_main",   "to": "git_snapshot"  },
      { "from": "git_snapshot",  "to": "validate"      },

      // Routing fan-out from validate
      { "type": "routing_to", "from": "validate", "to": "debate_brainstorm" },
      { "type": "routing_to", "from": "validate", "to": "debate_design"     },
      { "type": "routing_to", "from": "validate", "to": "apex_coder"        },

      // Terminal edges
      { "type": "no_routing", "from": "validate",     "to": "__end__"     },
      { "type": "on_error",   "from": "validate",     "to": "git_rollback"},

      // All subgraphs loop back to claude_main for synthesis
      { "from": "debate_brainstorm", "to": "claude_main" },
      { "from": "debate_design",     "to": "claude_main" },
      { "from": "apex_coder",        "to": "claude_main" },
      { "from": "git_rollback",      "to": "claude_main" }
    ]
  }
}
```

> **Rule:** Every subgraph must have an edge looping back to `claude_main`. That is how the conclusion gets synthesized into the final user reply.

---

### Step 4 — Create the instance in EdenGateway

```bash
mkdir -p /home/kingy/Foundation/EdenGateway/agents/my-agent
```

```jsonc
// EdenGateway/agents/my-agent/identity.json
{
  "name": "my-agent",
  "blueprint": "/home/kingy/Foundation/VoidDraft/blueprints/role_agents/my_role",
  "framework": "/home/kingy/Foundation/ZenithLoom",
  "persona_files": ["./IDENTITY.md", "./SOUL.md"],   // injected on top of blueprint personas
  "workspace": "/home/kingy/Foundation",
  "connector": "discord",
  "discord": {
    "token": "Bot TOKEN_HERE",
    "allowed_users": ["your-discord-user-id"]
  }
}
```

> `persona_files` in `identity.json` are injected **in addition to** the blueprint's own persona files. Use them for instance-specific identity (name, personality, greeting style).

---

### Step 5 — Validate and launch

```bash
# Validate entity.json syntax and persona references
cd /home/kingy/Foundation/VoidDraft/blueprints/role_agents/my_role
python3 -m json.tool entity.json > /dev/null && echo "JSON OK"
python3 -c "
import json, pathlib
d = json.load(open('entity.json'))
base = pathlib.Path('.')
for n in d['graph']['nodes']:
    for f in (n.get('persona_files') or []):
        p = base / f
        assert p.exists(), f'Missing: {f}'
print('Persona files OK')
"

# Launch via CLI for testing
python3 /home/kingy/Foundation/ZenithLoom/awaken.py \
  --entity /home/kingy/Foundation/EdenGateway/agents/my-agent \
  --connector cli \
  --debug

# Launch via systemd for production
# (create a systemd user service that runs awaken.py with --entity flag)
```

---

### The Routing Loop in Practice

Here is what happens when a user message triggers a subgraph:

```
Turn 1 — User: "Should I use microservices or monolith?"
  → claude_main decides this needs a debate
  → Emits (first line): {"route": "debate_brainstorm", "context": "microservices vs monolith..."}
  → git_snapshot runs
  → validate parses signal → routing_target = "debate_brainstorm"
  → debate_brainstorm subgraph runs (Gemini proposes → Claude critiques → conclusion)
  → debate_conclusion written to state
  → Loop back to claude_main

Turn 2 — [same thread, no user input]
  → claude_main sees [Debate Conclusion] injected in prompt
  → Synthesizes the conclusion into a final answer
  → Does NOT emit a routing signal
  → validate sees no routing → no_routing edge → __end__
  → User receives the final reply
```

---

## 9. Minimal Functional Graph Example

A two-node pipeline: Claude plans, Gemini critiques, Claude concludes.

```jsonc
// VoidDraft/blueprints/functional_graphs/my_pipeline/entity.json
{
  "persona_files": ["PERSONA.md"],
  "graph": {
    "entry": "claude_propose",
    "exit": "claude_conclude",
    "nodes": [
      {
        "id": "claude_propose",
        "type": "CLAUDE_SDK",
        "session_key": "claude_session",
        "tools": [],
        "system_prompt": "You are a solution architect. Propose a design for the given problem."
      },
      {
        "id": "gemini_critique",
        "type": "GEMINI_API",
        "model": "gemini-2.5-pro"
      },
      {
        "id": "claude_conclude",
        "type": "CLAUDE_SDK",
        "session_key": "claude_session",
        "tools": [],
        "system_prompt": "Review the critique and produce a final refined design."
      }
    ],
    "edges": [
      { "from": "claude_propose",  "to": "gemini_critique" },
      { "from": "gemini_critique", "to": "claude_conclude" }
    ]
  }
}
```

Launch it:

```bash
# identity.json points blueprint to this pipeline
python awaken.py --entity ~/Foundation/EdenGateway/agents/my-pipeline --connector cli
```

---

---

## 10. Blueprint Editing Guide for Agents

> This section is for agents that need to create or modify blueprints in VoidDraft. Read it entirely before touching any file.

### 10.1 — Blueprint Directory Layout

A blueprint directory contains:

```
VoidDraft/blueprints/
├── role_agents/
│   └── my_role/
│       ├── entity.json        ← REQUIRED: graph definition
│       ├── ROLE.md            ← persona: role identity (who I am)
│       ├── PROTOCOL.md        ← persona: operating rules (how I behave)
│       └── graph.py           ← OPTIONAL: custom Python graph (overrides entity.json)
│
└── functional_graphs/
    └── my_subgraph/
        ├── entity.json        ← REQUIRED: graph + routing_hint
        └── validators.py      ← OPTIONAL: functions for DETERMINISTIC nodes
```

**Do NOT edit these runtime files** (they appear in blueprint dirs but are instance state):
```
*.db            ← SQLite checkpoint database
sessions.json   ← LLM session ID registry
```

### 10.2 — Full `entity.json` Field Reference

```jsonc
{
  // ── Top-level fields ─────────────────────────────────────────────────
  "llm": "claude",                   // default LLM type hint (optional)
  "max_retries": 2,                  // node-level retry limit on exception
  "channel_history_limit": 20,       // how many Discord messages to inject as history
  "permission_mode": "bypassPermissions",  // default for all nodes unless overridden
  "routing_hint": "Use when ...",    // auto-injected into parent's system prompt

  "mcp": [                           // MCP servers to auto-start with this agent
    { "name": "heartbeat", "url": "http://127.0.0.1:8100/sse" }
  ],

  // ── Graph definition ─────────────────────────────────────────────────
  "graph": {
    "state_schema": "debate_schema", // OPTIONAL: custom state (default: BaseAgentState)
    "entry": "main_node",            // first node — used when no explicit __start__ edge
    "exit": "main_node",             // last node  — used when no explicit __end__ edge

    "nodes": [ ... ],
    "edges": [ ... ]
  }
}
```

### 10.3 — Node Definition Patterns

**LLM node (standard):**
```jsonc
{
  "id": "claude_main",
  "type": "CLAUDE_SDK",
  "session_key": "claude_main",
  "model": "claude-opus-4-5",         // optional; uses default if omitted
  "permission_mode": "bypassPermissions",
  "tools": null,                       // null = all tools; [] = read-only
  "channel_send_final": false,
  "output_field": "debate_conclusion", // write LLM output to this state field
  "system_prompt": "...",              // inline system prompt (or use persona_files)
  "persona_files": ["ROLE.md"],        // paths relative to blueprint_dir
  "inject_topic": true,                // inject routing_context as topic header
  "tombstone_enabled": true,           // inject prior-failure warnings
  "token_limit": 100000,
  "max_iterations": 10
}
```

**SUBGRAPH_REF node** — note: NO `type` field, just `agent_dir`:
```jsonc
{
  "id": "debate_brainstorm",
  "agent_dir": "/absolute/path/to/VoidDraft/blueprints/functional_graphs/debate_gemini_first",
  "session_mode": "fresh_per_call",
  "fresh_keep_fields": ["discovery_report"],  // preserve these fields through the reset
  "routing_hint": "Use when..."               // override subgraph's own routing_hint
}
```

**Special framework nodes** (no config needed):
```jsonc
{ "id": "git_snapshot", "type": "GIT_SNAPSHOT" }  // snapshot git state before LLM runs
{ "id": "validate",     "type": "VALIDATE"      }  // parse routing signal from prior node
{ "id": "flush_vram",   "type": "VRAM_FLUSH"    }  // clear GPU VRAM before local model
```

**DETERMINISTIC node:**
```jsonc
{
  "id": "my_validator",
  "type": "DETERMINISTIC",
  "agent_dir": "/path/to/blueprint"   // looks for validators.py here
}
// validators.py must define a function with the same name as the node id:
// def my_validator(state: dict) -> dict: ...
```

### 10.4 — Edge Definition Patterns

```jsonc
"edges": [
  // Direct unconditional edge
  { "id": "e1", "from": "__start__", "to": "claude_main" },

  // Unconditional — no id required
  { "from": "claude_main", "to": "git_snapshot" },

  // Routing edge — fires only when routing_target matches
  { "from": "validate", "type": "routing_to", "to": "debate_brainstorm" },

  // No-routing edge — fires when routing_target is empty (normal reply)
  { "from": "validate", "type": "no_routing", "to": "__end__" },

  // Error edge — fires when rollback_reason is non-empty
  { "from": "validate", "type": "on_error",   "to": "error_handler" }
]
```

> `__start__` and `__end__` are valid edge targets (equivalent to LangGraph's `START`/`END`). Use either; both work.

### 10.5 — Alias Fuse Table (Never Use Old Names)

Historical code may reference deprecated aliases. **Always write the current names:**

| ❌ Old name | ✅ Current name | Where used |
|---|---|---|
| `AgentNode` | `LlmNode` | Python base class |
| `AgentClaudeNode` | `LlmClaudeNode` | Python class |
| `AgentRefNode` | `SubgraphRefNode` | Python class |
| `AgentRunNode` | `HeartbeatNode` | Python class |
| `AGENT_REF` | `SUBGRAPH_REF` | `"type"` in entity.json |
| `AGENT_RUN` | `HEARTBEAT` | `"type"` in entity.json |
| `LLAMA` | `OLLAMA` | `"type"` in entity.json |
| `LOCAL_VLLM` | `OLLAMA` | `"type"` in entity.json (alias still works) |
| `agent.json` | `entity.json` | filename |
| `from_json()` | `from_blueprint_and_instance()` | Python API |

### 10.6 — Subgraph State Leak Defensive Pattern

**Real production bug (2026-04-20):** Subgraphs leaked internal temporary state fields into the parent graph, corrupting the parent's `routing_target` and causing misroutes.

Safe patterns to prevent state pollution:

1. **Use `fresh_per_call`** (default) for all stateless subgraphs. This clears `messages` and `node_sessions` on entry and removes all internal messages on exit.

2. **Use `fresh_keep_fields`** only for fields that the parent intentionally passed IN and the subgraph must read:
   ```jsonc
   { "id": "tool_evaluate", "agent_dir": "...", "session_mode": "fresh_per_call",
     "fresh_keep_fields": ["discovery_report"] }
   ```

3. **Never rely on `routing_target` surviving a subgraph boundary.** It is cleared by `_subgraph_init` and by `_subgraph_exit`. The parent's validate/routing node re-parses the signal after the subgraph returns.

4. **Terminal nodes write via `output_field`**, not by directly setting parent state fields. This is the only safe handoff channel.

5. **Subgraph internal messages are always stripped** by `_subgraph_exit`. Parent `messages` list is not polluted regardless of what the subgraph's nodes add.

### 10.7 — Three-Phase Validation Before Committing

Before finalizing any blueprint edit, run these checks in order:

**Phase 1 — Anchor paths**
```bash
BLUEPRINT=/home/kingy/Foundation/VoidDraft/blueprints/role_agents/my_role
ls "$BLUEPRINT/entity.json"      # must exist
ls "$BLUEPRINT/"*.md             # persona files referenced in entity.json must exist
```

**Phase 2 — Atomic write**
Write `entity.json` and all persona files in one pass. Never leave entity.json referencing a persona file that doesn't exist yet.

**Phase 3 — Post-write validation**
```bash
# 1. JSON syntax check
python3 -m json.tool "$BLUEPRINT/entity.json" > /dev/null && echo "JSON OK"

# 2. All referenced persona files exist
python3 -c "
import json, pathlib
d = json.load(open('$BLUEPRINT/entity.json'))
base = pathlib.Path('$BLUEPRINT')
for f in (d.get('persona_files') or []):
    assert (base / f).exists(), f'Missing persona file: {f}'
for node in d.get('graph', {}).get('nodes', []):
    for f in (node.get('persona_files') or []):
        assert (base / f).exists(), f'Missing node persona: {f}'
print('All persona files exist OK')
"

# 3. Topology sanity — every non-start/end node has at least one in-edge and one out-edge
python3 -c "
import json
d = json.load(open('$BLUEPRINT/entity.json'))
nodes = {n['id'] for n in d['graph']['nodes']}
edges = d['graph']['edges']
in_deg  = {n: 0 for n in nodes}
out_deg = {n: 0 for n in nodes}
for e in edges:
    if e['to']   in nodes: in_deg[e['to']]   += 1
    if e['from'] in nodes: out_deg[e['from']] += 1
for n in nodes:
    assert in_deg[n]  > 0, f'Node {n!r} has no incoming edge (orphan)'
    assert out_deg[n] > 0, f'Node {n!r} has no outgoing edge (dead-end)'
print('Topology OK')
"
```

---

## 11. Quick Reference

| Want to... | How |
|---|---|
| Start an agent | `python awaken.py --entity <data_dir>` |
| Use Claude as main LLM | `"type": "CLAUDE_SDK"` with `"session_key"` |
| Use Gemini for critique/analysis | `"type": "GEMINI_API"` or `"GEMINI_CLI"` |
| Use a local model | `"type": "OLLAMA"` with `"model": "..."` |
| Add a deterministic step | `"type": "DETERMINISTIC"` + `validators.py` |
| Embed another pipeline | `"type": "SUBGRAPH_REF"` with `"agent_dir"` |
| Route conditionally | Emit `{"route": "node_id", "context": "..."}` from LLM |
| Isolate subgraph state | `"session_mode": "fresh_per_call"` (default) |
| Share LLM session with parent | `"session_mode": "inherit"` |
| Inspect graph topology | Send `!topology` in CLI/Discord |
| View session info | `!session`, `!sessions` |
| Roll back to last snapshot | `!rollback N` |
| Create a new blueprint | Add dir + `entity.json` + persona `.md` files in VoidDraft |
| Embed a subgraph node | Use `agent_dir` field, **no** `type` field |
| Pass state into fresh subgraph | Use `fresh_keep_fields: ["field_name"]` |
| Write subgraph output to parent | Set `output_field` on terminal node |
| Build a complete role agent | Follow Section 8 end-to-end walkthrough |
| Validate blueprint before commit | Run Phase 1-3 checks in Section 10.7 |
| Check deprecated names | See alias fuse table in Section 10.5 |
