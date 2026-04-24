# framework/loader

Loads agent blueprints, builds LangGraph state machines, and manages runtime controllers.

## Module layout

| File | Responsibility |
|---|---|
| `entity_loader.py` | `EntityLoader` — main public API |
| `graph_builder.py` | `_build_declarative`, `_DEFAULT` sentinel, node/edge wiring |
| `graph_validator.py` | ID deduplication, edge ref checks, reachability |
| `persona.py` | Assembles system prompt from `persona_files` + routing hints |
| `topology.py` | Mermaid flowchart renderer (`build_topology_mermaid`) |

---

## Quick start

```python
from pathlib import Path
from framework.loader import EntityLoader

loader = EntityLoader(
    Path("blueprints/role_agents/technical_architect"),
    data_dir=Path("~/Foundation/EdenGateway/agents/hani"),
)

# Recommended: get a GraphController (lazy-compiled, singleton)
controller = await loader.get_controller()
response = await controller.run("用户输入")
```

`data_dir` separates the **blueprint** (role definition, shared) from the **entity** (instance config, runtime data). When omitted, `blueprint_dir` is used for both.

---

## Directory layout expected

```
blueprints/role_agents/technical_architect/   ← blueprint_dir
    entity.json        # graph, persona_files, llm type, etc.
    PERSONA.md         # role definition
    graph.py           # optional custom graph builder

EdenGateway/agents/hani/                      ← data_dir
    identity.json      # name, discord_token, allowed_users
    SOUL.md            # instance-specific persona (auto-appended)
    hani.db            # SQLite checkpoint store
    sessions.json      # session envelopes
```

---

## Graph build priority

`build_graph()` checks these in order:

1. **`graph.py`** — if the blueprint dir contains `graph.py` with a `build_graph(loader, checkpointer)` function, it is called directly.
2. **Declarative** — if `entity.json["graph"]` has both `"nodes"` and `"edges"` keys, `_build_declarative` handles it.
3. **GraphSpec default** — otherwise, the legacy `GraphSpec` path builds a single-LLM graph.

---

## EntityLoader API

### Constructor

```python
EntityLoader(agent_dir: Path, data_dir: Path | None = None)
```

### Key methods

| Method | Returns | Notes |
|---|---|---|
| `load_config()` | `AgentConfig` | Blueprint + identity merged config |
| `load_system_prompt()` | `str` | Persona files concatenated with source headers |
| `await build_graph(...)` | compiled LangGraph | See parameters below |
| `await get_controller()` | `GraphController` | Lazy singleton — preferred entry point |
| `await start_mcp_servers()` | `list[str]` | Starts MCP servers from `entity.json["mcp"]` |
| `await stop_mcp_servers()` | — | Releases MCP refs and disconnects proxies |
| `await start_heartbeat()` | proxy or `None` | Connects Heartbeat MCP + loads blueprints |
| `await stop_heartbeat()` | — | Unloads heartbeat and disconnects |
| `build_topology_mermaid()` | `str` | Mermaid `flowchart LR` string |
| `invalidate_engine()` | — | Force rebuild on next `get_controller()` call |

### `build_graph` parameters

```python
await loader.build_graph(
    checkpointer=_DEFAULT,          # Pass None for no persistence, or a LangGraph checkpointer
    extra_persona_text="",          # Appended to every LLM node's system prompt
    is_subgraph=False,              # True → injects _subgraph_init / _subgraph_exit boundary nodes
    force_unique_session_keys=False,
    session_mode="fresh_per_call",  # "fresh_per_call" | "persistent" | "inherit" | "isolated"
    fresh_keep_fields=None,         # Fields to preserve across fresh_per_call resets
)
```

`_DEFAULT` causes `build_graph` to create a SQLite checkpointer from `config.db_path`. Pass `None` to skip checkpointing (e.g. for subgraph compilation).

---

## entity.json schema (declarative graph)

```jsonc
{
  "persona_files": ["PERSONA.md"],
  "graph": {
    "state_schema": "debate",       // optional: registers a named state schema
    "entry": "node_a",             // first node after START
    "exit": "node_b",              // last node before END
    "nodes": [
      {
        "id": "node_a",
        "type": "CLAUDE_SDK",      // LLM type or VRAM_FLUSH / SUBGRAPH_REF / HEARTBEAT
        "session_key": "claude_main",
        "system_prompt": "You are ...",
        "persona_files": ["PERSONA.md"],
        "extra_persona": true      // inject instance-level persona from identity.json
      },
      {
        "id": "subgraph_node",
        "type": "SUBGRAPH_REF",    // external subgraph
        "agent_dir": "blueprints/functional_graphs/apex_coder",
        "session_mode": "fresh_per_call"
      }
    ],
    "edges": [
      {"from": "node_a", "to": "node_b"},
      {
        "from": "node_a",
        "type": "conditional",
        "conditions": [
          {"target": "node_b", "when": "routing_target == 'node_b'"},
          {"target": "__end__", "when": "default"}
        ]
      }
    ]
  },
  "mcp": [
    {
      "name": "heartbeat",
      "url": "http://127.0.0.1:8100/sse",
      "proxy": "heartbeat"
    }
  ]
}
```

### Supported node types

| Type | Class |
|---|---|
| `CLAUDE_SDK` / `CLAUDE_CLI` | `ClaudeSDKNode` |
| `GEMINI_API` | `GeminiCodeAssistNode` |
| `GEMINI_CLI` | `GeminiCLINode` |
| `OLLAMA` / `LOCAL_VLLM` | `OllamaNode` |
| `SUBGRAPH_REF` | `SubgraphRefNode` (wraps external `EntityLoader`) |
| `HEARTBEAT` | `HeartbeatNode` |
| `VRAM_FLUSH` | VRAM flush utility node |

---

## Session modes (subgraphs)

| Mode | `_subgraph_init` | `_subgraph_exit` | Effect |
|---|---|---|---|
| `fresh_per_call` | ✓ | ✓ | Clears messages and node sessions before entry; clears on exit |
| `isolated` | ✓ | ✓ | Clears node sessions only (keeps messages) |
| `persistent` | — | ✓ | Keeps all state; exit still clears output fields |
| `inherit` | — | ✓ | Inherits parent state as-is |

---

## State schema registration

```python
from framework.loader import register_state_schema
from framework.schema.debate import DebateState

register_state_schema("debate", DebateState)
```

In `entity.json`, set `"state_schema": "debate"` to use it. Schemas can also be registered at import time from a schema module.
