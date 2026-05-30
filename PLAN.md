# ZenithLoom - Butler No. 0 - Basic Builder Agent Plan

**Note:** The core engine of the project resides in this `ZenithLoom` repository. However, all the `blueprints` have been moved to a separate repository called `VoidDraft`.

## Context

Set up "Boundless Wisdom Sphere Butler No. 0" in an empty `ZenithLoom/` directory.
The core is a LangGraph state machine (Claude for main control + Gemini for strategic advice), with memory persisted via SQLite,
allowing multiple interfaces to seamlessly share the same session state.

**The core design principle:** All interfaces share the same `thread_id` + SQLite DB → memory is automatically synchronized, and context is not lost when switching interfaces.

---

## Interface Priorities

| Priority | Interface       | Mechanism                                  | Notes                                       |
|----------|-----------------|--------------------------------------------|---------------------------------------------|
| P0       | Discord         | discord.py bot → LangGraph `astream()`     | The sole remote entry point                 |
| P1       | VSCode Terminal | CLI script + fixed thread_id               | Direct connection in VSCode integrated terminal |
| P1       | Tmux            | `libtmux` manages named session `bootstrap_boss` | No history loss on detach/reattach          |
| P2       | PixelOffice     | **No explicit integration needed**         | Automatically visualizes from Claude Code JSONL transcript |

> PixelOffice is a pure observation layer (a VSCode extension). As long as the CLI is run in the VSCode terminal, it will automatically render agent activity as pixel art avatars.

---

## Framework Choice: LangGraph + SQLite WAL

- `SqliteSaver` naturally provide `thread_id` → session mapping.
- `astream()` supports asynchronous streaming for Discord.
- `PRAGMA journal_mode=WAL` solves locking issues with concurrent access from Discord + CLI.

---

## Directory Structure

```
ZenithLoom/
├── agent/
│   ├── __init__.py
│   ├── core.py          # LangGraph state machine + engine instance
│   └── tools.py         # Tool definitions like consult_ceo_gemini
├── interfaces/
│   ├── discord_bot.py   # P0: Discord remote interface (async)
│   └── cli.py           # P1: Local CLI + Tmux integration
├── main.py              # Entry point: python main.py [discord|cli|tmux]
├── requirements.txt
├── .env.example
└── PLAN.md              # This file
```

---

## Key Discord Design (referencing openclaw)

1.  **Draft Streaming** — Send a placeholder message first, then edit it as content is generated, throttled to 1200ms.
2.  **Fence-Aware Chunking** — Code block-aware chunking to ensure ``` fences are not broken.
3.  **Long Session Sliding Window** — `trim_messages()` keeps the last 40 messages, the System Prompt is always preserved.

---

## Session Synchronization Diagram

```
Discord Bot ──┐
              ├──→ builder_engine (LangGraph)
VSCode CLI ───┤         ↕
              │    SQLite WAL (cyber_bootstrap.db)
Tmux CLI ─────┘    thread_id: "boss_bootstrap_session_01"

PixelOffice → Automatically listens to VSCode JSONL (no integration code needed)
```

---

## Verification Steps

1.  `python main.py cli` → Local conversation, see streaming output.
2.  Ctrl+C and restart → History is preserved.
3.  `python main.py tmux` → Creates tmux session `bootstrap_boss`, detach/reattach works correctly.
4.  `python main.py discord` → Send a message on Discord, the Bot replies and shares history with the CLI.
5.  Run CLI in VSCode terminal → PixelOffice automatically displays pixel art avatars.

---

## Interface Generalization + GChat Integration + ExternalToolNode (Implemented)

> Full plan: `/home/kingy/.claude/plans/tidy-churning-ocean.md`

### Background

There is a lot of duplicate code between the CLI and Discord interfaces (`invoke_agent`, session commands, token tracking).
This design abstracts that into a `BaseInterface`, and adds a GChat interface and a generic `EXTERNAL_TOOL` node.

### Goals (All completed)

1.  **`BaseInterface` Base Class** — Extract shared logic, to be inherited by CLI/Discord/GChat.
2.  **`GChatInterface`** — Receive GChat messages via `gws events +subscribe`, process through LangGraph, and reply with `gws chat +send`.
3.  **`ExternalToolNode` (`EXTERNAL_TOOL`)** — A generic CLI invocation node. Any external tool can be integrated by declaring its command list in `entity.json`.

### Architectural Principles

The interface layer architecture remains unchanged: the transport layer calls `graph.astream()` from the outside, and the LangGraph graph is unaware of the interface source.

```
Discord Bot         ─┐
CLI                  ├──→ graph.astream() → LangGraph
GChat Bot           ─┘

EXTERNAL_TOOL Node ──→ Subprocess calls any CLI (argument list, no shell) → JSON output injected into messages
```

### Three-Tier External Tool Strategy

| Scenario                               | Strategy                               | `command` Example                                     |
|----------------------------------------|----------------------------------------|-------------------------------------------------------|
| CLI-Anything harness already exists (blender, gimp, etc.) | Use `cli-anything-<tool> --json` directly | `["cli-anything-blender", "--json", "render", "animation"]` |
| Has a structured CLI (gws, obsidian, git) | Call directly, no wrapper needed         | `["gws", "gmail", "+triage", "--json"]`                 |
| GUI application with no CLI          | Generate a harness with /cli-anything then integrate | —                                                     |

**Decision criteria:** Does the tool have a `--json` flag + a stable subcommand structure? If yes, call it directly. If no, use CLI-Anything.

### Dual Role of `gws`

| Role                | Description                                                                          | Implementation Location                                  |
|---------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------|
| **Interface Layer** | `gws events +subscribe` listens for GChat messages as a conversation entry point; `gws chat +send` replies. | `interfaces/gchat_bot.py` → `GChatInterface`             |
| **Tool Node**       | `gws gmail +triage`, `gws drive list`, etc., execute GWS operations as an `EXTERNAL_TOOL` node. | `framework/nodes/external_tool_node.py` → `ExternalToolNode` |

The two do not interfere: `GChatInterface` is the **caller** of the graph (outside the graph), while `ExternalToolNode` is a **called node** of the graph (inside the graph).

### New Files Created

| File                                  | Description                                                                                             |
|---------------------------------------|---------------------------------------------------------------------------------------------------------|
| `framework/base_interface.py`         | `BaseInterface`: `invoke_agent()`, `handle_command()`, `split_fence_aware()`, `extract_attachments()`     |
| `interfaces/gchat_bot.py`             | `GChatInterface`: `gws events +subscribe` NDJSON stream → agent → `gws chat +send`                       |
| `framework/nodes/external_tool_node.py`| `ExternalToolNode`: Generic CLI invocation, `{field}` template injection from state, JSON pretty-printing |

### Modified Files

| File                        | Changes                                                                    |
|-----------------------------|----------------------------------------------------------------------------|
| `interfaces/cli.py`         | `_CliInterface(BaseInterface)`, `!topology`/`!debug`/`!snapshots`/`!rollback` remain in subclass |
| `interfaces/discord_bot.py` | `_DiscordInterface(BaseInterface)`, module-level `bot` is kept (discord.py constraint) |
| `framework/builtins.py`     | Register `EXTERNAL_TOOL` → `ExternalToolNode`                               |
| `main.py`                   | Add `gchat` mode                                                            |
| `framework/config.py`       | `AgentConfig` adds `gchat_space`, `gchat_gcp_project`, `gchat_event_types`  |

### `EXTERNAL_TOOL` Usage Example (`entity.json`)

```json
// gws: Has a structured CLI (call directly, no CLI-Anything needed)
{ "id": "gmail_reader",    "type": "EXTERNAL_TOOL",
  "node_config": { "command": ["gws", "gmail", "+triage", "--json"],
                   "description": "Read summaries of unread emails" } },

// obsidian: Has an official CLI (call directly)
{ "id": "obsidian_search", "type": "EXTERNAL_TOOL",
  "node_config": { "command": ["obsidian", "search", "--query", "{routing_context}"],
                   "description": "Search Obsidian vault" } },

// blender: CLI-Anything harness exists (call via wrapper)
{ "id": "blender_render",  "type": "EXTERNAL_TOOL",
  "node_config": { "command": ["cli-anything-blender", "--json", "render", "animation"],
                   "description": "Render Blender animation", "timeout": 120 } }
```

### GChat Configuration (`entity.json`)

```json
"gchat_space": "spaces/AAAA...",
"gchat_gcp_project": "my-gcp-project",
"gchat_event_types": "google.workspace.chat.message.v1.created"
```

### GChat Event Flow Architecture

```
GChat user message
  → Google Workspace Events API
    → GCP Pub/Sub Topic (must be configured in advance)
      → gws events +subscribe (long-running process, NDJSON poll)
        → GChatInterface._extract_chat_event()
          → invoke_agent() / handle_command()
            → gws chat +send reply
```

### Key Constraints

-   The actual NDJSON field paths for `gws events +subscribe` need to be confirmed by running a one-time pull with `--once`.
-   Must filter out the Bot's own messages (`sender.type == "BOT"`) to prevent echo loops.
-   `discord.py` requires the `bot` object to be registered at the module level; `DiscordInterface` injects the loader/controller via a closure.
-   External process calls should uniformly use an argument list format (not a shell string) to avoid injection risks.
