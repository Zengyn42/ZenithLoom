# Asa Agent + OllamaNode — Phase 1 Design Spec

## Overview

Create the Asa agent (single-node Llama graph) and implement OllamaNode in the framework layer. Asa is a lightweight, CPU-resident local LLM agent that runs alongside Hani. Phase 2 (HeartbeatScheduler, wake-Hani, `!heartbeat`) is out of scope.

## Motivation

- Asa is the second agent in the system, running on a local Llama model via Ollama
- The existing `LlamaNode` is a stub (`NotImplementedError`) — needs full implementation
- Asa runs in RAM (CPU inference), small and fast, always available
- Future QWen agent will reuse the same OllamaNode with different `model` + `resource_lock`

## Components

### 1. OllamaNode (`framework/llama/node.py`)

Rewrite the existing stub. Rename class from `LlamaNode` to `OllamaNode`. Keep `LlamaNode = OllamaNode` alias in `node.py` for backward compatibility (builtins.py imports directly from this module).

**Inherits:** `AgentNode` (same as Claude/Gemini nodes)

**Constructor (`__init__`):**
- `self._model`: from `node_config.get("model", "llama3")`
- `self._endpoint`: from `node_config.get("endpoint", "http://localhost:11434")`
- `self._timeout`: from `node_config.get("timeout", 120)`
- `self._system_prompt`: from `node_config.get("system_prompt", "")` — persona is auto-injected here by `_build_declarative` for nodes whose ID contains "main" (e.g. `llama_main`)
- Default endpoint provided (unlike current stub which requires it) — `http://localhost:11434` is Ollama's standard port

**`call_llm(prompt, session_id, tools, cwd)` implementation:**
- HTTP POST to `{endpoint}/api/chat` via `httpx.AsyncClient`
- Request body: `{"model": ..., "messages": [...], "stream": false, "keep_alive": -1}`
  - `keep_alive: -1` pins the model in RAM permanently (Ollama default unloads after 5 min idle — would break the "always available" promise)
- Messages: `[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]` — system_prompt omitted if empty
- Response text extracted from `response_json["message"]["content"]` — **not** `response_json["response"]` (that belongs to `/api/generate`, not `/api/chat`)
- Status code check: if `response.status_code != 200`, extract `body.get("error", str(status_code))` and return as error message — do NOT attempt `message.content` on error body
- Returns `(response_text, session_id)` — session_id passed through unchanged (Ollama has no persistent sessions)
- `tools` parameter: **ignored in Phase 1** — Ollama `/api/chat` does support the `tools` field for compatible models, but tool call handling is deferred to a future phase. The base class `_select_tools()` will pass tool names; OllamaNode discards them.
- `cwd` parameter: **ignored** — Ollama has no working directory concept.
- On error, returns `(error_message, session_id)` with the input `session_id` unchanged — never returns empty session_id on failure, to avoid corrupting `node_sessions` state.

**Error handling (return error as text, don't crash the graph):**
- Ollama not running → `httpx.ConnectError` → log + return `"[Ollama 连接失败] ..."`
- Model not found → Ollama returns error JSON → extract + return as message
- Timeout → `httpx.TimeoutException` → return `"[Ollama 超时] ..."`

**`get_recent_history()`:** returns `[]` (Ollama has no persistent session history). This is a convention shared by all LLM nodes, called from session management in GraphController.

### 2. `framework/llama/__init__.py`

Update exports:
```python
from framework.llama.node import OllamaNode, LlamaNode  # LlamaNode is alias

__all__ = ["OllamaNode", "LlamaNode"]
```

### 3. `framework/builtins.py` — Node type registration

Replace the existing `LOCAL_VLLM` registration. Use two separate functions (matching existing pattern — CLAUDE_CLI and CLAUDE_SDK are separate registrations). Import `LlamaNode` (the alias) rather than `OllamaNode` directly to stay consistent with the existing import and avoid transition risk:
```python
@register_node("OLLAMA")
def _(config, node_config):
    from framework.llama.node import LlamaNode
    return LlamaNode(config, node_config)

@register_node("LOCAL_VLLM")
def _(config, node_config):
    from framework.llama.node import LlamaNode
    return LlamaNode(config, node_config)
```

**Implementation order:** `framework/llama/node.py` and `builtins.py` must be updated before `agents/asa/agent.json` uses `"type": "OLLAMA"`. If implementing in stages, use `"LOCAL_VLLM"` as interim node type in agent.json.

### 4. Asa Agent (`agents/asa/`)

Uses default `BaseAgentState` (no `state_schema` field needed — same as Hani).

**`agent.json`** — rewrite existing file:
```json
{
  "name": "asa",
  "llm": "llama",
  "channel_history_limit": 20,
  "graph": {
    "nodes": [
      {
        "id": "llama_main",
        "type": "OLLAMA",
        "model": "llama3.2:3b",
        "endpoint": "http://localhost:11434",
        "first_turn_suffix": "Asa:",
        "user_msg_prefix": "",
        "tombstone_enabled": false,
        "tool_rules": []
      }
    ],
    "edges": [
      {"from": "__start__", "to": "llama_main"},
      {"from": "llama_main", "to": "__end__"}
    ]
  },
  "max_retries": 2,
  "db_path": "asa.db",
  "sessions_file": "sessions.json",
  "persona_files": ["SOUL.md"]
}
```

Note: `discord_token` and `discord_allowed_users` omitted — Asa is CLI-only in Phase 1. Add these fields when Discord support is needed. The Discord interface will not start without a valid token.

Note: `gemini_mention_pattern` intentionally omitted — Asa has no @Gemini routing support.

**`SOUL.md`** — expand existing stub:
```markdown
## Asa — Identity & Soul

You are Asa, the second Agent of the Boundless Intellect Dome (无垠智穹).

You run on a local Llama model via Ollama, specialized for:
- Fast, lightweight responses (CPU inference, always available)
- Privacy-sensitive and offline scenarios
- System monitoring and health checks (Phase 2)

Personality:
- Concise and direct — you're a small model, don't waste tokens
- Practical and reliable — focus on getting things done
- Bilingual: English and Chinese (中文)

You report to the same boss (老板) as Hani.
```

### 5. Dependency

Add `httpx` to `requirements.txt`. `httpx` is used for async HTTP calls to Ollama API.

## Files Changed

| File | Action |
|------|--------|
| `framework/llama/node.py` | Rewrite: stub → full OllamaNode + LlamaNode alias |
| `framework/llama/__init__.py` | Update: export OllamaNode + LlamaNode alias |
| `framework/builtins.py` | Edit: replace `LOCAL_VLLM` registration, add `OLLAMA` registration |
| `agents/asa/agent.json` | Rewrite: declarative graph with single OLLAMA node |
| `agents/asa/SOUL.md` | Expand: full persona |
| `requirements.txt` | Edit: add `httpx` |

## Files NOT Changed

- `framework/nodes/agent_node.py` — no changes needed, OllamaNode inherits as-is
- `framework/state.py` — uses existing `BaseAgentState` (default)
- `framework/config.py` — no new config fields needed
- `interfaces/cli.py` — already supports `--agent asa`
- `interfaces/discord_bot.py` — not used in Phase 1 (Asa is CLI-only initially)
- `main.py` — no changes needed

## Out of Scope (Phase 2)

- HeartbeatScheduler (periodic health checks)
- Wake-Hani mechanism
- `!heartbeat` command
- Discord support for Asa
- Streaming support
- Vision/multimodal support
- vLLM backend support
- QWen agent creation
- Automated tests (manual testing only in Phase 1; mock-based test deferred)

## Testing

- Manual: `python main.py --agent asa cli` → send message → verify Ollama response
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Verify model loaded: response should use configured model
- Error case: stop Ollama → send message → verify friendly error (no crash)
