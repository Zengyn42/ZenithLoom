Note: The core engine of the project resides in this ZenithLoom repository. However, all the blueprints have been moved to a separate repository called VoidDraft.

# ClaudeCLINode Design

> Date: 2026-04-08
> Status: Approved

## Summary

Add `ClaudeCLINode`, which calls the Claude CLI subprocess directly via `asyncio.create_subprocess_exec("claude", "-p", ...)`, as an alternative to `ClaudeSDKNode` (which uses `claude_agent_sdk`). Functionality is aligned with the SDK version, including streaming and session resume.

## Motivation

The existing `ClaudeSDKNode` is called via the `query()` function of `claude_agent_sdk`, which internally wraps the CLI subprocess. Adding `ClaudeCLINode` to call the CLI directly bypasses the SDK layer and provides:
- More direct CLI control (flag-level parameter passthrough)
- No dependency on the `claude_agent_sdk` Python package
- Architectural symmetry with `GeminiCLINode`

## Architecture

`ClaudeCLINode` exists alongside `ClaudeSDKNode` in `framework/nodes/llm/claude.py`:

```
claude.py
  ├── ClaudeSDKNode  (CLAUDE_SDK) — via claude_agent_sdk
  └── ClaudeCLINode  (CLAUDE_CLI) — via subprocess
```

### Inheritance Relationship

```
LlmNode (base)
  ├── ClaudeSDKNode   → call_llm() via claude_agent_sdk.query()
  ├── ClaudeCLINode   → call_llm() via asyncio.create_subprocess_exec("claude")
  ├── GeminiCLINode   → call_llm() via subprocess "gemini"
  ├── GeminiCodeAssistNode → call_llm() via HTTP API
  └── OllamaNode      → call_llm() via HTTP API
```

`ClaudeCLINode` **reuses the base class `LlmNode.__call__()`** and only implements `call_llm()`.

## CLI Invocation

```bash
claude -p \
  --output-format stream-json \
  --verbose \
  --include-partial-messages \
  --model <model> \
  --permission-mode <mode> \
  --system-prompt <prompt> \
  --allowedTools <tools...> \
  --disallowedTools <tools...> \
  --resume <session_id> \
  --add-dir <dirs...> \
  --setting-sources <sources> \
  --settings <json> \
  --mcp-config <json>
```

- Prompt is passed via **stdin** (to avoid yargs argument parsing issues, consistent with `GeminiCLINode`)
- Environment variable `CLAUDE_AGENT_SDK=1` is set to suppress hook sounds

## Stream-JSON Parsing

`--output-format stream-json --verbose --include-partial-messages` outputs line-by-line JSON:

| `type` | Handling |
|---|---|
| `"stream_event"` | Parse `event.content_block_delta`: `text_delta` → `cb(text, False)`, `thinking_delta` → `cb(thinking, True)` |
| `"result"` | Extract `result`, `session_id`, `is_error`, `usage`; call `update_token_stats()` |
| `"system"` | Skip (hook events, init info) |
| `"assistant"` | Skip (partial assembled messages) |
| `"rate_limit_event"` | Skip |

## Session Resume

- New session: Do not pass `--resume`
- Continuation: `--resume <session_id>`
- Resume failure (returncode != 0) → Retry with new session → Return error text + empty session_id if it fails again
- Exactly consistent with `ClaudeSDKNode` retry strategy

## Permission Mode

The CLI natively supports `--permission-mode`, which is passed directly from `self._permission_mode`.
`disallowed_tools` are passed via `--disallowedTools` (calculated by base class `_get_disallowed_tools()`).

## Timeout

Dynamic timeout, borrowed from `GeminiCLINode`:
- Baseline: 120s
- Upper limit: 600s
- Scaling: `prompt_len / 200` added seconds per character
- Formula: `min(600, max(120, prompt_len // 200))`

## SDK Options to CLI Flags Mapping

| ClaudeSDKNode (SDK option) | ClaudeCLINode (CLI flag) |
|---|---|
| `system_prompt` | `--system-prompt` |
| `permission_mode` | `--permission-mode` |
| `model` | `--model` |
| `allowed_tools` | `--allowedTools` |
| `disallowed_tools` | `--disallowedTools` |
| `resume` | `--resume` |
| `add_dirs` | `--add-dir` |
| `setting_sources` | `--setting-sources` |
| `settings` (JSON) | `--settings` |
| `mcp_servers` | `--mcp-config` (JSON string) |
| `env` | subprocess env vars |
| `cwd` | subprocess cwd |
| `thinking` | No direct CLI equivalent (depends on model default behavior) |
| `max_buffer_size` | Not applicable (manages stdout itself) |
| `include_partial_messages` | `--include-partial-messages` |

## Registry Changes

| builtins.py Registration | Points to |
|---|---|
| `CLAUDE_CLI` | `ClaudeCLINode` (New) |
| `CLAUDE_SDK` | `ClaudeSDKNode` (Unchanged) |

## Entity.json Migration

All existing `"type": "CLAUDE_CLI"` in `entity.json` are changed to `"type": "CLAUDE_SDK"` to ensure existing behavior remains unchanged:

- `VoidDraft/blueprints/role_agents/technical_architect/entity.json` (1 location)
- `VoidDraft/blueprints/functional_graphs/debate_claude_first/entity.json` (3 locations)
- `VoidDraft/blueprints/functional_graphs/debate_gemini_first/entity.json` (2 locations)
- `VoidDraft/blueprints/functional_graphs/apex_coder/entity.json` (1 location)
- `VoidDraft/blueprints/functional_graphs/tool_discovery/entity.json` (3 locations)

## Not Doing

- **Do not override `__call__`** — Reuse the base class `LlmNode` to avoid repeating `GeminiCLINode` logic
- **Do not use the `--bare` flag** — Maintain consistency with normal CLI behavior to allow hooks/CLAUDE.md to function normally
- **No model fallback chain** — `ClaudeSDKNode` doesn't have one, keep it consistent; fallback is a Gemini-specific requirement
- **No need for `get_recent_history` / `list_sessions`** — These are SDK-specific methods; the CLI node does not need them
