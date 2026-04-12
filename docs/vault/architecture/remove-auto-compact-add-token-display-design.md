# Design: Remove Auto-Compact, Add Inline Token Display

**Date**: 2026-04-10
**Scope**: `ClaudeSDKNode` and the `!compact` / `!tokens` commands

## Problem

`ClaudeSDKNode` has an auto-compact mechanism that triggers before any `call_llm` when the previously-observed context exceeds 100,000 tokens. It sends `/compact` to the Claude CLI via `sdk_query(prompt="/compact")`, breaks on the first `ResultMessage`, and calls `self._last_context_size.clear()` to reset tracking.

Observed symptom: after auto-compact fires, the Claude node "forgets" previous turns — user commands that were in the conversation immediately before the compact appear to be lost.

Two root causes:

1. **Premature `break`** — the loop exits on the first `ResultMessage`, which may be an intermediate progress event whose `session_id` is not the final compacted session. The returned `session_id` can be stale or empty, so the next `call_llm` tries to resume a session that is no longer the canonical one.
2. **Inherent `/compact` semantics** — even when the session_id is captured correctly, `/compact` replaces the Claude CLI's conversation history with an LLM-generated summary. Detail (exact wording, file paths, tool-call outputs) is lost by design; only the gist survives.

Additionally, `_last_context_size.clear()` wipes tracking for **all** sessions, not just the one that was compacted.

The user does not want an automatic mechanism that silently loses detail. They want:

1. Auto-compact removed.
2. Each Claude reply to end with an inline token-usage line so the user can see context growth in real time and decide when to act.
3. Manual `!compact` to do the compaction when they judge it appropriate, with explicit awareness of the tradeoff.
4. The inline token line must be toggleable on/off.

## Non-Goals

- Gemini and Ollama nodes are out of scope. Their `usage` payloads differ and the user did not report problems there. A TODO is noted below.
- `ClaudeCLINode` is out of scope. All current role agents (Hani, Asa, Jei) use `ClaudeSDKNode`.
- Per-session toggle state. The toggle is process-global.
- Reworking `/compact`'s summary behavior. Lossy compaction is Claude CLI's design; this spec surfaces it honestly rather than fighting it.

## Design

The work splits into three independent blocks.

### Block 1 — Remove Auto-Compact

File: `ZenithLoom/framework/nodes/llm/claude.py`

Deletions in `ClaudeSDKNode`:

- Class constant `_COMPACT_THRESHOLD = 100_000`.
- Instance field `self._last_context_size: dict[str, int] = {}` in `__init__`.
- The five lines inside the `ResultMessage` branch of `_run_once` that compute `_ctx` and assign into `self._last_context_size`.
- The entire auto-compact block before `_run_once` is invoked, including `self._last_context_size.clear()` and the stream-callback notifications.

Retained:

- `update_token_stats(msg.usage)` continues to feed the process-level `token_tracker`.
- All resume-failure retry logic.

After this block, `call_llm` makes exactly one `sdk_query` per invocation.

### Block 2 — `!compact` Compacts Both Checkpoint DB and Claude Session

The `!compact` command will perform two operations sequentially:

1. Compact the LangGraph checkpoint DB (existing behavior, unchanged).
2. Send `/compact` to every active `ClaudeSDKNode` session in the current thread.

#### 2A. New method `ClaudeSDKNode.compact_session()`

File: `ZenithLoom/framework/nodes/llm/claude.py`

```python
async def compact_session(self, session_id: str) -> tuple[str, str]:
    """
    Send /compact to the given Claude session.
    Returns (status_message, new_session_id).
    """
```

Implementation notes:

- Build `ClaudeAgentOptions` via the same helper as `call_llm` (resume=`session_id`, same cwd, same permission_mode, etc.).
- Iterate `async for msg in sdk_query(prompt="/compact", options=...)` to completion — **do not break on the first `ResultMessage`**. Remember the most recent non-empty `msg.session_id` from any `ResultMessage` seen, and return that as the new sid.
- If no `ResultMessage` with a session_id is observed, return `(session_id, "warning: no session_id returned")` so the caller keeps the old sid rather than blanking it.
- On exception, log and return `(session_id, f"error: {e}")` — never clear the sid on failure.
- Do **not** manipulate any `_last_context_size` field (it no longer exists).

#### 2B. New method `GraphController.compact_claude_session()`

File: `ZenithLoom/framework/graph_controller.py`

```python
async def compact_claude_session(self, thread_id: str) -> str:
    """
    Find all ClaudeSDKNode instances in the compiled graph, pull their
    session_ids from state["node_sessions"], send /compact to each, and
    write the new session_ids back via checkpointer update_state.
    Returns a human-readable summary.
    """
```

Implementation notes:

- Import `ClaudeSDKNode` lazily to avoid circular imports.
- Access compiled graph via `self._graph` (same field already used by `compact_checkpoint`).
- Get current state: `snapshot = await self._graph.aget_state(config={"configurable": {"thread_id": thread_id}})`. Extract `node_sessions = snapshot.values.get("node_sessions", {})`.
- Iterate the graph's nodes. For each node whose underlying callable is a `ClaudeSDKNode`, use the node's `_session_key` to find its session_id in `node_sessions`.
- For each `(key, sid)` with non-empty sid: call `await node.compact_session(sid)`. Collect results.
- If any sids changed, call `await self._graph.aupdate_state(config, {"node_sessions": updated_map})` to persist the new ids into the checkpoint.
- Return a summary string like `"压缩了 1 个 Claude session (claude_main: abc12345 → def67890)"` or `"无 Claude session 可压缩"` when nothing matched.
- Uncaught exceptions bubble up to the `!compact` handler, which wraps them into a user-visible error line.

Node-iteration detail: LangGraph compiled graphs expose `self._graph.nodes` (or similar) where each node's runnable can be an instance of the `ClaudeSDKNode`. The exact attribute path will be verified during implementation; if direct introspection is awkward, the fallback is to iterate `self._graph.builder.nodes` before compilation and stash a reference.

#### 2C. Extend `!compact` handler

File: `ZenithLoom/framework/base_interface.py` (around line 447)

```python
if cmd == "!compact":
    try:
        keep = int(arg) if arg else 20
    except ValueError:
        keep = 20
    thread_id = self._resolve_thread_id()
    deleted = await controller.compact_checkpoint(thread_id, keep_last=keep)
    try:
        claude_msg = await controller.compact_claude_session(thread_id)
    except Exception as e:
        claude_msg = f"❌ Claude 压缩失败: {e}"
    return (
        f"Compact 完成：\n"
        f"  checkpoint DB: 删除 {deleted} 条旧记录，保留最近 {keep} 条\n"
        f"  Claude session: {claude_msg}"
    )
```

### Block 3 — Inline Token Display with Toggle

#### 3A. New module `framework/token_display.py`

```python
"""Process-level toggle for inline token usage display after each Claude reply."""

_enabled: bool = True  # default on — matches original "show tokens after every message" intent

def is_token_display_enabled() -> bool:
    return _enabled

def set_token_display(value: bool) -> None:
    global _enabled
    _enabled = bool(value)
```

This module mirrors the pattern of `framework/debug.py`.

#### 3B. Modify `ClaudeSDKNode.call_llm` to emit the token line

File: `ZenithLoom/framework/nodes/llm/claude.py`

**Source-of-truth note**: Do NOT use `ResultMessage.usage` for the ctx/in/cache_read numbers in the inline line. `ResultMessage.usage` is cumulative across all API calls within one `sdk_query` (including tool-use sub-calls), so its `input_tokens` will be inflated to many times the real context window on tool-heavy turns. The existing code already addresses this by tracking context from `message_start` events (see the existing comment in `_run_once`: *"不能用 ResultMessage.usage — 那是累计值，复杂 tool use 会远超真实 context"*). We build on that mechanism.

`ResultMessage.usage` is still the correct source for the cumulative process-level `update_token_stats()` call (billing), and that call is kept unchanged. The distinction is:
- **Billing / `!tokens` cumulative**: `ResultMessage.usage` (cumulative is correct — you pay for every tool-use sub-call).
- **Inline context-fullness indicator**: last `message_start.usage` (represents the final API call's actual input context).

Two targeted changes:

1. In `_run_once`, replace the existing `_last_msg_ctx: int = 0` local with `_last_msg_usage: dict = {}`. In the `message_start` branch, replace the current computation `_last_msg_ctx = input_tokens + cache_read + cache_creation` with `_last_msg_usage = dict(_usage)` (shallow copy of the message_start usage dict). Keep `update_token_stats(msg.usage)` in the `ResultMessage` branch unchanged. Change `_run_once` return signature from `(text, sid, is_error)` to `(text, sid, is_error, last_msg_usage)` and return `_last_msg_usage` alongside. Update both `_run_once` call sites in `call_llm` (the initial call and the resume-retry call) to unpack the new 4-tuple.

2. After the successful path has populated `result_text`, `new_session_id`, `is_error`, and `last_msg_usage`, and before the final `return`, emit the token line:

```python
from framework.token_display import is_token_display_enabled

if is_token_display_enabled() and last_msg_usage:
    cb = _stream_cb.get()
    if cb is not None:
        ctx_total = (
            last_msg_usage.get("input_tokens", 0)
            + last_msg_usage.get("cache_read_input_tokens", 0)
            + last_msg_usage.get("cache_creation_input_tokens", 0)
        )
        line = (
            f"\n[tokens: ctx={ctx_total:,} "
            f"in={last_msg_usage.get('input_tokens', 0):,} "
            f"out={last_msg_usage.get('output_tokens', 0):,} "
            f"cache_read={last_msg_usage.get('cache_read_input_tokens', 0):,}]\n"
        )
        cb(line, False)
```

The `is_thinking=False` flag ensures downstream consumers treat it as normal output text, not ANSI-styled thinking.

Note on displayed fields: `ctx` sums `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` (the full context window sent to Claude on the last API call of this turn), but only three of those components appear individually in the bracket — `in`, `out`, `cache_read`. The fourth, `cache_creation_input_tokens`, is deliberately omitted from the inline line to keep it short; its value can be derived as `ctx - in - cache_read` when needed, and users who want the full breakdown can run `!tokens` for the cumulative process-level stats. Also note that `out` here is from the last `message_start.usage` and may render as `0` if message_start does not carry `output_tokens` — which is acceptable since cumulative billing output is tracked by `!tokens`.

**Known limitation**: when `channel_send_final=True` in the node config, `llm_node.py` intentionally sets `_stream_cb` to `None` to suppress Discord streaming drafts. In that mode, the token line is also suppressed. This is acceptable: channel_send_final is only used in debate subgraphs where intermediate chatter is unwanted, and the final compound message is posted by the subgraph orchestrator.

#### 3C. Extend `!tokens` command to toggle inline display

File: `ZenithLoom/framework/base_interface.py`

Current behavior: `!tokens` prints the cumulative process-level stats from `token_tracker.get_token_stats()`. Extend it:

- `!tokens` (no arg) → unchanged — print cumulative stats, append one line showing the current toggle state: `"内联显示: 开启"` or `"内联显示: 关闭"`.
- `!tokens on` → `set_token_display(True)`, reply `"✅ Token 内联显示：开启"`.
- `!tokens off` → `set_token_display(False)`, reply `"✅ Token 内联显示：关闭"`.
- `!tokens status` → reply `"Token 内联显示：开启"` or `"关闭"`.
- Any other arg → reply with usage hint `"用法：!tokens [on|off|status]"`.

File: `ZenithLoom/framework/command_registry.py`

Update the `!tokens` registration's usage hint from whatever it currently is to `"[on|off|status]"` and extend its description to mention the toggle.

## Testing

All three blocks must preserve existing test passes (`test_cli.py` 8/8, `test_e2e_debate.py` 8/8). No new tests are strictly required by this spec, but the following manual checks should be performed before declaring the work complete:

1. **Auto-compact gone**: grep the repo for `_COMPACT_THRESHOLD`, `_last_context_size`, and `auto-compact` — should return zero hits in `claude.py`.
2. **Token line appears**: start hani in CLI mode, issue a short prompt, confirm the reply ends with a line matching `^\[tokens: ctx=\d`.
3. **Toggle works**: `!tokens off` → next reply has no token line. `!tokens on` → token line returns.
4. **!compact runs both compactions**: `!compact` output mentions both the checkpoint DB count and the Claude session status. Immediately after, issue a follow-up prompt and confirm the Claude node still responds coherently (the reply may lose detail from before compaction — this is expected and spec-documented).
5. **Existing tests**: run `python -m pytest test_cli.py test_e2e_debate.py` and confirm no regressions.

## Risks and Open Questions

1. **`/compact` semantics loss**: manual `!compact` still triggers lossy summarization. This is documented and the user accepted the tradeoff, but we should warn clearly in the `!compact` reply message. Consider adding `"(Claude 对话细节已压缩为摘要，继续提问时如涉及旧细节请重新说明)"` to the output when compact_claude_session reports success.
2. **Graph node introspection**: the exact mechanism to iterate compiled graph nodes and identify `ClaudeSDKNode` instances needs implementation-time verification. If `self._graph.nodes` does not expose the underlying callable directly, we may need to keep a side-map in `AgentLoader.build_graph` from node_id → node instance.
3. **`channel_send_final` suppression**: token line does not appear in debate subgraphs. If this becomes a pain point, a follow-up could append the token line to the final compound message at the `llm_node` level.
4. **Gemini / Ollama token display**: marked as a follow-up. Not blocking the user's immediate need.

## Out-of-Scope Follow-ups (TODO, not this spec)

- Extend inline token display to `GeminiCodeAssistNode` and `OllamaNode` using their respective usage payloads.
- Consider a `!compact --dry-run` mode that previews what would be lost.
- Consider a "soft" compact that archives old turns to disk before telling Claude to summarize, enabling later restoration.
