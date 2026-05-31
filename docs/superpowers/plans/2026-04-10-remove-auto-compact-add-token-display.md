# Remove Auto-Compact & Add Inline Token Display — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the buggy auto-compact mechanism from `ClaudeSDKNode`, show an inline `[tokens: …]` line after each Claude reply (toggleable), and make `!compact` additionally send `/compact` to the live Claude session.

**Architecture:** Three decoupled blocks modifying a small set of files. Auto-compact state is purged; token display is a new process-global module; `!compact` is extended through a new controller method that looks up `ClaudeSDKNode` instances via a side-map attached to the compiled graph at build time.

**Tech Stack:** Python 3.x, LangGraph, claude_agent_sdk, existing ZenithLoom framework layer (`framework/nodes/llm/claude.py`, `framework/graph_controller.py`, `framework/base_interface.py`, `framework/command_registry.py`, `framework/agent_loader.py`).

**Spec:** `docs/vault/architecture/remove-auto-compact-add-token-display-design.md`

---

## File Structure

**Create:**
- `framework/token_display.py` — process-global toggle for the inline token line (mirrors `framework/debug.py`)
- `test_token_display.py` — unit test for the new module

**Modify:**
- `framework/nodes/llm/claude.py` — remove auto-compact, repurpose `_last_msg_ctx` → `_last_msg_usage`, add `compact_session()` method, emit token line
- `framework/graph_controller.py` — add `compact_claude_session()` method
- `framework/base_interface.py` — extend `!compact` handler; extend `!tokens` handler (on/off/status)
- `framework/command_registry.py` — update `!tokens` usage hint
- `framework/agent_loader.py` — collect LLM node instances into a side-map during `_build_declarative` and attach to the compiled graph
- `test_commands.py` — update existing `!compact` and `!tokens` test expectations; add coverage for toggle

---

## Task 1: Create `framework/token_display.py`

**Files:**
- Create: `framework/token_display.py`
- Test: `test_token_display.py`

- [ ] **Step 1: Write the failing test**

Create `test_token_display.py`:

```python
"""Unit test for framework.token_display process-global toggle."""

from framework import token_display


def test_default_is_enabled():
    # Reload to reset module-level state in case prior tests mutated it.
    import importlib
    importlib.reload(token_display)
    assert token_display.is_token_display_enabled() is True


def test_set_and_read_false():
    token_display.set_token_display(False)
    assert token_display.is_token_display_enabled() is False


def test_set_and_read_true():
    token_display.set_token_display(True)
    assert token_display.is_token_display_enabled() is True


def test_set_coerces_truthy_falsy():
    token_display.set_token_display(0)
    assert token_display.is_token_display_enabled() is False
    token_display.set_token_display("yes")
    assert token_display.is_token_display_enabled() is True


if __name__ == "__main__":
    test_default_is_enabled()
    test_set_and_read_false()
    test_set_and_read_true()
    test_set_coerces_truthy_falsy()
    print("✅ token_display OK")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_token_display.py`
Expected: `ModuleNotFoundError: No module named 'framework.token_display'`

- [ ] **Step 3: Write minimal implementation**

Create `framework/token_display.py`:

```python
"""
Process-level toggle for the inline token usage line emitted after each
Claude reply.

Pattern mirrors framework/debug.py: a single module-level boolean with
get/set helpers. Process-global scope is intentional — each agent
(technical_architect/administrative_officer/knowledge_curator) runs its own process, so the toggle naturally scopes to
"the current agent". Per-session granularity would double the surface
area with no observed benefit.

Default: True. The original intent expressed by the user was "every
message shows the token line"; the toggle exists so it can be silenced
when noisy, not because it should start off.
"""


_enabled: bool = True


def is_token_display_enabled() -> bool:
    """Return whether the inline [tokens: …] line should be emitted."""
    return _enabled


def set_token_display(value) -> None:
    """Enable or disable the inline token line (bool-coerced)."""
    global _enabled
    _enabled = bool(value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_token_display.py`
Expected: `✅ token_display OK`

- [ ] **Step 5: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/token_display.py test_token_display.py
git commit -m "feat(token_display): add process-global toggle module

New framework.token_display module mirrors framework.debug pattern:
_enabled flag + is_token_display_enabled() / set_token_display().
Default True. Will back the inline [tokens: ...] line emitted after
each Claude reply."
```

---

## Task 2: Remove auto-compact from `ClaudeSDKNode`

Pure deletion. No new behavior in this task — auto-compact code is torn out; the token-display emission is a later task. After this task the node no longer sends `/compact` on its own, and context tracking switches to a single repurposed local (`_last_msg_usage`, done in Task 3).

**Files:**
- Modify: `framework/nodes/llm/claude.py`

- [ ] **Step 1: Delete the `_COMPACT_THRESHOLD` class constant**

In `class ClaudeSDKNode(AgentNode):` (around line 57), remove the line:
```python
    _COMPACT_THRESHOLD = 100_000  # tokens
```

- [ ] **Step 2: Delete the `_last_context_size` instance field**

In `ClaudeSDKNode.__init__`, remove the two lines:
```python
        # session_id → last context size (tokens), used for auto-compact check
        self._last_context_size: dict[str, int] = {}
```

Also remove the class-level docstring paragraph about auto-compact (the block that reads *"Auto-compact: monitors the context size after each call…"* inside the class docstring, roughly 2 lines). Leave the rest of the docstring intact.

- [ ] **Step 3: Delete `_last_context_size` writes inside `_run_once`**

Inside `_run_once`'s `ResultMessage` branch, remove these two lines:
```python
                    # Use the last message_start context as the true session size
                    if _new_sid and _last_msg_ctx > 0:
                        self._last_context_size[_new_sid] = _last_msg_ctx
```

Leave `update_token_stats(msg.usage)` and the `if msg.result: _result = msg.result.strip()` lines intact. Leave the `_last_msg_ctx` local for now (Task 3 will repurpose it).

- [ ] **Step 4: Delete the auto-compact block before `_run_once` is invoked**

Remove the entire block that begins with the comment `# ── Auto-compact: when context exceeds threshold, let CLI compress session history first ──────` and includes the `if session_id and self._last_context_size.get(session_id, 0) > self._COMPACT_THRESHOLD:` guard plus everything inside its body (including the `try/except` that sends `/compact` and the `_last_context_size.clear()` call). After deletion, the line immediately before `try: result_text, new_session_id, is_error = await _run_once(session_id, prompt)` should be the preceding `is_error = False` assignment (with a blank line separator).

- [ ] **Step 5: Verify nothing still references the deleted symbols**

Run: `cd /home/kingy/Foundation/ZenithLoom && python -c "import framework.nodes.llm.claude; print('import ok')"`
Expected: `import ok`

Run grep:
```
```
Use the Grep tool with pattern `_COMPACT_THRESHOLD|_last_context_size|auto-compact` and path `framework/nodes/llm/claude.py`. Expected: no matches.

- [ ] **Step 6: Run existing Claude-adjacent tests**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_cli.py`
Expected: `8/8` or "All tests passed" — same result as the pre-change baseline.

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_e2e_debate.py`
Expected: same baseline output; no regressions.

- [ ] **Step 7: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/nodes/llm/claude.py
git commit -m "refactor(claude): remove buggy auto-compact mechanism

Auto-compact would silently send /compact to Claude CLI when the
tracked context exceeded 100K tokens, but it broke on the first
ResultMessage (often a progress event with a non-final session_id)
and cleared ALL sessions' tracking, not just the compacted one. The
net effect was that after firing, the node appeared to 'forget' the
immediately preceding user turn.

Removes:
  - _COMPACT_THRESHOLD class constant
  - self._last_context_size dict field
  - ResultMessage-branch writes into _last_context_size
  - The pre-call auto-compact block (sdk_query(prompt='/compact'))

Compaction moves to a manual command (follow-up commit).
Token tracking via update_token_stats() is preserved."
```

---

## Task 3: Repurpose `_last_msg_ctx` → `_last_msg_usage` and extend `_run_once` return

This task does **not** emit the token line yet — it only plumbs the `last_msg_usage` dict out of `_run_once` so Task 8 can emit it. Keeping the plumbing change in its own commit makes the subsequent emission diff tiny and reviewable.

**Files:**
- Modify: `framework/nodes/llm/claude.py`

- [ ] **Step 1: Change the local in `_run_once`**

In `_run_once`, replace:
```python
            # Track the context size of the last API call (each tool use loop has its own message_start).
            # Cannot use ResultMessage.usage — that is cumulative and would overstate real context for complex tool use.
            _last_msg_ctx = 0
```
with:
```python
            # Track the full usage dict of the last API call (each tool use loop has its own message_start).
            # Cannot use ResultMessage.usage — that is cumulative and would overstate real context for complex tool use.
            # This dict is surfaced through the return value for the inline token line display at the end of call_llm.
            _last_msg_usage: dict = {}
```

- [ ] **Step 2: Update the `message_start` branch**

Inside the `async for msg` loop, in the `message_start` handling:

Before:
```python
                    if etype == "message_start":
                        # At the start of each API call, extract the input context size for this call
                        _usage = ev.get("message", {}).get("usage", {})
                        if _usage:
                            _last_msg_ctx = (
                                _usage.get("input_tokens", 0)
                                + _usage.get("cache_read_input_tokens", 0)
                                + _usage.get("cache_creation_input_tokens", 0)
                            )
```

After:
```python
                    if etype == "message_start":
                        # At the start of each API call, capture the usage dict for this call (shallow copy).
                        # The last one is retained, reflecting the true context usage of the final API call in this round.
                        _usage = ev.get("message", {}).get("usage", {})
                        if _usage:
                            _last_msg_usage = dict(_usage)
```

- [ ] **Step 3: Change `_run_once` return signature and value**

Change the signature line:
```python
        async def _run_once(sid: str, msg_text: str) -> tuple[str, str, bool]:
```
to:
```python
        async def _run_once(sid: str, msg_text: str) -> tuple[str, str, bool, dict]:
```

Change the return statement at the end of `_run_once`:
```python
            return _result, _new_sid, _is_error
```
to:
```python
            return _result, _new_sid, _is_error, _last_msg_usage
```

Also update the docstring line that reads `"Returns (result_text, new_session_id, is_error)"` to `"Returns (result_text, new_session_id, is_error, last_msg_usage)"`.

- [ ] **Step 4: Update the initial `_run_once` call site in `call_llm`**

In `call_llm`, initialize a new local alongside the existing ones:
```python
        result_text = ""
        new_session_id = ""
        is_error = False
        last_msg_usage: dict = {}
```

Then update the first call to `_run_once`:
```python
        try:
            result_text, new_session_id, is_error = await _run_once(session_id, prompt)
```
to:
```python
        try:
            result_text, new_session_id, is_error, last_msg_usage = await _run_once(session_id, prompt)
```

- [ ] **Step 5: Update the resume-retry `_run_once` call site**

In the `except Exception as e:` branch that handles `_is_cli_exit_error(e) and session_id`, update the retry call:

```python
                try:
                    result_text, new_session_id, is_error = await _run_once("", prompt)
                except Exception as retry_err:
```
to:
```python
                try:
                    result_text, new_session_id, is_error, last_msg_usage = await _run_once("", prompt)
                except Exception as retry_err:
```

Inside the fallback where retry also fails (the block that sets `result_text = f"[Claude temporarily unavailable] {retry_err}"`), add one line resetting `last_msg_usage = {}` right after `is_error = True` to keep the local consistent (prevents stale data from a partial first attempt leaking into a later token-line emission).

- [ ] **Step 6: Import-level smoke test**

Run: `cd /home/kingy/Foundation/ZenithLoom && python -c "import framework.nodes.llm.claude; print('import ok')"`
Expected: `import ok`

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_cli.py`
Expected: same baseline pass count as before.

- [ ] **Step 7: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/nodes/llm/claude.py
git commit -m "refactor(claude): extend _run_once return with last_msg_usage

Replaces the _last_msg_ctx int local inside _run_once with a full
_last_msg_usage dict captured from the last message_start event
(shallow-copied). Adds it as a 4th element in _run_once's return
tuple and unpacks at both call sites (initial call + resume retry).

No behavior change yet — this is pure plumbing for the upcoming
inline token-usage line in call_llm. Using last message_start avoids
the tool-use inflation that affects ResultMessage.usage."
```

---

## Task 4: Add `ClaudeSDKNode.compact_session()` method

**Files:**
- Modify: `framework/nodes/llm/claude.py`

- [ ] **Step 1: Add the method**

Inside `ClaudeSDKNode` (after `call_llm` / the `call_claude = call_llm` alias, before `get_recent_history`), add:

```python
    async def compact_session(self, session_id: str) -> tuple[str, str]:
        """
        Send /compact to the specified Claude session, letting the CLI compress the conversation history.

        Returns (status_message, new_session_id).
          - status_message: human-readable status, e.g. "ok: sid abc12345 → def67890"
                            or "error: <exception msg>" / "warning: no session_id returned"
          - new_session_id: the new session_id after compaction; on failure, the original sid is preserved

        Fixes two bugs from the original auto-compact:
          1. Does not break on the first ResultMessage — iterates the iterator to completion,
             taking the last non-empty session_id seen as the new sid
          2. Does not clear any global tracking fields (those fields were already removed when auto-compact was deleted)

        Semantic note: Claude CLI's /compact is a **lossy** operation. It replaces conversation history
        with an LLM-generated summary; previous details (specific file paths, tool-call outputs, etc.) will be lost.
        Callers invoking this method manually should make this clear to the user in the UI.
        """
        if not session_id:
            return ("warning: empty session_id, nothing to compact", session_id)

        # Reuse the _make_options semantics from call_llm: permission_mode / cwd / settings etc.
        # Since _make_options is a closure inside call_llm, inline a minimal version here
        # to avoid the extra refactoring scope that extracting it to a method would entail.
        _sp = self.node_config.get("system_prompt") or self.system_prompt or None
        _allowed = self.node_config.get("tools") or self.config.tools
        _disallowed = self._get_disallowed_tools()
        _model = (
            self.node_config.get("model")
            or self.node_config.get("claude_model")
            or None
        )
        _cwd = self.config.workspace or None
        options = ClaudeAgentOptions(
            system_prompt=_sp,
            cwd=_cwd,
            allowed_tools=_allowed,
            disallowed_tools=_disallowed,
            permission_mode=self._permission_mode,
            resume=session_id,
            model=_model,
            env={"CLAUDECODE": "", "CLAUDE_CODE_SESSION": "", "CLAUDE_AGENT_SDK": "1"},
            include_partial_messages=False,
        )

        new_sid = session_id
        saw_result = False
        try:
            async for msg in sdk_query(prompt="/compact", options=options):
                if isinstance(msg, ResultMessage):
                    saw_result = True
                    if msg.session_id:
                        new_sid = msg.session_id
        except Exception as e:
            logger.warning(f"[claude] compact_session failed sid={session_id[:8]}: {e}")
            return (f"error: {e}", session_id)

        if not saw_result:
            return ("warning: no ResultMessage from /compact", session_id)

        logger.info(
            f"[claude] compact_session ok: sid={session_id[:8]} → {new_sid[:8]}"
        )
        return (
            f"ok: {session_id[:8]} → {new_sid[:8]}",
            new_sid,
        )
```

- [ ] **Step 2: Import-level smoke test**

Run: `cd /home/kingy/Foundation/ZenithLoom && python -c "from framework.nodes.llm.claude import ClaudeSDKNode; assert hasattr(ClaudeSDKNode, 'compact_session'); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/nodes/llm/claude.py
git commit -m "feat(claude): add compact_session() method

New async method sends /compact to the Claude CLI for a given
session_id and returns (status_message, new_session_id). Fixes two
bugs from the old auto-compact path:

  1. Iterates the sdk_query stream to completion instead of breaking
     on the first ResultMessage (which was often a progress event
     with a stale session_id).
  2. Never touches any global tracking state on failure — a failed
     compact leaves the caller's session_id intact.

Intended to be called from GraphController.compact_claude_session()
which is wired into the !compact user command in a later commit."
```

---

## Task 5: Attach `llm_node_instances` side-map to the compiled graph

**Files:**
- Modify: `framework/agent_loader.py`

**Why a side-map:** LangGraph wraps each `add_node` callable (via `_wrap_node_for_flow_log`), so the compiled graph does not directly expose the underlying `ClaudeSDKNode` instance. `_build_declarative` already has the instance in scope at node-creation time; we collect those instances into a dict and attach it to the compiled graph as a plain attribute for the controller to read.

- [ ] **Step 1: Initialize the collector near the top of `_build_declarative`**

Find the start of `_build_declarative` where `builder` is created (around line 760-764). Right after the `builder = StateGraph(...)` assignment, add:

```python
    # Collect LLM node instances keyed by node_id so GraphController can
    # reach them for /compact and other out-of-graph operations.
    _llm_node_instances: dict[str, object] = {}
```

- [ ] **Step 2: Populate when an LLM node is constructed**

Find the block that constructs an LLM node and calls `builder.add_node(node_id, _wrap_node_for_flow_log(node_id, node_instance))` (around line 989). Add one line directly after `node_instance = factory(config, effective_def)`:

```python
            node_instance = factory(config, effective_def)
            # Remember the raw instance so controller-level ops (e.g. /compact)
            # can reach past the LangGraph wrapping.
            _llm_node_instances[node_id] = node_instance
            builder.add_node(node_id, _wrap_node_for_flow_log(node_id, node_instance))
```

- [ ] **Step 3: Attach after compile**

Find where `_build_declarative` returns the compiled graph. Search for `return builder.compile` or `compiled = builder.compile`. Wherever the compile call produces the final object that is returned, attach the side-map before returning:

```python
    compiled = builder.compile(checkpointer=checkpointer)
    compiled._llm_node_instances = _llm_node_instances
    return compiled
```

If the current code has multiple return points or inlines the compile, pick one canonical exit and set the attribute there. If `builder.compile(...)` is the only exit, the edit is straightforward.

- [ ] **Step 4: Handle the Priority 3 (non-declarative) path**

`AgentLoader.build_graph` has a Priority 3 fallback that calls `build_agent_graph(...)` for non-declarative agents. That path lives in `framework/graph.py` and produces its own compiled graph with one `ClaudeSDKNode` — the instance bound to the local `agent_node` variable right above the call. After the `return await build_agent_graph(...)` call is made, we need the same attachment.

Refactor the end of `AgentLoader.build_graph` (around lines 273-278) from:

```python
        return await build_agent_graph(
            config=config,
            agent_node=agent_node,
            checkpointer=checkpointer,
            spec=spec,
        )
```

to:

```python
        compiled = await build_agent_graph(
            config=config,
            agent_node=agent_node,
            checkpointer=checkpointer,
            spec=spec,
        )
        # Expose the ClaudeSDKNode so controller ops (!compact /compact)
        # can reach it past the LangGraph wrapping. Priority 3 path has
        # exactly one Claude node; key it by its session_key so it lines
        # up with node_sessions lookups.
        compiled._llm_node_instances = {
            getattr(agent_node, "_session_key", "claude_main"): agent_node,
        }
        return compiled
```

Also verify the Priority 1 (custom `graph.py`) path (around line 195). That path returns whatever the user's custom `build_graph` returns; those graphs may or may not expose Claude nodes. No change is required — if `_llm_node_instances` is missing, the controller falls back to an empty dict (see Task 6 Step 3). Add a one-line comment above the `return await mod.build_graph(...)` call noting this:

```python
            if hasattr(mod, "build_graph"):
                logger.info(f"[agent_loader] using custom graph.py for {self.name!r}")
                # Custom graphs may not expose _llm_node_instances;
                # controller.compact_claude_session() tolerates absence.
                return await mod.build_graph(self, checkpointer)
```

- [ ] **Step 5: Smoke test**

Run: `cd /home/kingy/Foundation/ZenithLoom && python -c "
import asyncio
from framework.agent_loader import EntityLoader
from pathlib import Path

async def main():
    loader = EntityLoader(Path('blueprints/role_agents/technical_architect'))
    graph = await loader.build_graph(checkpointer=None)
    inst = getattr(graph, '_llm_node_instances', None)
    print('instances:', list(inst.keys()) if inst else 'MISSING')

asyncio.run(main())
"`
Expected: a non-empty list of node_ids, one of which corresponds to the main Claude node (e.g. `['claude_main', ...]` or similar — exact names depend on technical_architect's blueprint).

- [ ] **Step 6: Run existing tests**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_cli.py`
Expected: unchanged baseline pass.

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_e2e_debate.py`
Expected: unchanged baseline pass.

- [ ] **Step 7: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/agent_loader.py
git commit -m "feat(agent_loader): attach _llm_node_instances to compiled graph

LangGraph wraps each added node callable, hiding the underlying
instance. For controller-level ops that need to reach the raw node
(e.g. ClaudeSDKNode.compact_session()), we now collect instances
during _build_declarative and attach them to the compiled graph as
_llm_node_instances: dict[node_id, instance].

Priority 3 (non-declarative) path also attaches the single Claude
node keyed by its session_key. Priority 1 (custom graph.py) path is
untouched; GraphController tolerates missing attribute."
```

---

## Task 6: Add `GraphController.compact_claude_session()`

**Files:**
- Modify: `framework/graph_controller.py`

- [ ] **Step 1: Add the method**

Find `compact_checkpoint` (around line 190). Immediately after it, add:

```python
    async def compact_claude_session(self, thread_id: str) -> str:
        """
        Send /compact to all active ClaudeSDKNode sessions under thread_id.

        Process:
          1. Get node instances from the compiled graph's _llm_node_instances side-map
          2. Read each node's current session_id from state["node_sessions"]
          3. Call ClaudeSDKNode.compact_session(sid) for each match and get new_sid
          4. If any sid changed, write back to checkpoint via aupdate_state

        Returns a human-readable summary string.
        This method never raises — each node's failure is converted to a status string.
        """
        # Lazy import to avoid circular dependency between graph_controller and nodes.llm.claude
        from framework.nodes.llm.claude import ClaudeSDKNode

        instances: dict = getattr(self._graph, "_llm_node_instances", {}) or {}
        claude_nodes = {
            key: inst
            for key, inst in instances.items()
            if isinstance(inst, ClaudeSDKNode)
        }
        if not claude_nodes:
            return "No Claude session to compact (no ClaudeSDKNode instances registered in this graph)"

        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await self._graph.aget_state(config)
        except Exception as e:
            return f"❌ Failed to read state: {e}"

        node_sessions: dict = (snapshot.values or {}).get("node_sessions", {}) or {}
        if not node_sessions:
            return "No active Claude session (state.node_sessions is empty)"

        results: list[str] = []
        updated_sessions: dict[str, str] = {}
        any_changed = False

        for node_key, node in claude_nodes.items():
            # Node instances have a _session_key attribute pointing to their key in node_sessions.
            # Fall back to node_key (the id at registration time) if not present.
            lookup_key = getattr(node, "_session_key", None) or node_key
            sid = node_sessions.get(lookup_key, "")
            if not sid:
                results.append(f"{lookup_key}: no active session, skipping")
                continue

            status, new_sid = await node.compact_session(sid)
            results.append(f"{lookup_key}: {status}")
            if new_sid and new_sid != sid:
                updated_sessions[lookup_key] = new_sid
                any_changed = True

        if any_changed:
            merged = {**node_sessions, **updated_sessions}
            try:
                await self._graph.aupdate_state(config, {"node_sessions": merged})
            except Exception as e:
                results.append(f"⚠️ aupdate_state failed to write back new session_ids: {e}")
            else:
                # Sync sessions.json to keep consistent with sync_node_sessions behavior after run()
                self.sync_node_sessions({"node_sessions": merged}, thread_id=thread_id)

        logger.info(
            f"[controller] compact_claude_session thread={thread_id!r} "
            f"nodes={len(claude_nodes)} changed={any_changed}"
        )
        return "; ".join(results)
```

- [ ] **Step 2: Smoke test**

Run: `cd /home/kingy/Foundation/ZenithLoom && python -c "
from framework.graph_controller import GraphController
import inspect
assert inspect.iscoroutinefunction(GraphController.compact_claude_session)
print('ok')
"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/graph_controller.py
git commit -m "feat(controller): add compact_claude_session()

New async method on GraphController iterates the compiled graph's
_llm_node_instances side-map, calls ClaudeSDKNode.compact_session()
on each entry that has an active session in state.node_sessions,
and writes any changed session_ids back via aupdate_state plus
sync_node_sessions (for sessions.json parity).

All failures are caught and turned into status strings — the method
never raises. Returns a human-readable '；'-joined summary suitable
for direct display in the !compact reply."
```

---

## Task 7: Extend `!compact` handler + test

**Files:**
- Modify: `framework/base_interface.py`
- Modify: `test_commands.py`

- [ ] **Step 1: Update the existing `!compact` test**

In `test_commands.py`, find `test_discord_compact_reset` (around line 484). Adjust the `!compact` assertion block to tolerate the richer reply:

Before:
```python
        # !compact (default keep=20)
        reply = await iface.handle_command("!compact", "")
        assert "Compact" in reply
        print(f"   !compact: {reply}")
```

After:
```python
        # !compact (default keep=20) — now reports both checkpoint DB and Claude session
        reply = await iface.handle_command("!compact", "")
        assert "Compact" in reply
        assert "checkpoint DB" in reply
        assert "Claude session" in reply
        print(f"   !compact:\n{reply}")
```

The test's `loader` is a `_setup_bot`-backed fixture whose graph has no real ClaudeSDKNode wired through a live Claude CLI; the new `compact_claude_session()` should still run (returning `"No Claude session to compact…"` or similar) without raising. If running the test reveals that the fixture graph's compiled object lacks `_llm_node_instances`, the `getattr(..., {})` fallback in Task 6 Step 1 keeps it safe.

- [ ] **Step 2: Run the test to verify it fails on the new assertions**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_commands.py` (or narrow to the compact test if runner supports it)
Expected: `AssertionError` on `"checkpoint DB" in reply` (because the handler is still the old version).

- [ ] **Step 3: Update the `!compact` handler**

In `framework/base_interface.py` (around line 447), replace:

```python
        if cmd == "!compact":
            try:
                keep = int(arg) if arg else 20
            except ValueError:
                keep = 20
            thread_id = self._resolve_thread_id()
            deleted   = await controller.compact_checkpoint(thread_id, keep_last=keep)
            return f"Compact done: deleted {deleted} old records, kept the last {keep}."
```

with:

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
                claude_msg = f"❌ Call failed: {e}"
            return (
                "Compact done:\n"
                f"  checkpoint DB : deleted {deleted} old records, kept the last {keep}\n"
                f"  Claude session: {claude_msg}\n"
                "  (Note: /compact is a lossy summary — old details may no longer be recallable)"
            )
```

- [ ] **Step 4: Re-run the test**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_commands.py`
Expected: the compact/reset test passes. No regressions in other `test_commands.py` tests.

- [ ] **Step 5: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/base_interface.py test_commands.py
git commit -m "feat(base_interface): !compact also sends /compact to Claude

Extends the !compact command to invoke
controller.compact_claude_session() in addition to the existing
checkpoint-DB compaction. Reply now breaks out both parts and adds
a one-line warning that Claude's /compact is a lossy summary.

test_commands.test_discord_compact_reset updated to assert the new
reply structure."
```

---

## Task 8: Emit inline token line in `call_llm`

**Files:**
- Modify: `framework/nodes/llm/claude.py`

- [ ] **Step 1: Add the import**

Near the top of `framework/nodes/llm/claude.py`, next to the existing `from framework.debug import is_debug`, add:

```python
from framework.token_display import is_token_display_enabled
```

- [ ] **Step 2: Emit the token line at the end of `call_llm`**

Find the final lines of `call_llm`:

```python
        new_sid_short = new_session_id[:8] if new_session_id else "new"
        logger.info(f"[claude] done sid={new_sid_short} output_len={len(result_text)}")
        if is_debug():
            logger.debug(f"[claude] output_preview={result_text[:200]!r}")
        return result_text, new_session_id
```

Insert the token-line emission block **before** the `return`, after the debug log:

```python
        new_sid_short = new_session_id[:8] if new_session_id else "new"
        logger.info(f"[claude] done sid={new_sid_short} output_len={len(result_text)}")
        if is_debug():
            logger.debug(f"[claude] output_preview={result_text[:200]!r}")

        # Inline token usage line — last message_start.usage, NOT ResultMessage
        # (ResultMessage.usage is cumulative across tool-use sub-calls and
        # overstates the real context window on tool-heavy turns).
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

        return result_text, new_session_id
```

- [ ] **Step 3: Smoke test**

Run: `cd /home/kingy/Foundation/ZenithLoom && python -c "import framework.nodes.llm.claude; print('import ok')"`
Expected: `import ok`

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_cli.py`
Expected: baseline pass. The token line is suppressed by the test fixtures (no stream callback installed), so existing tests should be unaffected.

- [ ] **Step 4: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/nodes/llm/claude.py
git commit -m "feat(claude): emit inline [tokens: ...] line after each reply

After a successful call_llm, when token_display is enabled and a
stream callback is installed, emit a single-line summary pulled from
the last message_start.usage dict:

  [tokens: ctx=N in=N out=N cache_read=N]

Uses message_start (not ResultMessage.usage) to avoid tool-use
inflation. Suppressed when:
  - token_display is toggled off via !tokens off
  - No _stream_cb is bound (e.g. channel_send_final mode in debate
    subgraphs, where the downstream orchestrator owns output)
  - last_msg_usage is empty (defensive)

The !tokens cumulative stats command still reflects full billing
via ResultMessage-fed update_token_stats()."
```

---

## Task 9: Extend `!tokens` handler for on/off/status + test

**Files:**
- Modify: `framework/base_interface.py`
- Modify: `test_commands.py`

- [ ] **Step 1: Add the failing test**

In `test_commands.py`, find `test_discord_tokens` (around line 423). Extend it with toggle coverage. Replace the function body (keeping its signature, docstring if any, and the existing assertions where appropriate):

```python
async def test_discord_tokens():
    print("--- Discord !tokens ---")
    import interfaces.discord_bot as bot
    import framework.token_tracker as tt
    from framework import token_display

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _, loader = _setup_bot(sm)

        # Inject known stats
        tt._token_stats.update({
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 100,
            "calls": 5,
        })

        iface = bot._DiscordInterface(loader)

        # 1) no-arg — cumulative stats + toggle state line
        reply = await iface.handle_command("!tokens", "")
        assert "1,000" in reply or "1000" in reply
        assert "500" in reply
        assert "inline display" in reply.lower()
        print(f"   !tokens preview:\n{reply[:300]}")

        # 2) reset — preserved behavior
        reply2 = await iface.handle_command("!tokens", "reset")
        assert "reset" in reply2.lower()
        assert tt._token_stats["input_tokens"] == 0
        print(f"   !tokens reset: {reply2}")

        # 3) off — disable inline display
        reply3 = await iface.handle_command("!tokens", "off")
        assert "off" in reply3.lower()
        assert token_display.is_token_display_enabled() is False
        print(f"   !tokens off: {reply3}")

        # 4) status — report current state
        reply4 = await iface.handle_command("!tokens", "status")
        assert "off" in reply4.lower()
        print(f"   !tokens status (off): {reply4}")

        # 5) on — re-enable
        reply5 = await iface.handle_command("!tokens", "on")
        assert "on" in reply5.lower()
        assert token_display.is_token_display_enabled() is True
        print(f"   !tokens on: {reply5}")

        # 6) unknown arg — usage hint
        reply6 = await iface.handle_command("!tokens", "gibberish")
        assert "usage" in reply6.lower() or "on|off|status" in reply6
        print(f"   !tokens gibberish: {reply6}")

    # Restore default for test isolation
    token_display.set_token_display(True)
    print("✅ Discord !tokens OK\n")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_commands.py`
Expected: `AssertionError` on the inline display assertion in reply (the handler does not yet append the toggle-state line).

- [ ] **Step 3: Update the `!tokens` handler**

In `framework/base_interface.py`, find the `if cmd == "!tokens":` block (around line 360). Replace it with:

```python
        # ── Token stats ────────────────────────────────────────────────────
        if cmd == "!tokens":
            from framework.token_tracker import get_token_stats, reset_token_stats
            from framework.token_display import (
                is_token_display_enabled,
                set_token_display,
            )

            arg_norm = (arg or "").strip().lower()

            if arg_norm == "reset":
                reset_token_stats()
                return "Token counters reset."

            if arg_norm == "on":
                set_token_display(True)
                return "✅ Token inline display: on"

            if arg_norm == "off":
                set_token_display(False)
                return "✅ Token inline display: off"

            if arg_norm == "status":
                state = "on" if is_token_display_enabled() else "off"
                return f"Token inline display: {state}"

            if arg_norm and arg_norm not in {"", "reset", "on", "off", "status"}:
                return "Usage: !tokens [on|off|status|reset]"

            # no-arg: cumulative stats plus toggle state
            s = get_token_stats()
            inp = s["input_tokens"]
            out = s["output_tokens"]
            cr = s["cache_read_input_tokens"]
            cc = s["cache_creation_input_tokens"]
            calls = s["calls"]
            cost_usd = (inp * 3 + out * 15 + cr * 0.3 + cc * 3.75) / 1_000_000
            saved_usd = cr * (3 - 0.3) / 1_000_000
            state = "on" if is_token_display_enabled() else "off"
            return (
                f"Calls         : {calls}\n"
                f"Input tokens  : {inp:,}\n"
                f"Output tokens : {out:,}\n"
                f"Cache read    : {cr:,}  (saved ${saved_usd:.4f})\n"
                f"Cache create  : {cc:,}\n"
                f"Estimated cost: ~${cost_usd:.4f} USD\n"
                f"Inline display: {state}"
            )
```

- [ ] **Step 4: Re-run the test**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_commands.py`
Expected: `test_discord_tokens` passes. No regressions.

- [ ] **Step 5: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/base_interface.py test_commands.py
git commit -m "feat(base_interface): extend !tokens with on/off/status toggle

!tokens now recognizes:
  - no-arg    → cumulative stats + current toggle state line
  - reset     → reset cumulative counters (existing behavior)
  - on/off    → enable/disable the inline [tokens: ...] line in
                Claude replies (via framework.token_display)
  - status    → report current toggle state only
  - other arg → usage hint

test_discord_tokens updated to cover every arg form and restore the
default toggle state for cross-test isolation."
```

---

## Task 10: Update `command_registry.py` usage hint

**Files:**
- Modify: `framework/command_registry.py`

- [ ] **Step 1: Edit the registry entry**

Replace the line:
```python
_r("!tokens",     "View token usage stats",                    ALL,  "[reset]")
```
with:
```python
_r("!tokens",     "View token usage stats / toggle inline display",        ALL,  "[on|off|status|reset]")
```

- [ ] **Step 2: Smoke test**

Run: `cd /home/kingy/Foundation/ZenithLoom && python -c "
from framework.command_registry import REGISTRY
entry = REGISTRY['!tokens']
assert 'on|off' in entry.usage
assert 'inline display' in entry.description
print('ok:', entry.usage, '—', entry.description)
"`
Expected: `ok: [on|off|status|reset] — View token usage stats / toggle inline display`

Also verify `!help` picks it up by running (if it is quick to invoke) a mock help call:

```
```
Use the Grep tool with pattern `!tokens` and path `framework/command_registry.py`. Expected: a single matching line containing `[on|off|status|reset]`.

- [ ] **Step 3: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add framework/command_registry.py
git commit -m "docs(command_registry): update !tokens usage hint

Reflect the new on|off|status|reset argument surface in the
REGISTRY entry so !help renders an accurate hint."
```

---

## Task 11: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Repo-wide grep for leftover auto-compact references**

Use the Grep tool with pattern `_COMPACT_THRESHOLD|_last_context_size|auto-compact|auto_compact` and path `framework/`. Expected: **zero** matches.

Use the Grep tool with pattern `_COMPACT_THRESHOLD|_last_context_size` and path `ZenithLoom/`. Expected: zero matches outside `docs/` (the spec/plan/memory files may still mention the deleted names for historical context — those are acceptable).

- [ ] **Step 2: Full test suite**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_cli.py && python test_e2e_debate.py && python test_commands.py && python test_token_display.py`
Expected: every script reports its own success (e.g. `8/8`, `"All tests passed"`, or the `print("✅ ... OK")` lines each test uses). No tracebacks.

Note: `test_commands.py` has a pre-existing known failure unrelated to this work (`bot._channel_locks` attribute absence; documented in the project memory). That failure predates this plan and is **not** a regression — if it appears, confirm it matches the pre-existing signature and move on.

- [ ] Step 3: Live smoke test with technical_architect

This step requires a real Claude CLI and should only be run if available.

```bash
cd /home/kingy/Foundation
python -m ZenithLoom.interfaces.cli --agent technical_architect
```

Inside the CLI:

```
Hello, please briefly introduce yourself.
```

Expected: The technical_architect replies, and the reply **ends with** a line matching the shape `[tokens: ctx=N in=N out=N cache_read=N]` (numbers with commas).

Then run:

```
!tokens off
```

Expected reply: `✅ Token inline display: off`

Send another short prompt and confirm the token line is **absent** from the reply.

Then:

```
!tokens on
!compact
```

Expected reply of `!compact`: a two-section message mentioning both `checkpoint DB` and `Claude session`, plus the lossy-summary warning. Send another prompt and confirm the agent still responds coherently and the token line reappears.

Exit with `!quit` (or equivalent).

- [ ] **Step 4: Commit verification notes (optional)**

If the live smoke test produced any notable observations worth recording, add them to the memory file `operations_compact_behavior.md` (or similar). Otherwise no commit needed.

---

## Self-Review Notes

This section is for the plan author, not for the implementer. It verifies the plan against the spec before handoff.

**Spec coverage:**
- Spec Block 1 (Remove Auto-Compact) → Task 2
- Spec Block 2A (`compact_session()`) → Task 4
- Spec Block 2B (`compact_claude_session()`) → Task 5 (side-map) + Task 6
- Spec Block 2C (`!compact` handler) → Task 7
- Spec Block 3A (`token_display.py`) → Task 1
- Spec Block 3B (inline token line) → Task 3 (plumbing) + Task 8 (emission)
- Spec Block 3C (`!tokens` toggle) → Task 9 + Task 10
- Spec "Testing" section → Task 11

All spec sections have a corresponding task.

**Type consistency:**
- `compact_session()` returns `tuple[str, str]` — Task 4 defines it; Task 6 unpacks as `status, new_sid`. ✅
- `_run_once` returns 4-tuple `(text, sid, is_error, last_msg_usage)` — Task 3 defines; Task 8 consumes `last_msg_usage` locally in `call_llm`. ✅
- `_llm_node_instances: dict[str, object]` — Task 5 defines; Task 6 iterates with `isinstance(inst, ClaudeSDKNode)` filter. ✅
- `compact_claude_session()` returns `str` — Task 6 defines; Task 7 consumes via `claude_msg = await ...`. ✅

**Placeholders:** None. Every code block contains real code. Every assertion contains a literal expectation.
