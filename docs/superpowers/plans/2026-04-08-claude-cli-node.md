# ClaudeCLINode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ClaudeCLINode` — a subprocess-based Claude node using `claude -p --output-format stream-json`, and re-point the `CLAUDE_CLI` registry entry to it.

**Architecture:** `ClaudeCLINode` inherits `LlmNode`, implements `call_llm()` only (reuses base `__call__`). Streams stdout line-by-line parsing JSON events for real-time streaming callbacks. Existing entities migrate from `CLAUDE_CLI` to `CLAUDE_SDK` to preserve current behavior.

**Tech Stack:** Python 3.11+, asyncio subprocess, JSON line parsing, pytest

**Spec:** `docs/vault/architecture/claude-cli-node-design.md`

---

### Task 1: Add ClaudeCLINode to claude.py

**Files:**
- Modify: `framework/nodes/llm/claude.py` (append after ClaudeSDKNode class, add imports)

- [ ] **Step 1: Add missing imports**

Add `import asyncio` and `import os` to the import block at the top of `claude.py` (after `import json`).

- [ ] **Step 2: Add the ClaudeCLINode class**

Append after the `ClaudeNode = ClaudeSDKNode` alias. The class:
- Inherits `AgentNode` (alias for `LlmNode`)
- Implements `call_llm()` using `asyncio.create_subprocess_exec`
- `_build_cmd()` maps SDK options to CLI flags
- `_run_cli()` streams stdout, parses JSON events, feeds `_stream_cb`
- `call_llm()` handles resume retry (same pattern as `ClaudeSDKNode`)
- Dynamic timeout: `min(600, max(120, prompt_len // 200))`
- Env: `CLAUDE_AGENT_SDK=1` to suppress hook sounds

Full class implementation is in the spec at `docs/vault/architecture/claude-cli-node-design.md`.

- [ ] **Step 3: Run existing tests to verify no breakage**

Run: `cd /home/kingy/Foundation/ZenithLoom && python -m pytest tests/ -x -q 2>&1 | tail -20`
Expected: All existing tests pass

- [ ] **Step 4: Commit**

```bash
git add framework/nodes/llm/claude.py
git commit -m "feat: add ClaudeCLINode — subprocess-based Claude CLI node"
```

---

### Task 2: Update builtins.py registry

**Files:**
- Modify: `framework/builtins.py:45-54` (CLAUDE_CLI factory + docstring)

- [ ] **Step 1: Change CLAUDE_CLI registration to ClaudeCLINode**

Change the import in the `CLAUDE_CLI` factory from `ClaudeSDKNode` to `ClaudeCLINode`. Keep `CLAUDE_SDK` unchanged.

- [ ] **Step 2: Update the module docstring**

Update line 8 to reflect the new split:
- `CLAUDE_CLI` -> `ClaudeCLINode` (CLI subprocess)
- `CLAUDE_SDK` -> `ClaudeSDKNode` (Agent SDK)

- [ ] **Step 3: Commit**

```bash
git add framework/builtins.py
git commit -m "feat: CLAUDE_CLI registry -> ClaudeCLINode, CLAUDE_SDK stays as ClaudeSDKNode"
```

---

### Task 3: Migrate all entity.json files from CLAUDE_CLI to CLAUDE_SDK

**Files:**
- Modify: `blueprints/role_agents/technical_architect/entity.json` (1 occurrence)
- Modify: `blueprints/functional_graphs/debate_claude_first/entity.json` (3 occurrences)
- Modify: `blueprints/functional_graphs/debate_gemini_first/entity.json` (2 occurrences)
- Modify: `blueprints/functional_graphs/apex_coder/entity.json` (1 occurrence)
- Modify: `blueprints/functional_graphs/tool_discovery/entity.json` (3 occurrences)

- [ ] **Step 1: Replace all `"type": "CLAUDE_CLI"` with `"type": "CLAUDE_SDK"` in entity.json files**

Use `replace_all` on each file. Total: 10 occurrences across 5 files.

Note: test files that reference `CLAUDE_CLI` in mock configs are cosmetic and do not need changing.

- [ ] **Step 2: Verify migration complete**

Run: `grep -r '"type": "CLAUDE_CLI"' blueprints/`
Expected: No output

- [ ] **Step 3: Commit**

```bash
git add blueprints/
git commit -m "migrate: entity.json CLAUDE_CLI -> CLAUDE_SDK (preserve existing SDK behavior)"
```

---

### Task 4: Update README

**Files:**
- Modify: `README.md` (node type table, around line 107)

- [ ] **Step 1: Update the node type table**

Replace the single `CLAUDE_CLI` row with two rows:
- `CLAUDE_CLI` -> `ClaudeCLINode` (CLI subprocess)
- `CLAUDE_SDK` -> `ClaudeSDKNode` (Agent SDK)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update node type table for CLAUDE_CLI / CLAUDE_SDK split"
```

---

### Task 5: Integration test

**Files:**
- Modify: `test_claude_node.py` (add CLI node tests)

- [ ] **Step 1: Add ClaudeCLINode test functions**

Add three tests:
1. `test_cli_node_call_llm` — basic call, verify non-empty response and session_id
2. `test_cli_node_resume` — create session, resume it, verify continuity
3. `test_cli_node_streaming` — set `_stream_cb`, verify chunks received during call

- [ ] **Step 2: Add new tests to `run_all()`**

- [ ] **Step 3: Run integration tests**

Run: `cd /home/kingy/Foundation/ZenithLoom && python test_claude_node.py`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add test_claude_node.py
git commit -m "test: add ClaudeCLINode integration tests"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run unit tests**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 2: Verify registry**

Run: `python -c "from framework.builtins import *; from framework.registry import get_node_factory; print('CLAUDE_CLI:', get_node_factory('CLAUDE_CLI')); print('CLAUDE_SDK:', get_node_factory('CLAUDE_SDK'))"`
Expected: Different factories

- [ ] **Step 3: Verify no CLAUDE_CLI in entity files**

Run: `grep -r '"CLAUDE_CLI"' blueprints/ --include='*.json'`
Expected: No output
