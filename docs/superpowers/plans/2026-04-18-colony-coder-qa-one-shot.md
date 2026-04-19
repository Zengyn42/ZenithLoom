# Colony Coder QA One-Shot: E2E Tests Generated Once, Validated Many

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Colony Coder's QA loop so E2E tests are generated once by Claude, then reused as a fixed validation gate — eliminating repeated Claude calls and giving Qwen clear acceptance criteria upfront.

**Architecture:** The QA subgraph's `inject_e2e_context` node becomes a routing gate: first entry routes to `generate_e2e` (Claude writes tests), subsequent entries skip directly to `run_e2e`. The executor's `inject_task_context` is enhanced to inject the Planner's acceptance criteria into Qwen's prompt. The `e2e_route` node no longer routes back to `generate_e2e` — all retries go through the parent graph's execute loop. Rescue path is preserved for escalation.

**Tech Stack:** Python 3.12, LangGraph (ZenithLoom framework), Ollama (qwen3.5:27b), Claude SDK

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `blueprints/functional_graphs/colony_coder/state.py` | Modify | Add `e2e_tests_generated: bool` field |
| `blueprints/functional_graphs/colony_coder_qa/entity.json` | Modify | Change `inject_e2e_context` to routing node, remove `e2e_route → generate_e2e` edge |
| `blueprints/functional_graphs/colony_coder_qa/validators.py` | Modify | Add routing logic to `inject_e2e_context`, set flag after generation, simplify `e2e_route` |
| `blueprints/functional_graphs/colony_coder_executor/validators.py` | Modify | Inject acceptance criteria into `inject_task_context` prompt |
| `blueprints/functional_graphs/colony_coder_executor/entity.json` | No change | — |
| `blueprints/functional_graphs/colony_coder_planner/entity.json` | No change | Already outputs `e2e_plan` with `acceptance_criteria` |

---

### Task 1: Add `e2e_tests_generated` Flag to State

**Files:**
- Modify: `blueprints/functional_graphs/colony_coder/state.py:72-75`

- [ ] **Step 1: Add the boolean field**

In `state.py`, add `e2e_tests_generated` after the existing `intent_snippet` field (line 75):

```python
    # Context explosion fix: session reset state (2026-04-17)
    prev_test_results: Optional[dict]   # previous iteration's parsed pytest results
    prev_snapshot_hash: Optional[str]   # git commit hash of last good snapshot
    intent_snippet: str                 # first assistant message excerpt for deterministic summary

    # QA one-shot: E2E tests generated once, reused for all validation passes
    e2e_tests_generated: bool           # True after generate_e2e runs successfully
```

- [ ] **Step 2: Verify import**

No new imports needed — `bool` is a builtin. Run a quick syntax check:

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "from blueprints.functional_graphs.colony_coder.state import ColonyCoderState; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add blueprints/functional_graphs/colony_coder/state.py
git commit -m "feat(colony): add e2e_tests_generated flag to ColonyCoderState"
```

---

### Task 2: Restructure QA Subgraph Edges

**Files:**
- Modify: `blueprints/functional_graphs/colony_coder_qa/entity.json`

The key change: `inject_e2e_context` becomes a routing node with two outbound edges (to `generate_e2e` OR directly to `run_e2e`). The `e2e_route → generate_e2e` edge is removed — retries always go through the parent's execute loop.

- [ ] **Step 1: Replace entity.json edges**

Replace the entire `entity.json` with:

```json
{
  "name": "colony_coder_qa",
  "graph": {
    "state_schema": "colony_coder_schema",
    "nodes": [
      {
        "id": "inject_e2e_context",
        "type": "DETERMINISTIC"
      },
      {
        "id": "generate_e2e",
        "type": "CLAUDE_SDK",
        "session_key": "qa_e2e_session",
        "tools": ["Read", "Write", "Bash"],
        "permission_mode": "bypassPermissions",
        "system_prompt": "You are the QA ENGINEER for Colony Coder.\n\nYour job: write automated E2E tests and run them against the finished product.\nYou test from the USER perspective — you do NOT look at source code internals.\n\n## Your Process\n1. Read the e2e_plan carefully — it defines acceptance criteria and test scenarios.\n2. Examine what files exist in the working directory (use Read).\n3. Write E2E test scripts to test_tool/e2e_tests/ directory.\n4. Write a test runner: test_tool/run_e2e.sh\n5. Run: bash test_tool/run_e2e.sh\n6. If YOUR test scripts have bugs, fix them and re-run.\n7. Report results honestly.\n\n## Rules\n- Test the PRODUCT, not the code. You are a user, not a code reviewer.\n- E2E tests should launch the program, feed it input, and verify output/behavior.\n- For interactive programs (curses/pygame), write a headless test harness.\n- Be SPECIFIC about failures: exact error messages, wrong behaviors.\n- Do NOT rubber-stamp. Your job is to FIND problems.\n\n## CRITICAL: Test Speed Constraint\nThe test runner (run_e2e.sh) has a HARD 120-second timeout. If your tests exceed this, they are treated as FAIL regardless of results.\n\n- Each individual test MUST complete in under 10 seconds.\n- Total test suite MUST complete in under 90 seconds.\n- For interactive/curses programs: use short timeouts (2-3 seconds per test), send Q quickly, do NOT wait for natural game-over.\n- Do NOT write tests that wait for long game durations (e.g., 3000 frames at 10fps = 300 seconds).\n- Prefer fast smoke tests: launch -> verify output -> send quit -> check exit code.\n- If you need to test game-over behavior, use a deterministic seed that produces a quick death, or mock/patch the frame limit.\n\n## run_e2e.sh Template\n```bash\n#!/bin/bash\nset -e\ncd \"$(dirname \"$0\")/..\"\npython3 -m pytest test_tool/e2e_tests/ -v 2>&1\n```\n\n## Output Format\nYour response MUST contain exactly one of:\n  E2E_VERDICT: PASS\n  E2E_VERDICT: FAIL\n\nFollowed by detailed evaluation of each acceptance criterion."
      },
      {
        "id": "run_e2e",
        "type": "DETERMINISTIC"
      },
      {
        "id": "e2e_route",
        "type": "DETERMINISTIC"
      },
      {
        "id": "inject_rescue_context",
        "type": "DETERMINISTIC"
      },
      {
        "id": "qa_rescue",
        "type": "CLAUDE_SDK",
        "session_key": "qa_rescue_session",
        "tools": ["Write", "Edit", "Read", "Bash"],
        "permission_mode": "bypassPermissions",
        "system_prompt": "You are ApexCoder — the RESCUE agent for Colony Coder QA.\n\nThe code has failed E2E testing 5 times. The original coder could not fix it.\nYou have FULL AUTHORITY to rewrite any source file.\n\n## Your Process\n1. Read the e2e_plan — it defines what 'correct' means.\n2. Read the E2E test failure output — it shows what's broken.\n3. Read the source code files.\n4. Fix or rewrite the source code to pass E2E tests.\n5. Run the E2E tests yourself: bash test_tool/run_e2e.sh\n6. Verify they pass before finishing.\n\n## Rules\n- Do NOT modify E2E tests. Fix the SOURCE CODE to meet the tests.\n- Write COMPLETE, RUNNABLE code. No stubs.\n- After fixing, actually run the tests to confirm.\n\n## Output Format\nYour response MUST contain exactly one of:\n  RESCUE_VERDICT: PASS\n  RESCUE_VERDICT: FAIL\n\nFollowed by what you changed and test results."
      },
      {
        "id": "run_e2e_rescue",
        "type": "DETERMINISTIC"
      },
      {
        "id": "rescue_route",
        "type": "DETERMINISTIC"
      }
    ],
    "entry": "inject_e2e_context",
    "edges": [
      {"from": "inject_e2e_context",    "to": "generate_e2e",         "type": "routing_to"},
      {"from": "inject_e2e_context",    "to": "run_e2e",              "type": "routing_to"},
      {"from": "generate_e2e",          "to": "run_e2e"},
      {"from": "run_e2e",               "to": "e2e_route"},
      {"from": "e2e_route",             "to": "inject_rescue_context", "type": "routing_to"},
      {"from": "e2e_route",             "to": "__end__",              "type": "routing_to", "id": "no_routing"},
      {"from": "inject_rescue_context", "to": "qa_rescue"},
      {"from": "qa_rescue",             "to": "run_e2e_rescue"},
      {"from": "run_e2e_rescue",        "to": "rescue_route"},
      {"from": "rescue_route",          "to": "qa_rescue",            "type": "routing_to"},
      {"from": "rescue_route",          "to": "__end__",              "type": "routing_to", "id": "no_routing"}
    ]
  }
}
```

**Changes from original:**
1. `inject_e2e_context → generate_e2e` changed from static to `routing_to`
2. Added `inject_e2e_context → run_e2e` as `routing_to` (skip path)
3. Removed `e2e_route → generate_e2e` edge (no more re-generation)

- [ ] **Step 2: Validate JSON syntax**

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "import json; json.load(open('blueprints/functional_graphs/colony_coder_qa/entity.json')); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add blueprints/functional_graphs/colony_coder_qa/entity.json
git commit -m "feat(colony-qa): make inject_e2e_context a routing node, remove re-generation edge"
```

---

### Task 3: Modify QA Validators — One-Shot Generation Logic

**Files:**
- Modify: `blueprints/functional_graphs/colony_coder_qa/validators.py:27-102` (`inject_e2e_context`)
- Modify: `blueprints/functional_graphs/colony_coder_qa/validators.py:175-234` (`e2e_route`)

- [ ] **Step 1: Rewrite `inject_e2e_context` with routing gate**

Replace the entire `inject_e2e_context` function (lines 27-102) with:

```python
def inject_e2e_context(state: dict) -> dict:
    """Route QA entry: generate E2E tests (first time) or skip to run_e2e (subsequent).

    First entry: builds prompt for generate_e2e, sets e2e_tests_generated=True.
    Subsequent entries: routes directly to run_e2e (tests already exist on disk).
    """
    from langchain_core.messages import HumanMessage

    e2e_plan = state.get("e2e_plan") or {}
    qa_plan = state.get("qa_plan", "")  # fallback to legacy field
    working_dir = state.get("working_directory", "")
    e2e_tests_generated = state.get("e2e_tests_generated", False)

    logger.info(
        f"[inject_e2e_context] working_dir={working_dir!r} "
        f"e2e_tests_generated={e2e_tests_generated} "
        f"has_e2e_plan={bool(e2e_plan)}"
    )

    # ── Skip path: tests already exist, go straight to run_e2e ──
    if e2e_tests_generated:
        logger.info("[inject_e2e_context] E2E tests already generated → run_e2e")
        return {"routing_target": "run_e2e"}

    # ── First time: build prompt for generate_e2e ──
    lines = ["## E2E Test Generation Task\n"]
    lines.append(f"**Working directory**: `{working_dir}`\n")

    # E2E plan from planner
    if e2e_plan:
        lines.append("### E2E Test Plan (from Planner)")
        lines.append(f"```json\n{json.dumps(e2e_plan, indent=2, ensure_ascii=False)}\n```\n")

        criteria = e2e_plan.get("acceptance_criteria") or []
        if criteria:
            lines.append("### Acceptance Criteria")
            for i, c in enumerate(criteria, 1):
                lines.append(f"{i}. {c}")
            lines.append("")

        scenarios = e2e_plan.get("test_scenarios") or []
        if scenarios:
            lines.append("### Test Scenarios")
            for s in scenarios:
                lines.append(f"- {s}")
            lines.append("")

        run_cmd = e2e_plan.get("run_command", "")
        if run_cmd:
            lines.append(f"### Run Command: `{run_cmd}`\n")

        headless = e2e_plan.get("headless_notes", "")
        if headless:
            lines.append(f"### Headless Testing Notes\n{headless}\n")
    elif qa_plan:
        lines.append("### QA Plan (legacy format)")
        lines.append(f"```\n{qa_plan}\n```\n")

    # File listing
    if working_dir and os.path.isdir(working_dir):
        all_files = []
        for root, dirs, files in os.walk(working_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), working_dir)
                all_files.append(rel)
        logger.info(f"[inject_e2e_context] found {len(all_files)} files in working_dir")
        if all_files:
            lines.append("### Files in Working Directory")
            for f in sorted(all_files):
                lines.append(f"- `{f}`")
        else:
            lines.append("### Files in Working Directory\n(empty)")

    lines.append("\n### Your Task")
    lines.append("1. Read the acceptance criteria and test scenarios above.")
    lines.append("2. Examine the source files using Read.")
    lines.append("3. Write E2E test scripts to test_tool/e2e_tests/.")
    lines.append("4. Write test_tool/run_e2e.sh to run all E2E tests.")
    lines.append("5. Run: bash test_tool/run_e2e.sh")
    lines.append("6. Report E2E_VERDICT: PASS or E2E_VERDICT: FAIL.")

    prompt_len = sum(len(l) for l in lines)
    logger.info(f"[inject_e2e_context] built prompt ({prompt_len} chars) → generate_e2e")
    return {
        "messages": [HumanMessage(content="\n".join(lines))],
        "routing_target": "generate_e2e",
        "e2e_tests_generated": True,  # set flag BEFORE generation (generate_e2e will create files)
    }
```

- [ ] **Step 2: Simplify `e2e_route` — remove `generate_e2e` routing**

Replace the `e2e_route` function (lines 175-234) with:

```python
def e2e_route(state: dict) -> dict:
    """Route E2E test results.

    Pass (rc==0)                       → __end__ (success)
    Fail, qa_fail_count < QA_FAIL_CAP  → __end__ with routing_target="execute"
                                         (parent graph loops back to executor)
    Fail, qa_fail_count >= QA_FAIL_CAP → inject_rescue_context (escalate)
    """
    rc = state.get("execution_returncode")
    qa_fail_count = state.get("qa_fail_count", 0)

    logger.info(
        f"[e2e_route] rc={rc} qa_fail_count={qa_fail_count}/{QA_FAIL_CAP}"
    )

    if rc == 0:
        logger.info("[e2e_route] E2E tests PASSED → __end__ (success)")
        return {
            "routing_target": "__end__",
            "success": True,
        }

    # E2E failed
    logger.info(
        f"[e2e_route] E2E tests FAILED "
        f"(qa_fail_count will be {qa_fail_count + 1}/{QA_FAIL_CAP})"
    )

    if qa_fail_count + 1 >= QA_FAIL_CAP:
        logger.warning(
            f"[e2e_route] QA fail cap ({QA_FAIL_CAP}) reached → rescue"
        )
        return {"routing_target": "inject_rescue_context"}

    # Build qa_analysis with SPECIFIC failed test info for executor
    stdout = state.get("execution_stdout", "")
    stderr = state.get("execution_stderr", "")
    output = (stdout + "\n" + stderr).strip()

    # Extract failed test names for targeted feedback
    failed_tests = []
    for m in re.finditer(r"FAILED\s+([\w/.:]+)", output):
        failed_tests.append(m.group(1))

    qa_analysis_lines = [
        f"E2E test failed (attempt {qa_fail_count + 1}/{QA_FAIL_CAP}).\n",
    ]
    if failed_tests:
        qa_analysis_lines.append("Failed tests:")
        for t in failed_tests:
            qa_analysis_lines.append(f"  - {t}")
        qa_analysis_lines.append("")
    qa_analysis_lines.append(f"Test output:\n{output[-2500:]}")
    qa_analysis = "\n".join(qa_analysis_lines)

    logger.info(
        f"[e2e_route] → routing_target='execute' "
        f"(qa_analysis={len(qa_analysis)} chars, qa_fail_count→{qa_fail_count + 1})"
    )

    return {
        "routing_target": "execute",
        "qa_fail_count": qa_fail_count + 1,
        "qa_analysis": qa_analysis,
    }
```

- [ ] **Step 3: Add `import re` at top of file (if not already present)**

Check line 15 — the file already imports `re`? No, it only imports `json, logging, os, subprocess`. Add `re`:

```python
import json
import logging
import os
import re
import subprocess
```

- [ ] **Step 4: Verify syntax**

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "from blueprints.functional_graphs.colony_coder_qa.validators import inject_e2e_context, e2e_route; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add blueprints/functional_graphs/colony_coder_qa/validators.py
git commit -m "feat(colony-qa): one-shot E2E generation — skip regeneration on subsequent QA entries"
```

---

### Task 4: Inject Acceptance Criteria into Executor Prompt

**Files:**
- Modify: `blueprints/functional_graphs/colony_coder_executor/validators.py:334-418` (`inject_task_context`)

This is the key change that tells Qwen what it will be tested on. The Planner's `e2e_plan.acceptance_criteria` gets injected into the code_gen prompt.

- [ ] **Step 1: Add AC injection to `inject_task_context`**

In `inject_task_context` (line 334), after the QA feedback section (around line 373) and before the dependencies section, add acceptance criteria injection. Replace lines 360-406 with:

```python
    # Ensure git repo for stash snapshots
    if working_dir:
        _ensure_git_repo(working_dir)

    lines = ["## Coding Task\n"]
    lines.append(f"**Working directory**: `{working_dir}`\n")

    # ── Acceptance Criteria from Planner (tells Qwen what QA will test) ──
    e2e_plan = state.get("e2e_plan") or {}
    criteria = e2e_plan.get("acceptance_criteria") or []
    if criteria:
        lines.append("### Acceptance Criteria (your code MUST pass ALL of these)")
        lines.append("The QA engineer will test your code against these criteria.")
        lines.append("Your unit tests should cover each one.\n")
        for i, c in enumerate(criteria, 1):
            lines.append(f"  AC{i}: {c}")
        lines.append("")

    # QA feedback from previous cycle
    if qa_analysis:
        logger.info(
            f"[inject_task_context] injecting QA feedback "
            f"({len(qa_analysis)} chars, qa_fail_count={qa_fail_count})"
        )
        lines.append("### QA Feedback (from previous attempt)")
        lines.append("The QA engineer tested your code and found issues:")
        lines.append(f"```\n{qa_analysis[-3000:]}\n```\n")
        lines.append("Fix these issues in your implementation.\n")

    # Planner-declared dependencies
    deps = _extract_dependencies(state, working_dir)
    if deps:
        lines.append("### External Dependencies (interfaces you must use)")
        lines.append(deps + "\n")

    # Plan
    if refined_plan:
        lines.append("### Design Plan")
        lines.append(f"```\n{refined_plan[:2000]}\n```\n")

    # Tasks
    if tasks:
        lines.append("### Tasks to Implement")
        for tid in execution_order:
            task = next((t for t in tasks if t.get("id") == tid), None)
            if task:
                task_deps = task.get("dependencies") or []
                dep_str = f" (depends on: {', '.join(str(d) for d in task_deps)})" if task_deps else ""
                lines.append(
                    f"- **{tid}**: {task.get('description', '')}{dep_str}"
                )

    lines.append("\n### Instructions")
    lines.append("1. **FIRST: Write the PRODUCT CODE** — create the main source files described in the task.")
    lines.append("2. Use `read_file` to examine existing files (output includes line numbers).")
    lines.append("3. Use `write_file` for NEW files, `replace_lines` for EDITING existing files.")
    lines.append("4. AFTER product code is complete, write unit tests in test_tool/unit_tests/.")
    lines.append("5. Write test_tool/run_tests.sh to run all tests.")
    lines.append("6. Run tests via `bash_exec` and fix until all pass.")
    if criteria:
        lines.append(f"\n**REMINDER**: Your code will be tested against {len(criteria)} acceptance criteria listed above. Make sure every AC is satisfied.")
    lines.append(
        "\n**IMPORTANT**: Always `read_file` before `replace_lines` to get current line numbers."
    )
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "from blueprints.functional_graphs.colony_coder_executor.validators import inject_task_context; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add blueprints/functional_graphs/colony_coder_executor/validators.py
git commit -m "feat(colony-executor): inject acceptance criteria from e2e_plan into Qwen prompt"
```

---

### Task 5: Integration Test — Full Flow Verification

**Files:**
- Read: All modified files for consistency check

- [ ] **Step 1: Verify state field is accessible across subgraphs**

The `e2e_tests_generated` field must survive across the parent graph's `execute → qa → execute` loop. Since all subgraphs share `colony_coder_schema`, and the field is on `ColonyCoderState`, it persists in LangGraph checkpoints.

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "
from blueprints.functional_graphs.colony_coder.state import ColonyCoderState
import typing
hints = typing.get_type_hints(ColonyCoderState)
assert 'e2e_tests_generated' in hints, 'missing field'
assert hints['e2e_tests_generated'] is bool, f'wrong type: {hints[\"e2e_tests_generated\"]}'
print('State field OK')
"
```

Expected: `State field OK`

- [ ] **Step 2: Verify QA graph compiles**

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "
from framework.loader import AgentLoader
loader = AgentLoader('blueprints/functional_graphs/colony_coder_qa')
graph = loader.build_graph()
print(f'QA graph compiled: {len(graph.nodes)} nodes')
"
```

Expected: `QA graph compiled: 8 nodes`

- [ ] **Step 3: Verify routing edges exist**

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "
import json
qa = json.load(open('blueprints/functional_graphs/colony_coder_qa/entity.json'))
edges = qa['graph']['edges']
# Check inject_e2e_context has two routing_to edges
inject_routes = [e for e in edges if e['from'] == 'inject_e2e_context' and e.get('type') == 'routing_to']
assert len(inject_routes) == 2, f'Expected 2 routing edges from inject_e2e_context, got {len(inject_routes)}'
targets = {e['to'] for e in inject_routes}
assert targets == {'generate_e2e', 'run_e2e'}, f'Wrong targets: {targets}'
# Check e2e_route does NOT route to generate_e2e
e2e_route_targets = [e['to'] for e in edges if e['from'] == 'e2e_route']
assert 'generate_e2e' not in e2e_route_targets, 'e2e_route should NOT route to generate_e2e'
print('Edge validation OK')
"
```

Expected: `Edge validation OK`

- [ ] **Step 4: Verify full Colony Coder graph compiles**

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "
from framework.loader import AgentLoader
loader = AgentLoader('blueprints/functional_graphs/colony_coder')
graph = loader.build_graph()
print(f'Colony Coder graph compiled OK')
"
```

Expected: `Colony Coder graph compiled OK`

- [ ] **Step 5: Run existing tests**

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/ -v --timeout=30 2>&1 | tail -20
```

Expected: All existing tests pass (or pre-existing failures unrelated to this change).

- [ ] **Step 6: Commit verification results**

```bash
cd /home/kingy/Foundation/ZenithLoom
git add -A
git commit -m "test(colony): verify QA one-shot restructure compiles and existing tests pass"
```

---

## Flow Summary — Before vs After

### Before (current):
```
Planner → Executor(Qwen) → QA:
  inject_e2e_context → generate_e2e(Claude) → run_e2e → e2e_route
                                                           ↓ FAIL
                        generate_e2e(Claude) ← ← ← ← ← ← ↙  (re-generate!)
                                                    or → execute (parent loop)
                                                           ↓ FAIL x5
                                              → rescue(Claude)
```
- Claude called EVERY QA entry (4 retries = 4 Claude calls)
- Qwen never sees acceptance criteria
- Tests may drift between regenerations

### After (new):
```
Planner → Executor(Qwen, with AC injected) → QA:
  inject_e2e_context [first time] → generate_e2e(Claude) → run_e2e → e2e_route
                                                                        ↓ FAIL
  inject_e2e_context [skip] ─────────────────────────────→ run_e2e → e2e_route
                                                                        ↓ → execute (parent loop)
                                                                        ↓ FAIL x5
                                                              → rescue(Claude)
```
- Claude called ONCE for E2E generation
- Qwen sees AC list from first iteration
- Tests are fixed — no drift, consistent validation gate
