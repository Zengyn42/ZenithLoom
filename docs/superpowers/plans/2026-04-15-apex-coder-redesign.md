# ApexCoder TDD Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild ApexCoder from a single Claude SDK node into a 4-node TDD pipeline (splitter → ClaudeQA → reset_for_coder → ClaudeCoder) where QA writes tests first and Coder must pass them.

**Architecture:** Two DETERMINISTIC nodes (splitter, reset_for_coder) handle input normalization and message isolation. Two CLAUDE_SDK nodes (ClaudeQA, ClaudeCoder) run as independent sessions — QA writes tests from user requirements, Coder implements code that passes those tests. A PreToolUse hook prevents Coder from modifying QA test files.

**Tech Stack:** LangGraph, Claude Agent SDK, Python 3.12+, pytest

---

## File Map

### New Files

| File | Responsibility |
|------|----------------|
| `blueprints/functional_graphs/apex_coder/state.py` | `ApexCoderState` TypedDict + auto-register |
| `blueprints/functional_graphs/apex_coder/validators.py` | `splitter()` + `reset_for_coder()` DETERMINISTIC nodes |
| `blueprints/functional_graphs/apex_coder/CODER_ROLE.md` | Coder persona (TDD, must pass QA tests) |
| `blueprints/functional_graphs/apex_coder/QA_ROLE.md` | QA persona (test-first, real env, 5-10 tests) |
| `blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py` | PreToolUse hook blocking Write/Edit to qa_tests/ |
| `tests/test_apex_coder.py` | Unit tests for state, validators, graph compilation |

### Modified Files

| File | Change |
|------|--------|
| `blueprints/functional_graphs/apex_coder/entity.json` | Rewrite: 1 node → 4 nodes |

### Deleted Files

| File | Reason |
|------|--------|
| `blueprints/functional_graphs/apex_coder/ROLE.md` | Split into CODER_ROLE.md + QA_ROLE.md |

### Unchanged Files

| File | Reason |
|------|--------|
| `blueprints/functional_graphs/apex_coder/PROTOCOL.md` | Shared by both Coder and QA |
| `blueprints/functional_graphs/apex_coder/.claude/agents/*` | Coder can still spawn sub-agents |
| `blueprints/functional_graphs/apex_coder/.claude/skills/*` | Coder can still use skills |

---

## Task 1: ApexCoderState schema

**Files:**
- Create: `blueprints/functional_graphs/apex_coder/state.py`
- Create: `tests/test_apex_coder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_apex_coder.py
import pytest


def test_apex_coder_schema_registered():
    import blueprints.functional_graphs.apex_coder.state  # noqa: F401
    from framework.registry import get_all_schemas
    schemas = get_all_schemas()
    assert "apex_coder_schema" in schemas, f"missing apex_coder_schema, got: {list(schemas.keys())}"


def test_apex_coder_schema_has_required_fields():
    import typing
    from blueprints.functional_graphs.apex_coder.state import ApexCoderState
    hints = typing.get_type_hints(ApexCoderState, include_extras=True)
    for field in ("user_requirements", "working_directory", "qa_bypass",
                  "qa_tests_dir", "run_qa_script", "qa_summary", "apex_conclusion"):
        assert field in hints, f"ApexCoderState missing field: {field}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_apex_coder.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write the implementation**

```python
# blueprints/functional_graphs/apex_coder/state.py
"""ApexCoderState — apex_coder TDD pipeline state schema.

Shared by splitter / ClaudeQA / reset_for_coder / ClaudeCoder.

Auto-registers as "apex_coder_schema" on import.
"""

from __future__ import annotations

from typing import Annotated, Optional

from framework.schema.base import BaseAgentState
from framework.schema.reducers import _merge_dict
from framework.registry import register_schema


class ApexCoderState(BaseAgentState):
    # Splitter output
    user_requirements: str          # user requirements text (extracted from messages[0] or plan.md)
    working_directory: str          # working directory path

    # QA output
    qa_bypass: bool                 # whether QA is skipped (True when task does not require tests)
    qa_tests_dir: str               # QA tests directory path
    run_qa_script: str              # run_qa.sh path
    qa_summary: str                 # QA output summary

    # Coder output
    apex_conclusion: str            # Coder final report

    # Override node_sessions with merge reducer
    node_sessions: Annotated[dict, _merge_dict]


# Auto-register on import
register_schema("apex_coder_schema", ApexCoderState)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/state.py tests/test_apex_coder.py
git commit -m "feat(apex_coder): add ApexCoderState schema for TDD pipeline"
```

---

## Task 2: Validators — splitter + reset_for_coder

**Files:**
- Create: `blueprints/functional_graphs/apex_coder/validators.py`
- Modify: `tests/test_apex_coder.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_apex_coder.py`:

```python
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
import tempfile


def test_splitter_text_input():
    from blueprints.functional_graphs.apex_coder.validators import splitter
    result = splitter({
        "messages": [HumanMessage(content="Build a snake game\n\n## working_directory: /tmp/test_splitter_apex")]
    })
    assert result["user_requirements"] == "Build a snake game\n\n## working_directory: /tmp/test_splitter_apex"
    assert result["working_directory"] == "/tmp/test_splitter_apex"
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == result["user_requirements"]


def test_splitter_file_input():
    from blueprints.functional_graphs.apex_coder.validators import splitter
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("Build a todo app")
        f.flush()
        result = splitter({"messages": [HumanMessage(content=f.name)]})
    assert result["user_requirements"] == "Build a todo app"
    assert result["working_directory"].startswith("/tmp/apex_")


def test_splitter_auto_generates_working_dir():
    from blueprints.functional_graphs.apex_coder.validators import splitter
    result = splitter({"messages": [HumanMessage(content="Build something")]})
    assert result["working_directory"].startswith("/tmp/apex_")
    assert Path(result["working_directory"]).exists()
    assert Path(result["working_directory"], "test_tool", "qa_tests").exists()


def test_splitter_creates_directories():
    from blueprints.functional_graphs.apex_coder.validators import splitter
    result = splitter({
        "messages": [HumanMessage(content="Task\n\n## working_directory: /tmp/test_splitter_dirs")]
    })
    assert Path("/tmp/test_splitter_dirs/test_tool/qa_tests").is_dir()


def test_reset_for_coder_clears_qa_messages():
    from blueprints.functional_graphs.apex_coder.validators import reset_for_coder
    result = reset_for_coder({
        "messages": [
            HumanMessage(content="user task", id="h1"),
            AIMessage(content="QA reasoning blah blah", id="a1"),
        ],
        "user_requirements": "user task",
        "working_directory": "/tmp/test_reset",
        "qa_bypass": False,
        "run_qa_script": "/tmp/test_reset/test_tool/run_qa.sh",
    })
    # Should have RemoveMessages for old messages + 1 new HumanMessage
    msgs = result["messages"]
    # New HumanMessage should contain user requirements
    human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 1
    assert "user task" in human_msgs[0].content
    assert "run_qa.sh" in human_msgs[0].content


def test_reset_for_coder_bypass_mode():
    from blueprints.functional_graphs.apex_coder.validators import reset_for_coder
    result = reset_for_coder({
        "messages": [HumanMessage(content="task", id="h1")],
        "user_requirements": "simple task",
        "working_directory": "/tmp/test_bypass",
        "qa_bypass": True,
        "run_qa_script": "",
    })
    human_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert "BYPASSED" in human_msgs[0].content
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 2 pass, 6 fail

- [ ] **Step 3: Write the implementation**

```python
# blueprints/functional_graphs/apex_coder/validators.py
"""DETERMINISTIC nodes for apex_coder TDD pipeline.

Nodes:
  splitter          — extract user_requirements, create working_directory
  reset_for_coder   — clear QA messages, build clean prompt for Coder
"""

import os
import re
import uuid
from pathlib import Path


def splitter(state: dict) -> dict:
    """Extract user_requirements and set up working_directory."""
    from langchain_core.messages import HumanMessage

    msg = state["messages"][0].content.strip()

    # Smart input: file path or raw text
    if msg.startswith("/") and Path(msg).is_file():
        user_requirements = Path(msg).read_text(encoding="utf-8")
    else:
        user_requirements = msg

    # Parse working_directory from task or auto-generate
    wd_match = re.search(
        r"[#]*\s*(?:working.?dir(?:ectory)?)[:：]\s*(\S+)",
        user_requirements,
        re.IGNORECASE,
    )
    if wd_match:
        working_directory = wd_match.group(1)
    else:
        working_directory = f"/tmp/apex_{uuid.uuid4().hex[:8]}"

    # Create directory structure
    Path(working_directory).mkdir(parents=True, exist_ok=True)
    Path(working_directory, "test_tool", "qa_tests").mkdir(parents=True, exist_ok=True)

    return {
        "user_requirements": user_requirements,
        "working_directory": working_directory,
        "messages": [HumanMessage(content=user_requirements)],
    }


def reset_for_coder(state: dict) -> dict:
    """Clear QA messages, build clean prompt for ClaudeCoder."""
    from langchain_core.messages import HumanMessage, RemoveMessage

    user_req = state.get("user_requirements", "")
    working_dir = state.get("working_directory", "")
    qa_bypass = state.get("qa_bypass", False)
    run_qa_script = state.get("run_qa_script", "")

    # Remove all previous messages (QA's conversation history)
    msgs = state.get("messages", [])
    removals = [RemoveMessage(id=m.id) for m in msgs]

    # Build clean prompt for Coder
    lines = [f"## User Requirements\n\n{user_req}"]
    lines.append(f"\n## Working Directory: `{working_dir}`")

    if qa_bypass:
        lines.append("\n## QA: BYPASSED (no tests to pass)")
    else:
        lines.append(f"\n## QA Tests")
        lines.append(f"- Runner script: `{run_qa_script}`")
        lines.append(f"- **Read the QA tests FIRST to understand what's expected.**")
        lines.append(f"- **Run `bash {run_qa_script}` and ensure ALL tests pass before finishing.**")
        lines.append(f"- **DO NOT modify any files in `test_tool/qa_tests/`.**")

        # List QA test files
        qa_dir = os.path.join(working_dir, "test_tool", "qa_tests")
        if os.path.isdir(qa_dir):
            test_files = [f for f in os.listdir(qa_dir) if f.endswith(".py") and f != "__init__.py"]
            if test_files:
                lines.append(f"\nQA test files ({len(test_files)}):")
                for f in sorted(test_files):
                    lines.append(f"  - `{qa_dir}/{f}`")

    prompt = "\n".join(lines)
    return {
        "messages": removals + [HumanMessage(content=prompt)],
    }
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/validators.py tests/test_apex_coder.py
git commit -m "feat(apex_coder): add splitter and reset_for_coder validators"
```

---

## Task 3: Persona files — CODER_ROLE.md + QA_ROLE.md

**Files:**
- Create: `blueprints/functional_graphs/apex_coder/CODER_ROLE.md`
- Create: `blueprints/functional_graphs/apex_coder/QA_ROLE.md`
- Delete: `blueprints/functional_graphs/apex_coder/ROLE.md`

- [ ] **Step 1: Create CODER_ROLE.md**

```markdown
# Coder Role

You are a senior P8 engineer. Implement features end-to-end.

## Your Inputs

1. **User Requirements** — user requirements (in the message)
2. **QA Tests** — automated tests written by an independent QA engineer (in the test_tool/qa_tests/ directory)

If QA is marked as BYPASSED, you only need to satisfy User Requirements — no need to run tests.

## Your Workflow

1. **Read QA tests first** — understand the expected behavior and acceptance criteria
2. **Then read User Requirements** — confirm the full requirements
3. **Implement** — write code to satisfy the requirements
4. **Self-test** — run `bash <run_qa_script>`
5. **Fix** — if tests fail, read the error output, fix the code, re-run
6. **Repeat 4-5 until all pass**
7. **Completion report**

## Iron Rules

- ❌ Never modify any files under `test_tool/qa_tests/` (a hook enforces this)
- ❌ Never delete or rename QA tests
- ✅ If you believe a QA test is wrong, note it in the final report but do not modify it
- ✅ All Bash commands must include a timeout (see PROTOCOL.md)

## Your Team

You have the Agent tool to spawn expert sub-agents:

| Agent | Purpose | When to spawn |
|-------|---------|--------------|
| planner | requirements analysis + implementation plan | before starting complex tasks |
| architect | system design + ADR | architecture decisions |
| code-reviewer | code review | after implementation is complete |
| build-error-resolver | build error fixes | when build fails |
| pua-debugger | extreme debugging | when repeatedly failing |

## Report Format

When done, output:
- Which files were created/modified
- QA test results (all passing or which still fail + reasons)
- Brief code architecture description
```

- [ ] **Step 2: Create QA_ROLE.md**

```markdown
# QA Engineer Role

You are an independent QA engineer. Your task is to write automated tests based on user requirements.
You do not know how the code will be implemented and should not care. You only care about: what does the user want? How do you verify it?

## Your Work

1. Read the user requirements
2. Determine whether QA tests are needed:
   - Not needed (pure refactor with no behavior change, documentation edits, config changes, code style adjustments, modifying parts of a large codebase where E2E testing is impractical): output `QA_BYPASS: <reason>`, do not write tests
   - Needed: continue with the steps below
3. Write 5-10 tests in the `<working_directory>/test_tool/qa_tests/` directory
4. Write the `<working_directory>/test_tool/run_qa.sh` execution script
5. Output a test summary

## Testing Rules

- Tests verify functionality from the **user's perspective** — do not test internal implementation
- Each test < 10 seconds, total tests < 90 seconds
- curses/terminal programs: must use the pty module to test in a **real terminal environment**
  - ❌ Forbidden: headless tests like `python3 -c "import X; X.Game().tick()"`
  - ✅ Must use pty + 24x80 terminal to launch a real process
- Coverage: core functionality + key edge cases + exit behavior
- Do not test implementation details (internal class names, function signatures, etc.)

## run_qa.sh Template

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")/.."
timeout 120 python3 -m pytest test_tool/qa_tests/ -v 2>&1
```

## Output Format

If tests were written:
  QA_READY: <number of tests> tests written to <path>

If skipped:
  QA_BYPASS: <reason>
```

- [ ] **Step 3: Delete old ROLE.md**

```bash
git rm blueprints/functional_graphs/apex_coder/ROLE.md
```

- [ ] **Step 4: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/CODER_ROLE.md \
       blueprints/functional_graphs/apex_coder/QA_ROLE.md
git commit -m "feat(apex_coder): split ROLE.md into CODER_ROLE.md + QA_ROLE.md"
```

---

## Task 4: PreToolUse hook

**Files:**
- Create: `blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py`
- Modify: `tests/test_apex_coder.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_apex_coder.py`:

```python
import subprocess
import json


def test_hook_blocks_qa_test_write():
    hook_path = "blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py"
    data = {"tool_input": {"file_path": "/tmp/game/test_tool/qa_tests/test_foo.py"}}
    result = subprocess.run(
        ["python3", hook_path],
        input=json.dumps(data),
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["decision"] == "block"


def test_hook_allows_source_write():
    hook_path = "blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py"
    data = {"tool_input": {"file_path": "/tmp/game/main.py"}}
    result = subprocess.run(
        ["python3", hook_path],
        input=json.dumps(data),
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["decision"] == "allow"


def test_hook_allows_unit_test_write():
    hook_path = "blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py"
    data = {"tool_input": {"file_path": "/tmp/game/test_tool/unit_tests/test_main.py"}}
    result = subprocess.run(
        ["python3", hook_path],
        input=json.dumps(data),
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["decision"] == "allow"
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python3 -m pytest tests/test_apex_coder.py::test_hook_blocks_qa_test_write -v`
Expected: FAIL — file not found or JSON error

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""PreToolUse hook: block Write/Edit to test_tool/qa_tests/ directory.

Reads tool_input from stdin (JSON), checks if file_path targets QA tests.
Outputs JSON decision: {"decision": "allow"} or {"decision": "block", "reason": "..."}.
"""
import json
import sys

data = json.load(sys.stdin)
tool_input = data.get("tool_input", {})
path = tool_input.get("file_path", "") or tool_input.get("path", "")

if "test_tool/qa_tests" in path:
    print(json.dumps({
        "decision": "block",
        "reason": f"BLOCKED: Cannot modify QA test file '{path}'. Fix your source code instead.",
    }))
else:
    print(json.dumps({"decision": "allow"}))
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py tests/test_apex_coder.py
git commit -m "feat(apex_coder): add PreToolUse hook to protect QA test files"
```

---

## Task 5: entity.json rewrite + graph compilation test

**Files:**
- Modify: `blueprints/functional_graphs/apex_coder/entity.json`
- Modify: `tests/test_apex_coder.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_apex_coder.py`:

```python
@pytest.mark.asyncio
async def test_apex_coder_graph_compiles():
    import blueprints.functional_graphs.apex_coder.state  # noqa: F401
    from framework.agent_loader import EntityLoader
    g = await EntityLoader(Path("blueprints/functional_graphs/apex_coder")).build_graph(checkpointer=None)
    node_ids = set(g.nodes) - {"__start__", "__end__"}
    required = {"splitter", "claude_qa", "reset_for_coder", "claude_coder"}
    assert required <= node_ids, f"Missing nodes: {required - node_ids}, got: {node_ids}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apex_coder.py::test_apex_coder_graph_compiles -v`
Expected: FAIL — old entity.json has `apex_main` not 4 nodes

- [ ] **Step 3: Rewrite entity.json**

```json
{
  "name": "apex_coder",
  "routing_hint": "Use when a programming problem is extremely complex and requires exhausting all possible approaches. Applicable for: bugs that have failed to resolve after multiple attempts, full architecture refactors, cross-file system changes, environment/config issues requiring deep debugging.",
  "llm": "claude",
  "graph": {
    "state_schema": "apex_coder_schema",
    "entry": "splitter",
    "exit": "claude_coder",
    "nodes": [
      {
        "id": "splitter",
        "type": "DETERMINISTIC"
      },
      {
        "id": "claude_qa",
        "type": "CLAUDE_SDK",
        "model": "claude-opus-4-6",
        "session_key": "apex_qa",
        "tools": ["Read", "Write", "Bash"],
        "permission_mode": "bypassPermissions",
        "persona_files": ["./QA_ROLE.md", "./PROTOCOL.md"],
        "output_field": "qa_summary",
        "setting_sources": null
      },
      {
        "id": "reset_for_coder",
        "type": "DETERMINISTIC"
      },
      {
        "id": "claude_coder",
        "type": "CLAUDE_SDK",
        "model": "claude-opus-4-6",
        "extra_persona": true,
        "session_key": "apex_coder",
        "tools": [
          "Read", "Write", "Edit", "Bash",
          "Glob", "Grep", "NotebookEdit", "Agent"
        ],
        "permission_mode": "bypassPermissions",
        "max_turns": 50,
        "persona_files": ["./CODER_ROLE.md", "./PROTOCOL.md"],
        "output_field": "apex_conclusion",
        "setting_sources": null,
        "settings_override": {
          "enabledPlugins": [],
          "hooks": {
            "PreToolUse": [{
              "matcher": "Write|Edit",
              "hooks": [{
                "type": "command",
                "command": "python3 blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py"
              }]
            }]
          }
        },
        "add_dirs": [
          "blueprints/functional_graphs/apex_coder"
        ]
      }
    ],
    "edges": [
      {"from": "splitter",         "to": "claude_qa"},
      {"from": "claude_qa",        "to": "reset_for_coder"},
      {"from": "reset_for_coder",  "to": "claude_coder"}
    ]
  },
  "max_retries": 0,
  "db_path": "apex_coder.db",
  "sessions_file": "sessions.json"
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 12 passed

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python3 -m pytest tests/test_apex_coder.py test_colony_coder.py test_e2e_colony_coder.py test_e2e_debate.py test_cli.py tests/test_debug_reporter.py tests/test_subgraph_init_node.py -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/entity.json tests/test_apex_coder.py
git commit -m "feat(apex_coder): rewrite entity.json to 4-node TDD pipeline"
```

---

## Task 6: Update debug runner + smoke test

**Files:**
- Modify: `run_apex_coder_debug.py`
- Modify: `run_benchmark_apex.py`

- [ ] **Step 1: Update run_apex_coder_debug.py**

Replace the import of `apex_coder.state` and update the init_state to include `ApexCoderState` fields:

```python
#!/usr/bin/env python3
"""ApexCoder debug runner — TDD pipeline (QA writes tests → Coder passes them)."""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)

# Register state schema
import blueprints.functional_graphs.apex_coder.state  # noqa: F401

from framework.agent_loader import EntityLoader
from framework.debug import set_debug
from framework.debug_reporter import DebugConsoleReporter
from langchain_core.messages import HumanMessage

SNAKE_TASK = (
    "Write a two-snake battle game (Snake Battle) in Python.\n"
    "\n"
    "## Core Requirements\n"
    "1. Use the curses library for terminal UI\n"
    "2. Two snakes appear simultaneously on the board, both controlled by AI (no human player), the player is only a spectator\n"
    "3. Multiple food items exist on screen simultaneously; snakes grow longer after eating food\n"
    "4. A snake dies if it hits a wall, itself, or the other snake's body\n"
    "5. The last surviving snake wins; if both are alive, compare by length\n"
    "\n"
    "## AI Design\n"
    "Design two AIs with different strategies (AI-Alpha and AI-Beta), each controlling one snake.\n"
    "AI goal: eat as much food as possible to grow, while trying to eliminate the opponent.\n"
    "The two AIs must use different strategies to make the battle interesting.\n"
    "\n"
    "## UI Requirements\n"
    "- Top status bar showing both snakes' info and current frame count\n"
    "- Game area has a border\n"
    "- Two snakes distinguished by different colors\n"
    "- Display winner when game ends\n"
    "- Press Q to quit\n"
    "- Default frame rate ~10 FPS\n"
    "\n"
    "## Technical Requirements\n"
    "- Single-file implementation, save to /tmp/snake_battle_apex/snake_battle.py\n"
    "- Clear code structure, two AIs as separate independent classes\n"
    "- Can be run directly with python3 snake_battle.py\n"
    "- Must run correctly in a standard 24x80 terminal\n"
    "\n"
    "## working_directory: /tmp/snake_battle_apex\n"
)


async def main():
    set_debug(True)

    loader = EntityLoader(Path("blueprints/functional_graphs/apex_coder"))
    graph = await loader.build_graph(checkpointer=None)

    reporter = DebugConsoleReporter("apex_coder")

    print("=" * 70)
    print("  ApexCoder TDD — Snake Battle")
    print("=" * 70)
    print(flush=True)

    init_state = {"messages": [HumanMessage(content=SNAKE_TASK)]}

    async for namespace, event in graph.astream(
        init_state, stream_mode="updates", subgraphs=True
    ):
        reporter.on_event(namespace, event)

    reporter.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Update run_benchmark_apex.py**

Add `import blueprints.functional_graphs.apex_coder.state` to the benchmark runner (replace the old plan.md logic — splitter now handles file path input natively):

Replace the `main()` function body. The key change: pass task text directly as message (no plan.md file needed — splitter handles both formats):

```python
async def main():
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(TASKS.keys())}>")
        sys.exit(1)

    task_name = sys.argv[1]
    task_desc, working_dir, _ = TASKS[task_name]

    set_debug(True)

    # Register state schema
    import blueprints.functional_graphs.apex_coder.state  # noqa: F401

    loader = EntityLoader(Path("blueprints/functional_graphs/apex_coder"))
    graph = await loader.build_graph(checkpointer=None)
    reporter = DebugConsoleReporter(f"apex_{task_name}")

    print("=" * 70)
    print(f"  ApexCoder TDD — {task_name}")
    print("=" * 70, flush=True)

    # Append working_directory hint to task
    task_with_wd = f"{task_desc}\n## working_directory: {working_dir}\n"
    init_state = {"messages": [HumanMessage(content=task_with_wd)]}

    async for ns, event in graph.astream(init_state, stream_mode="updates", subgraphs=True):
        reporter.on_event(ns, event)
    reporter.print_summary()
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('run_apex_coder_debug.py').read()); ast.parse(open('run_benchmark_apex.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add run_apex_coder_debug.py run_benchmark_apex.py
git commit -m "feat(apex_coder): update debug and benchmark runners for TDD pipeline"
```
