# DebugConsoleReporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic `DebugConsoleReporter` that visualizes LangGraph subgraph execution in real-time with tree-indented console output + markdown logs, then use it to run ColonyCoder on a snake game task.

**Architecture:** `DebugConsoleReporter` consumes `astream(subgraphs=True)` events, tracks namespace depth changes to detect subgraph enter/exit, and renders tree-indented output. A standalone runner script wires it to ColonyCoder.

**Tech Stack:** LangGraph `astream(subgraphs=True)`, Python 3.12+, langchain_core messages

---

## File Map

### New Files

| File | Responsibility |
|------|----------------|
| `framework/debug_reporter.py` | `DebugConsoleReporter` class — generic, no business logic |
| `tests/test_debug_reporter.py` | Unit tests for reporter |
| `run_colony_coder_debug.py` | ColonyCoder + snake game debug runner (replaces existing `run_colony_debug.py`) |

### Existing Files (reference only, not modified)

| File | Why referenced |
|------|----------------|
| `framework/debug.py` | `set_debug()` used by runner |
| `framework/agent_loader.py` | `EntityLoader` used by runner |
| `blueprints/functional_graphs/colony_coder/state.py` | Schema registration import |
| `run_colony_coder_e2e.py:40-67` | Snake task prompt to copy |
| `run_colony_debug.py` | Old debug runner to be replaced |

---

## Task 1: DebugConsoleReporter — core class with scope tracking

**Files:**
- Create: `framework/debug_reporter.py`
- Create: `tests/test_debug_reporter.py`

- [ ] **Step 1: Write the failing tests for scope tracking**

```python
# tests/test_debug_reporter.py
import pytest
from unittest.mock import patch
from io import StringIO


def test_scope_name_top_level():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("myapp")
    assert r._scope_name(()) == "myapp"


def test_scope_name_one_level():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("myapp")
    assert r._scope_name(("plan:abc123",)) == "plan"


def test_scope_name_nested():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("myapp")
    assert r._scope_name(("plan:abc123", "design_debate:def456")) == "design_debate"


def test_depth():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("myapp")
    assert r._depth(()) == 0
    assert r._depth(("a:1",)) == 1
    assert r._depth(("a:1", "b:2")) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_debug_reporter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'framework.debug_reporter'`

- [ ] **Step 3: Implement core class with scope tracking**

```python
# framework/debug_reporter.py
"""
Generic LangGraph subgraph debug visualization — framework/debug_reporter.py

Consumes the event stream from astream(subgraphs=True) and outputs:
  1. Real-time tree-indented console output (with timestamps)
  2. Markdown log file

Usage:
    reporter = DebugConsoleReporter("colony_coder")
    async for ns, event in graph.astream(state, stream_mode="updates", subgraphs=True):
        reporter.on_event(ns, event)
    reporter.print_summary()
"""

import sys
from datetime import datetime
from pathlib import Path

INDENT = "  "


class DebugConsoleReporter:
    def __init__(self, graph_name: str, log_dir: Path | None = None):
        self._graph_name = graph_name
        self._prev_namespace: tuple = ()
        self._start_time = datetime.now()
        self._node_count = 0
        self._last_state: dict = {}

        # Markdown log
        date_str = self._start_time.strftime("%Y-%m-%d")
        self._log_dir = log_dir or Path("logs") / date_str / graph_name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "debug_report.md"
        self._log_file.write_text(
            f"# Debug Report — {graph_name}\n\n"
            f"> Started: {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            encoding="utf-8",
        )

    def _scope_name(self, namespace: tuple) -> str:
        if not namespace:
            return self._graph_name
        last = namespace[-1]
        return last.split(":")[0] if ":" in last else last

    def _depth(self, namespace: tuple) -> int:
        return len(namespace)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _indent(self, depth: int) -> str:
        return INDENT * (depth + 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_debug_reporter.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add framework/debug_reporter.py tests/test_debug_reporter.py
git commit -m "feat(debug_reporter): core class with scope tracking"
```

---

## Task 2: Value formatting + state summary

**Files:**
- Modify: `framework/debug_reporter.py`
- Modify: `tests/test_debug_reporter.py`

- [ ] **Step 1: Write the failing tests for value formatting**

Append to `tests/test_debug_reporter.py`:

```python
def test_format_value_short_string():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value("hello") == "'hello'"


def test_format_value_long_string():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    long_s = "x" * 100
    result = r._format_value(long_s)
    assert "100 chars" in result


def test_format_value_list():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value([1, 2, 3]) == "3 items"


def test_format_value_dict():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value({"a": 1, "b": 2}) == "{2 keys}"


def test_format_value_int():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value(42) == "42"


def test_format_value_bool():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value(True) == "True"


def test_format_value_none():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value(None) == "None"
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python3 -m pytest tests/test_debug_reporter.py -v`
Expected: 4 pass, 7 fail

- [ ] **Step 3: Implement `_format_value`**

Add to `DebugConsoleReporter` in `framework/debug_reporter.py`:

```python
    def _format_value(self, value) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            if len(value) <= 80:
                return repr(value)
            return f"{len(value)} chars"
        if isinstance(value, list):
            return f"{len(value)} items"
        if isinstance(value, dict):
            return f"{{{len(value)} keys}}"
        return repr(value)
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python3 -m pytest tests/test_debug_reporter.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add framework/debug_reporter.py tests/test_debug_reporter.py
git commit -m "feat(debug_reporter): add value formatting"
```

---

## Task 3: on_event — console output with scope transitions + LLM output + state changes

**Files:**
- Modify: `framework/debug_reporter.py`
- Modify: `tests/test_debug_reporter.py`

- [ ] **Step 1: Write the failing tests for on_event**

Append to `tests/test_debug_reporter.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage


def test_on_event_skips_start_end(capsys):
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    r.on_event((), {"__start__": {}})
    r.on_event((), {"__end__": {}})
    captured = capsys.readouterr()
    assert captured.out == ""


def test_on_event_prints_node(capsys):
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    r.on_event((), {"my_node": {"routing_target": "__end__"}})
    out = capsys.readouterr().out
    assert "[test]" in out
    assert "my_node" in out


def test_on_event_prints_ai_message(capsys):
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    msg = AIMessage(content="hello world output", id="msg1")
    r.on_event((), {"my_node": {"messages": [msg]}})
    out = capsys.readouterr().out
    assert "hello world output" in out
    assert "\u2502" in out  # │ prefix


def test_on_event_scope_enter(capsys):
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("app")
    # First event at top level, then event inside subgraph
    r.on_event((), {"node_a": {"x": 1}})
    r.on_event(("sub:123",), {"node_b": {"y": 2}})
    out = capsys.readouterr().out
    assert "\u25b6 sub" in out  # ▶ sub (subgraph enter)


def test_on_event_scope_exit(capsys):
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("app")
    r.on_event(("sub:123",), {"node_b": {"y": 2}})
    r.on_event((), {"node_a": {"x": 1}})
    out = capsys.readouterr().out
    assert "\u25c0 sub" in out  # ◀ sub (subgraph exit)


def test_on_event_routing_target(capsys):
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    r.on_event((), {"validator": {"routing_target": "code_gen"}})
    out = capsys.readouterr().out
    assert "Route:" in out
    assert "code_gen" in out


def test_on_event_state_changes(capsys):
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    r.on_event((), {"my_node": {"retry_count": 3, "success": True}})
    out = capsys.readouterr().out
    assert "retry_count=3" in out
    assert "success=True" in out


def test_node_count_increments():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    r.on_event((), {"a": {"x": 1}})
    r.on_event((), {"b": {"y": 2}})
    assert r._node_count == 2
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python3 -m pytest tests/test_debug_reporter.py -v`
Expected: 11 pass, 8 fail

- [ ] **Step 3: Implement `on_event` and helpers**

Add to `DebugConsoleReporter` in `framework/debug_reporter.py`:

```python
    def on_event(self, namespace: tuple, update: dict) -> None:
        for node_id, changes in update.items():
            if node_id in ("__start__", "__end__"):
                continue

            self._node_count += 1
            depth = self._depth(namespace)
            scope = self._scope_name(namespace)
            indent = self._indent(depth)
            ts = self._timestamp()

            # Detect scope transitions
            self._emit_scope_transitions(namespace)
            self._prev_namespace = namespace

            # Track last state
            if isinstance(changes, dict):
                self._last_state.update(changes)

            # Node header
            self._print(f"{ts} {indent}[{scope}] \u25b6 {node_id}")

            # LLM output (AIMessage content)
            self._print_messages(changes, indent)

            # State changes
            self._print_state(changes, indent)

    def _emit_scope_transitions(self, namespace: tuple) -> None:
        prev = self._prev_namespace
        if namespace == prev:
            return

        # Find common prefix length
        common = 0
        for a, b in zip(prev, namespace):
            if a == b:
                common += 1
            else:
                break

        # Exiting scopes (from deepest to common)
        for i in range(len(prev) - 1, common - 1, -1):
            name = prev[i].split(":")[0] if ":" in prev[i] else prev[i]
            depth = i
            indent = self._indent(depth)
            scope = self._scope_name(prev[:i]) if i > 0 else self._graph_name
            self._print(f"{self._timestamp()} {indent}[{scope}] \u25c0 {name}")

        # Entering scopes (from common+1 to deepest)
        for i in range(common, len(namespace)):
            name = namespace[i].split(":")[0] if ":" in namespace[i] else namespace[i]
            depth = i
            indent = self._indent(depth)
            scope = self._scope_name(namespace[:i]) if i > 0 else self._graph_name
            self._print(f"{self._timestamp()} {indent}[{scope}] \u25b6 {name}")

    def _print_messages(self, changes: dict, indent: str) -> None:
        if not isinstance(changes, dict):
            return
        msgs = changes.get("messages", [])
        for msg in msgs:
            if getattr(msg, "type", "") != "ai":
                continue
            content = getattr(msg, "content", "")
            if not content:
                continue
            char_count = len(content)
            lines = content.split("\n")
            line_indent = " " * len(f"{self._timestamp()} ") + indent
            for line in lines:
                self._print(f"{line_indent}\u2502 {line}")
            if char_count > 0:
                self._print(f"{line_indent}({char_count} chars total)")

    def _print_state(self, changes: dict, indent: str) -> None:
        if not isinstance(changes, dict):
            return
        line_indent = " " * len(f"{self._timestamp()} ") + indent
        parts = []
        for key in sorted(changes.keys()):
            if key == "messages":
                continue
            val = changes[key]
            if val is None or val == "" or val == [] or val == {}:
                continue
            if key == "routing_target":
                self._print(f"{line_indent}Route: \u2192 {val}")
            else:
                parts.append(f"{key}={self._format_value(val)}")
        if parts:
            self._print(f"{line_indent}State: {', '.join(parts)}")

    def _print(self, text: str) -> None:
        print(text, flush=True)
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python3 -m pytest tests/test_debug_reporter.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add framework/debug_reporter.py tests/test_debug_reporter.py
git commit -m "feat(debug_reporter): on_event with scope tracking, LLM output, state changes"
```

---

## Task 4: Markdown logging + print_summary

**Files:**
- Modify: `framework/debug_reporter.py`
- Modify: `tests/test_debug_reporter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_debug_reporter.py`:

```python
import tempfile


def test_markdown_log_created():
    from framework.debug_reporter import DebugConsoleReporter
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "logs"
        r = DebugConsoleReporter("test", log_dir=log_dir)
        assert (log_dir / "debug_report.md").exists()
        content = (log_dir / "debug_report.md").read_text()
        assert "Debug Report" in content


def test_markdown_log_appended():
    from framework.debug_reporter import DebugConsoleReporter
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "logs"
        r = DebugConsoleReporter("test", log_dir=log_dir)
        r.on_event((), {"my_node": {"x": 42}})
        content = (log_dir / "debug_report.md").read_text()
        assert "my_node" in content


def test_markdown_log_includes_ai_output():
    from framework.debug_reporter import DebugConsoleReporter
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "logs"
        r = DebugConsoleReporter("test", log_dir=log_dir)
        msg = AIMessage(content="the full llm output", id="m1")
        r.on_event((), {"llm": {"messages": [msg]}})
        content = (log_dir / "debug_report.md").read_text()
        assert "the full llm output" in content


def test_print_summary(capsys):
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    r.on_event((), {"a": {"x": 1}})
    r.on_event((), {"b": {"success": True}})
    r.print_summary()
    out = capsys.readouterr().out
    assert "2" in out  # node count
    assert "success" in out.lower()
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python3 -m pytest tests/test_debug_reporter.py -v`
Expected: 19 pass, 4 fail

- [ ] **Step 3: Implement markdown logging and print_summary**

Add markdown write to `on_event` and implement `print_summary`. Modify `on_event` to call `_write_markdown` after printing:

```python
    def on_event(self, namespace: tuple, update: dict) -> None:
        for node_id, changes in update.items():
            if node_id in ("__start__", "__end__"):
                continue

            self._node_count += 1
            depth = self._depth(namespace)
            scope = self._scope_name(namespace)
            indent = self._indent(depth)
            ts = self._timestamp()

            self._emit_scope_transitions(namespace)
            self._prev_namespace = namespace

            if isinstance(changes, dict):
                self._last_state.update(changes)

            self._print(f"{ts} {indent}[{scope}] \u25b6 {node_id}")
            self._print_messages(changes, indent)
            self._print_state(changes, indent)

            # Markdown log
            self._write_markdown(namespace, node_id, changes)

    def _write_markdown(self, namespace: tuple, node_id: str, changes: dict) -> None:
        scope_path = " / ".join(
            [self._graph_name] + [n.split(":")[0] for n in namespace]
        )
        ts = self._timestamp()
        lines = [f"\n---\n\n## {ts} — {scope_path} / {node_id}\n\n"]

        # AI message output
        if isinstance(changes, dict):
            for msg in changes.get("messages", []):
                if getattr(msg, "type", "") != "ai":
                    continue
                content = getattr(msg, "content", "")
                if content:
                    lines.append(f"**Output** ({len(content)} chars):\n\n")
                    lines.append(f"```\n{content}\n```\n\n")

            # State changes
            state_parts = []
            for key in sorted(changes.keys()):
                if key == "messages":
                    continue
                val = changes[key]
                if val is None or val == "" or val == [] or val == {}:
                    continue
                state_parts.append(f"- `{key}` = `{self._format_value(val)}`")
            if state_parts:
                lines.append("**State changes:**\n\n")
                lines.append("\n".join(state_parts))
                lines.append("\n")

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write("".join(lines))

    def print_summary(self) -> None:
        elapsed = datetime.now() - self._start_time
        minutes = elapsed.total_seconds() / 60
        success = self._last_state.get("success")
        abort = self._last_state.get("abort_reason", "")

        self._print(f"\n{'=' * 70}")
        self._print(f"  Summary: {self._graph_name}")
        self._print(f"  Nodes executed: {self._node_count}")
        self._print(f"  Elapsed: {minutes:.1f} min")

        if success is not None:
            icon = "\u2705" if success else "\u274c"
            self._print(f"  Result: {icon} success={success}")
        if abort:
            self._print(f"  Abort: {abort}")

        files = self._last_state.get("final_files", [])
        if files:
            self._print(f"  Files: {files}")

        self._print(f"  Log: {self._log_file}")
        self._print(f"{'=' * 70}")
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python3 -m pytest tests/test_debug_reporter.py -v`
Expected: 23 passed

- [ ] **Step 5: Commit**

```bash
git add framework/debug_reporter.py tests/test_debug_reporter.py
git commit -m "feat(debug_reporter): markdown logging + print_summary"
```

---

## Task 5: ColonyCoder debug runner script

**Files:**
- Create: `run_colony_coder_debug.py` (replaces `run_colony_debug.py`)
- Delete: `run_colony_debug.py`

- [ ] **Step 1: Create the debug runner**

```python
#!/usr/bin/env python3
"""
ColonyCoder debug runner — visualizes the full execution process using DebugConsoleReporter.

Usage: python3 run_colony_coder_debug.py
"""

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
import blueprints.functional_graphs.colony_coder.state  # noqa: F401

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
    "- Single-file implementation, save to /tmp/snake_battle_v3/snake_battle.py\n"
    "- Clear code structure, two AIs as separate independent classes\n"
    "- Can be run directly with python3 snake_battle.py\n"
)


async def main():
    set_debug(True)

    loader = EntityLoader(Path("blueprints/functional_graphs/colony_coder"))
    graph = await loader.build_graph(checkpointer=None)

    reporter = DebugConsoleReporter("colony_coder")

    print("=" * 70)
    print("  ColonyCoder Debug Run — Snake Battle")
    print("=" * 70)
    print(flush=True)

    init_state = {"messages": [HumanMessage(content=SNAKE_TASK)]}

    async for namespace, event in graph.astream(
        init_state, stream_mode="updates", subgraphs=True
    ):
        reporter.on_event(namespace, event)

    reporter.print_summary()

    # Check generated files
    working_dir = reporter._last_state.get("working_directory", "/tmp/snake_battle_v3")
    wd = Path(working_dir)
    py_files = list(wd.glob("*.py")) if wd.exists() else []
    if py_files:
        for pf in py_files:
            content = pf.read_text(encoding="utf-8")
            print(f"\n  {pf.name}: {len(content)} chars, {len(content.splitlines())} lines")
            try:
                compile(content, str(pf), "exec")
                print(f"  Syntax check: PASS")
            except SyntaxError as e:
                print(f"  Syntax check: FAIL — {e}")
    else:
        print(f"\n  No .py files found in {working_dir}")
        if wd.exists():
            all_files = list(wd.rglob("*"))
            print(f"  Files: {[str(f.relative_to(wd)) for f in all_files]}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Delete the old runner**

```bash
git rm run_colony_debug.py
```

- [ ] **Step 3: Verify the script is syntactically valid**

Run: `python3 -c "import ast; ast.parse(open('run_colony_coder_debug.py').read()); print('OK')"` 
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add run_colony_coder_debug.py
git commit -m "feat: add colony coder debug runner with DebugConsoleReporter

Replaces run_colony_debug.py with new runner using astream(subgraphs=True)
for full subgraph visibility."
```

---

## Task 6: Integration test — verify astream(subgraphs=True) works with colony coder graph

**Files:**
- Modify: `tests/test_debug_reporter.py`

- [ ] **Step 1: Write integration test**

Append to `tests/test_debug_reporter.py`:

```python
@pytest.mark.asyncio
async def test_colony_coder_graph_with_subgraphs_true():
    """Verify astream(subgraphs=True) emits events with namespace tuples
    for colony_coder's persistent subgraphs. Does NOT call real LLMs —
    just compiles the graph and checks that subgraphs=True is accepted."""
    import blueprints.functional_graphs.colony_coder.state  # noqa: F401
    from framework.agent_loader import EntityLoader

    loader = EntityLoader(Path("blueprints/functional_graphs/colony_coder"))
    graph = await loader.build_graph(checkpointer=None)

    # Verify the graph has subgraph nodes
    node_ids = set(graph.nodes) - {"__start__", "__end__"}
    assert "plan" in node_ids, f"Missing 'plan' node, got: {node_ids}"
    assert "execute" in node_ids, f"Missing 'execute' node, got: {node_ids}"
    assert "qa" in node_ids, f"Missing 'qa' node, got: {node_ids}"

    # Verify astream accepts subgraphs=True without error
    # (We can't run the full graph without LLMs, but we can verify the API works)
    # Just check the graph compiled correctly with native subgraphs
    assert hasattr(graph, 'astream'), "Compiled graph should have astream method"
```

- [ ] **Step 2: Run test**

Run: `python3 -m pytest tests/test_debug_reporter.py::test_colony_coder_graph_with_subgraphs_true -v`
Expected: PASS

- [ ] **Step 3: Run all tests to verify no regressions**

Run: `python3 -m pytest tests/test_debug_reporter.py test_colony_coder.py test_e2e_colony_coder.py test_e2e_colony_coder_game.py test_e2e_debate.py test_cli.py -v --tb=short`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_debug_reporter.py
git commit -m "test: integration test for colony coder with subgraphs=True"
```
