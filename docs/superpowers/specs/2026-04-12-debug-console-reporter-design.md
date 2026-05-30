Note: The core engine of the project resides in this ZenithLoom repository. However, all the blueprints have been moved to a separate repository called VoidDraft.

# DebugConsoleReporter — Universal Subgraph Debug Visualization

> Date: 2026-04-12
> Status: Approved

## Goals

Provide real-time tree-indented debug output + markdown logs for any LangGraph graph,
leveraging LangGraph's native `astream(subgraphs=True)` to track all subgraph levels.

Prerequisites: unified subgraph integration refactor completed (all session_modes use native subgraphs).

## Scope

### Additions

| File | Responsibility |
|------|------|
| `framework/debug_reporter.py` | `DebugConsoleReporter` — General debug output class |
| `run_colony_coder_debug.py` | Debug runner for ColonyCoder + snake game |

### No Changes

| File | Reason |
|------|------|
| `framework/agent_loader.py` | Subgraph access is already native, no changes needed |
| `framework/debug.py` | Retain existing `_wrap_node_for_flow_log` flow.md logs |
| `run_colony_coder_e2e.py` | Retain, the new runner is an independent script |
| All entity.json | No changes |

---

## DebugConsoleReporter Design

### Data Source

LangGraph `astream(stream_mode="updates", subgraphs=True)` returns:

```python
async for namespace, event in graph.astream(state, stream_mode="updates", subgraphs=True):
    # namespace: tuple[str, ...]
    #   () = top level
    #   ("plan:abc123",) = plan subgraph
    #   ("plan:abc123", "design_debate:def456") = nested subgraph
    # event: dict[str, dict]
    #   {node_id: {state_changes...}}
```

### Console Output Format

Tree indentation + timestamps, full LLM output:

```
14:23:01   [colony_coder] ▶ plan
14:23:01     [plan] ▶ _subgraph_init
14:23:01     [plan] ◀ _subgraph_init
14:23:02       [design_debate] ▶ claude_propose
14:23:15       [design_debate] ◀ claude_propose (2,341 chars)
               │ I suggest using MVC architecture to implement the snake battle game...
               │ Core classes: Snake, Food, GameBoard, AIController
               │ ... (2,341 chars total)
14:23:15       [design_debate] ▶ gemini_critique_1
14:23:28       [design_debate] ◀ gemini_critique_1 (1,892 chars)
               │ There are several issues to consider with this proposal...
14:25:03     [plan] ◀ design_debate
14:25:03     [plan] ▶ claude_swarm
14:25:30     [plan] ◀ claude_swarm (3,201 chars)
               │ Summary of the review from three perspectives...
14:25:30     [plan] ▶ task_decompose
14:25:45     [plan] ◀ task_decompose
               │ {"tasks": [{"id": "t1", ...}], ...}
               State: tasks=4 items, execution_order=['t1','t2','t3','t4']
14:25:45     [plan] ▶ decomposition_validator
               Route: → __end__
14:25:45     [plan] ◀ _subgraph_exit
14:25:45   [colony_coder] ◀ plan

14:25:45   [colony_coder] ▶ execute
14:25:45     [execute] ▶ inject_task_context
               State: current_task_id='t1', retry_count=0
14:25:46     [execute] ▶ code_gen
14:27:45     [execute] ◀ code_gen (8,234 chars)
               │ I'll implement the Snake class first...
14:27:46     [execute] ▶ run_tests
               State: execution_returncode=0
               Route: → __end__
14:27:46     [execute] ◀ _subgraph_exit
14:27:46   [colony_coder] ◀ execute
```

### Output Rules

1. **Full AIMessage output**: Extract AIMessage from `changes["messages"]`, print full text with `│` prefix.
2. **routing_target**: Print `Route: → target` when not empty.
3. **Other non-empty state changes**: `key=value` format, skipping messages and empty values.
4. **Subgraph enter/exit**: Detected via namespace depth changes, print `▶ subgraph_name` / `◀ subgraph_name`.
5. **Skip `__start__` / `__end__` nodes**.

### Value Formatting Rules

| Type | Format |
|------|------|
| str (short, ≤80 chars) | `key='value'` |
| str (long, >80 chars) | `key=value (N chars)` |
| list | `key=N items` |
| dict | `key={N keys}` |
| int / float / bool / None | `key=value` |

### API

```python
class DebugConsoleReporter:
    def __init__(self, graph_name: str):
        """
        Args:
            graph_name: Top-level graph name, used for console output and markdown log directory.
        """
    
    def on_event(self, namespace: tuple, update: dict) -> None:
        """Process an astream(subgraphs=True) event.
        
        Callers call this method directly in the async for loop.
        """
    
    def print_summary(self) -> None:
        """Print final execution summary: duration, node count, success/failure."""
```

### Internal Methods

```python
def _scope_name(self, namespace: tuple) -> str:
    """Extract scope name from namespace.
    ("plan:abc123",) → "plan"
    ("plan:abc123", "design_debate:def456") → "design_debate"
    () → self._graph_name
    """

def _depth(self, namespace: tuple) -> int:
    """namespace length = indentation depth."""

def _format_value(self, value) -> str:
    """General value formatting."""

def _print_node_event(self, namespace, node_id, changes):
    """Print entry + output + state changes for a single node."""

def _detect_scope_change(self, prev_namespace, curr_namespace):
    """Detect subgraph enter/exit events."""

def _write_markdown(self, namespace, node_id, changes):
    """Write to markdown log file."""
```

### Markdown Logs

Written to `logs/YYYY-MM-DD/<graph_name>/debug_report.md`, single file, appended chronologically.

Format:

```markdown
# Debug Report — colony_coder

## 14:23:01 — plan / design_debate / claude_propose

**Output** (2,341 chars):

```
I suggest using MVC architecture to implement the snake battle game...
```

**State changes:**
- `routing_target` = `__end__`
- `tasks` = 4 items

---
```

Independent of and non-conflicting with existing `flow.md` / `thinking.md` / `state_snapshots.md`.

---

## run_colony_coder_debug.py

Independent debug runner script for running ColonyCoder + snake game tasks:

```python
#!/usr/bin/env python3
"""ColonyCoder debug runner — run snake game with full debug output."""

import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

import blueprints.functional_graphs.colony_coder.state  # noqa: F401
from framework.agent_loader import EntityLoader
from framework.debug import set_debug
from framework.debug_reporter import DebugConsoleReporter

SNAKE_TASK = """Write a two-player snake battle game in Python (Snake Battle).
... (using existing prompt from run_colony_coder_e2e.py)
"""

async def main():
    set_debug(True)
    
    loader = EntityLoader(Path("blueprints/functional_graphs/colony_coder"))
    graph = await loader.build_graph(checkpointer=None)
    
    reporter = DebugConsoleReporter("colony_coder")
    
    async for namespace, event in graph.astream(
        {"messages": [HumanMessage(content=SNAKE_TASK)]},
        stream_mode="updates",
        subgraphs=True,
    ):
        reporter.on_event(namespace, event)
    
    reporter.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
```

### Script Behavior

1. Enable debug mode.
2. Build colony_coder graph (without checkpointer).
3. Execute streaming with `subgraphs=True`.
4. Real-time printing of each event to the console.
5. Concurrent writing to the markdown log.
6. Print execution summary upon completion.

---

## Relationship with Existing Debug Systems

| System | Data Source | Output | Status |
|------|--------|------|------|
| `_wrap_node_for_flow_log` | Node wrapper | `flow.md` + `state_snapshots.md` | Retained, continues working |
| `log_node_thinking` | Inside LlmNode | `thinking.md` | Retained, continues working |
| `DebugConsoleReporter` (New) | `astream(subgraphs=True)` | console + `debug_report.md` | New |

All three can work simultaneously, outputting to different locations without conflict.
The advantage of `DebugConsoleReporter` is seeing the full subgraph hierarchy (via namespace),
whereas `_wrap_node_for_flow_log`'s `flow.md` lacks subgraph hierarchy information (as `push_graph_scope` has been removed).
