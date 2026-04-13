# Unified Subgraph Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace async wrapper-based subgraph invocation with LangGraph native subgraph + symmetric init/exit cleanup nodes. Replace custom `_keep_last_2` with native `add_messages`. Implement `inherit` session_mode. Delete dead `SubgraphMapperNode`.

**Architecture:** `make_subgraph_init(session_mode)` produces entry cleanup (only for `fresh_per_call` / `isolated`). `make_subgraph_exit()` produces exit cleanup (for ALL subgraphs — removes internal messages via `RemoveMessage`). Injected at build time:
```
START → [_subgraph_init] → entry → ... → exit → _subgraph_exit → END
```
Exit is always injected. Init is only injected for `fresh_per_call` and `isolated`.

**Tech Stack:** LangGraph 1.0.10, langchain-core 1.2.17, Python 3, pytest, `RemoveMessage`

**Design doc:** `docs/vault/architecture/unified-subgraph-integration.md`

---

### Task 1: Create `subgraph_init_node.py` with unit tests

**Files:**
- Create: `framework/nodes/subgraph_init_node.py`
- Create: `tests/test_subgraph_init_node.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_subgraph_init_node.py
"""
Unit tests for make_subgraph_init / make_subgraph_exit — factories that produce
deterministic cleanup nodes for subgraph boundaries.
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage


# ── make_subgraph_init tests ──────────────────────────────────────────────


def test_init_persistent_returns_none():
    """persistent needs no init cleanup."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    assert make_subgraph_init("persistent") is None


def test_init_inherit_returns_none():
    """inherit needs no init cleanup (inherits parent state as-is)."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    assert make_subgraph_init("inherit") is None


def test_init_fresh_per_call_clears_sessions_and_messages():
    """fresh_per_call must clear node_sessions, trim messages to last human, clear output fields."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("fresh_per_call")
    assert fn is not None

    state = {
        "messages": [
            HumanMessage(content="old question", id="h1"),
            AIMessage(content="old answer", id="a1"),
            HumanMessage(content="current topic", id="h2"),
        ],
        "node_sessions": {"claude_main": "uuid-old"},
        "routing_context": "some routing signal",
        "subgraph_topic": "",
        "debate_conclusion": "stale conclusion",
        "apex_conclusion": "stale apex",
        "knowledge_result": "stale knowledge",
        "discovery_report": "stale discovery",
        "previous_node_output": "stale output",
    }
    result = fn(state)

    assert result["node_sessions"] == {}
    assert result["debate_conclusion"] == ""
    assert result["apex_conclusion"] == ""
    assert result["knowledge_result"] == ""
    assert result["discovery_report"] == ""
    assert result["previous_node_output"] == ""
    assert result["routing_context"] == ""
    assert result["subgraph_topic"] == "some routing signal"

    msgs = result["messages"]
    removes = [m for m in msgs if isinstance(m, RemoveMessage)]
    humans = [m for m in msgs if isinstance(m, HumanMessage)]
    assert len(removes) == 3
    assert len(humans) == 1
    assert humans[0].content == "current topic"


def test_init_fresh_per_call_empty_messages():
    """fresh_per_call with no messages should not crash."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("fresh_per_call")
    state = {
        "messages": [],
        "node_sessions": {"x": "y"},
        "routing_context": "",
        "subgraph_topic": "topic",
        "debate_conclusion": "",
        "apex_conclusion": "",
        "knowledge_result": "",
        "discovery_report": "",
        "previous_node_output": "",
    }
    result = fn(state)
    assert len(result["messages"]) == 0
    assert result["subgraph_topic"] == "topic"


def test_init_fresh_per_call_no_human_message_keeps_last():
    """If no HumanMessage exists, keep the last message (whatever type)."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("fresh_per_call")
    state = {
        "messages": [AIMessage(content="only ai msg", id="a1")],
        "node_sessions": {},
        "routing_context": "",
        "subgraph_topic": "",
        "debate_conclusion": "",
        "apex_conclusion": "",
        "knowledge_result": "",
        "discovery_report": "",
        "previous_node_output": "",
    }
    result = fn(state)
    msgs = result["messages"]
    removes = [m for m in msgs if isinstance(m, RemoveMessage)]
    non_removes = [m for m in msgs if not isinstance(m, RemoveMessage)]
    assert len(removes) == 1
    assert len(non_removes) == 1
    assert non_removes[0].content == "only ai msg"


def test_init_fresh_per_call_topic_fallback():
    """If routing_context is empty, preserve existing subgraph_topic."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("fresh_per_call")
    state = {
        "messages": [],
        "node_sessions": {},
        "routing_context": "",
        "subgraph_topic": "existing topic",
        "debate_conclusion": "",
        "apex_conclusion": "",
        "knowledge_result": "",
        "discovery_report": "",
        "previous_node_output": "",
    }
    result = fn(state)
    assert result["subgraph_topic"] == "existing topic"


def test_init_isolated_clears_only_sessions():
    """isolated mode only clears node_sessions."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("isolated")
    assert fn is not None
    state = {
        "node_sessions": {"claude_main": "uuid-123", "gemini": "uuid-456"},
        "messages": [HumanMessage(content="hello", id="h1")],
    }
    result = fn(state)
    assert result == {"node_sessions": {}}


def test_init_unknown_mode_returns_none():
    """Unknown session_mode returns None."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    assert make_subgraph_init("some_future_mode") is None


# ── make_subgraph_exit tests ──────────────────────────────────────────────


def test_exit_removes_all_messages():
    """Exit must RemoveMessage all internal messages."""
    from framework.nodes.subgraph_init_node import make_subgraph_exit
    fn = make_subgraph_exit()
    assert fn is not None
    state = {
        "messages": [
            HumanMessage(content="topic", id="h1"),
            AIMessage(content="propose", id="a1"),
            AIMessage(content="critique", id="a2"),
            AIMessage(content="revise", id="a3"),
            AIMessage(content="conclusion", id="a4"),
        ],
    }
    result = fn(state)
    msgs = result["messages"]
    removes = [m for m in msgs if isinstance(m, RemoveMessage)]
    non_removes = [m for m in msgs if not isinstance(m, RemoveMessage)]
    assert len(removes) == 5
    assert len(non_removes) == 0
    assert {m.id for m in removes} == {"h1", "a1", "a2", "a3", "a4"}


def test_exit_empty_messages():
    """Exit with no messages returns empty list."""
    from framework.nodes.subgraph_init_node import make_subgraph_exit
    fn = make_subgraph_exit()
    result = fn({"messages": []})
    assert result == {"messages": []}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_subgraph_init_node.py -v`
Expected: `ModuleNotFoundError: No module named 'framework.nodes.subgraph_init_node'`

- [ ] **Step 3: Implement `make_subgraph_init` and `make_subgraph_exit`**

```python
# framework/nodes/subgraph_init_node.py
"""
subgraph_init_node — symmetric init/exit cleanup nodes for subgraph boundaries.

Injected by _build_declarative() at subgraph build time:

    START → [_subgraph_init] → entry → ... → exit → _subgraph_exit → END

_subgraph_init:  entry cleanup, only for fresh_per_call / isolated (persistent/inherit = None)
_subgraph_exit:  exit cleanup, for ALL subgraphs — RemoveMessage all internal messages

See docs/vault/architecture/unified-subgraph-integration.md for design.
"""

import logging
from langchain_core.messages import HumanMessage, RemoveMessage

logger = logging.getLogger(__name__)


def make_subgraph_init(session_mode: str):
    """Return entry cleanup function per session_mode.

    Returns None for persistent, inherit, and unknown modes (no init needed).
    """

    if session_mode == "fresh_per_call":

        def _fresh_init(state: dict) -> dict:
            msgs = state.get("messages", [])
            removals = [RemoveMessage(id=m.id) for m in msgs]
            human_msgs = [m for m in reversed(msgs) if getattr(m, "type", "") == "human"]
            if human_msgs:
                fresh = [HumanMessage(content=human_msgs[0].content)]
            elif msgs:
                last = msgs[-1]
                fresh = [type(last)(content=last.content)]
            else:
                fresh = []
            _topic = state.get("routing_context", "") or state.get("subgraph_topic", "")
            logger.debug(
                "[subgraph_init:fresh_per_call] clearing sessions + output fields + "
                "trimming messages %d → %d", len(msgs), len(fresh),
            )
            return {
                "node_sessions": {},
                "messages": removals + fresh,
                "routing_context": "",
                "debate_conclusion": "",
                "apex_conclusion": "",
                "knowledge_result": "",
                "discovery_report": "",
                "previous_node_output": "",
                "subgraph_topic": _topic,
            }

        return _fresh_init

    elif session_mode == "isolated":

        def _isolated_init(state: dict) -> dict:
            logger.debug("[subgraph_init:isolated] clearing node_sessions")
            return {"node_sessions": {}}

        return _isolated_init

    else:  # persistent, inherit, unknown
        return None


def make_subgraph_exit():
    """Return exit cleanup function — uniform for ALL subgraphs.

    Removes all internal messages via RemoveMessage so they don't
    pollute the parent graph's message list.
    """

    def _exit_cleanup(state: dict) -> dict:
        msgs = state.get("messages", [])
        removals = [RemoveMessage(id=m.id) for m in msgs]
        logger.debug("[subgraph_exit] removing %d internal messages", len(msgs))
        return {"messages": removals}

    return _exit_cleanup
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_subgraph_init_node.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add framework/nodes/subgraph_init_node.py tests/test_subgraph_init_node.py
git commit -m "feat: add make_subgraph_init/exit factories for native subgraph cleanup"
```

---

### Task 2: Replace `_keep_last_2` with `add_messages` in BaseAgentState

**Files:**
- Modify: `framework/schema/base.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_subgraph_init_node.py`:

```python
def test_base_agent_state_uses_add_messages():
    """BaseAgentState must use add_messages reducer, not _keep_last_2."""
    from framework.schema.base import BaseAgentState
    from typing import get_type_hints
    hints = get_type_hints(BaseAgentState, include_extras=True)
    messages_hint = hints["messages"]
    metadata = getattr(messages_hint, "__metadata__", ())
    from langgraph.graph.message import add_messages
    assert add_messages in metadata, (
        f"BaseAgentState.messages should use add_messages reducer, got {metadata}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_subgraph_init_node.py::test_base_agent_state_uses_add_messages -v`
Expected: FAIL — currently uses `_keep_last_2`

- [ ] **Step 3: Edit `framework/schema/base.py`**

Changes:
1. Update module docstring: remove "只保留最近 2 条消息", describe new behavior
2. Add `from langgraph.graph.message import add_messages`
3. Delete the entire `_keep_last_2` function (lines 17-26)
4. Change `messages: Annotated[list[BaseMessage], _keep_last_2]` to `messages: Annotated[list[BaseMessage], add_messages]`

```python
# BEFORE (line 31):
    messages: Annotated[list[BaseMessage], _keep_last_2]

# AFTER:
    messages: Annotated[list[BaseMessage], add_messages]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_subgraph_init_node.py::test_base_agent_state_uses_add_messages -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add framework/schema/base.py tests/test_subgraph_init_node.py
git commit -m "refactor: replace _keep_last_2 with add_messages in BaseAgentState"
```

---

### Task 3: Inject `_subgraph_init` and `_subgraph_exit` in `_build_declarative`

**Files:**
- Modify: `framework/agent_loader.py:720-728` (signature)
- Modify: `framework/agent_loader.py:1035-1047` (START/END logic)

- [ ] **Step 1: Write failing tests for injection**

Add to `tests/test_subgraph_init_node.py`:

```python
@pytest.mark.asyncio
async def test_fresh_per_call_injects_init_and_exit():
    """fresh_per_call subgraph must have both _subgraph_init and _subgraph_exit."""
    from framework.agent_loader import _build_declarative
    from framework.config import AgentConfig
    graph_spec = {
        "state_schema": "debate_schema",
        "entry": "fake_node",
        "exit": "fake_node",
        "nodes": [{"id": "fake_node", "type": "DETERMINISTIC"}],
        "edges": [],
    }
    graph = await _build_declarative(
        graph_spec, AgentConfig(), checkpointer=None,
        is_subgraph=True, session_mode="fresh_per_call",
    )
    node_names = [n.name for n in graph.get_graph(xray=True).nodes.values()]
    assert "_subgraph_init" in node_names, f"missing _subgraph_init: {node_names}"
    assert "_subgraph_exit" in node_names, f"missing _subgraph_exit: {node_names}"


@pytest.mark.asyncio
async def test_persistent_injects_exit_only():
    """persistent subgraph must have _subgraph_exit but NOT _subgraph_init."""
    from framework.agent_loader import _build_declarative
    from framework.config import AgentConfig
    graph_spec = {
        "entry": "fake_node",
        "exit": "fake_node",
        "nodes": [{"id": "fake_node", "type": "DETERMINISTIC"}],
        "edges": [],
    }
    graph = await _build_declarative(
        graph_spec, AgentConfig(), checkpointer=None,
        is_subgraph=True, session_mode="persistent",
    )
    node_names = [n.name for n in graph.get_graph(xray=True).nodes.values()]
    assert "_subgraph_init" not in node_names, f"persistent should NOT have _subgraph_init: {node_names}"
    assert "_subgraph_exit" in node_names, f"missing _subgraph_exit: {node_names}"


@pytest.mark.asyncio
async def test_inherit_injects_exit_only():
    """inherit subgraph must have _subgraph_exit but NOT _subgraph_init."""
    from framework.agent_loader import _build_declarative
    from framework.config import AgentConfig
    graph_spec = {
        "entry": "fake_node",
        "exit": "fake_node",
        "nodes": [{"id": "fake_node", "type": "DETERMINISTIC"}],
        "edges": [],
    }
    graph = await _build_declarative(
        graph_spec, AgentConfig(), checkpointer=None,
        is_subgraph=True, session_mode="inherit",
    )
    node_names = [n.name for n in graph.get_graph(xray=True).nodes.values()]
    assert "_subgraph_init" not in node_names
    assert "_subgraph_exit" in node_names


@pytest.mark.asyncio
async def test_non_subgraph_has_no_boundary_nodes():
    """Top-level graph (is_subgraph=False) must NOT inject init/exit."""
    from framework.agent_loader import _build_declarative
    from framework.config import AgentConfig
    graph_spec = {
        "entry": "fake_node",
        "exit": "fake_node",
        "nodes": [{"id": "fake_node", "type": "DETERMINISTIC"}],
        "edges": [],
    }
    graph = await _build_declarative(
        graph_spec, AgentConfig(), checkpointer=None,
        is_subgraph=False,
    )
    node_names = [n.name for n in graph.get_graph(xray=True).nodes.values()]
    assert "_subgraph_init" not in node_names
    assert "_subgraph_exit" not in node_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_subgraph_init_node.py::test_fresh_per_call_injects_init_and_exit -v`
Expected: FAIL — `_build_declarative` does not accept `session_mode`

- [ ] **Step 3: Add `session_mode` parameter to `_build_declarative`**

In `framework/agent_loader.py`, line 720 — add to signature:

```python
# AFTER:
async def _build_declarative(
    graph_spec: dict,
    config: AgentConfig,
    checkpointer,
    blueprint_dir: str = "",
    is_subgraph: bool = False,
    force_unique_session_keys: bool = False,
    extra_persona_text: str = "",
    session_mode: str = "persistent",
) -> object:
```

- [ ] **Step 4: Replace the START/END injection logic (lines 1035-1047)**

```python
# AFTER:
    _graph_entry = graph_spec.get("entry")
    _graph_exit  = graph_spec.get("exit")
    _has_start_edge = any(e["from"] == "__start__" for e in graph_spec.get("edges", []))
    _has_end_edge   = any(e["to"]   == "__end__"   for e in graph_spec.get("edges", []))

    _needs_init = is_subgraph and session_mode in ("fresh_per_call", "isolated")
    _needs_exit = is_subgraph  # ALL subgraphs get exit cleanup

    # ── Entry side ────────────────────────────────────────────────────
    if _needs_init and _graph_entry and not _has_start_edge:
        from framework.nodes.subgraph_init_node import make_subgraph_init
        _init_fn = make_subgraph_init(session_mode)
        builder.add_node("_subgraph_init", _init_fn)
        builder.add_edge(START, "_subgraph_init")
        builder.add_edge("_subgraph_init", _graph_entry)
        logger.debug(f"[agent_loader] subgraph_init injected: START → _subgraph_init → {_graph_entry!r} (mode={session_mode})")
    elif _graph_entry and not _has_start_edge:
        builder.add_edge(START, _graph_entry)
        logger.debug(f"[agent_loader] dynamic entry: START → {_graph_entry!r}")

    # ── Exit side ─────────────────────────────────────────────────────
    if _needs_exit and _graph_exit and not _has_end_edge:
        from framework.nodes.subgraph_init_node import make_subgraph_exit
        _exit_fn = make_subgraph_exit()
        builder.add_node("_subgraph_exit", _exit_fn)
        builder.add_edge(_graph_exit, "_subgraph_exit")
        builder.add_edge("_subgraph_exit", END)
        logger.debug(f"[agent_loader] subgraph_exit injected: {_graph_exit!r} → _subgraph_exit → END")
    elif _graph_exit and not _has_end_edge:
        builder.add_edge(_graph_exit, END)
        logger.debug(f"[agent_loader] dynamic exit: {_graph_exit!r} → END")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_subgraph_init_node.py -v`
Expected: All 16 tests PASS

- [ ] **Step 6: Commit**

```bash
git add framework/agent_loader.py tests/test_subgraph_init_node.py
git commit -m "feat: inject _subgraph_init + _subgraph_exit in _build_declarative"
```

---

### Task 4: Unify external subgraph branch — replace wrappers with native subgraph

**Files:**
- Modify: `framework/agent_loader.py:169` (`build_graph` signature)
- Modify: `framework/agent_loader.py:841-953` (session_mode branch)
- Modify: `framework/agent_loader.py:706-717` (delete `_get_subgraph_session_keys`)
- Modify: `framework/agent_loader.py:35` (imports)

- [ ] **Step 1: Add `session_mode` to `build_graph` signature**

Line 169:
```python
# AFTER:
    async def build_graph(self, checkpointer=_DEFAULT, extra_persona_text: str = "", is_subgraph: bool = False, force_unique_session_keys: bool = False, session_mode: str = "persistent"):
```

Find where `build_graph` calls `_build_declarative` (Priority 2 branch) and add `session_mode=session_mode`.

- [ ] **Step 2: Replace session_mode branch (lines 841-953)**

Delete `_fresh_wrapper`, `_isolated_wrapper`, and the 4-branch if/elif. Replace with:

```python
            _force_unique = session_mode == "isolated"
            inner_graph = await inner_loader.build_graph(
                checkpointer=None, extra_persona_text=_child_extra_persona, is_subgraph=True,
                force_unique_session_keys=_force_unique,
                session_mode=session_mode,
            )

            if session_mode in ("persistent", "fresh_per_call", "inherit", "isolated"):
                # All modes: native subgraph. Init/exit nodes injected inside by _build_declarative.
                builder.add_node(node_id, inner_graph)
            else:
                raise ValueError(
                    f"subgraph '{node_id}': unknown session_mode {session_mode!r} "
                    f"(valid: persistent, fresh_per_call, inherit, isolated)"
                )
```

- [ ] **Step 3: Remove unused imports**

Line 35 — remove `push_graph_scope, pop_graph_scope`:
```python
# AFTER:
from framework.debug import is_debug, log_graph_flow, log_state_snapshot
```

- [ ] **Step 4: Delete `_get_subgraph_session_keys` (lines 706-717)**

Delete the entire function.

- [ ] **Step 5: Run integration tests**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_session_mode_integration.py -v`

- [ ] **Step 6: Commit**

```bash
git add framework/agent_loader.py
git commit -m "refactor: replace async wrappers with native subgraph for all session_mode"
```

---

### Task 5: Delete SubgraphMapperNode dead code

**Files:**
- Delete: `framework/nodes/subgraph/subgraph_mapper.py`
- Modify: `framework/builtins.py`
- Modify: `framework/schema/base.py` (comments)
- Modify: `framework/nodes/llm/llm_node.py` (comments)
- Modify: `framework/nodes/llm/gemini.py` (comments)

- [ ] **Step 1: Delete `subgraph_mapper.py`**

```bash
rm framework/nodes/subgraph/subgraph_mapper.py
```

- [ ] **Step 2: Remove SUBGRAPH_MAPPER from `builtins.py`**

Delete the `@register_node("SUBGRAPH_MAPPER")` factory (around line 106-109) and the line in the module docstring (line 18).

- [ ] **Step 3: Update comments referencing SubgraphMapperNode**

In `framework/schema/base.py`:
- Line 43: `SubgraphMapperNode 入口写入、出口清空` → `_subgraph_init 入口写入`
- Line 44: `SubgraphMapperNode 入口清空` → `_subgraph_init 入口清空`
- Line 74: `SubgraphMapperNode 或 LLM 节点` → `_subgraph_init 或 LLM 节点`

In `framework/nodes/llm/llm_node.py`:
- Line 321: `SubgraphMapperNode 入口已清空` → `_subgraph_init 入口已清空`
- Line 507: `SubgraphMapperNode 入口负责清空` → `_subgraph_init 入口负责清空`

In `framework/nodes/llm/gemini.py`:
- Line 887: `SubgraphMapperNode 入口已清空` → `_subgraph_init 入口已清空`

- [ ] **Step 4: Run tests**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/ -v --tb=short 2>&1 | head -80`

- [ ] **Step 5: Commit**

```bash
git add -u framework/nodes/subgraph/subgraph_mapper.py framework/builtins.py framework/schema/base.py framework/nodes/llm/llm_node.py framework/nodes/llm/gemini.py
git commit -m "refactor: delete unused SubgraphMapperNode dead code"
```

---

### Task 6: Update existing tests

**Files:**
- Modify: `tests/test_session_mode.py`
- Modify: `tests/test_session_mode_integration.py`

- [ ] **Step 1: Update `test_session_mode.py`**

1. **DELETE** `test_get_subgraph_session_keys_basic` and `test_get_subgraph_session_keys_empty`
2. **REPLACE** `test_fresh_per_call_wrapper_clears_node_sessions` with:

```python
@pytest.mark.asyncio
async def test_fresh_per_call_uses_subgraph_init_node():
    """fresh_per_call must inject _subgraph_init/_subgraph_exit (not async wrapper)."""
    import framework.agent_loader as al
    src = inspect.getsource(al._build_declarative)
    assert "_fresh_wrapper" not in src, "_fresh_wrapper should be removed"
    assert "_subgraph_init" in src
    assert "_subgraph_exit" in src
```

3. **REPLACE** `test_isolated_wrapper_clears_node_sessions` with:

```python
@pytest.mark.asyncio
async def test_isolated_uses_native_subgraph():
    """isolated must use native subgraph (not async wrapper)."""
    import framework.agent_loader as al
    src = inspect.getsource(al._build_declarative)
    assert "_isolated_wrapper" not in src, "_isolated_wrapper should be removed"
```

4. **DELETE** `test_inherit_wrapper_injects_parent_session` — inherit is now implemented, not a conceptual test
5. **KEEP** `test_all_four_modes_referenced`, `test_unknown_session_mode_raises`, entity.json checks, `test_force_unique_session_keys_overrides_shared_keys`

- [ ] **Step 2: Update `test_session_mode_integration.py`**

1. Replace `test_inherit_raises_not_implemented` with a test that verifies inherit WORKS:

```python
@pytest.mark.asyncio
async def test_inherit_preserves_parent_sessions(tmp_path):
    """inherit: subgraph inherits parent's node_sessions without clearing."""
    inner_dir = _make_recorder_dir(tmp_path, "inherit_inner")
    parent = await _build_parent(_parent_spec(inner_dir, "inherit"))
    cfg = {"configurable": {"thread_id": f"inherit-{uuid.uuid4().hex[:6]}"}}

    result = await parent.ainvoke(dict(BASE_STATE), config=cfg)
    sid = result.get("debate_conclusion", "")
    # inherit: node_sessions not cleared, inner graph sees parent's sessions.
    # The recorder checks for "test_session" key which is empty in parent →
    # generates new UUID. This is expected — inherit means "don't clear", not "inject".
    assert sid, "inherit: debate_conclusion is empty (inner graph did not run?)"
```

2. Add `session_mode=session_mode` assertion to `test_isolated_clears_sessions_and_unique_keys_applied`.

- [ ] **Step 3: Run all session_mode tests**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_session_mode.py tests/test_session_mode_integration.py tests/test_subgraph_init_node.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_session_mode.py tests/test_session_mode_integration.py
git commit -m "test: update session_mode tests for native subgraph + init/exit + inherit"
```

---

### Task 7: Update comments, docs, and design doc status

**Files:**
- Modify: `framework/agent_loader.py:765` (comment)
- Modify: `docs/vault/architecture/unified-subgraph-integration.md` (status)

- [ ] **Step 1: Update comment at line 765**

```python
# BEFORE:
    # 自定义 schema 的跨调用字段污染由 session_mode wrapper（_fresh_wrapper 等）清理。

# AFTER:
    # 自定义 schema 的跨调用字段污染由 _subgraph_init 节点清理（注入在 START → entry 之间）。
    # 子图内部 messages 由 _subgraph_exit 节点清理（注入在 exit → END 之间），防止污染父图。
```

- [ ] **Step 2: Update design doc status**

In `docs/vault/architecture/unified-subgraph-integration.md`, line 4:
```markdown
> Status: Implemented
```

- [ ] **Step 3: Commit**

```bash
git add framework/agent_loader.py docs/vault/architecture/unified-subgraph-integration.md
git commit -m "docs: update comments and design doc for unified subgraph integration"
```

---

### Task 8: Full regression test suite and verification

**Files:** (no changes — verification only)

- [ ] **Step 1: Run all tests**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/ -v --tb=short 2>&1 | head -120`
Expected: All tests pass.

- [ ] **Step 2: Verify xray visibility**

```bash
cd /home/kingy/Foundation/ZenithLoom && python3 -c "
import asyncio
from framework.agent_loader import EntityLoader
async def check():
    loader = EntityLoader('blueprints/role_agents/technical_architect')
    graph = await loader.build_graph(checkpointer=None)
    nodes = [n.name for n in graph.get_graph(xray=True).nodes.values()]
    print('xray nodes:', nodes)
    print(f'_subgraph_init visible: {\"_subgraph_init\" in nodes}')
    print(f'_subgraph_exit visible: {\"_subgraph_exit\" in nodes}')
asyncio.run(check())
"
```

Expected: Both `_subgraph_init` and `_subgraph_exit` visible.

- [ ] **Step 3: Final commit (if any fixups needed)**

```bash
git add -A && git commit -m "fix: regression fixes from unified subgraph integration"
```
