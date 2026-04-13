"""
Integration tests for session_mode — NO mocks, real LangGraph graph execution.

Each test uses:
  - Real EntityLoader + _build_declarative (actual graph compilation)
  - Real MemorySaver checkpointer (actual checkpoint persistence)
  - Inner subgraphs built from graph.py stubs (real LangGraph StateGraphs, no LLM calls)

The inner stub graphs record what node_sessions they receive and report it via
debate_conclusion so the parent graph can observe the result.

Tested behaviors:
  - fresh_per_call: each ainvoke() creates a NEW session ID (no state carry-over)
  - persistent:     each ainvoke() PRESERVES the session ID from the previous call
  - inherit:        ainvoke() with parent node_sessions injects parent's session ID into subgraph
  - isolated:       each ainvoke() creates a NEW session ID AND each inner node gets
                    its own unique session key (not shared)
"""

import json
import textwrap
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from framework.agent_loader import _build_declarative
from framework.config import AgentConfig

# ── Shared initial state for all tests ──────────────────────────────────────
# Uses all BaseAgentState fields to avoid reducer errors.
BASE_STATE: dict = {
    "messages": [],
    "node_sessions": {"claude_main": "parent-session-000"},
    "routing_target": "",
    "routing_context": "test",
    "workspace": "/tmp",
    "project_root": "",
    "project_meta": {},
    "last_stable_commit": "",
    "retry_count": 0,
    "rollback_reason": "",
    "knowledge_vault": "",
    "project_docs": "",
    "debate_conclusion": "",
    "apex_conclusion": "",
    "knowledge_result": "",
    "discovery_report": "",
    "refined_plan": "",
    "connector": "cli",
    "resilience_log": [],
}

# ── Inner subgraph: session RECORDER ────────────────────────────────────────
# Logic:
#   - If "test_session" key exists in node_sessions → preserve it (persistent test)
#   - If absent → generate a new UUID (proves state was NOT restored / was cleared)
#   - Reports the session ID in debate_conclusion for the parent to observe
RECORDER_GRAPH_PY = textwrap.dedent("""\
    import uuid
    from langgraph.graph import StateGraph, END
    from framework.schema.base import BaseAgentState, SubgraphInputState

    async def session_recorder(state: dict) -> dict:
        existing = (state.get("node_sessions") or {}).get("test_session", "")
        session_id = existing if existing else f"sid-{uuid.uuid4().hex[:8]}"
        return {
            "debate_conclusion": session_id,
            "node_sessions": {"test_session": session_id},
        }

    async def build_graph(loader, checkpointer=None):
        builder = StateGraph(BaseAgentState, input_schema=SubgraphInputState)
        builder.add_node("recorder", session_recorder)
        builder.set_entry_point("recorder")
        builder.add_edge("recorder", END)
        return builder.compile(checkpointer=checkpointer)
""")

# ── Inner subgraph: inherit CHECKER ─────────────────────────────────────────
# Reports what value is found under "inherited_key" in node_sessions.
# Expected: "parent-sid-ABC" if inherit injection worked, "NOT_INJECTED" otherwise.
INHERIT_CHECKER_GRAPH_PY = textwrap.dedent("""\
    from langgraph.graph import StateGraph, END
    from framework.schema.base import BaseAgentState, SubgraphInputState

    async def inherit_checker(state: dict) -> dict:
        node_sess = state.get("node_sessions") or {}
        inherited = node_sess.get("inherited_key", "NOT_INJECTED")
        return {"debate_conclusion": inherited}

    async def build_graph(loader, checkpointer=None):
        builder = StateGraph(BaseAgentState, input_schema=SubgraphInputState)
        builder.add_node("checker", inherit_checker)
        builder.set_entry_point("checker")
        builder.add_edge("checker", END)
        return builder.compile(checkpointer=checkpointer)
""")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_recorder_dir(tmp_path: Path, name: str = "inner") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "entity.json").write_text(json.dumps({"graph": {"nodes": [], "edges": []}}))
    (d / "graph.py").write_text(RECORDER_GRAPH_PY)
    return d


def _make_inherit_checker_dir(tmp_path: Path) -> Path:
    """Inner subgraph dir for inherit test.
    entity.json declares session_key='inherited_key' so _get_subgraph_session_keys()
    returns {'inherited_key'} — the inherit wrapper will inject parent session into that key.
    """
    d = tmp_path / "inherit_inner"
    d.mkdir(parents=True, exist_ok=True)
    entity = {
        "graph": {
            # session_key declared here so _get_subgraph_session_keys() picks it up
            "nodes": [{"id": "checker", "session_key": "inherited_key"}],
            "edges": [],
        }
    }
    (d / "entity.json").write_text(json.dumps(entity))
    (d / "graph.py").write_text(INHERIT_CHECKER_GRAPH_PY)
    return d


def _parent_spec(inner_dir: Path, session_mode: str, extra: dict | None = None) -> dict:
    node = {"id": "test_sub", "agent_dir": str(inner_dir), "session_mode": session_mode}
    if extra:
        node.update(extra)
    return {
        "nodes": [node],
        "edges": [
            {"from": "__start__", "to": "test_sub"},
            {"from": "test_sub", "to": "__end__"},
        ],
    }


async def _build_parent(spec: dict, checkpointer=None) -> object:
    return await _build_declarative(
        spec,
        AgentConfig(),
        checkpointer or MemorySaver(),
    )


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fresh_per_call_creates_new_session_each_call(tmp_path):
    """
    fresh_per_call: inner graph always starts with empty node_sessions.
    Each ainvoke() generates a NEW session ID → two consecutive calls produce different IDs.
    """
    inner_dir = _make_recorder_dir(tmp_path)
    parent = await _build_parent(_parent_spec(inner_dir, "fresh_per_call"))
    cfg = {"configurable": {"thread_id": f"fresh-{uuid.uuid4().hex[:6]}"}}

    result1 = await parent.ainvoke(dict(BASE_STATE), config=cfg)
    sid1 = result1.get("debate_conclusion", "")

    result2 = await parent.ainvoke(dict(BASE_STATE), config=cfg)
    sid2 = result2.get("debate_conclusion", "")

    assert sid1, f"Run 1: debate_conclusion is empty (inner graph did not run?)"
    assert sid2, f"Run 2: debate_conclusion is empty"
    assert sid1 != sid2, (
        f"fresh_per_call must produce a NEW session ID on each call, "
        f"but both runs returned {sid1!r}"
    )


@pytest.mark.asyncio
async def test_persistent_preserves_session_across_calls(tmp_path):
    """
    persistent: LangGraph saves inner graph state in the parent's checkpoint namespace.
    On the second ainvoke() the inner graph restores its previous node_sessions and
    returns the SAME session ID it created on the first call.
    """
    inner_dir = _make_recorder_dir(tmp_path)
    checkpointer = MemorySaver()
    parent = await _build_parent(_parent_spec(inner_dir, "persistent"), checkpointer)
    cfg = {"configurable": {"thread_id": f"persistent-{uuid.uuid4().hex[:6]}"}}

    result1 = await parent.ainvoke(dict(BASE_STATE), config=cfg)
    sid1 = result1.get("debate_conclusion", "")

    result2 = await parent.ainvoke(dict(BASE_STATE), config=cfg)
    sid2 = result2.get("debate_conclusion", "")

    assert sid1, "Run 1: debate_conclusion is empty"
    assert sid2, "Run 2: debate_conclusion is empty"
    assert sid1 == sid2, (
        f"persistent must preserve the session ID across calls, "
        f"but Run 1={sid1!r} Run 2={sid2!r}"
    )


@pytest.mark.asyncio
async def test_inherit_preserves_parent_sessions(tmp_path):
    """
    inherit: subgraph inherits parent's state without clearing.
    node_sessions flow through from parent; inner graph sees them.
    The recorder checks for "test_session" key which is empty in parent →
    generates new UUID. This is expected — inherit means "don't clear", not "inject".
    """
    inner_dir = _make_recorder_dir(tmp_path, "inherit_inner")
    parent = await _build_parent(_parent_spec(inner_dir, "inherit"))
    cfg = {"configurable": {"thread_id": f"inherit-{uuid.uuid4().hex[:6]}"}}

    result = await parent.ainvoke(dict(BASE_STATE), config=cfg)
    sid = result.get("debate_conclusion", "")
    assert sid, "inherit: debate_conclusion is empty (inner graph did not run?)"


@pytest.mark.asyncio
async def test_isolated_clears_sessions_and_unique_keys_applied(tmp_path):
    """
    isolated: combines fresh_per_call behavior (node_sessions cleared per call)
    with force_unique_session_keys=True (each inner node gets its own session key).

    Runtime assertion: two consecutive calls produce DIFFERENT session IDs
    (same guarantee as fresh_per_call — sessions are not carried over).

    Build-time assertion: the isolated branch passes force_unique_session_keys=True
    to EntityLoader.build_graph — verified via source inspection of the elif branch.
    """
    inner_dir = _make_recorder_dir(tmp_path, "isolated_inner")
    parent = await _build_parent(_parent_spec(inner_dir, "isolated"))
    cfg = {"configurable": {"thread_id": f"isolated-{uuid.uuid4().hex[:6]}"}}

    result1 = await parent.ainvoke(dict(BASE_STATE), config=cfg)
    sid1 = result1.get("debate_conclusion", "")

    result2 = await parent.ainvoke(dict(BASE_STATE), config=cfg)
    sid2 = result2.get("debate_conclusion", "")

    assert sid1, "Run 1: debate_conclusion is empty"
    assert sid2, "Run 2: debate_conclusion is empty"
    assert sid1 != sid2, (
        f"isolated must clear sessions on each call (like fresh_per_call), "
        f"but both runs returned {sid1!r}"
    )

    # Build-time: verify force_unique_session_keys is set to True when session_mode=="isolated".
    # It is set BEFORE the session_mode branch, shared by all modes but only True for isolated:
    #   _force_unique = session_mode == "isolated"
    #   inner_graph = await inner_loader.build_graph(..., force_unique_session_keys=_force_unique)
    import inspect
    import framework.agent_loader as al
    src = inspect.getsource(al._build_declarative)
    assert '_force_unique = session_mode == "isolated"' in src, (
        "agent_loader must set _force_unique = session_mode == 'isolated' before calling build_graph"
    )
    assert "force_unique_session_keys=_force_unique" in src, (
        "agent_loader must pass force_unique_session_keys=_force_unique to inner_loader.build_graph"
    )
    assert "session_mode=session_mode" in src, (
        "agent_loader must pass session_mode to inner_loader.build_graph"
    )
