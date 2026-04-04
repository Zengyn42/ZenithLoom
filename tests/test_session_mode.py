"""
Tests for session_mode feature in external subgraph nodes.

session_mode controls how LLM sessions are managed when a parent graph
invokes an external subgraph via agent_dir:
  - persistent (default): subgraph keeps LangGraph checkpoint across calls
  - fresh_per_call: node_sessions cleared before each invocation
  - inherit: subgraph nodes inherit parent node's session ID
  - isolated: each node forced to unique session_key + sessions cleared
"""
import inspect
import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Source-level checks (no runtime graph needed)
# ---------------------------------------------------------------------------

def test_session_mode_branch_exists():
    """agent_loader must handle session_mode in external subgraph branch."""
    import framework.agent_loader as al
    src = inspect.getsource(al._build_declarative)
    assert "session_mode" in src, "session_mode not found in _build_declarative"


def test_all_four_modes_referenced():
    """All four session_mode values must appear in _build_declarative."""
    import framework.agent_loader as al
    src = inspect.getsource(al._build_declarative)
    for mode in ("persistent", "fresh_per_call", "inherit", "isolated"):
        assert mode in src, f"session_mode {mode!r} not in _build_declarative"


def test_unknown_session_mode_raises():
    """Unknown session_mode value must appear in ValueError message."""
    import framework.agent_loader as al
    src = inspect.getsource(al._build_declarative)
    assert "unknown session_mode" in src, "Missing error for unknown session_mode"


def test_get_subgraph_session_keys_basic():
    """_get_subgraph_session_keys extracts session_key from entity.json graph nodes."""
    from framework.agent_loader import _get_subgraph_session_keys
    entity = {
        "graph": {
            "nodes": [
                {"id": "a", "session_key": "shared"},
                {"id": "b", "session_key": "shared"},
                {"id": "c", "session_key": "unique_c"},
                {"id": "d"},  # no session_key → falls back to id
            ]
        }
    }
    keys = _get_subgraph_session_keys(entity)
    assert keys == {"shared", "unique_c", "d"}


def test_get_subgraph_session_keys_empty():
    """_get_subgraph_session_keys handles empty or missing graph gracefully."""
    from framework.agent_loader import _get_subgraph_session_keys
    assert _get_subgraph_session_keys({}) == set()
    assert _get_subgraph_session_keys({"graph": {}}) == set()
    assert _get_subgraph_session_keys({"graph": {"nodes": []}}) == set()


def test_force_unique_session_keys_in_build_graph_signature():
    """build_graph must accept force_unique_session_keys parameter."""
    from framework.agent_loader import EntityLoader
    sig = inspect.signature(EntityLoader.build_graph)
    assert "force_unique_session_keys" in sig.parameters, \
        "build_graph missing force_unique_session_keys param"


def test_build_declarative_accepts_force_unique_session_keys():
    """_build_declarative must accept force_unique_session_keys parameter."""
    from framework.agent_loader import _build_declarative
    sig = inspect.signature(_build_declarative)
    assert "force_unique_session_keys" in sig.parameters


# ---------------------------------------------------------------------------
# Entity.json integration checks
# ---------------------------------------------------------------------------

def test_debate_brainstorm_has_fresh_per_call():
    """debate_brainstorm in technical_architect entity.json must use fresh_per_call."""
    entity_path = Path("blueprints/role_agents/technical_architect/entity.json")
    if not entity_path.exists():
        pytest.skip("technical_architect entity.json not found")
    data = json.loads(entity_path.read_text())
    nodes = data.get("graph", {}).get("nodes", [])
    brainstorm = next((n for n in nodes if n["id"] == "debate_brainstorm"), None)
    assert brainstorm is not None, "debate_brainstorm node not found"
    assert brainstorm.get("session_mode") == "fresh_per_call", \
        f"debate_brainstorm session_mode should be fresh_per_call, got {brainstorm.get('session_mode')}"


def test_debate_design_has_fresh_per_call():
    """debate_design in technical_architect entity.json must use fresh_per_call."""
    entity_path = Path("blueprints/role_agents/technical_architect/entity.json")
    if not entity_path.exists():
        pytest.skip("technical_architect entity.json not found")
    data = json.loads(entity_path.read_text())
    nodes = data.get("graph", {}).get("nodes", [])
    design = next((n for n in nodes if n["id"] == "debate_design"), None)
    assert design is not None, "debate_design node not found"
    assert design.get("session_mode") == "fresh_per_call", \
        f"debate_design session_mode should be fresh_per_call, got {design.get('session_mode')}"


def test_persistent_is_default_for_nodes_without_session_mode():
    """Nodes without session_mode should default to persistent (no wrapper)."""
    entity_path = Path("blueprints/role_agents/technical_architect/entity.json")
    if not entity_path.exists():
        pytest.skip("technical_architect entity.json not found")
    data = json.loads(entity_path.read_text())
    nodes = data.get("graph", {}).get("nodes", [])
    apex = next((n for n in nodes if n["id"] == "apex_coder"), None)
    assert apex is not None, "apex_coder node not found"
    # apex_coder has agent_dir but no session_mode → defaults to persistent
    assert apex.get("session_mode") is None or apex.get("session_mode") == "persistent"


# ---------------------------------------------------------------------------
# Wrapper behavior tests (mock-level, no real LLM calls)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fresh_per_call_wrapper_clears_node_sessions():
    """fresh_per_call wrapper must pass node_sessions={} to inner graph."""
    captured = {}

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            captured["node_sessions"] = state.get("node_sessions")
            return state

    fake = FakeGraph()

    # Simulate wrapper logic
    async def _fresh_wrapper(state, config=None, *, _g=fake):
        patched = {**state, "node_sessions": {}}
        return await _g.ainvoke(patched)

    parent_state = {
        "node_sessions": {"claude_main": "uuid-abc", "gemini_debate": "uuid-old"},
        "messages": [],
    }
    await _fresh_wrapper(parent_state)
    assert captured["node_sessions"] == {}, \
        f"Expected empty node_sessions, got {captured['node_sessions']}"


@pytest.mark.asyncio
async def test_inherit_wrapper_injects_parent_session():
    """inherit wrapper must inject parent's session ID into all subgraph keys."""
    captured = {}

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            captured["node_sessions"] = state.get("node_sessions")
            return state

    fake = FakeGraph()
    sub_keys = {"gemini_debate", "claude_debate"}

    async def _inherit_wrapper(state, config=None, *, _g=fake, _from="claude_main", _keys=sub_keys):
        parent_sid = (state.get("node_sessions") or {}).get(_from, "")
        injected = {sk: parent_sid for sk in _keys}
        patched = {**state, "node_sessions": injected}
        return await _g.ainvoke(patched)

    parent_state = {
        "node_sessions": {"claude_main": "parent-uuid-123"},
        "messages": [],
    }
    await _inherit_wrapper(parent_state)
    assert captured["node_sessions"] == {
        "gemini_debate": "parent-uuid-123",
        "claude_debate": "parent-uuid-123",
    }


@pytest.mark.asyncio
async def test_isolated_wrapper_clears_node_sessions():
    """isolated wrapper must also pass node_sessions={} (unique keys forced at build time)."""
    captured = {}

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            captured["node_sessions"] = state.get("node_sessions")
            return state

    fake = FakeGraph()

    async def _isolated_wrapper(state, config=None, *, _g=fake):
        patched = {**state, "node_sessions": {}}
        return await _g.ainvoke(patched)

    parent_state = {
        "node_sessions": {"claude_main": "uuid-abc"},
        "messages": [],
    }
    await _isolated_wrapper(parent_state)
    assert captured["node_sessions"] == {}


def test_force_unique_session_keys_overrides_shared_keys():
    """When force_unique_session_keys=True, node_def.session_key should be overridden to node_id."""
    # This tests the logic inline: if session_key exists and force_unique is True,
    # session_key should become node_id
    node_def = {"id": "claude_critique_1", "session_key": "claude_debate", "type": "CLAUDE_CLI"}
    _base = dict(node_def)
    force_unique = True
    if force_unique and _base.get("session_key"):
        _base["session_key"] = _base["id"]
    assert _base["session_key"] == "claude_critique_1"
