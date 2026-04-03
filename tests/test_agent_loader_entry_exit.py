"""
Tests for subgraph dynamic entry/exit support in agent_loader.

Covers:
  1. _check_edge_refs: entry references unknown node → ValueError
  2. _check_edge_refs: exit references unknown node → ValueError
  3. _check_reachable: entry field allows all nodes reachable → passes
  4. _check_reachable: entry field declared but node not reachable → ValueError
  5. _build_declarative: graph with entry/exit fields builds correctly
  6. Backward compatibility: __start__/__end__ edges still work
"""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from framework.agent_loader import (
    _check_edge_refs,
    _check_reachable,
    _collect_all_ids,
    EntityLoader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(nodes: list[str], edges: list[tuple[str, str]],
               entry: str | None = None, exit_node: str | None = None) -> dict:
    """Build a minimal graph_spec dict."""
    spec: dict = {
        "nodes": [{"id": nid, "type": "CLAUDE_CLI"} for nid in nodes],
        "edges": [{"from": src, "to": dst} for src, dst in edges],
    }
    if entry is not None:
        spec["entry"] = entry
    if exit_node is not None:
        spec["exit"] = exit_node
    return spec


def _all_ids(spec: dict) -> set:
    return _collect_all_ids(spec)


# ---------------------------------------------------------------------------
# 1. _check_edge_refs: entry references unknown node → ValueError
# ---------------------------------------------------------------------------

class TestCheckEdgeRefsEntry:
    """entry field referencing unknown node must raise ValueError."""

    def test_entry_references_missing_node(self):
        spec = _make_spec(
            nodes=["node_a", "node_b"],
            edges=[("node_a", "node_b")],
            entry="nonexistent_node",
        )
        all_ids = _all_ids(spec)
        with pytest.raises(ValueError, match="'entry' references unknown node"):
            _check_edge_refs(spec, all_ids)

    def test_entry_references_existing_node_passes(self):
        spec = _make_spec(
            nodes=["node_a", "node_b"],
            edges=[("node_a", "node_b")],
            entry="node_a",
        )
        all_ids = _all_ids(spec)
        # Should not raise
        _check_edge_refs(spec, all_ids)

    def test_entry_none_passes(self):
        """No entry field → no extra validation."""
        spec = _make_spec(
            nodes=["node_a"],
            edges=[("__start__", "node_a"), ("node_a", "__end__")],
        )
        all_ids = _all_ids(spec)
        _check_edge_refs(spec, all_ids)

    def test_entry_error_message_contains_node_name(self):
        spec = _make_spec(
            nodes=["real_node"],
            edges=[],
            entry="ghost_node",
        )
        all_ids = _all_ids(spec)
        with pytest.raises(ValueError, match="ghost_node"):
            _check_edge_refs(spec, all_ids)


# ---------------------------------------------------------------------------
# 2. _check_edge_refs: exit references unknown node → ValueError
# ---------------------------------------------------------------------------

class TestCheckEdgeRefsExit:
    """exit field referencing unknown node must raise ValueError."""

    def test_exit_references_missing_node(self):
        spec = _make_spec(
            nodes=["node_a", "node_b"],
            edges=[("node_a", "node_b")],
            exit_node="does_not_exist",
        )
        all_ids = _all_ids(spec)
        with pytest.raises(ValueError, match="'exit' references unknown node"):
            _check_edge_refs(spec, all_ids)

    def test_exit_references_existing_node_passes(self):
        spec = _make_spec(
            nodes=["node_a", "node_b"],
            edges=[("node_a", "node_b")],
            exit_node="node_b",
        )
        all_ids = _all_ids(spec)
        _check_edge_refs(spec, all_ids)

    def test_exit_none_passes(self):
        """No exit field → no extra validation."""
        spec = _make_spec(
            nodes=["node_a"],
            edges=[("__start__", "node_a"), ("node_a", "__end__")],
        )
        all_ids = _all_ids(spec)
        _check_edge_refs(spec, all_ids)

    def test_exit_error_message_contains_node_name(self):
        spec = _make_spec(
            nodes=["real_node"],
            edges=[],
            exit_node="phantom_exit",
        )
        all_ids = _all_ids(spec)
        with pytest.raises(ValueError, match="phantom_exit"):
            _check_edge_refs(spec, all_ids)

    def test_both_entry_and_exit_missing_raises_on_entry_first(self):
        """When both entry and exit are bad, entry is checked first."""
        spec = _make_spec(
            nodes=["node_a"],
            edges=[],
            entry="bad_entry",
            exit_node="bad_exit",
        )
        all_ids = _all_ids(spec)
        with pytest.raises(ValueError, match="'entry' references unknown node"):
            _check_edge_refs(spec, all_ids)


# ---------------------------------------------------------------------------
# 3. _check_reachable: entry field → all nodes reachable → passes
# ---------------------------------------------------------------------------

class TestCheckReachableEntryPasses:
    """When entry field is set and all nodes are reachable, no error."""

    def test_entry_field_linear_chain(self):
        """entry: node_a → node_b → node_c — all reachable."""
        spec = _make_spec(
            nodes=["node_a", "node_b", "node_c"],
            edges=[("node_a", "node_b"), ("node_b", "node_c")],
            entry="node_a",
        )
        all_ids = _all_ids(spec)
        # Should not raise
        _check_reachable(spec, all_ids)

    def test_entry_field_with_exit_no_end_edge(self):
        """entry+exit without __start__/__end__ edges — nodes still reachable."""
        spec = _make_spec(
            nodes=["first", "middle", "last"],
            edges=[("first", "middle"), ("middle", "last")],
            entry="first",
            exit_node="last",
        )
        all_ids = _all_ids(spec)
        _check_reachable(spec, all_ids)

    def test_entry_and_start_edge_coexist(self):
        """Both __start__ edge and entry field → all nodes reachable."""
        spec = {
            "entry": "node_a",
            "nodes": [
                {"id": "node_a", "type": "CLAUDE_CLI"},
                {"id": "node_b", "type": "CLAUDE_CLI"},
            ],
            "edges": [
                {"from": "__start__", "to": "node_a"},
                {"from": "node_a", "to": "node_b"},
                {"from": "node_b", "to": "__end__"},
            ],
        }
        all_ids = _all_ids(spec)
        _check_reachable(spec, all_ids)

    def test_empty_graph_with_entry_passes(self):
        """Empty nodes + entry field that doesn't exist as node — no nodes to check."""
        spec = {
            "entry": "some_node",
            "nodes": [],
            "edges": [],
        }
        # No nodes to check reachability for
        all_ids = _all_ids(spec)
        _check_reachable(spec, all_ids)


# ---------------------------------------------------------------------------
# 4. _check_reachable: entry field but node not reachable → ValueError
# ---------------------------------------------------------------------------

class TestCheckReachableEntryFails:
    """When entry is declared but some node is not reachable, raise ValueError."""

    def test_isolated_node_not_reachable_via_entry(self):
        """node_a → node_b chain, but node_c is isolated — not reachable."""
        spec = _make_spec(
            nodes=["node_a", "node_b", "node_c"],
            edges=[("node_a", "node_b")],
            entry="node_a",
        )
        all_ids = _all_ids(spec)
        with pytest.raises(ValueError, match="Unreachable nodes"):
            _check_reachable(spec, all_ids)

    def test_no_start_edge_and_no_entry_means_unreachable(self):
        """Without __start__ edge AND no entry → nodes unreachable."""
        spec = _make_spec(
            nodes=["node_a", "node_b"],
            edges=[("node_a", "node_b")],
            # No entry, no __start__ edge
        )
        all_ids = _all_ids(spec)
        with pytest.raises(ValueError, match="Unreachable nodes"):
            _check_reachable(spec, all_ids)

    def test_entry_node_disconnected_from_rest(self):
        """entry=node_a but node_a has no edges — node_b unreachable."""
        spec = _make_spec(
            nodes=["node_a", "node_b"],
            edges=[],
            entry="node_a",
        )
        all_ids = _all_ids(spec)
        with pytest.raises(ValueError, match="Unreachable nodes"):
            _check_reachable(spec, all_ids)


# ---------------------------------------------------------------------------
# 5. _build_declarative: graph with entry/exit builds correctly
# ---------------------------------------------------------------------------

def _make_agent_dir_with_entry_exit(tmp: Path, name: str,
                                     entry: str, exit_node: str) -> Path:
    """Create a minimal agent dir with entry/exit fields (no __start__/__end__ edges)."""
    agent_dir = tmp / name
    agent_dir.mkdir(parents=True)
    entity = {
        "name": name,
        "graph": {
            "entry": entry,
            "exit": exit_node,
            "nodes": [
                {"id": entry, "type": "CLAUDE_CLI", "model": "claude-opus-4-6",
                 "session_key": "test_session", "tools": []},
            ],
            "edges": [],
        },
    }
    if entry != exit_node:
        entity["graph"]["nodes"].append(
            {"id": exit_node, "type": "CLAUDE_CLI", "model": "claude-opus-4-6",
             "session_key": "test_session", "tools": []}
        )
        entity["graph"]["edges"].append({"from": entry, "to": exit_node})
    (agent_dir / "entity.json").write_text(json.dumps(entity))
    return agent_dir


class TestBuildDeclarativeEntryExit:
    """_build_declarative correctly adds START→entry and exit→END edges."""

    def test_single_node_entry_exit_same_node(self):
        """Minimal graph: one node that is both entry and exit."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = _make_agent_dir_with_entry_exit(
                Path(tmp), "single_node_graph", "only_node", "only_node"
            )
            loader = EntityLoader(agent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_two_node_entry_exit(self):
        """entry=first_node, exit=last_node, edge first→last."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = _make_agent_dir_with_entry_exit(
                Path(tmp), "two_node_graph", "first_node", "last_node"
            )
            loader = EntityLoader(agent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_entry_only_no_exit(self):
        """Only entry field declared, exit uses __end__ edge normally."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / "entry_only_graph"
            agent_dir.mkdir()
            entity = {
                "name": "entry_only",
                "graph": {
                    "entry": "worker",
                    "nodes": [
                        {"id": "worker", "type": "CLAUDE_CLI", "model": "claude-opus-4-6",
                         "session_key": "s", "tools": []}
                    ],
                    "edges": [
                        {"from": "worker", "to": "__end__"}
                    ],
                },
            }
            (agent_dir / "entity.json").write_text(json.dumps(entity))
            loader = EntityLoader(agent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_exit_only_no_entry(self):
        """Only exit field declared, entry uses __start__ edge normally."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / "exit_only_graph"
            agent_dir.mkdir()
            entity = {
                "name": "exit_only",
                "graph": {
                    "exit": "worker",
                    "nodes": [
                        {"id": "worker", "type": "CLAUDE_CLI", "model": "claude-opus-4-6",
                         "session_key": "s", "tools": []}
                    ],
                    "edges": [
                        {"from": "__start__", "to": "worker"}
                    ],
                },
            }
            (agent_dir / "entity.json").write_text(json.dumps(entity))
            loader = EntityLoader(agent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_entry_exit_skipped_when_start_end_edges_present(self):
        """When __start__/__end__ edges already exist, entry/exit fields are ignored for edge injection."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / "both_styles"
            agent_dir.mkdir()
            entity = {
                "name": "both_styles",
                "graph": {
                    # entry/exit declared but __start__/__end__ edges also exist
                    "entry": "worker",
                    "exit": "worker",
                    "nodes": [
                        {"id": "worker", "type": "CLAUDE_CLI", "model": "claude-opus-4-6",
                         "session_key": "s", "tools": []}
                    ],
                    "edges": [
                        {"from": "__start__", "to": "worker"},
                        {"from": "worker", "to": "__end__"},
                    ],
                },
            }
            (agent_dir / "entity.json").write_text(json.dumps(entity))
            loader = EntityLoader(agent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None


# ---------------------------------------------------------------------------
# 6. Backward compatibility: __start__/__end__ edges still work
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Original __start__/__end__ edge syntax must still work unchanged."""

    def test_classic_start_end_edges(self):
        """Standard graph with __start__ and __end__ edges builds normally."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / "classic_graph"
            agent_dir.mkdir()
            entity = {
                "name": "classic_graph",
                "graph": {
                    "nodes": [
                        {"id": "main_worker", "type": "CLAUDE_CLI",
                         "model": "claude-opus-4-6", "session_key": "s", "tools": []}
                    ],
                    "edges": [
                        {"from": "__start__", "to": "main_worker"},
                        {"from": "main_worker", "to": "__end__"},
                    ],
                },
            }
            (agent_dir / "entity.json").write_text(json.dumps(entity))
            loader = EntityLoader(agent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_classic_multi_node_chain(self):
        """Multi-node chain with __start__/__end__ edges builds normally."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / "classic_chain"
            agent_dir.mkdir()
            entity = {
                "name": "classic_chain",
                "graph": {
                    "nodes": [
                        {"id": "step_one", "type": "CLAUDE_CLI",
                         "model": "claude-opus-4-6", "session_key": "s", "tools": []},
                        {"id": "step_two", "type": "CLAUDE_CLI",
                         "model": "claude-opus-4-6", "session_key": "s2", "tools": []},
                    ],
                    "edges": [
                        {"from": "__start__", "to": "step_one"},
                        {"from": "step_one", "to": "step_two"},
                        {"from": "step_two", "to": "__end__"},
                    ],
                },
            }
            (agent_dir / "entity.json").write_text(json.dumps(entity))
            loader = EntityLoader(agent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_check_edge_refs_classic_passes(self):
        """__start__/__end__ are valid pseudo-nodes — no error."""
        spec = {
            "nodes": [{"id": "worker", "type": "CLAUDE_CLI"}],
            "edges": [
                {"from": "__start__", "to": "worker"},
                {"from": "worker", "to": "__end__"},
            ],
        }
        all_ids = _all_ids(spec)
        _check_edge_refs(spec, all_ids)

    def test_check_reachable_classic_start_edge(self):
        """__start__ edge provides reachability as before."""
        spec = {
            "nodes": [
                {"id": "a", "type": "X"},
                {"id": "b", "type": "X"},
            ],
            "edges": [
                {"from": "__start__", "to": "a"},
                {"from": "a", "to": "b"},
                {"from": "b", "to": "__end__"},
            ],
        }
        all_ids = _all_ids(spec)
        _check_reachable(spec, all_ids)

    def test_real_world_migrated_debate_spec(self):
        """Mirrors the migrated debate_gemini_first structure (entry/exit, no __start__/__end__)."""
        spec = {
            "state_schema": "debate_schema",
            "entry": "gemini_propose",
            "exit": "gemini_conclusion",
            "nodes": [
                {"id": "gemini_propose",   "type": "GEMINI_CLI"},
                {"id": "claude_critique_1","type": "CLAUDE_CLI"},
                {"id": "gemini_revise",    "type": "GEMINI_CLI"},
                {"id": "claude_critique_2","type": "CLAUDE_CLI"},
                {"id": "gemini_conclusion","type": "GEMINI_CLI"},
            ],
            "edges": [
                {"from": "gemini_propose",    "to": "claude_critique_1"},
                {"from": "claude_critique_1", "to": "gemini_revise"},
                {"from": "gemini_revise",     "to": "claude_critique_2"},
                {"from": "claude_critique_2", "to": "gemini_conclusion"},
            ],
        }
        all_ids = _all_ids(spec)
        _check_edge_refs(spec, all_ids)
        _check_reachable(spec, all_ids)
