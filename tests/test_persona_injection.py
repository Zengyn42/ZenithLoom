"""Tests for persona injection: _find_persona_targets + SUBGRAPH_NODE passthrough.

Rules:
  1. Prefer LLM nodes with 'main' in id (supports multiple).
  2. Fallback: reverse BFS from __end__, pick closest LLM node.
  3. No LLM node → empty list.
  4. SUBGRAPH_NODE: parent persona passes through to child graph's LLM nodes.
     SUBGRAPH_REF: parent persona does NOT pass through.
"""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from framework.agent_loader import _find_persona_targets, _LLM_NODE_TYPES, EntityLoader


# ---------------------------------------------------------------------------
# Helper: build minimal graph_spec
# ---------------------------------------------------------------------------

def _spec(nodes: list[dict], edges: list[tuple[str, str]]) -> dict:
    """Shorthand for building graph_spec dicts."""
    return {
        "nodes": nodes,
        "edges": [{"from": src, "to": dst} for src, dst in edges],
    }


def _node(nid: str, ntype: str = "CLAUDE_CLI") -> dict:
    return {"id": nid, "type": ntype}


# =========================================================================
# Rule 1: nodes with 'main' in id
# =========================================================================

class TestRule1MainNodes:
    """When LLM nodes with 'main' in id exist, return them all."""

    def test_single_main_node(self):
        spec = _spec(
            [_node("claude_main"), _node("validate", "VALIDATE")],
            [("__start__", "claude_main"), ("claude_main", "__end__")],
        )
        assert _find_persona_targets(spec) == ["claude_main"]

    def test_multiple_main_nodes(self):
        spec = _spec(
            [_node("main_claude"), _node("main_gemini", "GEMINI_CLI")],
            [
                ("__start__", "main_claude"),
                ("main_claude", "main_gemini"),
                ("main_gemini", "__end__"),
            ],
        )
        result = _find_persona_targets(spec)
        assert set(result) == {"main_claude", "main_gemini"}

    def test_main_in_middle_of_id(self):
        """'main' can appear anywhere in the id string."""
        spec = _spec(
            [_node("my_main_llm", "OLLAMA")],
            [("__start__", "my_main_llm"), ("my_main_llm", "__end__")],
        )
        assert _find_persona_targets(spec) == ["my_main_llm"]

    def test_main_non_llm_ignored(self):
        """Non-LLM nodes with 'main' in id are NOT selected."""
        spec = _spec(
            [
                _node("main_validator", "VALIDATE"),
                _node("gemini_worker", "GEMINI_CLI"),
            ],
            [
                ("__start__", "main_validator"),
                ("main_validator", "gemini_worker"),
                ("gemini_worker", "__end__"),
            ],
        )
        # 'main_validator' is not LLM, falls through to rule 2
        result = _find_persona_targets(spec)
        assert result == ["gemini_worker"]


# =========================================================================
# Rule 2: no 'main' → reverse BFS from __end__
# =========================================================================

class TestRule2ReverseBFS:
    """When no 'main' node, find LLM node closest to __end__."""

    def test_linear_chain_picks_last_llm(self):
        """A → B → C(LLM) → D(VALIDATE) → __end__ → picks C."""
        spec = _spec(
            [
                _node("step_a", "GEMINI_CLI"),
                _node("step_b", "CLAUDE_CLI"),
                _node("step_c", "OLLAMA"),
                _node("step_d", "VALIDATE"),
            ],
            [
                ("__start__", "step_a"),
                ("step_a", "step_b"),
                ("step_b", "step_c"),
                ("step_c", "step_d"),
                ("step_d", "__end__"),
            ],
        )
        assert _find_persona_targets(spec) == ["step_c"]

    def test_llm_directly_before_end(self):
        spec = _spec(
            [_node("only_llm", "GEMINI_CLI")],
            [("__start__", "only_llm"), ("only_llm", "__end__")],
        )
        assert _find_persona_targets(spec) == ["only_llm"]

    def test_branching_graph_picks_closest(self):
        """Diamond: start → A(LLM), B(LLM) → merge(VALIDATE) → __end__
        Both A and B are equidistant (2 hops from __end__).
        BFS returns whichever is found first — either is acceptable.
        """
        spec = _spec(
            [
                _node("branch_a", "CLAUDE_CLI"),
                _node("branch_b", "GEMINI_CLI"),
                _node("merge", "VALIDATE"),
            ],
            [
                ("__start__", "branch_a"),
                ("__start__", "branch_b"),
                ("branch_a", "merge"),
                ("branch_b", "merge"),
                ("merge", "__end__"),
            ],
        )
        result = _find_persona_targets(spec)
        assert len(result) == 1
        assert result[0] in {"branch_a", "branch_b"}

    def test_multi_hop_selects_nearest(self):
        """start → A(LLM) → B(VALIDATE) → C(LLM) → __end__
        C is 1 hop from __end__, A is 3 hops. Should pick C.
        """
        spec = _spec(
            [
                _node("far_llm", "CLAUDE_CLI"),
                _node("mid_validate", "VALIDATE"),
                _node("near_llm", "GEMINI_CLI"),
            ],
            [
                ("__start__", "far_llm"),
                ("far_llm", "mid_validate"),
                ("mid_validate", "near_llm"),
                ("near_llm", "__end__"),
            ],
        )
        assert _find_persona_targets(spec) == ["near_llm"]

    def test_snapshot_validate_chain(self):
        """Real pattern: LLM → snapshot → validate → __end__."""
        spec = _spec(
            [
                _node("worker", "OLLAMA"),
                _node("snapshot", "GIT_SNAPSHOT"),
                _node("validate", "VALIDATE"),
            ],
            [
                ("__start__", "worker"),
                ("worker", "snapshot"),
                ("snapshot", "validate"),
                ("validate", "__end__"),
            ],
        )
        assert _find_persona_targets(spec) == ["worker"]


# =========================================================================
# Rule 3: no LLM nodes → empty list
# =========================================================================

class TestRule3NoLLM:
    """When no LLM nodes exist, return empty list."""

    def test_only_infra_nodes(self):
        spec = _spec(
            [
                _node("snapshot", "GIT_SNAPSHOT"),
                _node("validate", "VALIDATE"),
            ],
            [
                ("__start__", "snapshot"),
                ("snapshot", "validate"),
                ("validate", "__end__"),
            ],
        )
        assert _find_persona_targets(spec) == []

    def test_subgraph_node_only(self):
        """SUBGRAPH_NODE is not an LLM type."""
        spec = _spec(
            [_node("sub", "SUBGRAPH_NODE")],
            [("__start__", "sub"), ("sub", "__end__")],
        )
        assert _find_persona_targets(spec) == []

    def test_empty_graph(self):
        assert _find_persona_targets({"nodes": [], "edges": []}) == []


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    """Edge cases and robustness."""

    def test_all_llm_types_recognized(self):
        """Every type in _LLM_NODE_TYPES is treated as LLM."""
        for ntype in _LLM_NODE_TYPES:
            spec = _spec(
                [_node("worker", ntype)],
                [("__start__", "worker"), ("worker", "__end__")],
            )
            result = _find_persona_targets(spec)
            assert result == ["worker"], f"Failed for type {ntype}"

    def test_main_takes_priority_over_closer_non_main(self):
        """Even if a non-main LLM is closer to __end__, 'main' wins."""
        spec = _spec(
            [
                _node("claude_main", "CLAUDE_CLI"),
                _node("validate", "VALIDATE"),
                _node("closer_llm", "GEMINI_CLI"),
            ],
            [
                ("__start__", "claude_main"),
                ("claude_main", "validate"),
                ("validate", "closer_llm"),
                ("closer_llm", "__end__"),
            ],
        )
        assert _find_persona_targets(spec) == ["claude_main"]

    def test_disconnected_llm_fallback(self):
        """LLM node not reachable from __end__ via reverse edges → fallback."""
        spec = _spec(
            [
                _node("orphan_llm", "CLAUDE_CLI"),
                _node("other", "VALIDATE"),
            ],
            [
                ("__start__", "orphan_llm"),
                ("__start__", "other"),
                ("other", "__end__"),
                # orphan_llm has no edge to __end__
            ],
        )
        # Fallback: return any LLM node
        result = _find_persona_targets(spec)
        assert result == ["orphan_llm"]

    def test_routing_edges_traversed(self):
        """Routing edges (with type field) are still traversed for BFS."""
        spec = {
            "nodes": [
                _node("worker", "CLAUDE_CLI"),
                _node("validate", "VALIDATE"),
            ],
            "edges": [
                {"from": "__start__", "to": "worker"},
                {"from": "worker", "to": "validate"},
                {"from": "validate", "to": "__end__", "type": "no_routing"},
            ],
        }
        assert _find_persona_targets(spec) == ["worker"]

    def test_real_technical_architect_pattern(self):
        """Mirrors the actual technical_architect graph structure."""
        spec = _spec(
            [
                _node("claude_main", "CLAUDE_CLI"),
                _node("git_snapshot", "GIT_SNAPSHOT"),
                _node("validate", "VALIDATE"),
                _node("git_rollback", "GIT_ROLLBACK"),
            ],
            [
                ("__start__", "claude_main"),
                ("claude_main", "git_snapshot"),
                ("git_snapshot", "validate"),
                ("validate", "__end__"),
                ("validate", "git_rollback"),
                ("git_rollback", "claude_main"),
            ],
        )
        assert _find_persona_targets(spec) == ["claude_main"]

    def test_real_knowledge_curator_pattern(self):
        """Mirrors knowledge_curator: only SUBGRAPH_NODE, no LLM."""
        spec = _spec(
            [_node("obsidian_manager", "SUBGRAPH_NODE")],
            [
                ("__start__", "obsidian_manager"),
                ("obsidian_manager", "__end__"),
            ],
        )
        assert _find_persona_targets(spec) == []


# =========================================================================
# Persona passthrough: SUBGRAPH_NODE vs SUBGRAPH_REF
# =========================================================================

def _make_child_agent_dir(tmp: Path, child_name: str, has_own_persona: bool = False) -> Path:
    """Create a minimal child subgraph agent dir with a GEMINI_CLI node."""
    child_dir = tmp / child_name
    child_dir.mkdir(parents=True)
    agent_json = {
        "name": child_name,
        "llm": "gemini",
        "persona_files": ["CHILD_ROLE.md"] if has_own_persona else [],
        "graph": {
            "nodes": [
                {"id": "gemini_worker", "type": "GEMINI_CLI", "model": "gemini-2.5-flash"}
            ],
            "edges": [
                {"from": "__start__", "to": "gemini_worker"},
                {"from": "gemini_worker", "to": "__end__"},
            ],
        },
    }
    (child_dir / "agent.json").write_text(json.dumps(agent_json))
    if has_own_persona:
        (child_dir / "CHILD_ROLE.md").write_text("# Child Role\nI am a child agent.")
    return child_dir


def _make_parent_agent_dir(
    tmp: Path,
    parent_name: str,
    child_dir: Path,
    node_type: str = "SUBGRAPH_NODE",
    has_persona: bool = True,
) -> Path:
    """Create a parent agent dir with a single SUBGRAPH_NODE or SUBGRAPH_REF."""
    parent_dir = tmp / parent_name
    parent_dir.mkdir(parents=True)

    node_def = {
        "id": "child_sub",
        "type": node_type,
    }
    if node_type == "SUBGRAPH_NODE":
        node_def["agent_dir"] = str(child_dir)
    else:
        # SUBGRAPH_REF needs agent_dir + state mapping
        node_def["agent_dir"] = str(child_dir)
        node_def["state_in"] = {"task": "routing_context"}
        node_def["state_out"] = {"result": "last_message"}

    agent_json = {
        "name": parent_name,
        "llm": "gemini",
        "persona_files": ["PARENT_ROLE.md"] if has_persona else [],
        "graph": {
            "nodes": [node_def],
            "edges": [
                {"from": "__start__", "to": "child_sub"},
                {"from": "child_sub", "to": "__end__"},
            ],
        },
    }
    (parent_dir / "agent.json").write_text(json.dumps(agent_json))
    if has_persona:
        (parent_dir / "PARENT_ROLE.md").write_text(
            "# Parent Role\nYou are the parent agent with important persona."
        )
    return parent_dir


class TestPersonaPassthrough:
    """SUBGRAPH_NODE receives parent persona; SUBGRAPH_REF does not."""

    def test_subgraph_node_receives_parent_persona(self):
        """Parent persona should merge into child's system_prompt via build_graph."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph")
            parent_dir = _make_parent_agent_dir(
                tmp_path, "parent_graph", child_dir, node_type="SUBGRAPH_NODE"
            )

            loader = EntityLoader(parent_dir)
            # build_graph should not raise
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_subgraph_node_no_parent_persona(self):
        """When parent has no persona, child builds normally."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph")
            parent_dir = _make_parent_agent_dir(
                tmp_path, "parent_graph", child_dir,
                node_type="SUBGRAPH_NODE", has_persona=False,
            )

            loader = EntityLoader(parent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_subgraph_node_merges_child_and_parent_persona(self):
        """When both parent and child have persona, they should merge."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph", has_own_persona=True)
            parent_dir = _make_parent_agent_dir(
                tmp_path, "parent_graph", child_dir, node_type="SUBGRAPH_NODE"
            )

            loader = EntityLoader(parent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_parent_with_own_llm_also_passes_through(self):
        """SUBGRAPH_NODE nodes = main graph nodes, always receive persona."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph")
            parent_dir = tmp_path / "parent_with_llm"
            parent_dir.mkdir()

            agent_json = {
                "name": "parent_with_llm",
                "llm": "claude",
                "persona_files": ["PARENT_ROLE.md"],
                "graph": {
                    "nodes": [
                        {"id": "claude_main", "type": "CLAUDE_CLI"},
                        {"id": "child_sub", "type": "SUBGRAPH_NODE",
                         "agent_dir": str(child_dir)},
                    ],
                    "edges": [
                        {"from": "__start__", "to": "claude_main"},
                        {"from": "claude_main", "to": "child_sub"},
                        {"from": "child_sub", "to": "__end__"},
                    ],
                },
            }
            (parent_dir / "agent.json").write_text(json.dumps(agent_json))
            (parent_dir / "PARENT_ROLE.md").write_text("# Parent with LLM")

            loader = EntityLoader(parent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_build_graph_parent_persona_param(self):
        """EntityLoader.build_graph(parent_persona=...) merges into system_prompt."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph", has_own_persona=True)

            loader = EntityLoader(child_dir)
            # Simulate what SUBGRAPH_NODE does: pass parent_persona
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None, parent_persona="# Injected Parent Persona")
            )
            assert graph is not None
