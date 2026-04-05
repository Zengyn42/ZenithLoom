"""Tests for per-node persona injection: _load_persona_text + extra_persona opt-in.

Design (per-node persona):
  - Each LLM node assembles its own system_prompt from:
      extra_persona (opt-in, parent/instance level, only if extra_persona:true)
      + node's own persona_files
      + node's own system_prompt
  - extra_persona:true on an LLM node = receives parent/instance extra persona
  - extra_persona:{...} on a subgraph node = passes specific persona to child
  - No automatic persona passthrough; nodes must explicitly opt in.
"""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from framework.agent_loader import _load_persona_text, _LLM_NODE_TYPES, EntityLoader


# ---------------------------------------------------------------------------
# Tests for _load_persona_text helper
# ---------------------------------------------------------------------------

class TestLoadPersonaText:
    """Unit tests for _load_persona_text helper."""

    def test_single_file(self, tmp_path):
        (tmp_path / "ROLE.md").write_text("I am a role.", encoding="utf-8")
        result = _load_persona_text(["ROLE.md"], tmp_path, label="myagent")
        assert "<!-- [source: myagent/ROLE.md] -->" in result
        assert "I am a role." in result

    def test_multiple_files(self, tmp_path):
        (tmp_path / "A.md").write_text("Part A", encoding="utf-8")
        (tmp_path / "B.md").write_text("Part B", encoding="utf-8")
        result = _load_persona_text(["A.md", "B.md"], tmp_path)
        assert "Part A" in result
        assert "Part B" in result
        assert "---" in result  # separator between parts

    def test_missing_file_warns(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="framework.agent_loader"):
            result = _load_persona_text(["MISSING.md"], tmp_path)
        assert result == ""
        assert "persona file not found" in caplog.text

    def test_prompt_only(self, tmp_path):
        result = _load_persona_text([], tmp_path, prompt="Hello prompt")
        assert result == "Hello prompt"

    def test_file_and_prompt(self, tmp_path):
        (tmp_path / "ROLE.md").write_text("Role content", encoding="utf-8")
        result = _load_persona_text(["ROLE.md"], tmp_path, prompt="Extra prompt")
        assert "Role content" in result
        assert "Extra prompt" in result

    def test_empty_inputs(self, tmp_path):
        result = _load_persona_text([], tmp_path, prompt="")
        assert result == ""

    def test_label_defaults_to_dirname(self, tmp_path):
        (tmp_path / "X.md").write_text("content", encoding="utf-8")
        result = _load_persona_text(["X.md"], tmp_path)
        assert f"<!-- [source: {tmp_path.name}/X.md] -->" in result


# ---------------------------------------------------------------------------
# Helper: build minimal graph_spec and agent directories
# ---------------------------------------------------------------------------

def _spec(nodes: list[dict], edges: list[tuple[str, str]]) -> dict:
    return {
        "nodes": nodes,
        "edges": [{"from": src, "to": dst} for src, dst in edges],
    }


def _node(nid: str, ntype: str = "CLAUDE_CLI") -> dict:
    return {"id": nid, "type": ntype}


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
                {"id": "gemini_worker", "type": "GEMINI_CLI", "model": "gemini-2.5-flash",
                 "extra_persona": True}
            ],
            "edges": [
                {"from": "__start__", "to": "gemini_worker"},
                {"from": "gemini_worker", "to": "__end__"},
            ],
        },
    }
    (child_dir / "entity.json").write_text(json.dumps(agent_json))
    if has_own_persona:
        (child_dir / "CHILD_ROLE.md").write_text("# Child Role\nI am a child agent.")
    return child_dir


def _make_parent_agent_dir(
    tmp: Path,
    parent_name: str,
    child_dir: Path,
    node_type: str = "",
    has_persona: bool = True,
) -> Path:
    """Create a parent agent dir with a single external subgraph."""
    parent_dir = tmp / parent_name
    parent_dir.mkdir(parents=True)

    node_def = {
        "id": "child_sub",
        "agent_dir": str(child_dir),
    }

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
    (parent_dir / "entity.json").write_text(json.dumps(agent_json))
    if has_persona:
        (parent_dir / "PARENT_ROLE.md").write_text(
            "# Parent Role\nYou are the parent agent with important persona."
        )
    return parent_dir


# ---------------------------------------------------------------------------
# Tests for build_graph basic functionality
# ---------------------------------------------------------------------------

class TestBuildGraphBasic:
    """basic build_graph tests (no persona assertions, just verify they build)."""

    def test_subgraph_node_builds_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph")
            parent_dir = _make_parent_agent_dir(
                tmp_path, "parent_graph", child_dir, node_type=""
            )
            loader = EntityLoader(parent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_subgraph_node_no_parent_persona(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph")
            parent_dir = _make_parent_agent_dir(
                tmp_path, "parent_graph", child_dir,
                node_type="", has_persona=False,
            )
            loader = EntityLoader(parent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_subgraph_merges_child_and_parent_persona(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph", has_own_persona=True)
            parent_dir = _make_parent_agent_dir(
                tmp_path, "parent_graph", child_dir, node_type=""
            )
            loader = EntityLoader(parent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None

    def test_extra_persona_text_param(self):
        """EntityLoader.build_graph(extra_persona_text=...) is accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph", has_own_persona=True)

            loader = EntityLoader(child_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None, extra_persona_text="# Injected Extra Persona")
            )
            assert graph is not None

    def test_parent_with_own_llm_and_subgraph(self):
        """Graph with both LLM nodes and subgraph nodes builds correctly."""
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
                        {"id": "child_sub", "agent_dir": str(child_dir)},
                    ],
                    "edges": [
                        {"from": "__start__", "to": "claude_main"},
                        {"from": "claude_main", "to": "child_sub"},
                        {"from": "child_sub", "to": "__end__"},
                    ],
                },
            }
            (parent_dir / "entity.json").write_text(json.dumps(agent_json))
            (parent_dir / "PARENT_ROLE.md").write_text("# Parent with LLM")

            loader = EntityLoader(parent_dir)
            graph = asyncio.get_event_loop().run_until_complete(
                loader.build_graph(checkpointer=None)
            )
            assert graph is not None


# ---------------------------------------------------------------------------
# Persona Spy: intercepts node factory calls to capture system_prompt
# ---------------------------------------------------------------------------

class _PersonaSpy:
    """Context manager that patches node factories to capture system_prompt per node id."""

    def __init__(self):
        self.prompts: dict[str, str] = {}

    def __enter__(self):
        from framework.registry import _NODE_REGISTRY

        original_factories = dict(_NODE_REGISTRY)

        def _make_spy_factory(orig_factory):
            def _spy(config, node_config):
                node_id = node_config.get("id", "")
                sp = node_config.get("system_prompt", "")
                if sp:
                    self.prompts[node_id] = sp
                return orig_factory(config, node_config)
            return _spy

        for ntype in _LLM_NODE_TYPES:
            if ntype in _NODE_REGISTRY:
                _NODE_REGISTRY[ntype] = _make_spy_factory(original_factories[ntype])

        self._original = original_factories
        return self

    def __exit__(self, *args):
        from framework.registry import _NODE_REGISTRY
        _NODE_REGISTRY.update(self._original)


# ---------------------------------------------------------------------------
# Persona content markers
# ---------------------------------------------------------------------------

_PERSONA_A = "<!-- PERSONA_A_MARKER -->\n# Persona A\nI am subgraph A's persona."
_PERSONA_B = "<!-- PERSONA_B_MARKER -->\n# Persona B\nI am subgraph B's persona."
_PERSONA_C = "<!-- PERSONA_C_MARKER -->\n# Persona C\nI am subgraph C's persona."


def _make_subgraph_dir(
    tmp: Path,
    name: str,
    persona_content: str,
    nodes: list[dict],
    edges: list[dict],
) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    persona_fname = f"{name}_ROLE.md"
    agent_json = {
        "name": name,
        "llm": "gemini",
        "persona_files": [persona_fname] if persona_content else [],
        "graph": {"nodes": nodes, "edges": edges},
    }
    (d / "entity.json").write_text(json.dumps(agent_json))
    if persona_content:
        (d / persona_fname).write_text(persona_content)
    return d


def _make_subgraph_A(tmp: Path) -> Path:
    """Subgraph A: 1 LLM node (A1_main) with extra_persona:true and own persona_files, personaA."""
    d = tmp / "subgraph_A"
    d.mkdir(parents=True, exist_ok=True)
    (d / "subgraph_A_ROLE.md").write_text(_PERSONA_A)
    agent_json = {
        "name": "subgraph_A",
        "llm": "gemini",
        "graph": {
            "nodes": [
                {
                    "id": "A1_main",
                    "type": "GEMINI_CLI",
                    "model": "gemini-2.5-flash",
                    "extra_persona": True,
                    "persona_files": ["subgraph_A_ROLE.md"],
                }
            ],
            "edges": [
                {"from": "__start__", "to": "A1_main"},
                {"from": "A1_main", "to": "__end__"},
            ],
        },
    }
    (d / "entity.json").write_text(json.dumps(agent_json))
    return d


def _make_subgraph_B(tmp: Path) -> Path:
    """Subgraph B: no LLM nodes (just a VALIDATE node), personaB."""
    return _make_subgraph_dir(
        tmp, "subgraph_B", _PERSONA_B,
        nodes=[{"id": "B_validate", "type": "VALIDATE"}],
        edges=[
            {"from": "__start__", "to": "B_validate"},
            {"from": "B_validate", "to": "__end__"},
        ],
    )


def _make_subgraph_C(tmp: Path) -> Path:
    """Subgraph C: 2 LLM nodes (C1_main, C2_main) with extra_persona:true and own persona_files."""
    d = tmp / "subgraph_C"
    d.mkdir(parents=True, exist_ok=True)
    (d / "subgraph_C_ROLE.md").write_text(_PERSONA_C)
    agent_json = {
        "name": "subgraph_C",
        "llm": "gemini",
        "graph": {
            "nodes": [
                {
                    "id": "C1_main",
                    "type": "GEMINI_CLI",
                    "model": "gemini-2.5-flash",
                    "extra_persona": True,
                    "persona_files": ["subgraph_C_ROLE.md"],
                },
                {
                    "id": "C2_main",
                    "type": "CLAUDE_CLI",
                    "extra_persona": True,
                    "persona_files": ["subgraph_C_ROLE.md"],
                },
            ],
            "edges": [
                {"from": "__start__", "to": "C1_main"},
                {"from": "C1_main", "to": "C2_main"},
                {"from": "C2_main", "to": "__end__"},
            ],
        },
    }
    (d / "entity.json").write_text(json.dumps(agent_json))
    return d


def _make_composite_dir(
    tmp: Path,
    name: str,
    persona_content: str,
    own_nodes: list[dict],
    child_refs: list[dict],
    edges: list[dict],
) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    persona_fname = f"{name}_ROLE.md"
    all_nodes = own_nodes + child_refs
    agent_json = {
        "name": name,
        "llm": "gemini",
        "persona_files": [persona_fname] if persona_content else [],
        "graph": {"nodes": all_nodes, "edges": edges},
    }
    (d / "entity.json").write_text(json.dumps(agent_json))
    if persona_content:
        (d / persona_fname).write_text(persona_content)
    return d


# ---------------------------------------------------------------------------
# Per-node persona content integration tests
# ---------------------------------------------------------------------------

class TestPerNodePersonaContent:
    """Integration tests verifying per-node persona content assembly.

    Each LLM node with extra_persona:true receives the parent/instance extra_persona.
    Nodes without extra_persona:true receive only their own persona_files + system_prompt.
    """

    def test_llm_node_with_extra_persona_true_receives_instance_persona(self):
        """LLM node with extra_persona:true receives identity.json persona from instance."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            blueprint_dir = tmp_path / "bp"
            blueprint_dir.mkdir()
            instance_dir = tmp_path / "inst"
            instance_dir.mkdir()

            node_persona_content = "# Node's own persona"
            instance_persona_content = "# Instance persona injected"

            agent_json = {
                "name": "test_agent",
                "llm": "gemini",
                "graph": {
                    "nodes": [
                        {"id": "llm_main", "type": "GEMINI_CLI", "model": "gemini-2.5-flash",
                         "extra_persona": True,
                         "persona_files": ["NODE_ROLE.md"],
                         "system_prompt": "Node sys prompt"},
                    ],
                    "edges": [
                        {"from": "__start__", "to": "llm_main"},
                        {"from": "llm_main", "to": "__end__"},
                    ],
                },
            }
            (blueprint_dir / "entity.json").write_text(json.dumps(agent_json))
            (blueprint_dir / "NODE_ROLE.md").write_text(node_persona_content)

            identity_json = {
                "name": "test_instance",
                "persona_files": ["INSTANCE.md"],
            }
            (instance_dir / "identity.json").write_text(json.dumps(identity_json))
            (instance_dir / "INSTANCE.md").write_text(instance_persona_content)

            with _PersonaSpy() as spy:
                loader = EntityLoader(blueprint_dir, data_dir=instance_dir)
                asyncio.get_event_loop().run_until_complete(
                    loader.build_graph(checkpointer=None)
                )

            assert "llm_main" in spy.prompts
            prompt = spy.prompts["llm_main"]
            assert instance_persona_content in prompt
            assert node_persona_content in prompt
            assert "Node sys prompt" in prompt

    def test_llm_node_without_extra_persona_skips_instance_persona(self):
        """LLM node without extra_persona:true does NOT receive instance persona."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            blueprint_dir = tmp_path / "bp"
            blueprint_dir.mkdir()
            instance_dir = tmp_path / "inst"
            instance_dir.mkdir()

            agent_json = {
                "name": "test_agent",
                "llm": "gemini",
                "graph": {
                    "nodes": [
                        {"id": "llm_worker", "type": "GEMINI_CLI", "model": "gemini-2.5-flash",
                         "system_prompt": "Worker sys prompt"},
                    ],
                    "edges": [
                        {"from": "__start__", "to": "llm_worker"},
                        {"from": "llm_worker", "to": "__end__"},
                    ],
                },
            }
            (blueprint_dir / "entity.json").write_text(json.dumps(agent_json))

            identity_json = {"name": "test_instance", "persona_files": ["INSTANCE.md"]}
            (instance_dir / "identity.json").write_text(json.dumps(identity_json))
            (instance_dir / "INSTANCE.md").write_text("# SECRET INSTANCE PERSONA")

            with _PersonaSpy() as spy:
                loader = EntityLoader(blueprint_dir, data_dir=instance_dir)
                asyncio.get_event_loop().run_until_complete(
                    loader.build_graph(checkpointer=None)
                )

            # Worker has no extra_persona:true, so it should NOT get instance persona
            prompt = spy.prompts.get("llm_worker", "")
            assert "SECRET INSTANCE PERSONA" not in prompt

    def test_case1_parent_llm_no_subgraph_passthrough(self):
        """Case 1: Parent has A1_main(LLM) + B(no LLM subgraph, no extra_persona dict).

        A (parent, personaA)
        ├── A1_main (LLM, extra_persona:true)
        └── B (subgraph, no extra_persona dict)

        Expected: A1_main.load(personaA from identity level)
        B receives no extra persona (no extra_persona dict on subgraph node).
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subB = _make_subgraph_B(tmp_path)

            a_dir = _make_composite_dir(
                tmp_path, "graph_A", _PERSONA_A,
                own_nodes=[
                    {"id": "A1_main", "type": "GEMINI_CLI", "model": "gemini-2.5-flash",
                     "extra_persona": True},
                ],
                child_refs=[
                    {"id": "sub_B", "agent_dir": str(subB)},
                ],
                edges=[
                    {"from": "__start__", "to": "A1_main"},
                    {"from": "A1_main", "to": "sub_B"},
                    {"from": "sub_B", "to": "__end__"},
                ],
            )

            # Use a as both blueprint and data dir so persona_files in entity.json is loaded
            # by identity-level mechanism → inject into extra_persona nodes
            # We place identity.json in a_dir with persona_files pointing to A's ROLE.md
            identity = {
                "name": "graph_A",
                "persona_files": ["graph_A_ROLE.md"],
            }
            (a_dir / "identity.json").write_text(json.dumps(identity))

            with _PersonaSpy() as spy:
                loader = EntityLoader(a_dir)
                asyncio.get_event_loop().run_until_complete(
                    loader.build_graph(checkpointer=None)
                )

            # A1_main is an extra_persona:true node; receives instance-level persona
            assert "A1_main" in spy.prompts
            assert "PERSONA_A_MARKER" in spy.prompts["A1_main"]

    def test_case2_subgraph_with_extra_persona_dict(self):
        """Case 2: Parent passes persona to child via extra_persona dict on subgraph node.

        B (parent, personaB)
        └── sub_A (subgraph node, extra_persona: {persona_files: [B_ROLE.md]})
            └── A1_main (LLM, extra_persona:true) ← receives B's persona
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subA = _make_subgraph_A(tmp_path)

            b_dir = tmp_path / "graph_B"
            b_dir.mkdir()
            (b_dir / "B_ROLE.md").write_text(_PERSONA_B)

            agent_json = {
                "name": "graph_B",
                "llm": "gemini",
                "graph": {
                    "nodes": [
                        {
                            "id": "sub_A",
                            "agent_dir": str(subA),
                            "extra_persona": {
                                "persona_files": ["B_ROLE.md"],
                            },
                        },
                    ],
                    "edges": [
                        {"from": "__start__", "to": "sub_A"},
                        {"from": "sub_A", "to": "__end__"},
                    ],
                },
            }
            (b_dir / "entity.json").write_text(json.dumps(agent_json))

            # subA's A1_main also has its own persona from subgraph_A_ROLE.md
            with _PersonaSpy() as spy:
                loader = EntityLoader(b_dir)
                asyncio.get_event_loop().run_until_complete(
                    loader.build_graph(checkpointer=None)
                )

            assert "A1_main" in spy.prompts
            # A1_main has extra_persona:true → receives B's persona passed via extra_persona dict
            assert "PERSONA_B_MARKER" in spy.prompts["A1_main"]
            # A1_main also has its own persona from entity.json persona_files
            assert "PERSONA_A_MARKER" in spy.prompts["A1_main"]

    def test_extra_persona_bool_true_on_subgraph_node_raises(self):
        """extra_persona:true on a subgraph node (not LLM) must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            child_dir = _make_child_agent_dir(tmp_path, "child_graph")
            parent_dir = tmp_path / "parent"
            parent_dir.mkdir()

            agent_json = {
                "name": "parent",
                "llm": "gemini",
                "graph": {
                    "nodes": [
                        {
                            "id": "bad_sub",
                            "agent_dir": str(child_dir),
                            "extra_persona": True,  # invalid: bool on subgraph node
                        },
                    ],
                    "edges": [
                        {"from": "__start__", "to": "bad_sub"},
                        {"from": "bad_sub", "to": "__end__"},
                    ],
                },
            }
            (parent_dir / "entity.json").write_text(json.dumps(agent_json))

            loader = EntityLoader(parent_dir)
            with pytest.raises(ValueError, match="extra_persona:true 只能用于 LLM 节点"):
                asyncio.get_event_loop().run_until_complete(
                    loader.build_graph(checkpointer=None)
                )

    def test_node_persona_files_resolved_from_blueprint_dir(self):
        """LLM node persona_files are resolved relative to blueprint_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            blueprint_dir = tmp_path / "bp"
            blueprint_dir.mkdir()

            (blueprint_dir / "NODE_ROLE.md").write_text("# Node role from blueprint")

            agent_json = {
                "name": "test",
                "llm": "gemini",
                "graph": {
                    "nodes": [
                        {"id": "llm_node", "type": "GEMINI_CLI", "model": "gemini-2.5-flash",
                         "persona_files": ["NODE_ROLE.md"]},
                    ],
                    "edges": [
                        {"from": "__start__", "to": "llm_node"},
                        {"from": "llm_node", "to": "__end__"},
                    ],
                },
            }
            (blueprint_dir / "entity.json").write_text(json.dumps(agent_json))

            with _PersonaSpy() as spy:
                loader = EntityLoader(blueprint_dir)
                asyncio.get_event_loop().run_until_complete(
                    loader.build_graph(checkpointer=None)
                )

            assert "llm_node" in spy.prompts
            assert "Node role from blueprint" in spy.prompts["llm_node"]
