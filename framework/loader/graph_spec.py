"""
AgentGraph representation layer — framework/loader/graph_spec.py

Typed intermediate representation of an entity.json["graph"] spec.
Sits between raw dict parsing and graph_builder / graph_validator / persona.

Classes
-------
NodeSpec    — one node (typed, with is_subgraph / is_llm / is_framework properties)
EdgeSpec    — one edge (with is_conditional / is_routing properties)
AgentGraph  — full graph: nodes + edges + entry/exit/state_schema, plus accessors
              and validate() ported from graph_validator.py.

Usage
-----
    graph = AgentGraph.from_dict(entity_json["graph"])
    graph.validate()
    for node in graph.llm_nodes():
        ...

Backward compatibility
----------------------
All three classes expose ``to_dict()`` / ``from_dict()`` so any code that
previously passed a raw dict can rebuild one from an ``AgentGraph``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node type sets (must stay in sync with graph_builder._LLM_NODE_TYPES)
# ---------------------------------------------------------------------------

_LLM_NODE_TYPES: frozenset[str] = frozenset({
    "CLAUDE_CLI", "CLAUDE_SDK", "GEMINI_CLI", "GEMINI_API",
    "OLLAMA", "LOCAL_VLLM",
})

_FRAMEWORK_NODE_TYPES: frozenset[str] = frozenset({
    "VALIDATE", "GIT_SNAPSHOT", "GIT_ROLLBACK", "VRAM_FLUSH",
    "HEARTBEAT", "DETERMINISTIC", "EXTERNAL_TOOL",
})


# ---------------------------------------------------------------------------
# NodeSpec
# ---------------------------------------------------------------------------

@dataclass
class NodeSpec:
    """Typed representation of a single node definition from entity.json["graph"]["nodes"]."""

    id: str
    type: str = ""                              # "" → SUBGRAPH_REF (detected by agent_dir)
    agent_dir: str = ""                         # set for external subgraph nodes
    config: dict[str, Any] = field(default_factory=dict)  # all remaining fields

    # ── Classification properties ─────────────────────────────────────────

    @property
    def is_subgraph(self) -> bool:
        """True when node is an external subgraph reference (agent_dir set, no type)."""
        return bool(self.agent_dir) and not self.type

    @property
    def is_llm(self) -> bool:
        """True when node type is one of the known LLM execution types."""
        return self.type in _LLM_NODE_TYPES

    @property
    def is_framework(self) -> bool:
        """True when node type is a built-in framework utility node."""
        return self.type in _FRAMEWORK_NODE_TYPES

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Round-trip back to the raw entity.json node dict format."""
        d: dict[str, Any] = {"id": self.id}
        if self.type:
            d["type"] = self.type
        if self.agent_dir:
            d["agent_dir"] = self.agent_dir
        d.update(self.config)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NodeSpec":
        """Parse a single node dict from entity.json["graph"]["nodes"]."""
        node_id = raw["id"]
        node_type = raw.get("type", "")
        agent_dir = raw.get("agent_dir", "")
        config = {k: v for k, v in raw.items() if k not in ("id", "type", "agent_dir")}
        return cls(id=node_id, type=node_type, agent_dir=agent_dir, config=config)


# ---------------------------------------------------------------------------
# EdgeSpec
# ---------------------------------------------------------------------------

@dataclass
class EdgeSpec:
    """Typed representation of a single edge definition from entity.json["graph"]["edges"]."""

    source: str
    target: str
    condition: str = ""   # "routing_to" | "no_routing" | "on_error" | "" (unconditional)
    config: dict[str, Any] = field(default_factory=dict)  # max_retry, id, etc.

    # ── Classification properties ─────────────────────────────────────────

    @property
    def is_conditional(self) -> bool:
        """True when this edge has a condition type."""
        return bool(self.condition)

    @property
    def is_routing(self) -> bool:
        """True when this edge is a dynamic routing target edge."""
        return self.condition == "routing_to"

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Round-trip back to the raw entity.json edge dict format."""
        d: dict[str, Any] = {"from": self.source, "to": self.target}
        if self.condition:
            d["type"] = self.condition
        d.update(self.config)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EdgeSpec":
        """Parse a single edge dict from entity.json["graph"]["edges"]."""
        config = {k: v for k, v in raw.items() if k not in ("from", "to", "type")}
        return cls(
            source=raw["from"],
            target=raw["to"],
            condition=raw.get("type", ""),
            config=config,
        )


# ---------------------------------------------------------------------------
# AgentGraph
# ---------------------------------------------------------------------------

@dataclass
class AgentGraph:
    """
    Typed representation of an entity.json["graph"] spec.

    Constructed once from the raw dict via ``from_dict()``, then shared by
    graph_validator, persona, topology renderer, and graph_builder — eliminating
    four independent re-parses of the same raw dict.
    """

    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    entry: str = ""
    exit: str = ""
    state_schema: str = "base_schema"
    meta: dict[str, Any] = field(default_factory=dict)  # routing_hint and any extra top-level keys

    # ── Convenience accessors ──────────────────────────────────────────────

    def node(self, node_id: str) -> NodeSpec | None:
        """Return the NodeSpec with the given id, or None."""
        return next((n for n in self.nodes if n.id == node_id), None)

    def llm_nodes(self) -> list[NodeSpec]:
        """All nodes whose type is an LLM execution type."""
        return [n for n in self.nodes if n.is_llm]

    def subgraph_nodes(self) -> list[NodeSpec]:
        """All external subgraph reference nodes."""
        return [n for n in self.nodes if n.is_subgraph]

    def edges_from(self, node_id: str) -> list[EdgeSpec]:
        """All edges originating from node_id."""
        return [e for e in self.edges if e.source == node_id]

    def edges_to(self, node_id: str) -> list[EdgeSpec]:
        """All edges terminating at node_id."""
        return [e for e in self.edges if e.target == node_id]

    def routing_targets(self, node_id: str) -> list[str]:
        """All routing_to target node IDs from node_id."""
        return [e.target for e in self.edges_from(node_id) if e.is_routing]

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Round-trip back to the raw entity.json graph dict format."""
        d: dict[str, Any] = {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }
        if self.state_schema != "base_schema":
            d["state_schema"] = self.state_schema
        if self.entry:
            d["entry"] = self.entry
        if self.exit:
            d["exit"] = self.exit
        d.update(self.meta)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AgentGraph":
        """Parse entity.json["graph"] dict into a typed AgentGraph."""
        nodes = [NodeSpec.from_dict(n) for n in raw.get("nodes", [])]
        edges = [EdgeSpec.from_dict(e) for e in raw.get("edges", [])]
        meta = {
            k: v for k, v in raw.items()
            if k not in ("nodes", "edges", "entry", "exit", "state_schema")
        }
        return cls(
            nodes=nodes,
            edges=edges,
            entry=raw.get("entry", ""),
            exit=raw.get("exit", ""),
            state_schema=raw.get("state_schema", "base_schema"),
            meta=meta,
        )

    # ── Validation (ported from graph_validator.py) ────────────────────────

    def validate(self) -> None:
        """
        Run three-pass structural validation.

        Raises ValueError on:
          - missing/duplicate node IDs
          - edges referencing unknown nodes
          - entry/exit referencing unknown nodes
          - nodes unreachable from __start__ or entry
        """
        all_ids = self._collect_all_ids()
        self._check_edge_refs(all_ids)
        self._check_reachable(all_ids)

    def _collect_all_ids(self, seen: set[str] | None = None) -> set[str]:
        """
        Recursively collect all node IDs (including inline SUBGRAPH children).
        Raises ValueError on missing or duplicate IDs.
        """
        if seen is None:
            seen = set()
        for node in self.nodes:
            nid = node.id
            if not nid:
                raise ValueError(f"Node missing 'id': {node.to_dict()}")
            if nid in seen:
                raise ValueError(f"Duplicate node ID: {nid!r}")
            seen.add(nid)
            # Inline SUBGRAPH (type="SUBGRAPH") may carry a nested graph dict
            if node.type == "SUBGRAPH" and "graph" in node.config:
                inner = AgentGraph.from_dict(node.config["graph"])
                inner._collect_all_ids(seen)
        return seen

    def _check_edge_refs(self, all_ids: set[str]) -> None:
        """Verify all edge endpoints and entry/exit reference known node IDs."""
        valid_ids = all_ids | {"__start__", "__end__"}
        for edge in self.edges:
            for ref_name, ref_val in (("from", edge.source), ("to", edge.target)):
                if ref_val not in valid_ids:
                    raise ValueError(
                        f"Edge references unknown node: {ref_val!r} "
                        f"(known: {sorted(valid_ids)})"
                    )
        if self.entry and self.entry not in all_ids:
            raise ValueError(
                f"'entry' references unknown node: {self.entry!r} (known: {sorted(all_ids)})"
            )
        if self.exit and self.exit not in all_ids:
            raise ValueError(
                f"'exit' references unknown node: {self.exit!r} (known: {sorted(all_ids)})"
            )

    def _check_reachable(self, all_ids: set[str]) -> None:
        """BFS reachability check from __start__ and/or entry."""
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            adjacency[edge.source].add(edge.target)

        start_nodes: set[str] = {"__start__"}
        if self.entry:
            start_nodes.add(self.entry)

        visited = set(start_nodes)
        queue: list[str] = []
        for s in start_nodes:
            queue.extend(adjacency.get(s, []))

        while queue:
            node_id = queue.pop()
            if node_id not in visited:
                visited.add(node_id)
                queue.extend(adjacency.get(node_id, []))

        unreachable = all_ids - visited - {"__end__"}
        if unreachable:
            raise ValueError(f"Unreachable nodes from start: {unreachable}")
