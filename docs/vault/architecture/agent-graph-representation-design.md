# AgentGraph Representation Layer — Design Decision

**Date:** 2026-06-05
**Status:** Decided, pending implementation
**Authors:** technical_architect, debate_design subgraph

---

## Problem

ZenithLoom currently has no intermediate graph representation. The pipeline flows directly from raw dict to execution:

```
entity.json (raw dict)
    └──► graph_builder.py (_build_declarative)
              └──► LangGraph StateGraph (compiled, opaque)
```

`graph_spec` is passed around as a plain `dict[str, Any]` with no type safety. The same dict is re-parsed independently by four different parts of the codebase:

| Location | What it re-parses |
|---|---|
| `graph_validator.py` | Node IDs, edge refs, reachability |
| `persona.py` | `agent_dir` nodes for routing_hint collection |
| `build_topology_mermaid()` | Node/edge structure for Mermaid rendering |
| `graph_builder.py` | Everything, for LangGraph compilation |

### Specific failures

**1. Duplicated, inconsistent SUBGRAPH_REF detection**
```python
# graph_builder.py
if node_def.get("agent_dir") and not node_type:   # SUBGRAPH_REF

# persona.py
if not agent_dir or node_def.get("type"):          # inverse of the same check
```
Same logical condition, written in opposite form, no shared source of truth.

**2. `graph_builder` does 7 unrelated things in one 464-line async function**
Validation → persona assembly → subgraph recursive loading → routing_hint injection → edge building → checkpointer setup → LangGraph compile. Impossible to unit-test any single step.

**3. Topology is not a first-class object**
To answer "which nodes in this graph are LLM nodes?" you must re-parse the raw dict every time. There is no object that can answer this directly.

**4. No serialization of the built graph**
Once LangGraph compiles the graph, the topology is opaque. You cannot round-trip: compiled LangGraph → graph definition → compiled LangGraph.

---

## Decision

Introduce a thin **AgentGraph representation layer** between `entity.json` and `graph_builder`. This layer:

- Is the single parse of `graph_spec` dict into typed Python objects
- Is shared by all consumers (validator, persona, topology renderer, builder)
- Does **not** replace LangGraph as the execution engine
- Does **not** add a new persistence mechanism (GraphStore) — entity.json remains the source of truth

```
entity.json
    └──► AgentGraph.from_dict()          ← one parse, typed objects
              ├──► validate()            ← graph_validator logic
              ├──► routing_hints()       ← persona.py logic
              ├──► flatten_view()        ← topology rendering
              └──► compile_langgraph()   ← graph_builder logic
```

---

## Why not NetworkX

NetworkX adds zero algorithmic value for graphs of 10–20 nodes with intentional cycles (subgraph → claude_main loop). Its strengths are community detection, centrality, shortest-path on large static DAGs — none of which apply here.

The API is also an impedance mismatch:
```python
# NetworkX — untyped attribute dict
G.nodes['claude_main']['type']
G.nodes['claude_main'].get('agent_dir', '')

# Proposed dataclass — typed, explicit
node.is_subgraph    # property
node.is_llm         # property
```

All graph traversal logic we need (reachability, edge-ref validation) is already written in `graph_validator.py`. We are porting it, not replacing it.

## Why not Haystack

Haystack's pipeline model is **data-flow** (output port → input port). ZenithLoom's pipeline model is **control-flow** (conditional state routing). Incompatible semantics.

Forcing Haystack would require wrapping every ZenithLoom node type (CLAUDE_SDK, GEMINI_API, SUBGRAPH_REF, etc.) as Haystack Component subclasses — more work than writing 150 lines of dataclasses, plus a large framework dependency with no benefit.

We borrow Haystack's **pattern** (`type:` field for node identification, `to_dict()/from_dict()` for round-trip), not its code.

## Why not a GraphStore / UUID / full version system (yet)

The debate identified these as Phase 2 concerns:

- **UUID on nodes/edges**: Needed for stable cross-version diff and UI animation. Not needed until a Web UI exists. entity.json node `id` strings are sufficient for current Discord + CLI use.
- **GraphStore**: A content-addressable history store. Adds ~300 lines and changes entity.json from source-of-truth to import-only seed. Out of scope for the problem being solved (eliminating duplicate dict parsing).
- **Dynamic StateSchema synthesis**: Rejected outright. Adding/removing nodes that require new State keys is a code-level change, not a runtime mutation. The graph layer's mutation API (if added) covers topology changes only.

---

## Proposed Interface

```python
# framework/graph/spec.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

_LLM_NODE_TYPES = frozenset({
    "CLAUDE_CLI", "CLAUDE_SDK", "GEMINI_CLI", "GEMINI_API",
    "OLLAMA", "LOCAL_VLLM",
})

_FRAMEWORK_NODE_TYPES = frozenset({
    "VALIDATE", "GIT_SNAPSHOT", "GIT_ROLLBACK", "VRAM_FLUSH",
    "HEARTBEAT", "DETERMINISTIC", "EXTERNAL_TOOL",
})


@dataclass
class NodeSpec:
    id: str
    type: str = ""                          # "" = SUBGRAPH_REF (detected by agent_dir)
    agent_dir: str = ""                     # set for SUBGRAPH_REF nodes
    config: dict[str, Any] = field(default_factory=dict)  # all other node fields

    @property
    def is_subgraph(self) -> bool:
        return bool(self.agent_dir) and not self.type

    @property
    def is_llm(self) -> bool:
        return self.type in _LLM_NODE_TYPES

    @property
    def is_framework(self) -> bool:
        return self.type in _FRAMEWORK_NODE_TYPES

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id}
        if self.type:
            d["type"] = self.type
        if self.agent_dir:
            d["agent_dir"] = self.agent_dir
        d.update(self.config)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NodeSpec":
        node_id = raw["id"]
        node_type = raw.get("type", "")
        agent_dir = raw.get("agent_dir", "")
        config = {k: v for k, v in raw.items() if k not in ("id", "type", "agent_dir")}
        return cls(id=node_id, type=node_type, agent_dir=agent_dir, config=config)


@dataclass
class EdgeSpec:
    source: str
    target: str
    condition: str = ""     # "routing_to" | "no_routing" | "on_error" | "" (unconditional)
    config: dict[str, Any] = field(default_factory=dict)  # max_retry, id, etc.

    @property
    def is_conditional(self) -> bool:
        return bool(self.condition)

    @property
    def is_routing(self) -> bool:
        return self.condition == "routing_to"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"from": self.source, "to": self.target}
        if self.condition:
            d["type"] = self.condition
        d.update(self.config)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EdgeSpec":
        config = {k: v for k, v in raw.items() if k not in ("from", "to", "type")}
        return cls(
            source=raw["from"],
            target=raw["to"],
            condition=raw.get("type", ""),
            config=config,
        )


@dataclass
class AgentGraph:
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    entry: str = ""
    exit: str = ""
    state_schema: str = "base_schema"
    meta: dict[str, Any] = field(default_factory=dict)  # routing_hint, etc.

    # ── Convenience accessors ──────────────────────────────────────────

    def node(self, node_id: str) -> NodeSpec | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def llm_nodes(self) -> list[NodeSpec]:
        return [n for n in self.nodes if n.is_llm]

    def subgraph_nodes(self) -> list[NodeSpec]:
        return [n for n in self.nodes if n.is_subgraph]

    def edges_from(self, node_id: str) -> list[EdgeSpec]:
        return [e for e in self.edges if e.source == node_id]

    def edges_to(self, node_id: str) -> list[EdgeSpec]:
        return [e for e in self.edges if e.target == node_id]

    def routing_targets(self, node_id: str) -> list[str]:
        """All routing_to targets from node_id."""
        return [e.target for e in self.edges_from(node_id) if e.is_routing]

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
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

    # ── Validation (ported from graph_validator.py) ───────────────────

    def validate(self) -> None:
        """Raise ValueError on any structural problem."""
        node_ids = {n.id for n in self.nodes} | {"__start__", "__end__"}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source {edge.source!r} not in nodes")
            if edge.target not in node_ids:
                raise ValueError(f"Edge target {edge.target!r} not in nodes")
        # Reachability: every node reachable from entry or __start__
        # (logic ported from graph_validator._check_reachable)
        ...
```

---

## Implementation Plan

**Phase 1 — Introduce `AgentGraph`, wire into existing code (~150 lines)**

1. Create `framework/graph/spec.py` with `NodeSpec`, `EdgeSpec`, `AgentGraph`
2. Parse `entity.json["graph"]` into `AgentGraph` inside `entity_loader.py` before passing to `graph_builder`
3. Update `graph_validator.py` to consume `AgentGraph` instead of raw dict
4. Update `persona.py` (`_collect_routing_hints`) to consume `AgentGraph`
5. Update `build_topology_mermaid()` to consume `AgentGraph`
6. Update `graph_builder._build_declarative()` to accept `AgentGraph` (keep internal logic unchanged initially)

**Phase 2 — Refactor `graph_builder` internals**

Split the monolithic `_build_declarative()` into:
- `_build_nodes(graph: AgentGraph, config, ...)` — node instantiation
- `_build_edges(graph: AgentGraph, builder, ...)` — edge wiring
- `_compile(builder, checkpointer)` — final LangGraph compile

**Phase 3 — GraphStore + UUID (when Web UI is ready)**

Add `uid: str` to `NodeSpec`/`EdgeSpec`, introduce `GraphStore` with `history.jsonl`, promote entity.json to import-only seed.

---

## What This Does NOT Change

- LangGraph remains the execution engine. `compile_langgraph()` still produces a `CompiledStateGraph`.
- `entity.json` remains the source of truth. `AgentGraph` is parsed from it on startup.
- No new runtime dependencies.
- No changes to the node implementations (`ClaudeSDKNode`, `GeminiNode`, etc.).
- No changes to `BaseAgentState` or the state schema system.
