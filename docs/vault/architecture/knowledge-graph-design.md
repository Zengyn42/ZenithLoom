# Knowledge Graph Design

> Date: 2026-04-09
> Status: Design — guides Phase 2-4 implementation on top of the current Phase 1 RAG
> Companion: `rag-architecture-design.md` (Phase 1 — current production file-based RAG)

## Summary

Defines the knowledge representation layer for ZenithLoom: how to model "knowledge elements" as nodes in a graph, what counts as an edge, and where embeddings fit (or don't). Goals:

1. **Identity** — every piece of knowledge has a stable ID, independent of file path or content
2. **Lifecycle** — nodes have explicit status (active / superseded / invalidated / draft) with traceable history
3. **Explicit relations** — references / supersedes / depends_on / contradicts are typed edges, not implicit
4. **Incremental adoption** — current Markdown vault can evolve without big-bang migration
5. **Embeddings as accelerator, not foundation** — the system works without them; they're added when scale demands

---

## Conceptual Foundations

### 1.1 The Four Levels of "Knowledge Element"

The term *知识元* / *knowledge element* / *knowledge atom* has been used in literature to mean different things. We distinguish four levels:

| Level | Meaning | Example |
|---|---|---|
| **L1 — Physical** | A storage unit operable by the file system / MCP | a `.md` file, a `## ` section, a frontmatter field, a tag |
| **L2 — Semantic** | An atomic, independently-expressible, reusable claim | "LangGraph 1.0.10 forbids different reducers for shared schema fields" |
| **L3 — Domain** | A type-classified piece of knowledge with associated update semantics | `decision`, `fact`, `rule`, `concept`, `procedure`, `relation` |
| **L4 — Graph node** | An identified entity with properties and relations in a graph | `KNOW-042` — a node with `type=decision`, `status=active`, edges to other nodes |

**This design adopts L4** as the formal definition of "knowledge element". Lower levels are implementation details (L1) or analytical lenses (L2-L3) but not the unit of identity.

### 1.2 The Six Knowledge Types

Different types have fundamentally different update semantics — a fact being wrong is not the same as a decision being superseded. This matters because the framework should handle them differently.

| Type | Example | Update semantics |
|---|---|---|
| `fact` | "Session `32c591b1` was shared by two channels on 2026-04-09" | Append-only with timestamp; corrections via *invalidation* (status flip), not edit |
| `concept` | "What `session_mode` means and what its values do" | Edit in place, git history preserves prior state |
| `decision` | "Use `fresh_per_call` for the debate subgraph" | Can be **superseded**; original retained for audit; explicit `superseded_by` edge |
| `rule` | "Never write to `.git/`" | Edit in place; append changelog section in body |
| `procedure` | "How to restart Hani via systemd" | Edit in place; emphasize idempotency / no-breaking-change |
| `relation` | "Hani routes to `knowledge_shelf` via routing signal" | Must sync with system reality; both endpoints updated in lockstep |

These types are values for the `type` frontmatter field. They are not enforced by the framework but inform the LLM agent's update flow.

### 1.3 Documents vs Knowledge Nodes

The vault contains two file kinds:

- **Documents** — files **without** a `knowledge_id` frontmatter. Narrative, exploratory, free-form. Examples: design specs, post-mortems, meeting notes. Reference knowledge nodes as needed via wikilinks but are not themselves nodes in the graph.
- **Knowledge nodes** — files **with** a `knowledge_id` frontmatter. Atomic, identified, lifecycle-tracked. One file = one node.

A document may reference many knowledge nodes. A knowledge node may be referenced by many documents (tracked via the `mentioned_in` field).

This distinction lets long narrative design docs (like this file itself) coexist with atomic, tracked knowledge units.

---

## 2. Node Definition

### 2.1 Identity Rule

> **One file with `knowledge_id` frontmatter = one knowledge node.**

The file path is incidental to identity. Renaming or moving a file does not change its `knowledge_id`. Other nodes reference it by ID, never by path.

### 2.2 Why File-Level (Not Section / Block / Triple)

Considered alternatives and why they were rejected:

| Alternative | Why rejected |
|---|---|
| **Section as node** (under `#` heading) | Heading text is mutable; renaming "## 决策" → "## 最终决策" silently breaks references. ID instability is a fatal flaw for a knowledge graph. |
| **Obsidian block reference (`^block-id`)** | Pollutes Markdown source with noise; can't carry rich metadata (only ID); discouraged for human readability. |
| **Extracted RDF triples (mem0 / GraphRAG style)** | Requires LLM extraction step on every edit; extraction quality variable; two stores to keep in sync; too much complexity for current scale (~100s of nodes). Deferred to long-term roadmap. |
| **File with frontmatter** ✅ | MCP already parses frontmatter; ID lives in metadata, decoupled from content; clean Markdown source; one node per file is unambiguous. |

### 2.3 Frontmatter Schema

```yaml
---
knowledge_id: KNOW-042            # Immutable identity, format: KNOW-<integer>
type: decision                    # fact | concept | decision | rule | procedure | relation
title: 辩论子图采用 fresh_per_call # One-line title for indexing/listing
status: active                    # active | superseded | invalidated | draft
scope: ZenithLoom/framework       # Domain scope (project / module path)
created: 2026-04-08
updated: 2026-04-09
authors: [hani, frankwings]       # Who curated this — agent IDs and human handles

relations:
  supersedes: []                  # IDs of nodes this one replaces
  superseded_by: KNOW-100         # If this node was replaced, who replaced it
  references: [KNOW-007]          # Soft "I mention this" reference
  contradicts: []                 # I disagree with this
  depends_on: [KNOW-001]          # I assume this is true / I require this to make sense
  refines: []                     # I'm a more specific / precise version of these

mentioned_in:                     # Reverse index: which documents reference me
  - 设计细节/session-mode-design.md
  - 设计细节/rag-architecture-design.md

# Optional embedding metadata (added in Phase B)
embed: true                       # Should this node be in the embedding index?
embed_status: synced              # synced | dirty | none
embed_updated: 2026-04-09T03:24:00Z
embed_model: gemini-embedding-001
---

# 决策

采用 fresh_per_call，原因 ...

# 上下文

...

# 历史

2026-04-08 最初决策（KNOW-042 创建）
2026-04-09 [被 KNOW-100 推翻] 发现 ...
```

### 2.4 ID Assignment

- Format: `KNOW-<n>` where `n` is a monotonically increasing integer
- Allocated by the curator (Jei, or human) at creation time
- A registry file (`knowledge/REGISTRY.md` or similar) tracks the next free ID
- Once assigned, never reused (even after deletion)

Future option: namespace prefixes for multi-vault scenarios (e.g., `ZL-KNOW-042` for ZenithLoom-scoped, `EG-KNOW-042` for EdenGateway-scoped). Not needed yet.

### 2.5 Vault Layout

```
Vault/
├── 设计细节/                      # Documents (no knowledge_id)
├── 问题整理/                      # Documents
├── 操作手册/                      # Documents
├── ...
└── knowledge/                     # Knowledge nodes (one file each)
    ├── REGISTRY.md                # ID allocator state
    ├── KNOW-001-langgraph-state.md
    ├── KNOW-042-fresh-per-call-decision.md
    ├── KNOW-100-revised-fresh-decision.md
    └── ...
```

The `knowledge/` directory groups all knowledge nodes for easy listing and indexing. File names include the title for human readability but the ID is the only thing referenced.

---

## 3. Edges

### 3.1 Five Edge Sources

Edges in the knowledge graph can come from five sources, in order of preference:

| Source | Mechanism | Edge type | When to use |
|---|---|---|---|
| **Frontmatter declaration** | YAML `relations.supersedes / references / depends_on / contradicts / refines` | Typed, explicit, validated | **Default** — every meaningful relationship goes here |
| **Body wikilink** | `[[KNOW-042]]` in Markdown body | Untyped reference | When a node mentions another in passing without claiming a formal relationship |
| **Tag co-occurrence** | Both nodes have `#session-management` | Soft, set-based | Loose grouping, browsable but not authoritative |
| **Embedding-derived** *(future)* | `cosine(emb_A, emb_B) > threshold` | Untyped, automatic | Discover latent connections at scale (Phase C) |
| **LLM-classified** *(future)* | LLM reads pair, assigns relation type | Typed, expensive | Convert similarity edges into typed relations (Phase C) |

### 3.2 Current Scope

**Phase A uses only the first three.** Embedding-derived and LLM-classified edges are deferred until phase C (see §5).

### 3.3 Why Explicit Edges First

1. **Noise control** — at small graph scale (< 200 nodes), embedding-similarity edges produce many false positives. A graph with 1000 untrusted edges is harder to use than one with 100 trusted edges.
2. **Commitment vs observation** — explicit edges are *commitments* (a curator validated this relationship). Similarity edges are *observations* (a pattern was detected). Mixing them confuses downstream consumers — "is this edge here because someone said so, or because the embedder said so?"
3. **Forces curator discipline** — requiring explicit relations forces the LLM agent (Jei) to think about how a new node fits into the existing graph at creation time, surfacing potential contradictions / duplications early.
4. **Cheap to add later** — embeddings are a separable layer. Deferring them locks in nothing.

### 3.4 Edge Types: Semantics

```
A supersedes B       → A replaces B; B.status should be 'superseded'
A superseded_by B    → inverse of supersedes (auto-derivable but stored for fast lookup)
A references B       → A mentions B; weak relationship, no validation claim
A contradicts B      → A claims B is wrong; both can remain active for documentation
A depends_on B       → A is meaningful only if B is true; if B invalidated, A may need re-review
A refines B          → A is a more precise / scoped version of B; B may remain as the general statement
```

All relations are **directed**. Inverse traversal is supported by the indexer (e.g., "find everything that depends on KNOW-001").

### 3.5 Edge Lifecycle

When a node's `status` changes, the framework should:

| Source node status change | Effect on incoming edges |
|---|---|
| `active → superseded` | Notify nodes with `references` / `depends_on` edges to this node — they may need review. Auto-update `superseded_by` reverse index. |
| `active → invalidated` | Stronger notification — nodes with `depends_on` are now suspect. |
| `* → active` *(undeleting)* | No automatic action; curator handles re-validation manually. |

These notifications are **not implemented in Phase A**. In Phase A, the agent (Jei) is responsible for finding and reviewing affected nodes when she invalidates / supersedes a node.

---

## 4. Embedding's Role

### 4.1 What Embeddings Are For

Embeddings solve **one** problem: semantic similarity search across paraphrased queries / cross-language / vibe-matching.

### 4.2 What Embeddings Are NOT For

- **Identity lookup** (`get_node(KNOW-042)`) — trivial hash lookup, no vectors needed
- **Metadata filtering** (`type=decision AND status=active`) — relational query
- **Graph traversal** (`find all nodes that supersede KNOW-042`) — graph operation
- **Tag matching** — set intersection
- **Time queries** (`created > 2026-04-01`) — metadata sort
- **Wikilink reverse lookup** — already a graph edge

In practice, > 80% of vault queries fall into these categories. **Embedding is for the remaining < 20%.**

### 4.3 Tiered Embedding Policy

Not every node should be embedded. Recommended tiering by type:

| Type | Embed required? | Reason |
|---|---|---|
| `decision` | ✅ Yes | Frequently looked up via "what did we decide about X" semantic queries |
| `concept` | ✅ Yes | Cross-terminology mapping ("session 共享" vs "context 隔离") |
| `rule` | ✅ Yes | "Are there rules about X?" |
| `procedure` | ✅ Yes | "How do I X?" |
| `fact` | ⚠️ Optional | Most facts are queried by ID or metadata; embed only if heavily searched |
| `relation` | ❌ No | Always accessed via graph traversal, never via semantic search |

Controlled by frontmatter `embed: true / false`.

### 4.4 Embedding Storage

Embeddings live **outside** the Markdown source. Frontmatter only carries metadata pointing to the embedding (status, model, timestamp, optionally a vector ID). The actual vector is stored in:

- **Sidecar files** (`Vault/.embeddings/KNOW-042.npy`) for simple cases, OR
- **A separate vector DB** (LanceDB / Chroma / sqlite-vss) for richer querying

This keeps Markdown sources clean and human-editable. The `.embeddings/` directory is in `.gitignore` (rebuildable from sources).

### 4.5 Embedding Pipeline

Async, decoupled from the write path:

```
Trigger:
  - File mtime change in Vault/knowledge/, OR
  - Frontmatter shows embed_status: dirty

Action:
  1. Load file, parse frontmatter
  2. Skip if embed: false or embed_status: none
  3. Build embedding input text:
     - "{type}: {title}\n\n{body}" (or richer template per type)
  4. Call embedding API (e.g., Gemini Embedding 2)
  5. Store vector in sidecar / vector DB
  6. Update frontmatter: embed_status: synced, embed_updated: <now>
```

This is **not** inline in `obsidian_write_note`. Writes stay fast; indexing catches up asynchronously.

### 4.6 Why Not Use Embeddings at All

Phase A explicitly does not use embeddings because:

- Vault size is O(100s) — keyword search remains fast (sub-second)
- Embedding cost is non-zero (API calls + storage + index maintenance)
- Embedding quality varies across providers and models — locking in early creates migration burden
- Without embeddings, the system can be fully built and validated; embeddings are a clean add-on

---

## 5. Three-Phase Rollout

Adoption is incremental. Each phase produces a working system; later phases add capabilities without breaking earlier ones.

### Phase 1 — Current State: File-based RAG (✅ Implemented)

**What's already built:**
- Obsidian vault with ~100s of Markdown files as source of truth
- Full Obsidian MCP server (11 tools: read / write / patch / search / manage)
- Three-layer security (path sandbox + blacklist + soft delete to `.trash/`)
- CAS optimistic locking + per-file `asyncio.Lock`
- `knowledge_shelf` subgraph with Gemini integration
- Jei (Knowledge Curator) agent dedicated to vault operations
- WSL↔Windows rsync sync (one-way push)
- LangGraph state integration (`knowledge_vault`, `knowledge_result` auto-injection)
- Implicit edges (wikilinks `[[...]]`, tag co-occurrence)
- Implicit metadata (Obsidian frontmatter parsed by MCP)

**What's still missing for a true knowledge graph:**
- Formalized node identity (no `knowledge_id`, addressing is by file path)
- Typed relations (no `supersedes / depends_on / contradicts`)
- Lifecycle status (no `active / superseded / invalidated`)
- Type-aware update semantics (all edits are file-level, no concept of "this is a decision being superseded vs a fact being corrected")

Phase 1 is **production-ready as a file-based knowledge management system**, but does not yet treat knowledge as graph-native nodes. Phases 2-4 add that formalization.

See `rag-architecture-design.md` for the detailed Phase 1 architecture.

### Phase 2 — Formal knowledge graph nodes (📋 designed, not yet implemented)

Add the formalization layer described in §2-§3 of this document:

**Added:**
- `knowledge/` subdirectory in vault for atomic knowledge node files
- Knowledge node frontmatter schema (`knowledge_id`, `type`, `status`, `relations`)
- ID registry (`knowledge/REGISTRY.md`)
- All edges from frontmatter `relations` declarations (in addition to existing wikilinks + tags)
- New MCP tools: `knowledge_get(id)`, `knowledge_list(filter)`, `knowledge_create(...)`, `knowledge_update(...)`, `knowledge_supersede(old, new)`, `knowledge_neighbors(id)`
- Type-aware update flows (see §6)

**Unchanged from Phase 1:**
- Underlying file storage and MCP layer
- Search remains keyword/regex-based
- No embeddings yet

**Retrieval mechanisms:**
- Exact ID lookup (O(1))
- Metadata filtering (`type` / `status` / `scope` / `tags`)
- Keyword full-text search (existing `obsidian_search_files`)
- Graph traversal (BFS along typed edges)

**Hard rule for the LLM curator (Jei):**
> When creating a knowledge node, you MUST declare its `relations` (references, depends_on, supersedes if applicable). If you don't know related nodes, run `knowledge_search` first to find candidates from existing nodes. An undeclared relation = an orphan node that no one will find.

**Coexistence:** Phase 1 capabilities remain active. Documents (files without `knowledge_id`) and knowledge nodes coexist in the same vault.

### Phase 3 — Semantic retrieval (📋 designed, triggered when nodes > 200 OR semantic search becomes painful)

**Added:**
- Embedding pipeline (async indexer, decoupled from write path)
- Embedding storage (sidecar files or vector DB like LanceDB)
- New retrieval tool: `knowledge_search(query, mode="semantic")`
- `embed: true / false` frontmatter field on knowledge nodes (opt-in)

**Unchanged:**
- Edges still only explicit (frontmatter `relations` + wikilinks + tags)
- Embeddings serve retrieval only, not edge construction

**Behavior:**
- LLM agent uses semantic search to **find candidate related nodes** when creating a new node
- Then LLM decides which candidates to actually link via `relations` field
- Embeddings inform humans/LLMs but don't autonomously create edges

### Phase 4 — Automatic edge discovery (📋 designed, triggered when nodes > 1000 OR knowledge discovery features are wanted)

**Added:**
- Embedding-derived `semantic_similar` edges (with cosine weight)
- Optional: LLM classification of `semantic_similar` edges into typed relations
- Graph algorithms (Leiden community detection, PageRank, shortest-path queries)
- New tool: `knowledge_discover(node_id)` → suggests undeclared related nodes for curator review

**Behavior:**
- Indexer periodically scans node pairs above similarity threshold, adds `semantic_similar` edges
- These edges are visually distinguished from explicit edges (different color in graph view)
- Curators can promote `semantic_similar` edges to typed relations after review

---

## 6. Update Semantics by Type

When a node `K` with content `X` is updated to `X'`:

| Type | Recommended flow |
|---|---|
| `fact` | If correcting a wrong fact: set `K.status: invalidated`, append correction note. Optionally create new fact `K_new` with `references: [K]`. **Don't edit the body** — preserve "what we used to think" for audit. |
| `concept` | Edit body in place. Git history preserves prior state. Update `updated:` field. |
| `decision` | If revising the decision: set `K.status: superseded`, set `K.superseded_by: K_new`. Create `K_new` with `K_new.supersedes: K`. Old decision retained for audit. **Never edit a superseded decision in place.** |
| `rule` | Edit in place. Append a changelog section to body documenting what changed and why. |
| `procedure` | Edit in place. Emphasize: callers shouldn't need to know the procedure changed (no breaking changes to interface). |
| `relation` | Edit in place. **Must** also update both endpoint nodes — if `KNOW-X relates to KNOW-Y` was true and is no longer, both X and Y need their `relations` field updated. |

For all updates, consult `mentioned_in` to identify documents that reference this node. Those documents may need text updates.

---

## 7. Concurrency and CAS

The MCP server's CAS (compare-and-swap) protection extends to knowledge node operations:

- `knowledge_update(id, cas_hash, ...)` requires the caller to have read the current version first
- Concurrent updates fail with `conflict`, returning the new actual hash
- The caller (LLM or human) must re-read, merge, and retry

**Note:** CAS is at the file level. It does not understand semantic merge — two LLMs both updating disjoint frontmatter fields will still conflict. This is acceptable for current scale; richer 3-way merge is a Phase D concern.

---

## 8. Open Questions / Future Iterations

1. **Auto-maintenance of `mentioned_in`** — currently manual. Could be a periodic indexer job that scans the vault for `[[KNOW-XXX]]` patterns and rebuilds reverse indexes.
2. **Multi-vault scoping** — if multiple vaults exist (per-project knowledge), do we need namespace prefixes? Currently flat.
3. **Embedding model migration** — when moving from `gemini-embedding-001` to `-002`, how do we re-index without re-curation? Need a versioned re-embed flow.
4. **Privacy / retention** — which nodes never leave local? Currently no policy. May matter if vault is ever shared / synced beyond the local machine.
5. **Knowledge promotion from Claude/Gemini sessions** — sessions accumulate facts in dialogue. Should there be a tool for the agent to "promote this insight to a knowledge node"? Not yet designed.
6. **Conflict resolution UX** — if two curators (or agent + human) edit concurrently, current CAS just rejects the second. Could grow into:
   - Auto-merge for disjoint frontmatter fields
   - LLM-mediated 3-way merge for body conflicts
   - Branching ("propose change" workflow)

---

## 9. Bootstrap: Initial Knowledge Nodes

Suggested seed set, extracted from current architectural decisions, to validate the schema and flow before broader rollout:

| ID | Type | Title |
|---|---|---|
| KNOW-001 | concept | ZenithLoom uses LangGraph 1.0.10 for state orchestration |
| KNOW-002 | concept | Subgraph isolation has two layers: SubgraphInputState (declarative) + session_mode wrappers (imperative) |
| KNOW-003 | decision | `inherit` session_mode is NotImplementedError — cross-provider session UUID incompatibility |
| KNOW-004 | rule | Vault source of truth is WSL filesystem; Windows mirror via rsync push |
| KNOW-005 | rule | Obsidian MCP enforces three-layer path sandbox + CAS optimistic locking |
| KNOW-006 | decision | ClaudeSDKNode uses sdk_query (per-call subprocess), not persistent client |
| KNOW-007 | rule | `_fresh_wrapper` clears node_sessions + messages + routing_context + subgraph output fields |
| KNOW-008 | decision | Knowledge graph: file-as-node, explicit edges first, embedding later (this design) |

These seed nodes serve double duty:
1. Validate the schema works in practice
2. Document the most-referenced architectural commitments in the new format
3. Give the LLM agent (Jei) examples to follow when creating new nodes

---

## 10. Relationship to Other Designs

| Doc | Relationship |
|---|---|
| `rag-architecture-design.md` | Detailed description of Phase 1 (current production file-based RAG). This doc adds Phase 2 onward (formal knowledge graph, embeddings, automatic discovery). |
| `session-mode-design.md` | Governs how `knowledge_shelf` subgraph isolates context per call — relevant when curator agent (Jei) operates on the graph. |
| `claude-cli-node-design.md` | LLM tool execution layer used by Jei to interact with the MCP. |
| `Vault/知识/PrismRag-v4.0-设计文档.md` | The graph-first RAG paradigm explored long-term — Phase 4 aligns with its embedding-derived edge approach. |

---

## 11. Decision Log (this document's rationale)

For traceability of why this design was chosen over alternatives:

- **D1: Knowledge element = L4 (graph node) not L1/L2/L3** — Lower levels are implementation/analytical details. Identity belongs at L4.
- **D2: Node = file with frontmatter** — Section-based IDs are unstable. Block IDs pollute source. Triple extraction premature.
- **D3: Documents and knowledge nodes coexist** — Forcing all knowledge into atomic files would destroy long narrative docs. Allowing both lets each form serve its purpose.
- **D4: Edges explicit-first, embedding-later** — At small scale, similarity edges add more noise than value. Defer until needed.
- **D5: Embedding tiered, opt-in via `embed: true`** — Not every node benefits from semantic search. Don't pay the cost for nodes that won't use it.
- **D6: Embedding storage is sidecar, not in Markdown** — Source files remain pure. Embeddings are rebuildable / replaceable.
- **D7: Three-phase rollout** — Each phase produces a working system. No big-bang migration. Defer complexity until justified by scale.
- **D8: Update semantics differ by type** — A wrong fact and an outdated decision warrant different flows. The framework guides but doesn't enforce.
