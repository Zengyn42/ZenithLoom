# PrismRagMCP Inbox Architecture

> Date: 2026-04-09
> Status: Approved (architecture decision, not yet implemented)

## Summary

Knowledge notes (.md files) produced by LLMs enter the Obsidian Vault via a **producer-consumer model**.
Any LLM can write notes (regardless of whether it has MCP access). Curation and indexing are handled exclusively by the dedicated knowledge_curator role.

## Design Principle

**Zero barrier to write; curation delegated to a specialist.**

- Producers (technical_architect / administrative_officer / any LLM) do not need to understand the graph structure, tag taxonomy, or embedding mechanism
- Consumer (knowledge_curator + PrismRagMCP) handles all knowledge engineering: classification, tagging, embedding, graph maintenance
- Writing and indexing are decoupled: LLMs never call `embed()` — embedding is handled transparently by infrastructure

## Architecture

```
Producer (any LLM, with or without MCP)
    │
    ├── With MCP: call PrismRagMCP.write_note() → direct write + auto embedding
    │
    └── No MCP: use Write tool → ~/Foundation/Vault/inbox/*.md (raw file drop)
                    │
                    ▼
              ~/Foundation/Vault/inbox/          ← unsorted inbox
                    │
                    ▼
              Consumer (knowledge_curator, with PrismRagMCP)
                    │
                    ├── Periodically scans inbox/ for new .md files
                    ├── Understands content → assigns tags → selects category
                    ├── Calls PrismRagMCP tools to write to official Vault location
                    ├── Auto embedding + graph maintenance
                    └── Moves original to inbox/processed/ (for traceability)
```

## Two Modes of Writing

### Mode 1: With PrismRagMCP (direct)

LLMs with PrismRagMCP access write directly to the vault:

```
LLM calls prism_write_note(
    path="Architecture/session-mode-design.md",
    content="...",
    tags=["session", "subgraph", "langgraph"]
)
    │
    ▼
PrismRagMCP internally:
    1. Writes file to official Vault directory
    2. Auto-completes frontmatter (created_at, author, tags)
    3. Triggers embedding pipeline (chunk → embed → upsert vector store)
    4. Updates knowledge graph index
```

**Applicable roles**: knowledge_curator, any future role granted MCP access

### Mode 2: Without PrismRagMCP (inbox drop)

LLMs without PrismRagMCP use the native Write tool to drop files into inbox:

```python
LLM calls Write(
    path="~/Foundation/Vault/inbox/2026-04-09-session-design-notes.md",
    content="..."   # any format, plain text is fine
)
```

**Conventions**:
- Path: `~/Foundation/Vault/inbox/`
- Filename: `YYYY-MM-DD-<topic>.md` (recommended but not required)
- Content format: completely free — consumer will normalize
- Frontmatter: optional (consumer will complete)

**Applicable roles**: technical_architect, administrative_officer (no MCP access)

### Consumer: knowledge_curator's Inbox Processing

knowledge_curator processes the inbox periodically (via heartbeat or manual trigger):

```
1. Scan inbox/ for .md files (excluding processed/ subdirectory)
2. Read each file
3. Call prism_list_tags() to get existing tag taxonomy
4. Based on content understanding:
   - Select target directory (Architecture/ Projects/ Decisions/ etc.)
   - Generate tag list (reuse existing tags where possible, create new if needed)
   - Complete frontmatter (infer author from filename or content)
5. Call prism_write_note() to write to official location
6. Auto embedding (handled internally by PrismRagMCP)
7. Move original to inbox/processed/ (preserve traceability)
```

## PrismRagMCP Tool Interface

### LLM-facing tools

| Tool | Description |
|---|---|
| `prism_write_note(path, content, tags)` | Write note to official Vault location + normalize frontmatter |
| `prism_patch_note(path, section, content)` | Section-based partial update |
| `prism_read_note(path)` | Read complete note |
| `prism_search(query)` | Hybrid search: vector semantic + keyword |
| `prism_list_tags()` | Return existing tag taxonomy (reference when writing) |
| `prism_list_files(directory)` | List files in directory |
| `prism_get_links(path)` | Query bidirectional link relationships |

### Internal (auto-handled by PrismRagMCP, not called by LLMs)

| Internal Process | Trigger |
|---|---|
| Chunk + embed + upsert vector store | Automatically after `write_note()` / `patch_note()` |
| Tag index update | Automatically on tag changes |
| Knowledge graph update (link relationships) | Automatically after note write/edit |

**LLMs never need to know embedding exists.** They write notes, search notes, manage tags. Vectorization is transparent infrastructure.

## Role Summary

| Role | MCP Access | Write Method | Knowledge Responsibility |
|---|---|---|---|
| **technical_architect** | No PrismRagMCP | Write → inbox/ | Producer: records design decisions, technical learnings |
| **administrative_officer** | No PrismRagMCP | Write → inbox/ | Producer: records operational processes, management notes |
| **knowledge_curator** | Has PrismRagMCP | Direct write_note() | Consumer + Producer: curates inbox, manages knowledge base |
| **Future roles** | Configurable | Direct if MCP, else inbox | Depends on MCP assignment |

## Directory Convention

```
~/Foundation/Vault/
├── inbox/                    ← unsorted inbox (any LLM can write)
│   ├── 2026-04-09-<topic>.md
│   └── processed/            ← archived originals after knowledge_curator curation
├── Architecture/             ← official categories (managed by knowledge_curator via MCP)
├── Projects/
├── Decisions/
├── Operations/
└── ...
```

## Key Design Decisions

### Why not give all roles MCP access?

- **Separation of concerns**: technical_architect is a systems architect — managing tag taxonomy and vault organization is not its responsibility
- **Quality control**: knowledge_curator as the knowledge manager ensures consistent tags, categories, and embedding quality
- **Fault tolerance**: even if the MCP service is down, technical_architect and administrative_officer can still produce knowledge via inbox without data loss

### Why inbox + consumer, not direct write + async index?

- **LLMs don't know the vault structure**: requiring technical_architect to write directly to `Architecture/xxx.md` means it must understand the entire taxonomy first — not its job
- **Tag consistency**: if each LLM applies its own tags, fragmentation results (`#arch` vs `#architecture`). Centralized management is more controllable
- **Progressive upgrade**: inbox mode first, adding MCP to other roles later is seamless (adds one more write path, inbox still works)

### Why move to processed/ instead of delete?

- **Traceability**: can track diffs between original content and knowledge_curator's curated version
- **Recovery**: if knowledge_curator makes a mistake, originals can be restored from processed/
- **Audit**: can periodically review knowledge_curator's curation quality

## Future: PrismRagMCP = Obsidian MCP + RAG

PrismRagMCP consolidates the current `mcp_servers/obsidian/` with a future vector embedding system:

| Current (Obsidian MCP) | Future (PrismRagMCP) |
|---|---|
| Plain file read/write | File read/write + auto embedding |
| grep text search | Vector semantic search + keyword hybrid |
| Manual tag management | Auto tag suggestions on write (referencing existing taxonomy) |
| No knowledge graph | Bidirectional links + tags → knowledge graph |
| No inbox mechanism | Inbox + consumer pipeline |

## See Also

- `mcp_servers/obsidian/` — current Obsidian MCP implementation
- `blueprints/functional_graphs/knowledge_shelf/` — knowledge_curator's knowledge subgraph (in VoidDraft repo)
- `blueprints/role_agents/knowledge_curator/` — knowledge_curator role definition (in VoidDraft repo)
