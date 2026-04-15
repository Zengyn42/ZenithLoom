# RAG Architecture Design — Phase 1 (Current Production)

> Date: 2026-04-09
> Status: Describes the **currently deployed** Phase 1 file-based RAG (commit 6d55910)
> Scope: Obsidian-vault-backed retrieval system used by ZenithLoom agents
> Companion: `knowledge-graph-design.md` (Phase 2-4: formal knowledge graph + embeddings + auto-discovery)

## Phase Context

This document describes **Phase 1** of the RAG roadmap — a working, production-ready file-based knowledge management system. It includes a security-hardened MCP server, CAS concurrency control, a dedicated curator agent, and end-to-end LangGraph integration.

What Phase 1 deliberately does NOT have:
- Formalized knowledge node identity (no `knowledge_id`, no typed relations, no lifecycle status)
- Vector embeddings or semantic search
- Automatic edge discovery between knowledge units

These are planned for Phase 2-4. See `knowledge-graph-design.md`.

## Summary

ZenithLoom's RAG is a **lightweight, file-native, keyword-search knowledge layer** built around an Obsidian vault. It does **not** use vector embeddings or a vector database today. Retrieval is performed via structural file operations (read / list / regex search / wikilink traversal) exposed as an **MCP server**, consumed by LLM nodes through an optional **LangGraph subgraph** (`knowledge_shelf`).

The design optimizes for three properties:

1. **Human-editable source of truth** — the vault is plain Markdown, directly usable in the Obsidian desktop app without the framework.
2. **Strong write guarantees** — all writes go through a Compare-And-Swap-protected MCP server; concurrent LLM writes cannot silently clobber each other.
3. **Sandboxed LLM access** — agents never touch the vault filesystem directly through raw OS tools; all mutations are mediated by a path-constrained server that enforces blacklists and soft-delete.

Agents access the vault in one of two ways:
- **Mediated** (Jei, Knowledge Curator) — via the MCP server, with routing through a dedicated subgraph
- **Direct** (Hani, Technical Architect) — via Claude's native `Read / Glob / Grep` tools, read-only in practice

## Design Principles

1. **Files over databases** — knowledge stays as `.md` files so it survives framework replacement, works with existing Obsidian tooling, and can be version-controlled via git.
2. **Thin abstraction** — no embedding pipeline, no indexing daemon, no vector store. Search cost scales linearly with vault size (~100s of files is fine; thousands would need an index).
3. **Separation of retrieval and reasoning** — the MCP server returns raw content; the LLM decides what's relevant. No hardcoded relevance ranking.
4. **Write safety through CAS** — optimistic locking via SHA-256 content hash prevents lost updates under concurrent LLM operations.
5. **WSL as source of truth** — the vault lives in WSL; a rsync push keeps the Windows mirror fresh for the Obsidian desktop GUI.
6. **Agent-specific access layer** — different agents get different levels of vault access (full MCP / read-only file I/O / none) based on role.

---

# Layer 1 — Physical Storage

## 1.1 Location and Layout

The vault is a conventional Obsidian repository on the WSL filesystem:

```
/home/kingy/Foundation/Vault/
├── .git/                # Version control (not via MCP)
├── .obsidian/           # Obsidian GUI config, plugins, workspace state
├── .trash/              # Soft-delete staging area (created on first delete)
├── 设计细节/            # Design detail specs
├── 问题整理/            # Issue tracking and post-mortems
├── 操作手册/            # Operational runbooks
├── 概念介绍/            # Concept definitions
├── 项目管理/            # Project management docs
├── 知识/                # External knowledge imports (RAG research, papers)
├── superpowers/         # Skill specs
└── 未分类/              # Uncategorized
```

A Windows mirror exists at `/mnt/c/Users/kingy/Documents/Obsidian Vault/` for use by the Obsidian desktop app. See Layer 6 (Sync).

## 1.2 File Formats

Accepted file extensions (`mcp_servers/obsidian/core/vault.py`):

```python
_ALLOWED_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".canvas", ".base"})
```

Notes follow Obsidian conventions:

```markdown
---
tags:
  - Category1
  - nested/tag/path
created: 2026-03-21
aliases: [Alternative Name]
category: 设计细节
---
# Note Title

Content with [[wikilinks]] and #inline-tags.
```

The MCP server parses frontmatter (YAML) into a separate `frontmatter` dict in its read responses, so agents get structured metadata without having to re-parse the Markdown.

## 1.3 What Does NOT Live in Knowledge Storage

These are intentionally outside the vault RAG layer:

- **Agent runtime data** (`~/Foundation/EdenGateway/agents/*/`): SQLite checkpoints, sessions.json, identity.json — managed by the SessionManager, not the RAG system.
- **Claude CLI sessions** (`~/.claude/projects/*/`): raw conversation transcripts managed by Claude — accessible via `ClaudeSDKNode.get_recent_history()`, not through the RAG MCP.
- **Source code** (ZenithLoom repo itself): accessed through Claude's Read/Glob/Grep tools, not the Obsidian MCP.

The vault is **curated long-term knowledge**, not ephemeral state.

---

# Layer 2 — Indexing

## 2.1 Current State: No Index

There is no vector store, no embedding pipeline, no inverted index. Every search performs a fresh pass over the vault files.

**Why this works today:**
- Vault size: O(100) files, O(1-10) MB total
- Search cost: `rglob("*.md")` + regex match per line → sub-second on SSD
- Consistency: the filesystem *is* the index; no staleness window, no reindex job

**When this breaks:**
- > 10,000 files → `rglob` walks become the bottleneck
- Cross-lingual retrieval ("找上周讨论" where notes are in English) → regex can't bridge language gaps
- Semantic queries ("与 session 污染相关的讨论") where the exact keywords don't appear in the target docs

See `docs/vault/architecture/rag-architecture-design.md` future sections (out of scope for this doc) or the PrismRag v4.0 design in the vault itself for the graph-first evolution plan.

## 2.2 Search Implementation

`mcp_servers/obsidian/tools/search.py`:

```python
pattern = re.compile(re.escape(query), re.IGNORECASE)
for p in sorted(base.rglob("*.md")):
    content = p.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(content.split("\n"), 1):
        if pattern.search(line):
            matches.append({
                "line": i,
                "content": line.strip()[:200],
            })
    if matches:
        results.append({
            "path": str(p.relative_to(base)),
            "matches": matches[:max_matches_per_file],
        })
```

Two search modes:

| Mode | Scans | Use case |
|---|---|---|
| `content` | File body (line-by-line regex) | Phrase lookup across notes |
| `filename` | File paths (no body read) | Title-only lookup, cheap |

Results are grouped by file, with line-number context. Truncation at `max_results` (default 20) prevents context overflow.

## 2.3 Link Graph Traversal

`mcp_servers/obsidian/tools/search.py:obsidian_get_links`:

Parses `[[wikilink]]` and `[text](path.md)` syntax to build a lightweight link graph on demand:

- **Outgoing**: links written in the current note's body
- **Incoming**: other notes containing a `[[...]]` to this note (computed by scanning all vault files)

This is not a precomputed index; incoming lookup walks the whole vault each call. Cheap enough at current vault size.

---

# Layer 3 — MCP Protocol

The Obsidian MCP server is the sole mediated access point to the vault. It lives at `mcp_servers/obsidian/` and is started as a daemon process per host.

## 3.1 Process Lifecycle

**Startup**: Agents declare the server in their `entity.json`:

```json
{
  "name": "obsidian-vault",
  "module": "mcp_servers.obsidian.server",
  "module_args": [
    "--transport", "sse",
    "--port", "8101",
    "--vault", "/home/kingy/Foundation/Vault"
  ],
  "url": "http://localhost:8101/sse",
  "shared": true
}
```

`shared: true` means the first agent to boot spawns the daemon; subsequent agents connect to the same instance. The `MCPLauncher` (`framework/mcp_launcher.py`) handles:

1. PID-file-based running check
2. File-lock-guarded spawn (prevents two agents racing to launch)
3. Detached subprocess (`start_new_session=True`) with PID written to `data/obsidian/obsidian.pid`
4. SSE readiness polling before registering tools

**Transport**: SSE over HTTP by default. stdio mode exists for single-client embedding, but SSE is used in production because multiple agents (Hani, Jei) share one daemon.

## 3.2 Tool Surface

Eleven tools in four categories:

### Read
| Tool | Parameters | Returns |
|---|---|---|
| `obsidian_read_note` | `path` | `{content, frontmatter, cas_hash, mtime_ms, path}` |
| `obsidian_list_files` | `directory, pattern="*.md", recursive=true` | `{files: [{path, size_bytes, mtime_ms}], count}` |

### Write (CAS-protected)
| Tool | Parameters | Returns |
|---|---|---|
| `obsidian_write_note` | `path, content, cas_hash` | `{cas_hash_new}` on success |
| `obsidian_patch_note` | `path, cas_hash, operations[]` | `{cas_hash_new}` — section-level edits |

`operations` is a list of typed ops:
- `update_frontmatter` — merge YAML fields
- `replace_section` — replace content under a `#` heading
- `append_to_section` — append under a heading
- `insert_after_section` — insert between headings
- `delete_section` — remove heading + its content

Section-level patches are idempotent against reorderings and much cheaper than whole-file rewrites.

### Manage
| Tool | Parameters | Notes |
|---|---|---|
| `obsidian_move_note` | `source, destination` | Rename / move across directories |
| `obsidian_delete_note` | `path, cas_hash, permanent=false` | Default: soft delete to `.trash/` |
| `obsidian_get_frontmatter` | `path` | Parsed YAML dict |
| `obsidian_update_frontmatter` | `path, cas_hash, updates` | Merge into frontmatter |
| `obsidian_manage_tags` | `path, cas_hash, action, tags` | Add / remove tags from frontmatter |

### Search
| Tool | Parameters | Returns |
|---|---|---|
| `obsidian_search_files` | `query, search_type, directory, max_results` | Grouped matches with line numbers |
| `obsidian_get_links` | `path` | `{outgoing: [...], incoming: [...]}` |

## 3.3 Unified Return Envelope

Every tool returns the same shape:

```json
{
  "status": "success" | "error",
  "data": { ... },
  "error_code": "conflict" | "not_found" | "permission_denied" | "path_traversal" | ...,
  "message": "Human-readable",
  "metadata": {"cas_hash": "...", "index_status": "..."}
}
```

This lets the LLM handle errors programmatically (e.g., retry on `conflict` by re-reading for a fresh hash) without string parsing.

## 3.4 Security: Three-Layer Path Sandbox

`mcp_servers/obsidian/core/vault.py:resolve_path()`:

**L1 — Sandbox containment**:
```python
abs_path = (self.base_dir / cleaned).resolve()  # resolves symlinks
abs_path.relative_to(self.base_dir)             # raises if outside vault
```
Any attempt to escape (`../../../etc/passwd`, absolute path overrides, symlink traversal) is rejected before I/O.

**L2 — Directory blacklist**:
```python
_BLOCKED_DIRS = frozenset({
    ".obsidian", ".git", ".trash", "node_modules", ".DS_Store"
})
```
Write/delete is forbidden under these directories. Reads may still succeed for inspection purposes (controlled per-tool).

**L3 — Soft delete**:
```python
if permanent:
    resolved.unlink()
else:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    shutil.move(resolved, trash_dir / f"{resolved.stem}_{ts}{resolved.suffix}")
```
Default deletion path is `.trash/note_20260414_002345.md`. Permanent deletion requires an explicit flag, giving recovery time for accidental LLM mistakes.

## 3.5 Concurrency Control: CAS

Every write tool takes a `cas_hash` parameter. The server:

```python
def verify_cas(path, expected_hash) -> (bool, actual_hash):
    if expected_hash is None:       # create-only
        return (not path.exists()), compute_file_hash(path)
    if not path.exists():
        return False, ""
    actual = compute_file_hash(path)  # SHA-256 of file bytes
    return actual == expected_hash, actual
```

Behavior matrix:

| `cas_hash` value | File exists? | Result |
|---|---|---|
| `null` | No | ✅ Create new |
| `null` | Yes | ❌ `conflict` (returns actual hash) |
| `"abc..."` | Matches | ✅ Write |
| `"abc..."` | Mismatch | ❌ `conflict` (returns actual hash) |
| `"abc..."` | File gone | ❌ `not_found` |

In addition, per-file `asyncio.Lock` serializes writes to the same file within one server process, so CAS + file-lock gives a two-layer guarantee.

**Why this matters**: two LLM nodes routed into `knowledge_shelf` concurrently (e.g., user fires two questions while the first is mid-flight) cannot both write to the same note and silently drop one update.

---

# Layer 4 — Functional Subgraph (`knowledge_shelf`)

## 4.1 Purpose

Wraps the MCP-mediated vault access as a reusable LangGraph subgraph. Any parent agent can route into `knowledge_shelf` via the standard routing signal (`{"route": "knowledge_shelf", "context": "..."}`) without needing to know about MCP transports, tool schemas, or sync.

## 4.2 Graph Definition

`blueprints/functional_graphs/knowledge_shelf/entity.json`:

```json
{
  "name": "knowledge_shelf",
  "routing_hint": "当需要读写、搜索、管理 Obsidian Vault 中的笔记时使用...",
  "llm": "gemini",
  "graph": {
    "entry": "gemini_obsidian",
    "exit": "vault_sync_push",
    "nodes": [
      {
        "id": "gemini_obsidian",
        "type": "GEMINI_CLI",
        "model": "gemini-2.5-pro",
        "session_key": "knowledge_shelf",
        "output_field": "knowledge_result",
        "skill_files": ["skills/obsidian/obsidian_skill.md"],
        "system_prompt": "... 必须使用 obsidian_* MCP 工具 ..."
      },
      {
        "id": "vault_sync_push",
        "type": "EXTERNAL_TOOL",
        "node_config": {
          "command": [
            "rsync", "-a", "--delete",
            "/home/kingy/Foundation/Vault/",
            "/mnt/c/Users/kingy/Documents/Obsidian Vault/"
          ],
          "timeout": 30
        }
      }
    ],
    "edges": [
      {"from": "gemini_obsidian", "to": "vault_sync_push"}
    ]
  }
}
```

Two nodes, one edge. Entry is the Gemini LLM node; exit is the rsync step.

## 4.3 Node Behavior

**`gemini_obsidian`**:
- Connects to the Obsidian MCP server (already running, started by the parent agent's bootstrap)
- Receives the routing_context from the parent as its prompt
- Uses the `obsidian_*` MCP tools autonomously to read/search/write the vault
- Writes its final synthesized answer into `state["knowledge_result"]` via `output_field`

The `system_prompt` explicitly forbids raw file I/O and mandates MCP tool use, ensuring the CAS/sandbox guarantees are actually enforced.

**`vault_sync_push`**:
- Runs unconditionally after `gemini_obsidian`, even on read-only queries (redundant but idempotent)
- Pushes the WSL vault to the Windows mirror via rsync with `--delete` for exact mirroring
- Failure is logged but doesn't propagate — the subgraph returns the knowledge_result regardless

## 4.4 Why Gemini, Not Claude

The choice of Gemini for this subgraph is deliberate:

- Vault operations are largely I/O-bounded (search, read, minor edits) — no deep reasoning required, which plays to Gemini's cheaper per-token cost
- Gemini's longer context window accommodates multiple large notes without aggressive truncation
- Jei (the role agent owning this subgraph) is also Gemini-based, making MCP session sharing conceptually cleaner (though session_mode currently prevents cross-subgraph sharing)

## 4.5 Session Mode

When referenced from a parent agent, `knowledge_shelf` should be declared with `session_mode: "fresh_per_call"` (see `docs/vault/architecture/session-mode-design.md`) to ensure each query starts with a clean Gemini session. The current Hani blueprint doesn't yet reference knowledge_shelf; it accesses the vault through Jei instead.

---

# Layer 5 — Agent Layer

Three agents have three distinct access patterns to the vault.

## 5.1 Jei — Knowledge Curator (mediated access)

`blueprints/role_agents/knowledge_curator/`:

- **LLM**: Gemini (consistent with knowledge_shelf)
- **MCP attached**: `obsidian-vault` (shared daemon)
- **Access pattern**: Routes to `knowledge_shelf` subgraph for any vault operation
- **Additional subgraphs**: `render_slides` (Presenton), `render_docs` (Pandoc)

Jei is the **dedicated curator role**. Her `PROTOCOL.md` makes routing explicit — she emits `{"route": "knowledge_shelf", "context": "..."}` as the first line of her output for vault tasks, and the framework handles the actual MCP interaction inside the subgraph.

Routing hint in her agent blueprint reads:
> "当用户询问 Obsidian 笔记、知识库内容、文档整理时使用 knowledge_shelf 子图。"

**Why route instead of calling MCP directly from Jei's main node?** Isolation. The routing keeps Jei's top-level session clean — it sees only the final `knowledge_result`, not the multi-turn MCP tool-call dialogue. This prevents tool-call debris from polluting Jei's cross-session memory.

## 5.2 Hani — Technical Architect (direct file access)

`blueprints/role_agents/technical_architect/`:

- **LLM**: Claude
- **MCP attached**: `obsidian-vault` declared but… essentially unused in practice
- **Access pattern**: Direct `Read`, `Glob`, `Grep`, `Bash` tools on the filesystem
- **Routes to knowledge_shelf**: No (could, but not wired in current entity.json)

Hani reads the vault like any other directory using Claude's native tools. She doesn't write to it via MCP — her workflow is read-only in practice; any vault writes would either:

1. Go through raw `Write`/`Edit` (bypasses CAS but OK for single-agent use), or
2. Be delegated to Jei by conversation context (Hani tells the user "ask Jei to file this in the vault")

**Trade-off**: Hani gets faster, more direct access (no subgraph round-trip, no MCP serialization) but loses CAS safety and path sandboxing. This is acceptable because Hani is an architect role — her modifications to the vault are rare and always deliberate.

## 5.3 Asa — Administrative Officer (no vault access)

- **LLM**: Llama (local Ollama)
- **MCP attached**: None (no obsidian-vault entry in her config)
- **Access pattern**: Local file I/O only, confined to her working scope

Asa has no RAG role. She exists for operational tasks (heartbeat monitoring, system stats, background jobs) that don't involve knowledge retrieval.

## 5.4 Access Pattern Matrix

| Agent | LLM | MCP | Subgraph | Direct file I/O | Writes CAS-safe? |
|---|---|---|---|---|---|
| Jei | Gemini | ✅ obsidian-vault | ✅ knowledge_shelf | No | ✅ |
| Hani | Claude | ✅ obsidian-vault (declared, unused) | No | ✅ (Read/Glob/Grep) | ❌ (if writes via raw Write) |
| Asa | Llama | No | No | Local scope only | N/A |

---

# Layer 6 — Sync

## 6.1 Topology

```
┌──────────────────────────┐        ┌──────────────────────────────┐
│ WSL (Linux)              │        │ Windows                      │
│                          │ rsync  │                              │
│ /home/kingy/Foundation/  │───────▶│ /mnt/c/Users/kingy/Documents/│
│   Vault/                 │  push  │   Obsidian Vault/            │
│  (source of truth)       │        │  (mirror, opened in Obsidian) │
│                          │        │                              │
└──────────────────────────┘        └──────────────────────────────┘
         ▲
         │  git pull
         │
   (manual / CI)
```

**Source of truth**: WSL vault. This is what the MCP server serves, what Jei reads, what knowledge_shelf writes.

**Windows mirror**: Updated by rsync at the end of every `knowledge_shelf` invocation. Opened in the Obsidian desktop app for human editing.

## 6.2 rsync Command

Fixed in `knowledge_shelf/entity.json`:

```bash
rsync -a --delete \
  /home/kingy/Foundation/Vault/ \
  /mnt/c/Users/kingy/Documents/Obsidian\ Vault/
```

- `-a` (archive): preserves permissions, timestamps, recursive
- `--delete`: removes files in destination that don't exist in source (exact mirror)
- Timeout: 30s (set in EXTERNAL_TOOL `node_config`)

## 6.3 Return-Path Sync (Human → WSL)

**Currently manual**. If the user edits a note in the Windows Obsidian app:

1. Obsidian's git plugin auto-commits the change to the vault's git repo (on the Windows side)
2. The user runs `git pull` in `/home/kingy/Foundation/Vault/` on WSL — or a scheduled job does it
3. WSL now has the user's edits

There is **no automatic bidirectional sync**. The design deliberately trades convenience for simplicity: one rsync direction, one git direction, never conflicting.

## 6.4 The Missing `vault-sync` MCP

Hani's `entity.json` declares a second MCP:

```json
{
  "name": "vault-sync",
  "module": "mcp_servers.vault_sync.server",
  "module_args": ["--transport", "sse", "--port", "8105"],
  "url": "http://localhost:8105/sse",
  "shared": true
}
```

**This module does not exist yet.** It's a forward-looking declaration. The intent was to wrap the rsync operation behind an MCP tool (`sync_pull`, `sync_push`, `sync_status`) so agents could trigger sync explicitly rather than having it tied to the `knowledge_shelf` exit node.

Current behavior if Hani references this MCP: `MCPLauncher` will fail to start it and log a warning; Hani will continue without it since no tool actually depends on it.

---

# Layer 7 — LangGraph State Integration

## 7.1 Relevant Fields in `BaseAgentState`

From `framework/schema/base.py`:

```python
class BaseAgentState(TypedDict):
    ...
    knowledge_vault: str           # Hint: vault root path (read by agents)
    project_docs: str              # Current project's /docs/ path
    knowledge_result: str          # knowledge_shelf subgraph output
    ...
```

Three fields relevant to RAG:

- **`knowledge_vault`**: typically `/home/kingy/Foundation/Vault` — agents can use it to pass as the `directory` argument to `obsidian_*` tools, or to `Glob`/`Grep` for direct-access agents.
- **`project_docs`**: per-project docs path (e.g., `docs/` inside a repo being worked on). Independent of the vault; used by agents that need to reference the current repo's internal documentation.
- **`knowledge_result`**: the **output channel** from `knowledge_shelf`. Written by `gemini_obsidian` via `output_field`, consumed by parent agents via prompt injection.

## 7.2 Injection into Parent Prompt

`framework/nodes/llm/llm_node.py:_build_gemini_section()`:

```python
knowledge = state.get("knowledge_result", "")
if knowledge:
    parts.append(f"\n[知识库查询结果]\n{knowledge}\n")
```

When a parent LLM node runs **after** a knowledge_shelf invocation, it automatically sees the retrieved content injected into its system prompt as `[知识库查询结果]`. No tool-call replay, no re-routing — just a state read.

This is why `knowledge_result` is treated as a "subgraph output field" in `SubgraphInputState`: it must be **blocked from flowing back into** the subgraph on the next invocation, otherwise the subgraph would see its previous output in its own prompt and drift. See `session-mode-design.md` for the isolation mechanism.

## 7.3 Subgraph Input Filtering

`SubgraphInputState` deliberately **omits** `knowledge_result` along with other subgraph output fields. When a parent routes into `knowledge_shelf`:

- The subgraph gets: `routing_context`, `knowledge_vault`, `project_docs`, `workspace`, `node_sessions`, ...
- The subgraph does NOT get: `knowledge_result` (stale from last run), `debate_conclusion`, `messages`, etc.

This ensures each `knowledge_shelf` call starts clean — Gemini won't see its own previous answer in its input.

---

# End-to-End Flow

## Scenario: User asks Hani "查一下我们上周讨论的 session_mode"

### Step-by-step

```
1. User → Hani (Discord channel)
   "查一下我们上周讨论的 session_mode"

2. Hani's claude_main node runs
   - Reads messages[-1].content as latest_input
   - system_prompt includes Hani's persona (ROLE.md + PROTOCOL.md)
   - Decides: this is a vault lookup
   - Option A (current): Claude uses Glob + Grep directly on /home/kingy/Foundation/Vault
   - Option B (if wired): Claude emits {"route": "knowledge_shelf", "context": "..."}
                          → parent graph routes to Jei or knowledge_shelf subgraph

3a. Option A — direct access
   - Claude calls Glob("/home/kingy/Foundation/Vault/**/*.md")
   - Claude calls Grep("session_mode", ...)
   - Claude calls Read(...) on top matches
   - Claude synthesizes an answer, responds to user

3b. Option B — mediated access (future wiring)
   - Parent graph routes to knowledge_shelf subgraph
   - _fresh_wrapper clears node_sessions, previous_node_output, etc.
   - gemini_obsidian runs:
     * obsidian_search_files(query="session_mode", ...)
     * obsidian_read_note("docs/vault/architecture/session-mode-design.md")
     * Writes answer to state["knowledge_result"]
   - vault_sync_push runs rsync
   - Subgraph returns to parent
   - Parent's claude_main, next turn, sees [知识库查询结果] in prompt
   - Claude summarizes for user

4. User receives response
```

### State Snapshot at Each Step (Option B)

```
After user input:
  state = {
    messages: [HumanMessage("查一下我们上周讨论的 session_mode")],
    workspace: "/home/kingy/Foundation",
    knowledge_vault: "/home/kingy/Foundation/Vault",
    ...
  }

After claude_main:
  state.routing_target = "knowledge_shelf"
  state.routing_context = "查 session_mode 相关笔记"

After _fresh_wrapper (entering subgraph):
  state.node_sessions = {}
  state.messages = [HumanMessage("查 session_mode...")]   # trimmed
  state.routing_context = ""
  # knowledge_result, debate_conclusion, etc. are blocked by SubgraphInputState
  # or explicitly cleared by _fresh_wrapper

After gemini_obsidian:
  state.knowledge_result = "找到以下相关笔记: session-mode-design.md..."
  state.node_sessions = {"knowledge_shelf": "<gemini_session_id>"}

After vault_sync_push:
  # No state change, just side effect (rsync)

Back in parent graph (claude_main next turn):
  state.knowledge_result = "找到以下相关笔记..."   # merged back
  # claude_main's prompt now includes [知识库查询结果] \n 找到以下...

Final output:
  Claude responds to user with a synthesized answer referencing the found notes.
```

---

# Failure Modes and Recovery

## 8.1 MCP server not running

**Detection**: `MCPLauncher.wait_ready()` times out (default 10s).

**Current behavior**: Agent bootstrap raises; agent fails to start. No graceful degradation — without the MCP, Jei is crippled (all her vault tools are gone).

**Mitigation**: Hani's direct file access works independently of the MCP, so the user can at least ask Hani for vault lookups while the MCP is down.

## 8.2 CAS conflict (concurrent write)

**Detection**: `obsidian_write_note` returns `status: error, error_code: conflict, metadata.cas_hash: <actual>`.

**Expected LLM behavior** (by system prompt): retry with the fresh `cas_hash` from the error response, re-read if content changed materially.

**Practical limit**: LLM retry loops should be capped at 2-3 attempts. If repeated conflicts occur, the agent should surface a "human intervention needed" message rather than retry indefinitely.

## 8.3 Rsync failure

**Detection**: `vault_sync_push` EXTERNAL_TOOL node exit code non-zero.

**Current behavior**: Logged via framework node error path. Subgraph still returns `knowledge_result` normally. Windows mirror becomes stale until next successful sync.

**Consequence**: User editing in Windows Obsidian might miss recent WSL-side changes. Not a data loss risk — WSL remains source of truth.

## 8.4 Path traversal / blacklist violation

**Detection**: `resolve_path()` returns an error dict with `error_code: path_traversal` or `permission_denied`.

**Expected LLM behavior**: The system prompt for `gemini_obsidian` tells Gemini that vault access is sandboxed; it should not attempt to access `.obsidian/` or `.git/`. If this occurs in practice, it's a prompt regression — log and review.

## 8.5 Vault file corrupted / unreadable

**Detection**: `obsidian_read_note` returns decode error via `errors="replace"` (file reads always succeed but may contain replacement characters).

**Expected LLM behavior**: Notice the corruption in content, flag to user, do NOT attempt to rewrite (would CAS against the corrupted version).

---

# Known Limitations

1. **No semantic search**. Cross-language queries, synonym lookup, and "vibe" matches all fail.
2. **Linear search cost**. O(vault_size) per query; scales to thousands of files but not millions.
3. **No incremental index**. Every search is a fresh pass; no caching beyond OS page cache.
4. **One-way sync**. Windows edits require manual git pull to reach WSL.
5. **No authentication on MCP**. The SSE server listens on localhost:8101 with no auth. Acceptable in WSL (single-user host) but would need TLS + auth for multi-user or network deployment.
6. **`vault-sync` MCP is declared but not implemented** — Hani's entity.json references a module that doesn't exist. Currently harmless (the launcher logs a warning) but leaves a hanging reference.
7. **No version history exposed to LLM**. The vault has a git repo, but agents can't query "what did this note look like last week" through the MCP. A future `obsidian_note_history(path, since)` tool would close this gap.
8. **No structural queries**. "Find all notes tagged #architecture created after 2026-03" requires a frontmatter-indexing layer that doesn't exist. Workaround: list + read + filter in LLM.

---

# Future Direction (Summary)

A full graph-first RAG v4.0 design exists in the vault itself (`Vault/知识/PrismRag-v4.0-设计文档.md`). Headline plans:

- NetworkX knowledge graph built from wikilinks + embedding-derived edges
- Leiden community detection to create hierarchical topic clusters
- Query-time: graph traversal (BFS/DFS), not vector search
- Embeddings relegated to index-time only (edge weight computation)
- LanceDB for embedding cache
- Multi-modal extension: images via Vision API, audio via Whisper

That's a substantial architectural change — out of scope for this document, which describes the current simple-but-solid system.

---

# See Also

- `docs/vault/architecture/session-mode-design.md` — explains how `SubgraphInputState` blocks `knowledge_result` from parent→subgraph flow
- `docs/vault/architecture/claude-cli-node-design.md` — the Claude-side tool execution layer
- `mcp_servers/obsidian/server.py` — MCP server entry point
- `mcp_servers/obsidian/core/vault.py` — path sandbox and security
- `mcp_servers/obsidian/core/cas.py` — compare-and-swap implementation
- `blueprints/functional_graphs/knowledge_shelf/entity.json` — subgraph definition
- `blueprints/role_agents/knowledge_curator/` — Jei's persona and protocol
- `framework/mcp_launcher.py` — MCP lifecycle management
- `Vault/知识/PrismRag-v4.0-设计文档.md` — future graph-first RAG design (in vault)
