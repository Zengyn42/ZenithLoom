# Framework Design Decision Records

> Last updated: 2026-03-18

Records design rules established during real development of the ZenithLoom framework. Each rule has a corresponding bug or requirement as its source.

---

## Decision 1: State Field Ownership

> Source: colony_coder 392-iteration infinite loop

**Rule**: Each state field has exactly one type of node that can write to it.

| Field | Owner | Forbidden Writers |
|-------|-------|-----------------|
| retry_count | DETERMINISTIC validator | LLM nodes |
| routing_target | All nodes (each sets their own routing) | — |
| tasks, execution_order | DETERMINISTIC validator (parses and fills) | LLM nodes (output is in messages) |
| messages | All nodes | — |
| node_sessions | LLM nodes (write their own session_key) | — |

**Reason**: LlmNode returning `retry_count: 0` overwrote the validator's incremented value, causing infinite loops.

---

## Decision 2: tools=[] Means Read-Only Mode

> Source: planner/debate nodes ignoring system prompt and using tools to write files

**Rule**: `"tools": []` in node_config means "read-only mode" — forbids write/execute tools, retains read-only tools.

**Semantics**:
| Configuration | Meaning |
|--------------|---------|
| `"tools"` absent | Use global tool set (config.tools) |
| `"tools": ["Read", "Grep"]` | Only allow specified tools |
| `"tools": []` | Read-only mode: forbid Write/Edit/Bash etc., retain Read/Glob/Grep/WebSearch/WebFetch |

**Implementation (two layers)**:

1. `llm_node.py _select_tools()` — distinguishes "not configured" from "configured as empty list":
```python
_MISSING = object()
node_tools = self._cfg.get("tools", _MISSING)
if node_tools is _MISSING:
    tools = list(self.config.tools)  # not configured → use global
else:
    tools = list(node_tools or [])   # explicitly configured → use configured value
```

2. `claude.py _make_options()` — inject disallowed_tools when empty list:
```python
_WRITE_TOOLS = ["Write", "Edit", "MultiEdit", "Bash", "TodoWrite", "NotebookEdit"]
if isinstance(_allowed, list) and len(_allowed) == 0:
    _disallowed = _WRITE_TOOLS  # forbid writes, retain read-only
```

**Reason**:
- Claude SDK `allowed_tools=[]` means "unspecified" (all allowed by default), cannot be used to disable tools
- Debate nodes need to search the web and read files for research, but must never write files
- System prompt "do not use tools" is unreliable; Claude ignores it

---

## Decision 3: LLM Output Parsing is the Responsibility of DETERMINISTIC Nodes

> Source: task_decompose outputting JSON that the validator didn't recognize

**Rule**: LLM nodes only write AIMessage to `messages`. Extraction of structured data (JSON parsing, field filling) is handled by downstream DETERMINISTIC nodes.

**Reason**:
- LLM output format is unreliable (may include markdown fences, surrounding commentary)
- Parsing logic naturally couples with validation logic (parse failure = validation failure = retry)
- DETERMINISTIC nodes are deterministic, easy to test and debug

---

## Decision 4: Debate Subgraph Sessions Are Temporary

> Source: architecture design

**Rule**: When a SUBGRAPH_REF-referenced debate subgraph finishes, its internal session files are cleaned up. Debate conclusions are passed to the parent graph through `messages` and `state_out` mappings, not through session inheritance.

**Implementation**: `SubgraphRefNode._cleanup_orphan_sessions()` after subgraph ends:
1. Identifies newly added keys in `node_sessions`
2. Deletes corresponding Gemini session JSON files
3. Deletes corresponding Claude session directories

**Reason**: Debate subgraphs may be called multiple times (for different topics); old sessions would contaminate new debate contexts.

---

## Decision 5: SubgraphRefNode routing_context Fallback

> Source: debate subgraph in planner not receiving task content

**Rule**: When `state["routing_context"]` is empty, SubgraphRefNode automatically falls back to the parent graph's `messages[-1].content` as subgraph input.

**Reason**: `routing_context` is a dedicated field for main graph routing signals. When a subgraph is a fixed pipeline segment (not routing-triggered), the task content is in messages, not in routing_context.

---

## Decision 6: Token Safety Valve is Node-Level

> Source: colony_coder instantly exhausting tokens

**Rule**: Each LLM node checks cumulative tokens before `call_llm()`. Limits are configured at 3 priority levels:

```
node_config["token_limit"]  >  default by type  >  BB_TOKEN_LIMIT environment variable
```

**Reason**:
- Cloud APIs (Claude/Gemini) are billed; limits need to be tight (50k)
- Local inference (Ollama) is not billed; limits can be looser (1M)
- Special nodes may need larger/smaller limits; supports node_config override

---

## Decision 7: Debug Logs Stored in Directory Hierarchy Reflecting Graph Nesting

> Source: project requirements

**Rule**: Logs are stored as `.md` files; directory structure mirrors graph nesting levels.

```
logs/2026-03-18/
  colony_coder/                    ← master graph
    flow.md                        ← node enter/exit, routing
    thinking.md                    ← LLM thinking content
    colony_coder_planner/          ← planner subgraph
      flow.md
      thinking.md
      design_debate/               ← debate sub-subgraph
        flow.md
        thinking.md
```

**Implementation**: `ContextVar _graph_scope` maintains the current graph level stack; `push_graph_scope(name)` / `pop_graph_scope()` managed in GraphController and SubgraphRefNode.

---

## Decision 8: Heterogeneous Debate is Better than Self-Debate

> Source: planner test comparison

**Rule**: When design review is needed, prefer `debate_claude_first` (Claude proposes + Gemini reviews) over multiple Claude nodes debating themselves.

**Evidence**:
- Claude self-debate produces highly repetitive content across 3 nodes; no genuine opposition
- In Claude + Gemini debate, Gemini raised issues Claude hadn't considered from different dimensions (cross-platform compatibility, rendering performance, edge cases)
- Final task_decompose output was more complete and robust
