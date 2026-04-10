# Subgraph session_mode Design

> Date: 2026-04-09
> Status: Living document — tracks current design + known limitations

## Summary

`session_mode` is a property of **subgraph reference nodes** (nodes with `agent_dir` in the parent graph's `entity.json`) that controls how the parent graph's state — specifically `node_sessions` and related fields — flows into the referenced subgraph on each invocation.

It is NOT a property of LLM nodes. LLM nodes use `session_key` to decide which slot in `node_sessions` they read/write.

## The Two-Layer Isolation Model

Subgraph state isolation operates on two layers:

| Layer | Mechanism | Scope | Applies To |
|---|---|---|---|
| **Schema layer** | `SubgraphInputState` as LangGraph `input_schema` | Declarative field flow control | Only subgraphs using `base_schema` |
| **Wrapper layer** | `_fresh_wrapper` / `_inherit_wrapper` / `_isolated_wrapper` in `agent_loader.py` | Runtime per-invocation state patching | All subgraphs referenced with non-default `session_mode` |

### Schema layer: `SubgraphInputState`

Defined in `framework/schema/base.py`. This is LangGraph's `input_schema` — fields NOT in this TypedDict get filtered out when flowing from parent graph into the subgraph.

Deliberately omitted fields (blocked from parent flow):

| Field | Reason |
|---|---|
| `messages` | Subgraphs have their own conversation scope, parent history shouldn't flow in. |
| `debate_conclusion` / `apex_conclusion` / `knowledge_result` / `discovery_report` | These are subgraph OUTPUT fields. `LlmNode._build_gemini_section()` injects them into every Claude node's prompt (designed for `claude_main` to read subgraph results). If they flow into a subgraph from parent, the subgraph's internal Claude nodes would see the previous invocation's conclusion polluting their current context. |
| `refined_plan` | colony_coder_planner output, same reasoning. |
| `previous_node_output` / `subgraph_topic` | Per-invocation transient fields used for inter-node communication within a subgraph. Stale parent values should not leak into the subgraph's first node. |

`node_sessions` uses the same `_merge_dict` reducer as in `BaseAgentState` — LangGraph 1.0.10 requires matching reducers between input and state schemas for shared fields. This allows:
- `inherit` wrapper to inject session IDs that flow through to subgraph nodes
- `persistent` checkpoint restoration not to be zeroed by input schema
- `fresh_per_call` / `isolated` wrapper injecting `{}` — `_merge_dict({}, ...)` stays empty

### Wrapper layer: session_mode wrappers

Each non-default `session_mode` wraps the compiled inner subgraph with an `async` function that patches `state` before invoking the subgraph. Implemented in `agent_loader.py` under the `elif node_def.get("agent_dir") and not node_type:` branch.

## The Four Modes

### `persistent` (default)

**Semantics**: Subgraph's `node_sessions` survive across invocations. First call creates LLM sessions; subsequent calls resume them.

**Implementation**: No wrapper. The inner compiled graph is added directly as a LangGraph node. LangGraph's checkpointer manages the subgraph's namespace, persisting state automatically.

**Use case**: Subgraphs with long-term memory — e.g., an advisor subgraph that a user can repeatedly consult and it remembers prior advice.

**Example**: `apex_coder` referenced from `technical_architect` (no explicit `session_mode`, defaults to persistent). Claude continues its ApexCoder session across multiple invocations.

### `fresh_per_call`

**Semantics**: Every subgraph invocation starts from a clean state. All LLM sessions are reset, messages trimmed to the user's original input, transient fields and subgraph output fields cleared.

**Implementation**: `_fresh_wrapper` patches state with:
- `node_sessions: {}` — forces all LLM nodes to create new sessions
- `messages: [first HumanMessage]` — drops LangGraph history, keeps only the user's initial question
- `routing_context: ""` — prevents parent routing signal (which may be a file path) from being used as the subgraph's first prompt
- `debate_conclusion / apex_conclusion / knowledge_result / discovery_report: ""` — prevents previous-invocation subgraph outputs from being injected into current Claude nodes via `_build_gemini_section`
- `previous_node_output / subgraph_topic: ""` — transient fields cleared

**Use case**: One-shot analysis subgraphs, debates, tool discovery pipelines. Each call must be independent.

**Example**: `debate_brainstorm` / `debate_design` referenced from `technical_architect`.

### `inherit` — NOT IMPLEMENTED

**Intended semantics**: Inject a parent graph LLM node's session ID into all of the subgraph's session keys, so the subgraph's LLM nodes continue the parent's conversation.

**Status**: Raises `NotImplementedError` on graph compilation. See "Known Limitations" below for why.

### `isolated`

**Semantics**: Subgraph invocation gets maximally unique sessions — each internal node is forced to a unique `session_key` AT BUILD TIME (via `force_unique_session_keys=True`), AND `node_sessions` is cleared AT CALL TIME.

**Implementation**:
- Build-time: `EntityLoader.build_graph(force_unique_session_keys=True)` rewrites each LLM node's `session_key` to be unique even if the entity.json declared shared keys.
- Call-time: `_isolated_wrapper` clears `node_sessions` before invocation.

**Use case**: Parallel fan-out of the same subgraph where multiple instances must not share state (e.g., concurrent sandbox evaluations).

## Default Values

| Config location | Default |
|---|---|
| `session_mode` on subgraph reference node | `persistent` |
| `session_key` on LLM node | node id (`self._node_id`) |
| `inherit_from` on `inherit` mode subgraph node | `claude_main` |

## Known Limitations

### 1. `inherit` is cross-provider incompatible (NOT IMPLEMENTED)

The `inherit` wrapper blindly copies the parent's session UUID into all the subgraph's `session_key` slots:

```python
parent_sid = state["node_sessions"].get(inherit_from, "")
injected = {sk: parent_sid for sk in subgraph_session_keys}
```

Each LLM provider has its own session store:
- Claude: `~/.claude/projects/<hash>/<uuid>.jsonl`
- Gemini: `~/.gemini/tmp/<hash>/chats/<uuid>.json`
- Ollama: in-process state (no persistence)

A Claude UUID does not exist in Gemini's session store. If a parent Claude node passes its UUID to a Gemini subgraph node, `gemini-cli --resume <claude_uuid>` fails with session not found.

Additionally, even within the same provider, there are concerns:
- Token/context window implications — the inherited session may already be large.
- Permission mode mismatch — the subgraph may run with different `permission_mode` than the original session was created under.
- System prompt divergence — the subgraph's system prompt differs from the parent node's, but resuming a session preserves the original system prompt.

**Current state**: the wrapper code exists in `agent_loader.py` but is now gated behind `NotImplementedError`. If any entity.json declares `session_mode: "inherit"`, graph compilation fails with a clear error message pointing the user at alternatives.

**Workarounds for sharing context across subgraphs**:
1. `fresh_per_call` + pass context via `messages` (LangGraph native message flow) — the current debate pattern
2. `fresh_per_call` + `SubgraphMapperNode` explicit field mapping
3. Manual text injection — extract N recent turns from parent's session via `ClaudeSDKNode.get_recent_history()`, format as prompt prefix

### 2. `SubgraphInputState` only filters `base_schema` subgraphs

The input filter in `agent_loader.py` is gated to `_schema_name == "base_schema"`. Subgraphs using custom schemas bypass the filter:

| Schema | Reason |
|---|---|
| `debate_schema` | Overrides `messages` with `add_messages` reducer. `SubgraphInputState` omits `messages` entirely. Applying the filter drops the parent's (trimmed) `messages` and the subgraph's first node sees an empty prompt (observed: debate topic fails to reach gemini_propose). |
| `tool_discovery_schema` | Adds 7 custom business fields (`user_query`, `search_intent`, ...). `SubgraphInputState` doesn't cover them, so applying the filter would drop them on entry. |
| `colony_coder_schema` | Adds many custom fields for task management. Same issue as tool_discovery. |

**Consequence**: For custom-schema subgraphs, pollution prevention relies entirely on the wrapper layer (`_fresh_wrapper` field clearing). Modes that don't wrap — specifically `persistent` — have no pollution protection for custom schemas.

### 3. `persistent` + custom schema can retain stale subgraph output fields

If a custom-schema subgraph is used with `persistent` mode, the parent's `debate_conclusion` etc. from a previous subgraph invocation will flow in unfiltered, and `_build_gemini_section` will inject them into the current subgraph's Claude nodes' prompts.

**Currently not an issue in practice**: all custom-schema subgraph references (debate, tool_discovery) use `fresh_per_call`, where the wrapper clears these fields. But the framework does not prevent a future entity.json from declaring `persistent` on a custom-schema subgraph, which would silently regress.

**Potential future fix**: Either:
- Make the schema-layer filter more general — dynamically compose an input schema per subgraph that preserves custom fields but blocks pollution fields.
- Or centralize pollution cleanup in all wrappers (not just `_fresh_wrapper`), so `persistent` subgraphs also get cleaned up at wrap time. Trade-off: `persistent` is supposed to preserve state across calls.
- Or declare explicitly in entity.json which fields to clear at subgraph entry (declarative per-subgraph).

### 4. `_fresh_wrapper` field clearing list is a duplicate of `SubgraphInputState`'s omitted-field list

The set of "pollution fields" is currently declared in two places:
1. `SubgraphInputState` (fields omitted from TypedDict)
2. `_fresh_wrapper` (explicit `patched["field"] = ""` assignments)

If a new subgraph output field is added to `BaseAgentState` (e.g., a new conclusion type), both places must be updated in sync, otherwise subgraphs will be inconsistently protected.

**Potential future fix**: Define a single `SUBGRAPH_POLLUTION_FIELDS` constant and derive both `SubgraphInputState`'s omissions and `_fresh_wrapper`'s clearing list from it.

## Configuration Locations

| What | Where |
|---|---|
| `session_mode` on a subgraph node | Parent graph's `entity.json`, on the node with `agent_dir` |
| `session_key` on an LLM node | The subgraph's own `entity.json`, on the LLM node |
| `inherit_from` for `inherit` mode | Parent graph's `entity.json`, alongside `session_mode: "inherit"` (currently errors) |
| Default `session_mode` | `persistent` (in `agent_loader.py`) |

## See Also

- `framework/agent_loader.py` — the external subgraph branch with all four wrappers
- `framework/schema/base.py` — `BaseAgentState` and `SubgraphInputState` definitions
- `framework/nodes/llm/llm_node.py` — `_build_gemini_section()` which injects subgraph outputs into Claude prompts
- `docs/vault/architecture/claude-cli-node-design.md` — related design for Claude node implementations
