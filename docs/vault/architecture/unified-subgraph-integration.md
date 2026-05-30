Note: The core engine of the project resides in this ZenithLoom repository. However, all the blueprints have been moved to a separate repository called VoidDraft.

# Unified Subgraph Integration — Native Subgraph + Symmetric Init/Exit Nodes

> Date: 2026-04-12
> Status: Implemented
> Supersedes: The wrapper layer design in `session-mode-design.md` (the wrapper layer will be replaced by this solution)

## Problem

Currently, the three non-default `session_mode` settings use different access methods:

| session_mode | Access Method | LangGraph Perspective |
|---|---|---|
| persistent | `builder.add_node(id, CompiledStateGraph)` | Native Subgraph ✅ |
| fresh_per_call | `builder.add_node(id, async_wrapper)` | Plain Function ❌ |
| isolated | `builder.add_node(id, async_wrapper)` | Plain Function ❌ |

`fresh_per_call` and `isolated` use async wrappers around subgraphs to implement state cleanup. This causes LangGraph to lose subgraph visibility:

1. `astream(subgraphs=True)` cannot track internal nodes.
2. `get_graph(xray=True)` cannot see the subgraph structure.
3. Debugging/tracing tools fail.

Additionally, `BaseAgentState` uses a custom `_keep_last_2` reducer to truncate messages instead of LangGraph's native `add_messages`. This is because messages from within subgraphs flow back to the parent graph and cause pollution.

## Core Idea

1. **All `session_mode` settings now use native LangGraph subgraph integration (`builder.add_node(id, CompiledStateGraph)`).**
2. **State cleanup logic no longer uses external wrappers; instead, it is injected as DETERMINISTIC nodes symmetrically before the entry and after the exit of the subgraph.**
3. **`_subgraph_exit` uniformly cleans up all subgraph exit messages, allowing the parent graph to safely use the native LangGraph `add_messages` reducer.**

## Architecture

### Symmetric Init/Exit Nodes

Boundary nodes are automatically injected into all subgraphs (regardless of `session_mode`) during construction by `_build_declarative()`:

```
START → _subgraph_init → entry → ... → exit → _subgraph_exit → END
         (Cleanup input state)                   (Cleanup output messages)
```

- `_subgraph_init`: Performs different input cleanups based on `session_mode`.
- `_subgraph_exit`: Uniform for all modes; uses `RemoveMessage` to clear all internal subgraph messages.

### Four session_mode Rules

| session_mode | Initial LLM Call | Retry LLM Call | `_subgraph_exit` Behavior | Use Case |
|---|---|---|---|---|
| **persistent** | Create new session | Resume own session | RemoveMessage all msgs | Subgraph maintains session across calls |
| **inherit** | **Fork parent session** (`fork_session=True`) | Resume own fork | RemoveMessage all msgs + clear subgraph session keys | Subgraph inherits parent context without interference |
| **fresh_per_call** | Create new session | N/A (Each call is fresh) | RemoveMessage all msgs | Fresh on each call (debate, etc.) |
| **isolated** | Create new session (unique key) | N/A | RemoveMessage all msgs | Complete isolation |

### persistent vs. inherit Differences

Neither mode cleans up `node_sessions` during initialization, but their **session creation method** and **exit cleanup** are entirely different:

- **persistent**: The subgraph LLM node creates a completely new session (unrelated to the parent). The checkpoint saves the subgraph's session UUID. Resumed on the next call → the subgraph "remembers" the previous conversation. `_subgraph_exit` does not clear session keys.

- **inherit**: The subgraph LLM node **forks the parent graph's session** (via Claude SDK `fork_session=True`). The subgraph sees the parent's full conversation history, but subsequent dialogue does not affect the parent session. `_subgraph_exit` **clears the subgraph's session keys** → the next call forks again.

#### inherit Fork Semantics

```
Parent graph session: uuid-A (Full conversation between technical_architect and user)

Initial call to subgraph LLM node:
  node_sessions["apex_qa"] = "" (empty)
  → sdk_query(resume=uuid-A, fork_session=True)
  → Obtain uuid-fork-qa (independent session, starting point = full history of uuid-A)
  → node_sessions["apex_qa"] = "uuid-fork-qa"

Retry of subgraph LLM node:
  node_sessions["apex_qa"] = "uuid-fork-qa" (non-empty)
  → sdk_query(resume=uuid-fork-qa, fork_session=False)
  → Continue own fork session (remembers what was done last time, won't repeat mistakes)

Subgraph exit (_subgraph_exit):
  → Clear node_sessions["apex_qa"], ["apex_coder"]
  → Parent graph only retains {"claude_main": "uuid-A"}
  → Next call to subgraph forks again
```

#### Independence of inherit

Multiple LLM nodes within the same subgraph (e.g., QA and Coder) each fork the parent session:

- QA fork → sees parent conversation + QA's own conversation.
- Coder fork → sees parent conversation + Coder's own conversation (**cannot see QA's reasoning**).
- A `reset_for_coder` (DETERMINISTIC node) clears QA's messages to ensure the Coder's prompt doesn't contain QA's thinking.

#### inherit_from Configuration

A subgraph LLM node declares which parent session to inherit from in `entity.json`:

```json
{
  "id": "claude_qa",
  "type": "CLAUDE_SDK",
  "session_key": "apex_qa",
  "inherit_from": "claude_main"
}
```

In `inherit` mode, the framework looks up `node_sessions[inherit_from]` to get the parent session UUID and passes it as the `inherit_from` parameter to `call_llm`.

#### ClaudeSDKNode Implementation

```python
async def call_llm(self, prompt, session_id="", inherit_from="", ...):
    if not session_id and inherit_from:
        # Initial call + inherit → fork parent session
        options = ClaudeAgentOptions(resume=inherit_from, fork_session=True, ...)
    elif session_id:
        # Has own session → resume (retry scenario)
        options = ClaudeAgentOptions(resume=session_id, ...)
    else:
        # Completely new session (standalone execution)
        options = ClaudeAgentOptions(...)
```

### `_subgraph_exit` Differentiation by Mode

| mode | _subgraph_exit Behavior |
|---|---|
| persistent | RemoveMessage all msgs |
| **inherit** | RemoveMessage all msgs + **clear subgraph session keys** |
| fresh_per_call | RemoveMessage all msgs |
| isolated | RemoveMessage all msgs |

In `inherit` mode, `_subgraph_exit` needs to know which `session_key`s belong to the subgraph (extracted from `entity.json` during construction):

```python
def make_subgraph_exit(session_mode="persistent", subgraph_session_keys=None):
    def _exit_cleanup(state):
        msgs = state.get("messages", [])
        result = {"messages": [RemoveMessage(id=m.id) for m in msgs]}
        
        if session_mode == "inherit" and subgraph_session_keys:
            ns = dict(state.get("node_sessions", {}))
            for key in subgraph_session_keys:
                ns.pop(key, None)
            result["node_sessions"] = ns
        
        return result
    return _exit_cleanup
```

### Why `_subgraph_exit` Uniformly Cleans Up Messages

Internal messages within a subgraph (debate rounds, coder step outputs, etc.) are noise to the parent graph. The parent only needs to obtain the subgraph's conclusion through state fields (e.g., `debate_conclusion`).

Uniform exit cleanup allows:
1. The parent graph to safely use the native LangGraph `add_messages` reducer (replacing the custom `_keep_last_2`).
2. The parent's messages to contain only outputs from its own nodes, unpolluted by subgraph internals.
3. `persistent` subgraph session persistence to remain unaffected (`node_sessions` is managed by the checkpoint and is independent of messages).

### Impact of `_subgraph_exit` on `persistent` Checkpoints

```
First call:
  Subgraph execution → node_sessions={"claude_propose": "uuid-1"}, messages=[m1, m2, m3]
  _subgraph_exit → messages=[] (cleared by RemoveMessage)
  Checkpoint saved: node_sessions={"claude_propose": "uuid-1"}, messages=[]

Second call:
  Checkpoint restored: node_sessions={"claude_propose": "uuid-1"}, messages=[]
  Subgraph execution → LLM restores session via uuid-1 (dialogue history managed internally by SDK)
```

Loss of messages does not affect LLM dialogue continuity — history is managed by the LLM SDK session (Claude SDK → `~/.claude/`, Gemini CLI → `~/.gemini/`); LangGraph messages are just the recent context window.

### `_subgraph_init` Implementation

```python
# framework/nodes/subgraph_init_node.py

def make_subgraph_init(session_mode: str):
    """Returns an input cleanup function based on session_mode. Returns None (no injection) for persistent/inherit."""

    if session_mode == "fresh_per_call":
        def _fresh_init(state: dict) -> dict:
            msgs = state.get("messages", [])
            removals = [RemoveMessage(id=m.id) for m in msgs]
            human_msgs = [m for m in reversed(msgs) if getattr(m, "type", "") == "human"]
            fresh = [HumanMessage(content=human_msgs[0].content)] if human_msgs else (
                [type(msgs[-1])(content=msgs[-1].content)] if msgs else []
            )
            _topic = state.get("routing_context", "") or state.get("subgraph_topic", "")
            return {
                "node_sessions": {},
                "messages": removals + fresh,
                "routing_context": "",
                "debate_conclusion": "",
                "apex_conclusion": "",
                "knowledge_result": "",
                "discovery_report": "",
                "previous_node_output": "",
                "subgraph_topic": _topic,
            }
        return _fresh_init

    elif session_mode == "isolated":
        def _isolated_init(state: dict) -> dict:
            return {"node_sessions": {}}
        return _isolated_init

    else:  # persistent, inherit
        return None
```

### `_subgraph_exit` Implementation

```python
def make_subgraph_exit():
    """Unified exit cleanup for all subgraphs: RemoveMessage clears all internal messages."""
    def _exit_cleanup(state: dict) -> dict:
        msgs = state.get("messages", [])
        return {"messages": [RemoveMessage(id=m.id) for m in msgs]}
    return _exit_cleanup
```

### `_build_declarative` Injection Logic

```python
_needs_init = is_subgraph and session_mode in ("fresh_per_call", "isolated")
_needs_exit = is_subgraph  # Injected for all subgraphs

# Entry side
if _needs_init and _graph_entry and not _has_start_edge:
    _init_fn = make_subgraph_init(session_mode)
    builder.add_node("_subgraph_init", _init_fn)
    builder.add_edge(START, "_subgraph_init")
    builder.add_edge("_subgraph_init", _graph_entry)
elif _graph_entry and not _has_start_edge:
    builder.add_edge(START, _graph_entry)

# Exit side
if _needs_exit and _graph_exit and not _has_end_edge:
    _exit_fn = make_subgraph_exit()
    builder.add_node("_subgraph_exit", _exit_fn)
    builder.add_edge(_graph_exit, "_subgraph_exit")
    builder.add_edge("_subgraph_exit", END)
elif _graph_exit and not _has_end_edge:
    builder.add_edge(_graph_exit, END)
```

### Parent Graph Integration (Unified)

```python
# agent_loader.py, external subgraph branch — no longer differentiates access methods by session_mode
inner_graph = await inner_loader.build_graph(
    checkpointer=None,
    is_subgraph=True,
    session_mode=session_mode,
    force_unique_session_keys=(session_mode == "isolated"),
)
builder.add_node(node_id, inner_graph)  # Always native subgraph
```

## Side Changes

### Remove `_keep_last_2` → Use `add_messages`

`BaseAgentState.messages` changed from `Annotated[list[BaseMessage], _keep_last_2]` to `Annotated[list[BaseMessage], add_messages]`.

`_keep_last_2` existed to prevent subgraph message pollution. Now handled at the source by `_subgraph_exit`, it is no longer needed.

### Remove `SubgraphMapperNode` Dead Code

`SubgraphMapperNode` (`framework/nodes/subgraph/subgraph_mapper.py`) is not referenced by any blueprint and is dead code. Its `subgraph_topic` management responsibility is now handled by the `fresh_per_call` branch of `_subgraph_init`.

Deleted:
- `framework/nodes/subgraph/subgraph_mapper.py`
- `SUBGRAPH_MAPPER` registration in `framework/builtins.py`

### Remove async wrappers

Deleted from `agent_loader.py`:
- `_fresh_wrapper` async function
- `_isolated_wrapper` async function
- `_get_subgraph_session_keys()` helper
- Calls to `push_graph_scope` / `pop_graph_scope` within wrappers

### Implement `inherit` session_mode

`inherit` is no longer a `NotImplementedError`. Semantics: No init cleanup (inherits parent state), has exit cleanup (RemoveMessage). Differs from `persistent` in checkpoint behavior.

## Benefits

1. **All subgraphs visible to LangGraph** — `astream(subgraphs=True)` works for all modes.
2. **Observable cleanup logic** — `_subgraph_init` / `_subgraph_exit` are normal nodes; visible in debug flow logs.
3. **Testable cleanup logic** — Factory functions can be tested independently.
4. **Removal of async wrappers** — Eliminated `_fresh_wrapper` and `_isolated_wrapper`.
5. **Native LangGraph `add_messages`** — Removed custom `_keep_last_2`.
6. **Symmetric design** — `init` handles input, `exit` handles output; clear responsibilities.
7. **`inherit` functionality** — No longer `NotImplementedError`.

## Impact on Existing Code

### Deletions

- `agent_loader.py`: `_fresh_wrapper`, `_isolated_wrapper`, `_get_subgraph_session_keys()`
- `framework/schema/base.py`: `_keep_last_2` function
- `framework/nodes/subgraph/subgraph_mapper.py`: Entire file
- `framework/builtins.py`: `SUBGRAPH_MAPPER` registration

### Modifications

- `agent_loader.py: _build_declarative()` — Added `session_mode` parameter, injected init/exit nodes.
- `agent_loader.py: build_graph()` — Added `session_mode` parameter.
- External subgraph branch in `agent_loader.py` — Unified use of `builder.add_node(id, inner_graph)`.
- `framework/schema/base.py` — `_keep_last_2` → `add_messages`.
- `framework/builtins.py` — Removed `SUBGRAPH_MAPPER`.
- Updated comments: References to `SubgraphMapperNode` in `llm_node.py`, `gemini.py`, and `base.py`.

### Additions

- `framework/nodes/subgraph_init_node.py` — `make_subgraph_init()` + `make_subgraph_exit()` factory functions.

### Unchanged

- All `entity.json` — Declarative configurations remain the same.
- `SubgraphInputState` — Temporarily retained; simplify later if possible.

## Risks

1. **persistent checkpoint + exit cleanup** — After `_subgraph_exit` clears messages, the messages saved in the checkpoint will be empty. Dialogue history is managed by the LLM SDK session and doesn't depend on LangGraph messages. However, verify that persistent subgraphs (`colony_coder`) work correctly when restored with `messages=[]`.

2. **`add_messages` reducer + RemoveMessage compatibility** — Verified: in LangGraph 1.0.10, `RemoveMessage` only takes effect within its own reducer scope and does not propagate to the parent graph. Subgraph internal `RemoveMessage` does not affect parent graph messages. Note: `RemoveMessage` throws an exception for non-existent IDs; `_subgraph_exit` only deletes actual messages, which is safe.

3. **`_subgraph_init` state schema compatibility** — The dict returned by the cleanup function must be compatible with the subgraph's `state_schema`. `debate_schema`'s `add_messages` reducer accepts `RemoveMessage` + new message list (verified). `base_schema` behavior is consistent after changing to `add_messages`.
