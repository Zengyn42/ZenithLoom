# ApexCoder Redesign — TDD Pipeline with Inherit Session Fork

> Date: 2026-04-15
> Status: Implemented (phase 1) + Designed (phase 2: inherit fork)

## Goal

Redesign ApexCoder from a single Claude SDK node to a TDD pipeline:
QA writes tests first → Coder implements and passes tests → Executor mechanically verifies → Retry on failure.
Two LLM nodes inherit parent graph conversation context via `fork_session` but are isolated from each other.

## Motivation

Snake Battle experiments exposed core flaws of single-node ApexCoder:
- Claude wrote headless tests to verify itself → 3 bugs bypassed completely
- `appendleft` body direction was reversed (bypassed by headless AI, but died on 1st frame in curses)
- Hardcoded `BOARD_H=24` required 28 lines of terminal (standard 24x80 couldn't fit)
- AI snake didn't eat food for 49 frames (pathfinding logic issue)

Fundamental reason: **Letting developers verify their own code = letting students create their own test questions and answer them.**

---

## Architecture (Phase 1 — Implemented)

```
setup → ClaudeQA(Sonnet) → reset_for_coder → ClaudeCoder(Opus)
                                                  ↓
                                              executor
                                                  ↓
                                               route
                                              ↙     ↘
                              inject_error_context   __end__
                                     ↓
                              ClaudeCoder (retry)
```

7 nodes: 2 CLAUDE_SDK + 5 DETERMINISTIC

| Node                 | Type          | Model           | Responsibility                                 |
|----------------------|---------------|-----------------|------------------------------------------------|
| setup                | DETERMINISTIC | -               | Extract user_requirements, parse/create working_directory |
| claude_qa            | CLAUDE_SDK    | claude-sonnet-4-6 | Write 5-10 QA tests based on requirements (independent session) |
| reset_for_coder      | DETERMINISTIC | -               | Clear QA messages, build Coder prompt          |
| claude_coder         | CLAUDE_SDK    | claude-opus-4-6 | Implement code + run QA tests internally       |
| executor             | DETERMINISTIC | -               | Run run_qa.sh subprocess, capture stdout/stderr/returncode |
| route                | DETERMINISTIC | -               | PASS → END, FAIL + retry<5 → retry, FAIL + retry≥5 → abort |
| inject_error_context | DETERMINISTIC | -               | Build retry prompt (error stack + iteration_history) |

### Anti-Cheat Mechanism

PreToolUse hook prevents ClaudeCoder from modifying files under `test_tool/qa_tests/`.
Configured in `entity.json`'s `settings_override.hooks.PreToolUse`.

### State Schema

`ApexCoderState` extends `BaseAgentState`:

| Field                | Source               | Purpose                     |
|----------------------|----------------------|-----------------------------|
| user_requirements    | setup                | User requirement text       |
| working_directory    | setup                | Working directory path      |
| qa_bypass            | claude_qa            | QA bypass flag              |
| qa_tests_dir         | claude_qa            | Test directory path         |
| run_qa_script        | claude_qa            | run_qa.sh path              |
| qa_summary           | claude_qa            | QA summary                  |
| apex_conclusion      | claude_coder         | Coder's final report        |
| execution_stdout     | executor             | Test standard output        |
| execution_stderr     | executor             | Test error output           |
| execution_returncode | executor             | Test return code            |
| iteration_history    | inject_error_context | History of failures, prevents repeated errors |
| status               | executor/route       | PENDING/PASS/FAIL           |

---

## Architecture (Phase 2 — To Be Implemented: Inherit Session Fork)

### Problem

In Phase 1, QA and Coder each create a brand new Claude session. When ApexCoder is called as a subgraph from `technical_architect`:
- QA and Coder cannot see the `technical_architect`'s conversation history with the user (requirement discussions, debate conclusions, design details).
- Only simplified context can be passed via state fields (`refined_plan`, `routing_context`).
- For complex tasks, losing the context chain can lead to QA writing unreasonable tests and Coder misunderstanding requirements.

### Solution: Claude SDK `fork_session`

Claude Agent SDK natively supports session fork:

```python
ClaudeAgentOptions(
    resume=parent_session_id,  # source forking
    fork_session=True,          # fork rather than direct resume
)
```

Effect: Creates a new independent session based on the parent session, inheriting the full conversation history, but subsequent conversations do not affect the original session.

### Session Lifecycle in Inherit Mode

```
technical_architect main graph:
  claude_main session = uuid-A (full conversation history)
  Route to apex_coder subgraph (session_mode: inherit)

ApexCoder subgraph:
  _subgraph_init: No cleanup (inherit mode)

  setup: Extract user_requirements, read state.refined_plan etc.

  claude_qa first call:
    node_sessions["apex_qa"] = "" (empty)
    → fork uuid-A → get uuid-fork-qa
    → QA sees full technical_architect conversation + its own test writing process
    → node_sessions["apex_qa"] = "uuid-fork-qa"

  reset_for_coder: Clear QA messages

  claude_coder first call:
    node_sessions["apex_coder"] = "" (empty)
    → fork uuid-A → get uuid-fork-coder
    → Coder sees full technical_architect conversation (without QA's reasoning)
    → node_sessions["apex_coder"] = "uuid-fork-coder"

  executor → route → FAIL → retry:

  claude_coder second call:
    node_sessions["apex_coder"] = "uuid-fork-coder" (not empty)
    → resume uuid-fork-coder (no fork, continue its own session)
    → Coder sees: technical_architect conversation + its own first written code + new error feedback
    → Will not repeat the same mistake

  _subgraph_exit:
    1. RemoveMessage clears all messages (prevents contamination of parent graph messages)
    2. Clears subgraph's session keys (apex_qa, apex_coder) from node_sessions
    → Parent graph node_sessions only retains claude_main = uuid-A
    → Next time ApexCoder is called, QA and Coder will fork again
```

### fork vs resume Decision Logic

In `ClaudeSDKNode.call_llm()`:

```python
async def call_llm(self, prompt, session_id="", inherit_from="", ...):
    if not session_id and inherit_from:
        # First call + inherit → fork parent session
        options = ClaudeAgentOptions(
            resume=inherit_from,
            fork_session=True,
            ...
        )
    elif session_id:
        # Has its own session → resume (retry scenario)
        options = ClaudeAgentOptions(
            resume=session_id,
            fork_session=False,
            ...
        )
    else:
        # New session (independent run scenario)
        options = ClaudeAgentOptions(...)
```

### Enhancement of `_subgraph_exit` in Inherit Mode

Currently `_subgraph_exit` only clears messages. Inherit mode requires additional cleanup of subgraph's session keys:

```python
def make_subgraph_exit(session_mode="persistent", subgraph_session_keys=None):
    """
    All subgraphs: RemoveMessage clears messages.
    Inherit mode: Additionally clears session keys created by the subgraph itself.
    """
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

`subgraph_session_keys` are extracted during build from `entity.json` node declarations (all LLM node's `session_key`).

### Full Comparison of Four `session_mode` (Updated)

| session_mode      | First LLM Call            | Retry LLM Call             | _subgraph_exit                      | Purpose                                       |
|-------------------|---------------------------|----------------------------|-------------------------------------|-----------------------------------------------|
| **persistent**    | New session               | Resume its own session     | Clear messages                      | Subgraph maintains session across calls       |
| **inherit**       | **Fork parent session**   | Resume its own fork        | Clear messages + Clear subgraph session keys | Subgraph inherits parent context, but doesn't affect parent session |
| **fresh_per_call**| New session               | N/A (fresh each time)      | Clear messages                      | New call each time                          |
| **isolated**      | New session (unique key)  | N/A                        | Clear messages                      | Completely isolated                         |

### Inherit Enhancement of `setup`

`setup` in inherit mode can read more parent graph context from state:

```python
def setup(state):
    # Prioritize refined_plan (refined design after technical_architect's debate)
    plan = state.get("refined_plan", "")
    debate = state.get("debate_conclusion", "")
    
    # routing_context or messages[0] as original requirement
    raw = state.get("routing_context", "") or state["messages"][0].content
    
    # Merge context
    if plan:
        user_requirements = f"{raw}

## Design Plan
{plan}"
    elif debate:
        user_requirements = f"{raw}

## Debate Conclusion
{debate}"
    else:
        user_requirements = raw
    
    # Optionally clear subgraph session keys (for next fork)
    ns = dict(state.get("node_sessions", {}))
    ns.pop("apex_qa", None)
    ns.pop("apex_coder", None)
    
    ...
    return {
        "user_requirements": user_requirements,
        "working_directory": working_directory,
        "node_sessions": ns,
        "messages": [HumanMessage(content=user_requirements)],
    }
```

### `inherit_from` Configuration

LLM node declaration `inherit_from` in `entity.json`:

```json
{
  "id": "claude_qa",
  "type": "CLAUDE_SDK",
  "session_key": "apex_qa",
  "inherit_from": "claude_main",
  ...
}
```

During framework build: if `session_mode == "inherit"` and node has `inherit_from`, it's passed to `ClaudeSDKNode`, used during `call_llm`.

---

## Phase 2 Implementation Plan

### List of Changes

| File                                | Change                                                           |
|-------------------------------------|------------------------------------------------------------------|
| `framework/nodes/llm/claude.py`     | `call_llm` adds `inherit_from` parameter, `fork_session` logic   |
| `framework/nodes/llm/llm_node.py`   | `__call__` passes `inherit_from` to `call_llm`                   |
| `framework/nodes/subgraph_init_node.py` | `make_subgraph_exit` accepts `session_mode` + `subgraph_session_keys` |
| `framework/agent_loader.py`         | Extract `subgraph_session_keys` in inherit mode, pass to exit node |
| `VoidDraft/blueprints/functional_graphs/apex_coder/entity.json` | Add `inherit_from` to QA and Coder nodes                 |
| `VoidDraft/blueprints/functional_graphs/apex_coder/validators.py` | `setup` adds inherit enhancement (read `refined_plan`, clear session keys) |

### No Changes Needed For

| File                          | Reason                            |
|-------------------------------|-----------------------------------|
| PROTOCOL.md                   | General engineering practices, unchanged |
| CODER_ROLE.md / QA_ROLE.md    | Role definitions unchanged        |
| hooks/protect_qa_tests.py     | Anti-cheat mechanism unchanged    |

---

## File Index

| File                                        | Responsibility                                             |
|---------------------------------------------|------------------------------------------------------------|
| `VoidDraft/blueprints/functional_graphs/apex_coder/entity.json` | 7-node graph definition                                    |
| `VoidDraft/blueprints/functional_graphs/apex_coder/state.py`    | ApexCoderState schema                                      |
| `VoidDraft/blueprints/functional_graphs/apex_coder/validators.py` | setup, reset_for_coder, executor, route, inject_error_context |
| `VoidDraft/blueprints/functional_graphs/apex_coder/CODER_ROLE.md` | Coder persona                                              |
| `VoidDraft/blueprints/functional_graphs/apex_coder/QA_ROLE.md`    | QA persona                                                 |
| `VoidDraft/blueprints/functional_graphs/apex_coder/PROTOCOL.md` | Shared engineering practices                               |
| `VoidDraft/blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py` | PreToolUse anti-cheat hook                                 |
| `tests/test_apex_coder.py`                  | 20 unit tests                                              |
| `run_apex_coder_debug.py`                   | Debug runner                                               |
| `run_benchmark_apex.py`                     | Benchmark runner                                           |
