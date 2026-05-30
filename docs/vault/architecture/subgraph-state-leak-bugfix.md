Note: The core engine of the project resides in this ZenithLoom repository. However, all the blueprints have been moved to a separate repository called VoidDraft.

# Subgraph State Leak Bugfix (Parent Graph Prompt Bloat)

> Date: 2026-04-20
> Status: Fixed
> Related: unified-subgraph-integration.md, subgraph-output-field-unification.md, session-mode-design.md

## Symptoms

In the Discord session of `technical_architect`, the `claude_main` node sees a prompt injected with several KB of historical context in every round, causing tokens to skyrocket. A user input like "restore gemma4" (8 characters) is expanded into a massive prompt.

Analysis of the latest checkpoint for session `hani_session_2d05f9a6`:

```
debate_conclusion    : 3262 chars   (Conclusion from the last debate subgraph)
subgraph_topic       : 1222 chars   (Topic anchor from the last debate)
previous_node_output : 107  chars
apex_conclusion      : 0
knowledge_result     : 0
discovery_report     : 0
```

These three non-empty fields are unconditionally spliced into the prefix of every round of the `claude_main` prompt by `LlmNode._build_gemini_section()`, `_topic_inject`, and `_prev_inject`.

## Root Causes (Three Independent Leak Paths)

### A. `subgraph_topic` Leak

- `framework/nodes/subgraph_init_node.py::_fresh_init` writes `subgraph_topic = routing_context or subgraph_topic` (only in `fresh_per_call` mode).
- The original implementation of `_subgraph_exit` only issued `RemoveMessage` to clear `messages` and **did not clear `subgraph_topic`**.
- Result: When a `fresh_per_call` subgraph returns, the parent graph state inherits this topic anchor.
- In the next round of the parent graph, any `LlmNode` reads `state["subgraph_topic"]` → `_topic_inject` splices the 1KB+ debate topic into the prompt.

### B. `previous_node_output` Leak

- After any `LlmNode` executes, it writes `result["previous_node_output"] = raw_output` ([llm_node.py:521](../../../framework/nodes/llm/llm_node.py)). The design intent is for chained reasoning between nodes **within** a subgraph (Gemini → Claude → Gemini ...).
- The final node of a subgraph also writes this, and `_subgraph_exit` did not clear it → it leaks to the parent graph.
- In the next round of the parent graph, `claude_main` reads `state["previous_node_output"]`; the gate for `_prev_inject` is `if _subgraph_topic`, so as long as Path A has already leaked, Path B is also injected.

### C. Persistent Injection of Subgraph Conclusion Fields

- These four fields (`debate_conclusion`, `apex_conclusion`, `knowledge_result`, `discovery_report`) are explicitly designed to be written by the final node of a subgraph via `node_config.output_field`, intended to be **consumed once by the next round of the parent graph's `claude_main`**.
- `LlmNode._build_gemini_section()` reads them and splices them into the prompt, but **did not clear them in the output**.
- Due to LangGraph checkpoint persistence, every subsequent round of the parent graph's `claude_main` repeatedly injects the same historical conclusion.

## Scope of Impact (Framework-level, not just Debate subgraph)

| Subgraph | session_mode | Path A Leak | Path B Leak | Path C Leak |
|---|---|---|---|---|
| debate_gemini_first (debate_brainstorm) | fresh_per_call | ✓ | ✓ | ✓ (`debate_conclusion`) |
| debate_claude_first (debate_design) | fresh_per_call | ✓ | ✓ | ✓ (`debate_conclusion`) |
| apex_coder | inherit | — | ✓ | ✓ (`apex_conclusion`) |
| tool_discovery | fresh_per_call | ✓ | ✓ | ✓ (`discovery_report`) |
| tool_evaluate | fresh_per_call | ✓ | ✓ | ✓ (`discovery_report`) |
| video_quality_loop | fresh_per_call | ✓ | ✓ | — |
| colony_coder (as Master) | fresh_per_call | ✓ | ✓ | — |

**Primary Victims**:

- **`technical_architect` Main Graph: Severely affected**. The call chains for `debate_brainstorm`, `debate_design`, `apex_coder`, `tool_discovery`, `tool_evaluate`, and `video_quality_loop` all trigger leaks.
- **`ColonyCoder` (as a standalone main graph): Indirectly and mildly affected**. The three `fresh_per_call` subgraphs (plan/execute/qa) trigger Path A+B leaks, but `ColonyCoder`'s internal LLM nodes use their own state fields (`refined_plan`, `qa_analysis`, `e2e_plan`, etc.) and do not read `debate_conclusion`/`apex_conclusion`, so Path C injection doesn't directly increase tokens.
- The path `colony_coder_planner → debate_claude_first` inside `ColonyCoder` will cause the planner's state to include `debate_conclusion`; if the planner runs an `LlmNode` again, it will be injected by Path C once.
- The "session reset + snapshot" context protection in `ColonyCoder` (2026-04-17) partially masked the symptoms of Path A/B leaks.

## Fixes

Three changes (applied on 2026-04-20):

### 1. `framework/agent_loader.py` — Fixed `Annotated is not defined`

The `state.py` for `apex_coder` and `colony_coder` uses `from __future__ import annotations`, so `Annotated[dict, _merge_dict]` is stored as a string. `get_type_hints()` uses `sys.modules[cls.__module__]` to find globals to evaluate these strings, but the old code using `importlib.util.module_from_spec` + `exec_module` **failed to register the module in `sys.modules`**, resulting in empty globals and a `NameError: name 'Annotated' is not defined`.

```python
_sys.modules[_mod_name] = _mod   # ← Added: Must register before exec_module
_spec.loader.exec_module(_mod)
```

### 2. `framework/nodes/subgraph_init_node.py::_subgraph_exit`

```python
result = {
    "messages": removals,
    "subgraph_topic": "",           # ← Added, clears Path A
    "previous_node_output": "",     # ← Added, clears Path B
}
```

Cleared upon exiting all `session_mode` subgraphs, fulfilling the promise in `BaseAgentState` comments: "written at `_subgraph_init` entry, cleared at exit".

### 3. `framework/nodes/llm/llm_node.py` — Clear Path C after consumption

```python
result["debate_conclusion"] = ""
result["apex_conclusion"]   = ""
result["knowledge_result"]  = ""
result["discovery_report"]  = ""

if self._output_field and raw_output:
    result[self._output_field] = raw_output   # Overwrite: Subgraph final producer nodes are unaffected
```

Since the `output_field` write occurs after the cleanup, nodes at the end of a subgraph that produce these fields can still pass conclusions back to the parent graph. The parent graph's `claude_main` will consume them and then automatically clear them in the next round, preventing accumulation.

## Residuals (Historical Checkpoint Data)

Code fixes only apply to new writes. Fields already leaked in existing session checkpoints will persist until:

- `debate_conclusion` / `apex_conclusion` / etc.: Automatically cleared after the next round of the parent graph's `LlmNode` (results in **one** additional large prompt).
- `subgraph_topic` / `previous_node_output`: Cleared during the next subgraph call (will remain if the user never triggers a subgraph again).

To clear them immediately, execute `!clear` or `!new` in the affected channel.

## Verification

- [framework/nodes/llm/llm_node.py](../../../framework/nodes/llm/llm_node.py)
- [framework/nodes/subgraph_init_node.py](../../../framework/nodes/subgraph_init_node.py)
- [framework/agent_loader.py](../../../framework/agent_loader.py)

Manual Smoke Test:
```python
# Reproduction of A→NameError passed
# Reproduction of B→get_type_hints(ApexCoderState, include_extras=True) returns correctly
```

`systemctl --user restart technical_architect` → `[Discord] controller initialized (graph compiled)`, no `Annotated` error.

## Long-term Improvement (Related to `subgraph-output-field-unification.md`)

One root cause of Path C leaks is the hardcoding of 4 output fields in `BaseAgentState`, requiring synchronized updates in `_build_gemini_section`, `_fresh_init`, etc., every time a new subgraph is added. Implementing the unified `subgraph_outputs: Annotated[dict, _merge_dict]` solution would centralize the consumption-cleanup logic and prevent similar leaks in the future.
