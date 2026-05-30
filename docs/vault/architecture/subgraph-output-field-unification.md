Note: The core engine of the project resides in this ZenithLoom repository. However, all the blueprints have been moved to a separate repository called VoidDraft.

# Subgraph Output Field Unification

> Date: 2026-04-12
> Status: Backlog
> Depends on: unified-subgraph-integration.md (Complete this first)

## Problem

Five subgraph output fields are currently hardcoded in `BaseAgentState`:

```python
debate_conclusion: str   # debate subgraph
apex_conclusion: str     # apex_coder subgraph
knowledge_result: str    # knowledge_shelf subgraph
discovery_report: str    # tool_discovery subgraph
refined_plan: str        # colony_coder_planner
```

Every time a new subgraph is added, it requires:
1. Adding a field to `BaseAgentState`
2. Excluding it in `SubgraphInputState`
3. Adding an `if` branch in `LlmNode._build_gemini_section()` to read it
4. Adding a cleanup line in `_session_init` / `_fresh_init`

## Proposed Solution

Unify them into a single dictionary field:

```python
subgraph_outputs: Annotated[dict, _merge_dict]
# {"debate_conclusion": "...", "apex_conclusion": "...", ...}
```

### Benefits

- Zero changes to `BaseAgentState` for new subgraphs
- `_build_gemini_section()` iterates over the dict instead of using hardcoded logic
- `_session_init` only needs one line for cleanup: `"subgraph_outputs": {}`
- `SubgraphInputState` only needs to exclude one field

### Scope of Impact

- `framework/schema/base.py` — Field definition
- `framework/nodes/llm/llm_node.py` — `output_field` write logic + `_build_gemini_section()` read logic
- `framework/nodes/session_init_node.py` — Cleanup logic
- `VoidDraft/blueprints/functional_graphs/*/entity.json` — `output_field` declaration (semantics unchanged, only the write target changes)
- `VoidDraft/blueprints/functional_graphs/colony_coder_*/validators.py` — Places where `refined_plan` is read
- All schema subclasses — Remove their respective output field overrides

### Note

`refined_plan` is somewhat special — it is read directly by validators in `colony_coder_executor` and `colony_coder_qa` (`state.get("refined_plan", "")`), so these will need to be updated to read from `subgraph_outputs`.

## Relationship with `_build_gemini_section`

Currently, `_build_gemini_section()` injects different labels for each field (e.g., `[Debate Conclusion]`, `[ApexCoder Conclusion]`). After unification, this can be changed to:

```python
_LABELS = {
    "debate_conclusion": "Debate Conclusion",
    "apex_conclusion": "ApexCoder Conclusion",
    "knowledge_result": "Knowledge Base Search Results",
    "discovery_report": "Tool Discovery Report",
}
for key, value in state.get("subgraph_outputs", {}).items():
    if value:
        label = _LABELS.get(key, key)
        parts.append(f"\n[{label}]\n{value}\n")
```
