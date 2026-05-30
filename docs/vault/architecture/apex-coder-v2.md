# ApexCoder v2 — TDD Pipeline Architecture Record

> Last updated: 2026-04-17
> Status: Architecture finalized, implementation in progress

---

## Design Goals

Refactor ApexCoder from a single Claude SDK node into a 4-node TDD pipeline, implementing a strict development flow where "QA writes tests first → Coder writes code to pass tests."

## Differences from v1

| Item | v1 (Colony Coder) | v2 (Apex Coder TDD) |
|------|-------------------|---------------------|
| Node count | 17 nodes (Master + Planner + Executor + QA) | 4 nodes (setup → ClaudeQA → reset → ClaudeCoder) |
| Test strategy | Post-hoc validation | TDD: QA writes tests first, Coder must pass them |
| Session mode | Independent session | `inherit_from: parent`, shared context |
| Debate integration | Internal Planner debate | External debate subgraph conclusion injection |
| Anti-tampering | None | PreToolUse hook prevents Coder from modifying QA test files |

## Current Topology

```
setup (DETERMINISTIC)
  → ClaudeQA (CLAUDE_SDK) — writes tests, defines acceptance criteria
    → reset_for_coder (DETERMINISTIC) — isolates messages, injects test summary
      → ClaudeCoder (CLAUDE_SDK) — implements code, must pass QA tests
```

## Key Design Decisions

### 1. inherit_from Mode
ApexCoder inherits session context from the parent graph (technical_architect) via `inherit_from`, without manually passing debate conclusions. The technical_architect's conversation history includes debate results, which ApexCoder automatically sees.

### 2. status Default Value
The `status` field's default value changed from none → `"PENDING"`. Fixes a routing anomaly bug caused by uninitialized fields during state merge.

### 3. Collaboration Flow with Debate Subgraphs
```
User requirement → technical_architect evaluates complexity
  → complex: debate_brainstorm / debate_design → debate conclusion injected into technical_architect context
  → technical_architect prepares implementation instructions → route to apex_coder
  → ApexCoder inherits context (including debate conclusion) → QA writes tests → Coder implements
  → technical_architect validates results (runs benchmark, checks output)
```

## State Schema

```python
class ApexCoderState(BaseAgentState):
    user_requirements: str
    working_directory: str
    qa_bypass: bool
    qa_tests_dir: str
    run_qa_script: str
    qa_summary: str
    apex_conclusion: str
    iteration_history: list
    retry_count: int
    status: str = "PENDING"  # "PENDING", "PASS", "FAIL"
    node_sessions: Annotated[dict, _merge_dict]
```

## Reference Files

- Design plan: `docs/superpowers/plans/2026-04-15-apex-coder-redesign.md`
- State: `blueprints/functional_graphs/apex_coder/state.py`
- Entity: `blueprints/functional_graphs/apex_coder/entity.json`
- Colony Coder v1 record: `docs/vault/architecture/colony-coder.md`
