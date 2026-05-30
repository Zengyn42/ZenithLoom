# Colony Coder — Architecture Record

> Last updated: 2026-03-22
> Status: Full pipeline verified (44/44 tests)

---

## Differences from Original Plan

The original plan (`docs/superpowers/plans/2026-03-17-colony-coder.md`) designed a 17-node pure Claude+Ollama system.
The actual implementation made the following major adjustments:

| Item | Original Plan | Actual Implementation | Reason |
|------|-------------|----------------------|--------|
| First 2 Planner nodes | `plan` + `design_debate` (CLAUDE_SDK x 2) | `design_debate` (SUBGRAPH_REF -> debate_claude_first) | Introducing Gemini for heterogeneous debate is more valuable than Claude debating itself |
| Debate rounds | 2 nodes each speaking once | 5 rounds (Claude->Gemini->Claude->Gemini->Claude) | debate_claude_first standard flow |
| Executor | 10 nodes (hard_validate/error_classifier/rescue_router etc.) | 4 nodes (inject_task_context/code_gen/run_tests/test_route) | Simplified architecture; code_gen's built-in tool loop implements write->test->fix cycle |
| Master phase 3 | `integrate` (AGENT_REF -> colony_coder_integrator) | `qa` (SUBGRAPH_NODE -> colony_coder_qa) | Added independent QA subgraph, E2E acceptance + rescue mechanism |
| Master node type | AGENT_REF | SUBGRAPH_NODE | Native LangGraph subgraph, state shared directly |
| Planner tools | Unlimited (inherits global tool set) | `"tools": []` explicitly disabled | Claude would ignore system prompt and secretly use tools to write code |
| decomposition_validator | Only checks state fields | Parses JSON from AI message → fills state fields | LLM output is in message content, not auto-populated to state fields |
| Token safety valve | None | Token check before each LLM node call | Added urgently after discovering 392-iteration infinite loop |
| retry_count management | Not specified | LLM nodes forbidden from resetting; only DETERMINISTIC nodes can write | Root cause of infinite loop: LlmNode zeroed retry_count |
| Schema name | colony_executor | colony_coder_schema | Unified schema name, all subgraphs share it |
| Config file name | agent.json | entity.json | Naming convention refactor: blueprint definition = entity.json |

---

## Current Topology

```
colony_coder (Master, colony_coder_schema)
├── plan (SUBGRAPH_NODE -> colony_coder_planner)
│   ├── design_debate (SUBGRAPH_REF -> debate_claude_first)
│   │   ├── claude_propose    [Claude Opus]     ─┐
│   │   ├── gemini_critique_1 [Gemini Pro]       │ 5 rounds of debate
│   │   ├── claude_revise     [Claude Opus]      │ session: claude_debate / gemini_debate
│   │   ├── gemini_critique_2 [Gemini Pro]       │ sessions cleaned up after debate
│   │   └── claude_conclusion [Claude Opus]     ─┘
│   ├── claude_swarm          [Claude SDK]  ─┐ session: planner_session
│   ├── task_decompose        [Claude SDK]  ─┘ shared session
│   └── decomposition_validator [DETERMINISTIC]   can retry -> task_decompose
│
├── execute (SUBGRAPH_NODE -> colony_coder_executor)
│   ├── inject_task_context   [DETERMINISTIC]  builds prompt with QA feedback
│   ├── code_gen              [Ollama Qwen3.5-27B]  session: executor_session
│   │                         tools: read_file, write_file, list_dir, bash_exec
│   │                         max_iterations: 25 (built-in write->test->fix cycle)
│   ├── run_tests             [DETERMINISTIC]  executes test_tool/run_tests.sh
│   └── test_route            [DETERMINISTIC]  pass->__end__ / fail->code_gen (max 5 retries)
│
└── qa (SUBGRAPH_NODE -> colony_coder_qa)
    ├── inject_e2e_context    [DETERMINISTIC]  builds E2E test prompt
    ├── generate_e2e          [Claude SDK]     session: qa_e2e_session
    │                         tools: Read, Write, Bash
    ├── run_e2e               [DETERMINISTIC]  executes test_tool/run_e2e.sh
    ├── e2e_route             [DETERMINISTIC]  pass->__end__ / fail->generate_e2e (max 5) / escalate
    ├── inject_rescue_context [DETERMINISTIC]  builds rescue full prompt
    ├── qa_rescue             [Claude SDK]     session: qa_rescue_session
    │                         tools: Write, Edit, Read, Bash
    ├── run_e2e_rescue        [DETERMINISTIC]  executes test_tool/run_e2e.sh (after rescue)
    └── rescue_route          [DETERMINISTIC]  pass->__end__ / fail->qa_rescue (max 5) / abort
```

Also has colony_coder_integrator subgraph (not connected to master, but usable standalone):

```
colony_coder_integrator
├── integration_test    [DETERMINISTIC]  executes test_tool/run_tests.sh
├── integration_rescue  [Claude SDK]     session: session_b
└── integration_route   [DETERMINISTIC]  pass->__end__ / fail->integration_rescue (max 2)
```

---

## Data Flow

### Master Graph Flow

```
__start__ -> plan -> execute -> qa -> execute (if QA fails) or __end__ (if QA passes)
```

The QA subgraph's e2e_route controls via `routing_target`:
- `"__end__"` + `success=True` — QA passed, master routes to `__end__`
- `"execute"` — QA failed, master routes back to `execute` (with qa_analysis feedback)
- `"inject_rescue_context"` — after 5 QA failures, escalate to rescue

### Executor Internal Loop

```
inject_task_context -> code_gen -> run_tests -> test_route
                          ^                        |
                          └── fail (retry < 5) ────┘
```

code_gen (OllamaNode) has a built-in 25-round tool-calling loop:
1. Read plan + task description
2. Use write_file to write source code
3. Use write_file to write tests
4. Use bash_exec to run tests
5. Test fails → read errors → fix code → rerun
6. Tests pass → return text response

run_tests is purely mechanical execution (no LLM), test_route does pass/fail routing.

---

## Session Architecture

### Design Principles

1. **Debate sessions are temporary** — SubgraphRefNode._cleanup_orphan_sessions() deletes disk files after subgraph ends
2. **planner_session is newly created** — claude_swarm gets debate conclusion text through messages, does not resume the debate session
3. **Ollama has no server-side session** — multi-turn history stored in state["ollama_sessions"] dict, persisted by LangGraph checkpoint
4. **Same session_key across subgraphs is isolated** — sessions in different subgraphs are different instances

### Data Flow: Debate Conclusion -> planner_session

```
debate_claude_first subgraph ends
    |
SubgraphRefNode:
    state_out: {"refined_plan": "last_message"}
    -> out["refined_plan"] = claude_conclusion text
    -> out["messages"] = [AIMessage("[Subgraph Conclusion]\n\n{full text}")]
    -> cleans up debate subgraph's claude_debate / gemini_debate session files
    |
claude_swarm (LlmNode.__call__):
    latest_input = msgs[-1].content = "[Subgraph Conclusion]\n\n..."
    session_id = ns.get("planner_session") = None -> create new session
    -> Claude starts new session, first message = debate conclusion
    -> ns["planner_session"] = "abc123..."
    |
task_decompose (LlmNode.__call__):
    latest_input = msgs[-1].content = claude_swarm's review output
    session_id = ns.get("planner_session") = "abc123..." -> resume
    -> Claude can see: turn1(debate conclusion->review) + turn2(review->JSON decomposition)
```

### Session Mapping Table

| Subgraph | Node | session_key | Model | Lifecycle |
|----------|------|-------------|-------|-----------|
| Planner/debate | claude_propose, claude_revise, claude_conclusion | claude_debate | Claude Opus | Temporary, cleaned after debate |
| Planner/debate | gemini_critique_1, gemini_critique_2 | gemini_debate | Gemini Pro | Temporary, cleaned after debate |
| Planner | claude_swarm, task_decompose | planner_session | Claude SDK | Follows planner subgraph |
| Executor | code_gen | executor_session | Ollama | State dict |
| QA | generate_e2e | qa_e2e_session | Claude SDK | Follows QA subgraph |
| QA | qa_rescue | qa_rescue_session | Claude SDK | Follows QA subgraph |
| Integrator | integration_rescue | session_b | Claude SDK | Independent |

---

## Token Safety Valve

### Trigger Conditions

Each LLM node checks cumulative session token count before call_llm().

### Limit Configuration (3 priority levels)

```
node_config["token_limit"]  >  default by type  >  BB_TOKEN_LIMIT environment variable
```

| Model type | Default limit | Notes |
|-----------|--------------|-------|
| CLAUDE_SDK | 50,000 | Cloud API, billed per token |
| GEMINI_API / GEMINI_CLI | 50,000 | Cloud API |
| OLLAMA / LOCAL_VLLM | 1,000,000 | Local inference, not billed |

### Implementation Location

- `framework/token_guard.py` — check_before_llm(), TokenLimitExceeded
- `framework/nodes/llm/llm_node.py` — __call__ calls check_before_llm
- `framework/nodes/llm/ollama.py` — additional check every iteration in tool loop

---

## Key Bug Fix Records

### 1. Infinite Loop (392 iterations) — 2026-03-17

**Symptom**: colony_coder immediately exhausts all tokens; task_decompose <-> decomposition_validator loops infinitely

**Root cause**: `LlmNode.__call__` line 274 returns `"retry_count": 0`, every LLM node execution zeroed the validator's retry_count

**Fix**: Remove retry_count from LlmNode return value. **retry_count is managed only by DETERMINISTIC nodes.**

### 2. tools=[] Ignored — 2026-03-18

**Symptom**: planner node's system prompt says "do not use tools", but Claude uses them anyway, directly writing snake game code

**Root cause**: `_select_tools` uses `self._cfg.get("tools") or self.config.tools`, empty list `[]` is falsy, falls back to global tool set

**Fix**: Distinguish "not configured" (key absent) from "explicitly disabled" (tools=[]). Use sentinel `_MISSING` for judgment. Also fix `_make_options` in claude.py.

### 3. Validator Doesn't Parse JSON — 2026-03-18

**Symptom**: task_decompose outputs valid JSON but decomposition_validator reports validation failure

**Root cause**: validator checks `state["tasks"]`, but JSON is in AI message content; nobody parses it into state fields

**Fix**: validator adds `_extract_json()` — extracts JSON from last AI message, fills tasks/execution_order/refined_plan/working_directory into state.

### 4. Architecture Refactor — 2026-03-22

**Changes**: Executor simplified from 10 nodes to 4 nodes; Master phase 3 changed from integrate to qa; Master node type changed from AGENT_REF to SUBGRAPH_NODE

**Reason**: code_gen's 25-round tool loop already contains write/test/fix cycle internally, external soft_validate/self_fix/error_classifier nodes not needed. QA subgraph is independent of integrator, responsibilities clearer.

---

## File Index

| File | Responsibility |
|------|--------------|
| `blueprints/functional_graphs/colony_coder/entity.json` | Master graph: plan -> execute -> qa |
| `blueprints/functional_graphs/colony_coder/state.py` | ColonyCoderState TypedDict (colony_coder_schema) |
| `blueprints/functional_graphs/colony_coder_planner/entity.json` | Planner: debate -> swarm -> decompose -> validate |
| `blueprints/functional_graphs/colony_coder_planner/validators.py` | decomposition_validator + JSON parsing |
| `blueprints/functional_graphs/colony_coder_executor/entity.json` | Executor: 4 nodes (inject -> code_gen -> run_tests -> test_route) |
| `blueprints/functional_graphs/colony_coder_executor/validators.py` | inject_task_context, run_tests, test_route |
| `blueprints/functional_graphs/colony_coder_qa/entity.json` | QA: 8-node E2E acceptance + rescue |
| `blueprints/functional_graphs/colony_coder_qa/validators.py` | inject_e2e_context, run_e2e, e2e_route, inject_rescue_context, run_e2e_rescue, rescue_route |
| `blueprints/functional_graphs/colony_coder_integrator/entity.json` | Integrator: 3-node integration test + fix loop |
| `blueprints/functional_graphs/colony_coder_integrator/validators.py` | integration_test, integration_route |
| `blueprints/functional_graphs/debate_claude_first/entity.json` | Heterogeneous debate subgraph (Claude x 3 + Gemini x 2) |
| `framework/token_guard.py` | Token safety valve |
| `framework/nodes/llm/llm_node.py` | LLM node base class, includes tools=[] fix + token guard |
| `framework/nodes/llm/ollama.py` | Ollama node, includes _chat_completions + _call_with_tools |
| `framework/nodes/subgraph/subgraph_ref_node.py` | Subgraph reference node, includes routing_context fallback |
