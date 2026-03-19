# Colony Coder — 架构实录

> 最后更新: 2026-03-18
> 状态: Planner 已验证通过，Executor/Integrator 待 Ollama 联调

---

## 与原计划的差异

原计划 (`docs/superpowers/plans/2026-03-17-colony-coder.md`) 设计了 17 节点的纯 Claude+Ollama 系统。
实际实现中做了以下重大调整：

| 项目 | 原计划 | 实际实现 | 原因 |
|------|--------|----------|------|
| Planner 前 2 节点 | `plan` + `design_debate` (CLAUDE_SDK × 2) | `design_debate` (SUBGRAPH_REF → debate_claude_first) | 引入 Gemini 做异构辩论，比 Claude 自我辩论更有价值 |
| 辩论轮次 | 2 节点各说一次 | 5 轮 (Claude→Gemini→Claude→Gemini→Claude) | debate_claude_first 标准流程 |
| 节点总数 | 17 | 22 | debate 子图贡献 +5 |
| Planner tools | 未限制（继承全局工具集）| `"tools": []` 显式禁用 | Claude 会无视 system prompt 偷用工具写代码 |
| decomposition_validator | 只检查 state 字段 | 从 AI 消息解析 JSON → 填充 state 字段 | LLM 输出在 message content 里，不会自动进 state |
| Token 安全阀 | 无 | 每个 LLM 节点调用前检查 | 发现 392 次死循环后紧急加入 |
| retry_count 管理 | 未明确 | LLM 节点禁止重置，仅 DETERMINISTIC 节点可写 | 死循环根因：LlmNode 把 retry_count 归零 |

---

## 当前拓扑 (22 节点)

```
colony_coder (Master)
├── plan (AGENT_REF → colony_coder_planner)
│   ├── design_debate (SUBGRAPH_REF → debate_claude_first)
│   │   ├── claude_propose    [Claude Opus]     ─┐
│   │   ├── gemini_critique_1 [Gemini Pro]       │ 共 5 轮辩论
│   │   ├── claude_revise     [Claude Opus]      │ session: claude_debate / gemini_debate
│   │   ├── gemini_critique_2 [Gemini Pro]       │ 辩论结束后 session 被清理
│   │   └── claude_conclusion [Claude Opus]     ─┘
│   ├── claude_swarm          [Claude SDK]  ─┐ session: planner_session
│   ├── task_decompose        [Claude SDK]  ─┘ 共享同一 session
│   └── decomposition_validator [DETERMINISTIC]   可 retry → task_decompose
│
├── execute (AGENT_REF → colony_coder_executor)
│   ├── code_gen              [Ollama Qwen3.5-27B] ─┐
│   ├── soft_validate         [Ollama Qwen3.5-27B]  │ session: executor_session
│   ├── self_fix              [Ollama Qwen3.5-27B] ─┘
│   ├── apply_patch           [EXTERNAL_TOOL]
│   ├── execute               [EXTERNAL_TOOL]
│   ├── hard_validate         [DETERMINISTIC]  → execute / error_classifier / __end__
│   ├── error_classifier      [DETERMINISTIC]  → self_fix / claude_rescue / rescue_router
│   ├── rescue_router         [DETERMINISTIC]  → rollback_state
│   ├── rollback_state        [DETERMINISTIC]  → claude_rescue
│   └── claude_rescue         [Claude SDK]     session: session_b (独立)
│
└── integrate (AGENT_REF → colony_coder_integrator)
    ├── integration_test      [Ollama Qwen3.5-27B]  session: integrator_session
    ├── integration_route     [DETERMINISTIC]        → __end__ / integration_rescue
    ├── integration_rescue    [Claude SDK]           session: session_b (独立)
    └── apply_patch           [EXTERNAL_TOOL]
```

---

## Session 架构

### 设计原则

1. **辩论 session 是临时的** — SubgraphRefNode._cleanup_orphan_sessions() 在子图结束后删除磁盘文件
2. **planner_session 是新建的** — claude_swarm 通过 messages 拿到辩论结论文本，不是 resume 辩论 session
3. **Ollama 无 server-side session** — 多轮历史存 state["ollama_sessions"] 字典，LangGraph checkpoint 持久化
4. **同名 session_key 跨子图隔离** — executor 和 integrator 的 session_b 是不同实例

### 数据流：辩论结论 → planner_session

```
debate_claude_first 子图结束
    ↓
SubgraphRefNode:
    state_out: {"refined_plan": "last_message"}
    → out["refined_plan"] = claude_conclusion 文本
    → out["messages"] = [AIMessage("[子图结论]\n\n{全文}")]
    → 清理 debate 子图的 claude_debate / gemini_debate session 文件
    ↓
claude_swarm (LlmNode.__call__):
    latest_input = msgs[-1].content = "[子图结论]\n\n..."
    session_id = ns.get("planner_session") = None → 新建 session
    → Claude 开新 session，首条消息 = 辩论结论
    → ns["planner_session"] = "abc123..."
    ↓
task_decompose (LlmNode.__call__):
    latest_input = msgs[-1].content = claude_swarm 的评审输出
    session_id = ns.get("planner_session") = "abc123..." → resume
    → Claude 能看到: turn1(辩论结论→评审) + turn2(评审→JSON分解)
```

### Session 映射表

| 子图 | 节点 | session_key | 模型 | 生命周期 |
|------|------|-------------|------|----------|
| Planner/debate | claude_propose, claude_revise, claude_conclusion | claude_debate | Claude Opus | 临时，辩论后清理 |
| Planner/debate | gemini_critique_1, gemini_critique_2 | gemini_debate | Gemini Pro | 临时，辩论后清理 |
| Planner | claude_swarm, task_decompose | planner_session | Claude SDK | 跟随 planner 子图 |
| Executor | code_gen, soft_validate, self_fix | executor_session | Ollama | state 内 dict |
| Executor | claude_rescue | session_b | Claude SDK | 独立 |
| Integrator | integration_test | integrator_session | Ollama | state 内 dict |
| Integrator | integration_rescue | session_b | Claude SDK | 独立 |

---

## Token 安全阀

### 触发条件

每个 LLM 节点在 call_llm() 之前检查 session 累计 token 数。

### 限额配置（3 级优先级）

```
node_config["token_limit"]  >  按 type 默认值  >  BB_TOKEN_LIMIT 环境变量
```

| 模型类型 | 默认限额 | 说明 |
|----------|----------|------|
| CLAUDE_SDK | 50,000 | 云端 API，按 token 计费 |
| GEMINI_API / GEMINI_CLI | 50,000 | 云端 API |
| OLLAMA / LOCAL_VLLM | 1,000,000 | 本地推理，不计费 |

### 实现位置

- `framework/token_guard.py` — check_before_llm(), TokenLimitExceeded
- `framework/nodes/llm/llm_node.py` — __call__ 中调用 check_before_llm
- `framework/nodes/llm/ollama.py` — tool loop 每次迭代额外检查

---

## 关键 Bug 修复记录

### 1. 死循环 (392 iterations) — 2026-03-17

**现象**: colony_coder 瞬间耗尽所有 token，task_decompose ↔ decomposition_validator 无限循环

**根因**: `LlmNode.__call__` 第 274 行返回 `"retry_count": 0`，每次 LLM 节点执行都把 validator 的 retry_count 归零

**修复**: 从 LlmNode 返回值中删除 retry_count。**retry_count 仅由 DETERMINISTIC 节点管理。**

### 2. tools=[] 被忽略 — 2026-03-18

**现象**: planner 节点的 system prompt 说"不许用工具"，但 Claude 照用不误，直接写了蛇游戏代码

**根因**: `_select_tools` 中 `self._cfg.get("tools") or self.config.tools`，空列表 `[]` 是 falsy，fallback 到全局工具集

**修复**: 区分"未配置"(key 不存在) 和"显式禁用"(tools=[])。用 sentinel `_MISSING` 判断。同步修复 claude.py 的 `_make_options`。

### 3. Validator 不解析 JSON — 2026-03-18

**现象**: task_decompose 输出了有效 JSON，但 decomposition_validator 报告验证失败

**根因**: validator 检查 `state["tasks"]`，但 JSON 在 AI message content 里，没人解析到 state 字段

**修复**: validator 新增 `_extract_json()` —— 从最后一条 AI 消息提取 JSON，填充 tasks/execution_order/refined_plan/working_directory 到 state。

---

## 文件索引

| 文件 | 职责 |
|------|------|
| `blueprints/functional_graphs/colony_coder/agent.json` | Master 图: plan → execute → integrate |
| `blueprints/functional_graphs/colony_coder_planner/agent.json` | Planner: debate → swarm → decompose → validate |
| `blueprints/functional_graphs/colony_coder_planner/validators.py` | decomposition_validator + JSON 解析 |
| `blueprints/functional_graphs/colony_coder_executor/agent.json` | Executor: 10 节点含 Ollama 代码生成 + 多级修复 |
| `blueprints/functional_graphs/colony_coder_executor/state.py` | ColonyCoderExecutorState TypedDict |
| `blueprints/functional_graphs/colony_coder_executor/validators.py` | hard_validate, error_classifier, rescue_router, rollback_state |
| `blueprints/functional_graphs/colony_coder_integrator/agent.json` | Integrator: 4 节点集成测试 + 修复循环 |
| `blueprints/functional_graphs/colony_coder_integrator/validators.py` | integration_route |
| `blueprints/functional_graphs/debate_claude_first/agent.json` | 异构辩论子图 (Claude×3 + Gemini×2) |
| `framework/token_guard.py` | Token 安全阀 |
| `framework/nodes/llm/llm_node.py` | LLM 节点基类，含 tools=[] 修复 + token guard |
| `framework/nodes/subgraph/subgraph_ref_node.py` | 子图引用节点，含 routing_context fallback |
