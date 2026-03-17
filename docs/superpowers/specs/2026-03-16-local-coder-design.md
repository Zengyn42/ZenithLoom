# LocalCoder (CodingSubgraph) 设计文档

> 版本：v1.0 | 日期：2026-03-16 | 状态：待实现
> 基于 Vault 编程子图设计 v1.6，结合 BootstrapBuilder 框架现有模式

---

## 一、设计目标

一个可复用的 LangGraph 编程子图，实现 Vault 编程子图设计 v1.6 的完整规格。

核心分工：
- **ApexCoder（CLAUDE_SDK）** — 决策层：规划、蜂群综合、任务分解、救援
- **Qwen3.5-27B（OLLAMA）** — 执行层：代码生成、语义验证、自修复、集成测试
- **DETERMINISTIC 节点** — 规则层：硬验证、错误分类、路由决策、状态回滚
- **EXTERNAL_TOOL 节点** — 工具层：subprocess 代码执行沙箱

**当前阶段：** 独立子图，不挂载到 Hani，待验证稳定后再接入。

---

## 二、架构概览

### 2.1 三层子图结构

主图 local_coder 是薄编排层，三层业务逻辑各成子图：

```
local_coder (主图，3个 AgentRefNode)
  ├── local_coder_planner     — 规划层（5节点）
  ├── local_coder_executor    — 执行循环（9节点）
  └── local_coder_integrator  — 集成层（3节点）
```

目录结构：

```
agents/
  local_coder/            <- 主图（薄编排）
    agent.json
  local_coder_planner/    <- 规划子图
    agent.json
  local_coder_executor/   <- 执行子图
    agent.json
    validators.py         <- DETERMINISTIC handler 函数
    executor.py           <- EXTERNAL_TOOL subprocess 实现
  local_coder_integrator/ <- 集成子图
    agent.json
    validators.py
```

### 2.2 新增框架组件

| 组件 | 位置 | 说明 |
|------|------|------|
| DeterministicNode | framework/nodes/deterministic_node.py | 按约定自动查找 validators.py:fn_name |
| ExternalToolNode 扩展 | framework/nodes/external_tool_node.py | 新增 executor: code_execution 模式 |
| OllamaNode tool-calling | framework/llama/node.py | 多轮工具调用循环 |
| OllamaNode session | framework/llama/node.py | ollama_sessions in state |
| framework/llama/tools.py | 新建 | 工具函数实现 |

### 2.3 框架节点大类分类原则

框架节点分为四大类：

| 大类 | 定义 | 例子 |
|------|------|------|
| LLM | 调用语言模型，有 session 管理 | CLAUDE_SDK, GEMINI_CLI, OLLAMA |
| EXTERNAL_TOOL | 调用外部系统（subprocess/HTTP/git），有副作用 | ExternalToolNode, GitSnapshotNode, VramFlushNode |
| DETERMINISTIC | 纯 Python 逻辑，无 subprocess，无网络，无文件写入 | DeterministicNode, SubgraphMapperNode |
| 编排 | 子图调度 | AgentRefNode, AgentRunNode |

**设计原则（何时拆分为子图）：**

> 任何非 LLM 操作都可以表达为 DETERMINISTIC + EXTERNAL_TOOL 的子图组合。
> 当步骤有独立复用价值或条件逻辑需要在图层面可见时应拆分；
> 否则用单一聚合类保持简洁（如 GitSnapshotNode）。

**DETERMINISTIC 的硬边界：** 有 subprocess 调用或文件写入即为 EXTERNAL_TOOL，不是 DETERMINISTIC。

---

## 三、节点拓扑（17 节点）

### 3.1 主图 local_coder

边：__start__ → plan → execute → integrate → __end__（abort_reason 非空时提前退出）

### 3.2 规划子图 local_coder_planner（5节点）

```
__start__ → plan → design_debate? → claude_swarm → task_decompose
                                                         |
                                              decomposition_validator
                                              pass↓  fail↑  abort→__end__
```

| 节点 | 类型 | 说明 |
|------|------|------|
| plan | CLAUDE_SDK | Session A，分析任务，输出方案 + 复杂度判定 |
| design_debate | CLAUDE_SDK | Session A，prompt-based 蜂群思考（complex 时触发） |
| claude_swarm | CLAUDE_SDK | Session A，4角色审查综合（feasibility/edge_cases/performance/testability） |
| task_decompose | CLAUDE_SDK | Session A，拆解为独立 task 列表 |
| decomposition_validator | DETERMINISTIC | 校验 task 列表，最多重试 2 次 |

**claude_swarm 实现：** 普通 CLAUDE_SDK 节点，system prompt 写明四个审查角色，Claude 自行蜂群思考并综合——无需新节点类型。

### 3.3 执行子图 local_coder_executor（9节点）

| 节点 | 类型 | 工具/模型 |
|------|------|------|
| code_gen | OLLAMA | qwen3.5:27b，tools: read_file/write_file/shell_run/list_dir |
| execute | EXTERNAL_TOOL | executor: code_execution，subprocess in tempdir |
| hard_validate | DETERMINISTIC | validators.py:hard_validate |
| soft_validate | OLLAMA | qwen3.5:27b，tools: submit_validation only |
| error_classifier | DETERMINISTIC | validators.py:error_classifier |
| self_fix | OLLAMA | qwen3.5:27b，tools: read_file/write_file/shell_run |
| claude_rescue | CLAUDE_SDK | Session B（每次独立新建） |
| rescue_router | DETERMINISTIC | validators.py:rescue_router |
| rollback_state | DETERMINISTIC | validators.py:rollback_state |

所有 CLAUDE_SDK 节点包装 @with_budget_check 熔断装饰器。

### 3.4 集成子图 local_coder_integrator（3节点）

| 节点 | 类型 | 说明 |
|------|------|------|
| integration_test | OLLAMA | Qwen 通过 shell_run 工具运行测试，再调用 submit_validation 输出结果（工具调用内部完成，不是独立 EXTERNAL_TOOL 节点） |
| integration_rescue | CLAUDE_SDK | Session B，输出 IntegrationPatch |
| apply_patch | EXTERNAL_TOOL | 应用 patch 到文件系统 |

---

## 四、State 与 Session

### 4.1 状态流动

```
local_coder master state
  task_description, context_files, working_directory
  → success, abort_reason, final_files, cost_summary

  local_coder_planner
    state_in:  task_description, context_files, working_directory
    state_out: tasks, execution_order, refined_plan, cost_apexcoder

  local_coder_executor
    state_in:  tasks, execution_order, refined_plan, working_directory
    state_out: completed_tasks, final_files, abort_reason, cost_qwen

  local_coder_integrator
    state_in:  completed_tasks, final_files, working_directory
    state_out: success, abort_reason, cost_integration
```

### 4.2 Session 作用域

| Session | 作用域 | 生命周期 |
|---------|--------|---------|
| Session A（规划链） | local_coder_planner 内部 | plan → task_decompose 完成后清理 |
| Session B（救援） | executor + integrator 各自独立 | 每次 rescue 触发时新建；子图退出时由 AgentRefNode._cleanup_orphan_sessions 自动清理（与 debate 子图相同机制） |
| Qwen session | local_coder_executor 内部 | 整个执行循环共享 session_key=qwen_executor |

### 4.3 OllamaNode Session 设计

Ollama HTTP API 无服务端 session，对话历史由客户端维护。

方案：消息历史存入 state（随 LangGraph checkpoint 序列化）

新增 state 字段：
  ollama_sessions: dict[str, list[dict]]
  key: session UUID（来自 node_sessions[session_key]）
  value: 完整消息历史 [{"role": "system/user/assistant/tool", "content": "..."}]

与 GeminiNode / ClaudeSDKNode 保持一致的接口：
- node_config["session_key"] 指定 session 共享键
- 首次运行：创建 UUID，初始化消息列表，存入 ollama_sessions
- 后续运行：从 node_sessions[session_key] 取 UUID，续接消息历史
- rollback_state 重置 task 状态时，LangGraph checkpoint 自动还原消息历史

### 4.4 ExecutorState 关键字段

```python
class ExecutorState(TypedDict):
    # 规划层传入
    tasks: list[dict]
    execution_order: list[str]
    refined_plan: str

    # 任务循环追踪
    current_task_index: int
    current_task_id: str
    retry_count: int               # per task，上限 5
    transient_retry_count: int     # per task，上限 3
    error_history: list[dict]
    completed_tasks: list[dict]
    cross_task_issues: list[dict]

    # 执行结果（per task）
    generated_code: str
    generated_files: list[str]
    execution_result: str
    execution_exit_code: int
    execution_stderr: str
    validation_output: dict | None  # ValidationOutput

    # Session
    node_sessions: dict[str, str]
    ollama_sessions: dict[str, list[dict]]  # 新增

    # 输出
    final_files: list[str]
    abort_reason: str | None
    cost_qwen: dict
```

---

## 五、OllamaNode 工具定义

### 5.1 工具集（framework/llama/tools.py）

五个工具函数：
- read_file(path) → str
- write_file(path, content) → str
- shell_run(cmd, cwd="") → dict {stdout, stderr, exit_code}
- list_dir(path) → list[str]
- submit_validation(status, rationale, affected_scope) → dict

### 5.2 节点工具集映射

| 节点 | tools | 理由 |
|------|-------|------|
| code_gen | read_file, write_file, shell_run, list_dir | 读现有代码、理解目录、写新文件 |
| soft_validate | submit_validation only | 结构化输出，不碰文件系统 |
| self_fix | read_file, write_file, shell_run | 读错误文件、修复、验证 |
| integration_test | shell_run, submit_validation | 运行测试命令 + 结构化结果 |

submit_validation 被调用后立即作为节点输出，不经过文本解析。

### 5.3 OllamaNode 工具调用循环（多轮 agentic loop）

OllamaNode._tool_loop 内部循环：
1. 发送消息给 Ollama
2. 若 response 包含 tool_calls：执行工具，追加 tool result，继续循环
3. 若无 tool_calls：返回最终文本，退出循环

---

## 六、错误处理与熔断

### 6.1 ValidationOutput 契约

ErrorCategory 四分类：
- TRANSIENT: 网络超时、文件锁、API 抖动
- LOGIC: 语法错误、类型错误、测试失败
- STRUCTURAL: 架构缺陷、方案本身有问题
- INFRASTRUCTURE: 磁盘满、环境损坏

ValidationOutput 字段：status, category, severity, rationale, affected_scope, is_regression, raw_stderr

### 6.2 error_classifier 路由规则

- INFRASTRUCTURE → __end__（直接终止）
- TRANSIENT → retry execute（独立计数 ≤3，不消耗 retry_count）
- LOGIC → self_fix（retry<3）或 claude_rescue（retry≥3）
- STRUCTURAL → claude_rescue（立即升级）

启发式升级 logic → structural：
1. 连续 3 次相同 rationale
2. is_regression = true
3. 连续 2 次 soft_validate fail

### 6.3 rescue_router 三条路径

| scope | 触发条件 | 动作 |
|-------|---------|------|
| current_task | 当前 task 设计有问题 | ApexCoder 修复 → 回 code_gen |
| previous_task | 上游 task 引入 bug | cascade_rollback → 回最早受影响 task |
| cross_task | 跨 task 架构冲突 | 惰性标记，继续，留给 integration_rescue |

### 6.4 重试计数器

| 计数器 | 上限 | 重置时机 |
|--------|------|---------|
| retry_count | 5（per task） | 进入下一个 task |
| transient_retry_count | 3（per task） | 进入下一个 task |
| decomposition_retry_count | 2 | 分解通过后 |
| integration_retry_count | 2 | 集成通过后 |

### 6.5 熔断装饰器

所有 CLAUDE_SDK 节点包装 @with_budget_check：
- APEX_TOKEN_BUDGET = 100_000 tokens
- TIME_BUDGET_SECONDS = 3_600（1 小时）

abort_reason 非空时主图通过全局边直接路由至 __end__。

---

## 七、代码执行沙箱（execute 节点）

选型：subprocess + tempdir（Phase 1）

理由：WSL2 原生兼容，零依赖。LocalCoder 生成自有代码，威胁模型是意外破坏文件而非恶意逃逸。

ExternalToolNode 新增 executor: code_execution 模式，返回结构化输出：
  execution_result (stdout), execution_exit_code (int), execution_stderr (str), execution_cwd (str)

node_config 支持 executor: subprocess | docker（docker 为 Phase 2 扩展）。

---

## 八、DeterministicNode 设计

按约定自动查找同目录 validators.py 中与 node_id 同名函数，或通过 handler 字段显式指定。

路由节点返回 {**state, "routing_target": "xxx"}，与现有 routing_to 模式兼容。
非路由节点返回修改后的 state 字典。

---

## 九、测试策略

### 9.1 单元测试（无 LLM，无网络）

- DETERMINISTIC 节点：直接调用纯函数
  - test_hard_validate_syntax_error
  - test_error_classifier_transient_retry
  - test_cascade_rollback
- DeterministicNode 框架：验证 handler 加载机制
- ExternalToolNode code_execution 模式：mock subprocess

### 9.2 集成测试（真实 Ollama，mock Claude）

- test_code_gen_writes_file: Qwen 调用 write_file，验证文件写入磁盘
- test_soft_validate_returns_structured: submit_validation 输出结构化数据

### 9.3 E2E 测试（分子图分阶段）

```
test_e2e_local_coder_planner.py   — mock Qwen，真实 Claude
test_e2e_local_coder_executor.py  — 真实 Qwen，mock Claude rescue
test_e2e_local_coder.py           — 完整链路
```

E2E 验收标准：
  TASK = "创建 Python 函数 add(a, b)，写单元测试，确保测试通过"
  result["success"] is True
  (tmp_path / "add.py").exists()
  result["abort_reason"] is None

---

## 十、与 Vault 设计的差异

| # | Vault 规格 | 本设计决策 | 理由 |
|---|-----------|-----------|------|
| 1 | Qwen3.5-27B via vLLM | Qwen3.5-27B via Ollama | 统一本地模型推理引擎 |
| 2 | 平铺 17 节点图 | 三层子图架构 | 独立可测，层次清晰 |
| 3 | CLAUDE_SWARM 新节点类型 | CLAUDE_SDK + prompt | CLI 原生蜂群通过 prompt 触发，零新节点类型 |
| 4 | 消息历史不在 state | ollama_sessions in state | LangGraph checkpoint rollback 自动还原历史 |
| 5 | execute 沙箱未指定 | subprocess + tempdir（Phase 1） | WSL2 兼容，可扩展 docker |

---

*— 无垠智穹 · BootstrapBuilder LocalCoder 设计 2026-03-16*
