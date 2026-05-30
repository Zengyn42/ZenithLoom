# ApexCoder Redesign — TDD Pipeline with Inherit Session Fork

> Date: 2026-04-15
> Status: Implemented (phase 1) + Designed (phase 2: inherit fork)

## 目标

重新设计 ApexCoder，从单 Claude SDK 节点改为 TDD pipeline：
QA 先写测试 → Coder 实现并通过测试 → Executor 机械验证 → 失败时 retry。
两个 LLM 节点通过 `fork_session` 继承父图对话上下文但互相隔离。

## 动机

Snake Battle 实验暴露了单节点 ApexCoder 的核心缺陷：
- Claude 自己写 headless 测试验证自己 → 3 个 bug 全部绕过
- `appendleft` body 方向反了（headless AI 绕过但 curses 下第 1 帧死）
- 硬编码 `BOARD_H=24` 需要 28 行终端（标准 24x80 放不下）
- AI 蛇 49 帧不吃食物（寻路逻辑有问题）

根本原因：**让开发者验证自己的代码 = 让学生出题自己答**。

---

## 架构（Phase 1 — 已实现）

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

7 个节点：2 CLAUDE_SDK + 5 DETERMINISTIC

| 节点 | 类型 | 模型 | 职责 |
|------|------|------|------|
| setup | DETERMINISTIC | - | 提取 user_requirements、解析/创建 working_directory |
| claude_qa | CLAUDE_SDK | claude-sonnet-4-6 | 根据需求写 5-10 个 QA 测试（独立 session） |
| reset_for_coder | DETERMINISTIC | - | 清掉 QA messages，构建 Coder prompt |
| claude_coder | CLAUDE_SDK | claude-opus-4-6 | 实现代码 + 内部跑 QA 测试 |
| executor | DETERMINISTIC | - | subprocess 跑 run_qa.sh，捕获 stdout/stderr/returncode |
| route | DETERMINISTIC | - | PASS → END，FAIL + retry<5 → retry，FAIL + retry≥5 → abort |
| inject_error_context | DETERMINISTIC | - | 构建 retry prompt（错误栈 + iteration_history） |

### 防作弊机制

PreToolUse hook 阻止 ClaudeCoder 修改 `test_tool/qa_tests/` 下的文件。
配置在 entity.json 的 `settings_override.hooks.PreToolUse`。

### State Schema

`ApexCoderState` 扩展 `BaseAgentState`：

| 字段 | 来源 | 用途 |
|------|------|------|
| user_requirements | setup | 用户需求文本 |
| working_directory | setup | 工作目录路径 |
| qa_bypass | claude_qa | QA 跳过标记 |
| qa_tests_dir | claude_qa | 测试目录路径 |
| run_qa_script | claude_qa | run_qa.sh 路径 |
| qa_summary | claude_qa | QA 摘要 |
| apex_conclusion | claude_coder | Coder 最终报告 |
| execution_stdout | executor | 测试标准输出 |
| execution_stderr | executor | 测试错误输出 |
| execution_returncode | executor | 测试返回码 |
| iteration_history | inject_error_context | 历史失败记录，防重复犯错 |
| status | executor/route | PENDING/PASS/FAIL |

---

## 架构（Phase 2 — 待实现：inherit session fork）

### 问题

Phase 1 中，QA 和 Coder 各自创建全新 Claude session。当 ApexCoder 作为子图从 technical_architect 调用时：
- QA 和 Coder 看不到 technical_architect 与用户的对话历史（需求讨论、辩论结论、设计细节）
- 只能通过 state 字段（refined_plan、routing_context）传递简化后的上下文
- 对复杂任务，丢失语境链会导致 QA 写出不合理的测试、Coder 误解需求

### 解决方案：Claude SDK `fork_session`

Claude Agent SDK 原生支持 session fork：

```python
ClaudeAgentOptions(
    resume=parent_session_id,  # fork 源
    fork_session=True,          # fork 而非直接 resume
)
```

效果：基于父 session 创建一个新的独立 session，继承完整对话历史，但后续对话不影响原 session。

### inherit 模式下的 session 生命周期

```
technical_architect 主图:
  claude_main session = uuid-A (完整对话历史)
  路由到 apex_coder 子图 (session_mode: inherit)

ApexCoder 子图:
  _subgraph_init: 不清理（inherit 模式）
  
  setup: 提取 user_requirements, 读 state.refined_plan 等
  
  claude_qa 首次调用:
    node_sessions["apex_qa"] = ""（空）
    → fork uuid-A → 得到 uuid-fork-qa
    → QA 看到 technical_architect 完整对话 + 自己写测试的过程
    → node_sessions["apex_qa"] = "uuid-fork-qa"
  
  reset_for_coder: 清 QA messages
  
  claude_coder 首次调用:
    node_sessions["apex_coder"] = ""（空）
    → fork uuid-A → 得到 uuid-fork-coder
    → Coder 看到 technical_architect 完整对话（不含 QA 的 reasoning）
    → node_sessions["apex_coder"] = "uuid-fork-coder"
  
  executor → route → FAIL → retry:
  
  claude_coder 第 2 次调用:
    node_sessions["apex_coder"] = "uuid-fork-coder"（非空）
    → resume uuid-fork-coder（不 fork，继续自己的 session）
    → Coder 看到：technical_architect 对话 + 自己第 1 次写的代码 + 新的错误反馈
    → 不会重复犯同一个错

  _subgraph_exit:
    1. RemoveMessage 清所有 messages（防父图 messages 污染）
    2. 清掉子图的 session keys（apex_qa, apex_coder）从 node_sessions
    → 父图 node_sessions 只保留 claude_main = uuid-A
    → 下次调用 ApexCoder 时，QA 和 Coder 重新 fork
```

### fork vs resume 判断逻辑

在 `ClaudeSDKNode.call_llm()` 中：

```python
async def call_llm(self, prompt, session_id="", inherit_from="", ...):
    if not session_id and inherit_from:
        # 首次 + inherit → fork 父 session
        options = ClaudeAgentOptions(
            resume=inherit_from,
            fork_session=True,
            ...
        )
    elif session_id:
        # 有自己的 session → resume（retry 场景）
        options = ClaudeAgentOptions(
            resume=session_id,
            fork_session=False,
            ...
        )
    else:
        # 全新 session（独立运行场景）
        options = ClaudeAgentOptions(...)
```

### inherit 模式下 _subgraph_exit 的增强

当前 `_subgraph_exit` 只清 messages。inherit 模式需要额外清理子图的 session keys：

```python
def make_subgraph_exit(session_mode="persistent", subgraph_session_keys=None):
    """
    所有子图：RemoveMessage 清 messages。
    inherit 模式：额外清掉子图自己创建的 session keys。
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

`subgraph_session_keys` 在构建时从 entity.json 的节点声明中提取（所有 LLM 节点的 session_key）。

### 四种 session_mode 完整对比（更新）

| session_mode | 首次 LLM 调用 | retry LLM 调用 | _subgraph_exit | 用途 |
|---|---|---|---|---|
| **persistent** | 新建 session | resume 自己的 session | 清 messages | 子图跨调用保持 session |
| **inherit** | **fork 父 session** | resume 自己的 fork | 清 messages + 清子图 session keys | 子图继承父图上下文，但不影响父 session |
| **fresh_per_call** | 新建 session | N/A（每次全新） | 清 messages | 每次调用全新 |
| **isolated** | 新建 session (unique key) | N/A | 清 messages | 完全隔离 |

### setup 的 inherit 增强

setup 在 inherit 模式下可以从 state 读取更多父图上下文：

```python
def setup(state):
    # 优先用 refined_plan（technical_architect 辩论后的精炼设计）
    plan = state.get("refined_plan", "")
    debate = state.get("debate_conclusion", "")
    
    # routing_context 或 messages[0] 作为原始需求
    raw = state.get("routing_context", "") or state["messages"][0].content
    
    # 合并上下文
    if plan:
        user_requirements = f"{raw}\n\n## 设计方案\n{plan}"
    elif debate:
        user_requirements = f"{raw}\n\n## 辩论结论\n{debate}"
    else:
        user_requirements = raw
    
    # 选择性清理子图 session keys（下次 fork 用）
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

### inherit_from 配置

entity.json 中 LLM 节点声明 `inherit_from`：

```json
{
  "id": "claude_qa",
  "type": "CLAUDE_SDK",
  "session_key": "apex_qa",
  "inherit_from": "claude_main",
  ...
}
```

框架在构建时：如果 `session_mode == "inherit"` 且节点有 `inherit_from`，将其传递给 ClaudeSDKNode，call_llm 时使用。

---

## Phase 2 实施计划

### 改动清单

| 文件 | 改动 |
|------|------|
| `framework/nodes/llm/claude.py` | `call_llm` 加 `inherit_from` 参数，fork_session 逻辑 |
| `framework/nodes/llm/llm_node.py` | `__call__` 传递 inherit_from 到 call_llm |
| `framework/nodes/subgraph_init_node.py` | `make_subgraph_exit` 接受 session_mode + subgraph_session_keys |
| `framework/agent_loader.py` | inherit 模式下提取 subgraph_session_keys，传递给 exit node |
| `VoidDraft/blueprints/functional_graphs/apex_coder/entity.json` | 添加 `inherit_from` 到 QA 和 Coder 节点 |
| `VoidDraft/blueprints/functional_graphs/apex_coder/validators.py` | setup 加 inherit 增强（读 refined_plan、清 session keys） |

### 不需要改的

| 文件 | 原因 |
|------|------|
| PROTOCOL.md | 通用工程实践，不变 |
| CODER_ROLE.md / QA_ROLE.md | 角色定义不变 |
| hooks/protect_qa_tests.py | 防作弊机制不变 |

---

## 文件索引

| 文件 | 职责 |
|------|------|
| `VoidDraft/blueprints/functional_graphs/apex_coder/entity.json` | 7 节点图定义 |
| `VoidDraft/blueprints/functional_graphs/apex_coder/state.py` | ApexCoderState schema |
| `VoidDraft/blueprints/functional_graphs/apex_coder/validators.py` | setup, reset_for_coder, executor, route, inject_error_context |
| `VoidDraft/blueprints/functional_graphs/apex_coder/CODER_ROLE.md` | Coder persona |
| `VoidDraft/blueprints/functional_graphs/apex_coder/QA_ROLE.md` | QA persona |
| `VoidDraft/blueprints/functional_graphs/apex_coder/PROTOCOL.md` | 共享工程实践 |
| `VoidDraft/blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py` | PreToolUse 防作弊 hook |
| `tests/test_apex_coder.py` | 20 个单元测试 |
| `run_apex_coder_debug.py` | Debug runner |
| `run_benchmark_apex.py` | Benchmark runner |
