# ApexCoder v2 — TDD Pipeline 架构实录

> 最后更新: 2026-04-17
> 状态: 架构已定型，实现进行中

---

## 设计目标

将 ApexCoder 从单一 Claude SDK 节点重构为 4 节点 TDD 流水线，实现「QA 先写测试 → Coder 写代码通过测试」的严格开发流程。

## 与 v1 的差异

| 项目 | v1 (Colony Coder) | v2 (Apex Coder TDD) |
|------|-------------------|---------------------|
| 节点数 | 17 节点 (Master + Planner + Executor + QA) | 4 节点 (setup → ClaudeQA → reset → ClaudeCoder) |
| 测试策略 | 事后验证 | TDD: QA 先写测试，Coder 必须通过 |
| 会话模式 | 独立 session | `inherit_from: parent`，共享上下文 |
| 辩论集成 | Planner 内部辩论 | 外部辩论子图结论注入 |
| 防篡改 | 无 | PreToolUse hook 禁止 Coder 修改 QA 测试文件 |

## 当前拓扑

```
setup (DETERMINISTIC)
  → ClaudeQA (CLAUDE_SDK) — 写测试、定义验收标准
    → reset_for_coder (DETERMINISTIC) — 隔离消息，注入测试摘要
      → ClaudeCoder (CLAUDE_SDK) — 实现代码，必须通过 QA 测试
```

## 关键设计决策

### 1. inherit_from 模式
ApexCoder 通过 `inherit_from` 从父图（technical_architect）继承 session 上下文，无需手动传递辩论结论。technical_architect 的对话历史中包含辩论结果，ApexCoder 自动可见。

### 2. status 默认值
`status` 字段默认值从无 → `"PENDING"`。解决 state merge 时未初始化字段导致路由异常的 bug。

### 3. 与辩论子图的协作流程
```
用户需求 → technical_architect 评估复杂度
  → 复杂: debate_brainstorm / debate_design → 辩论结论注入 technical_architect context
  → technical_architect 整理实现指令 → route to apex_coder
  → ApexCoder 继承 context（含辩论结论）→ QA 写测试 → Coder 实现
  → technical_architect 验证结果（跑 benchmark、检查输出）
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

## 参考文件

- 设计计划: `docs/superpowers/plans/2026-04-15-apex-coder-redesign.md`
- State: `blueprints/functional_graphs/apex_coder/state.py`
- Entity: `blueprints/functional_graphs/apex_coder/entity.json`
- Colony Coder v1 实录: `docs/vault/architecture/colony-coder.md`
