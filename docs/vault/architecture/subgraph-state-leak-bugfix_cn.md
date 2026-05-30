# Subgraph State Leak Bugfix (父图 prompt 膨胀)

> Date: 2026-04-20
> Status: Fixed
> Related: unified-subgraph-integration.md, subgraph-output-field-unification.md, session-mode-design.md

## 症状

technical_architect 的 Discord session 里，Claude Main 每一轮看到的 prompt 都被塞进几 KB 的历史上下文，token 持续飙升。用户输入 "恢复gemma4"（8 字符）也会被扩成巨大 prompt。

实测 session `hani_session_2d05f9a6` 最新 checkpoint：

```
debate_conclusion    : 3262 chars   （上一次 debate 子图的结论）
subgraph_topic       : 1222 chars   （上一次辩题锚点）
previous_node_output : 107  chars
apex_conclusion      : 0
knowledge_result     : 0
discovery_report     : 0
```

这三个非空字段都由 `LlmNode._build_gemini_section()` / `_topic_inject` / `_prev_inject` 无条件拼进 Claude Main 的每一轮 prompt 前缀。

## 根因（三个独立泄漏路径）

### A. `subgraph_topic` 泄漏

- `framework/nodes/subgraph_init_node.py::_fresh_init` 会写入 `subgraph_topic = routing_context or subgraph_topic`（仅 `fresh_per_call` 模式）。
- `_subgraph_exit` 原实现只发 `RemoveMessage` 清 `messages`，**没有清空 `subgraph_topic`**。
- 结果：`fresh_per_call` 子图返回时，父图 state 继承了这个主题锚点。
- 父图下一轮任何 LlmNode 读 `state["subgraph_topic"]` → `_topic_inject` 把 1KB+ 的辩题拼到 prompt。

### B. `previous_node_output` 泄漏

- 任何 LlmNode 执行后都会写 `result["previous_node_output"] = raw_output`（[llm_node.py:521](../../../framework/nodes/llm/llm_node.py)）。设计意图是子图**内部**节点间链式推理（Gemini → Claude → Gemini ...）。
- 子图末尾节点也会写，`_subgraph_exit` 没清理 → 泄漏到父图。
- 父图下一轮 Claude Main 读 `state["previous_node_output"]`；`_prev_inject` 的 gate 是 `if _subgraph_topic`，所以只要 A 已经泄漏，B 就跟着被注入。

### C. `debate_conclusion` / `apex_conclusion` / `knowledge_result` / `discovery_report` 持续注入

- 这四个字段是子图末尾节点通过 `node_config.output_field` 显式写入的设计，意图是**供父图下一轮的 Claude Main 消费一次**。
- `LlmNode._build_gemini_section()` 读它们后拼进 prompt，但**没有在输出里清空**。
- LangGraph checkpoint 持久化 → 每一轮父图 Claude Main 都重复注入同一份历史结论。

## 影响范围（不是辩论子图独有，是框架层）

| 子图 | session_mode | A 泄漏 | B 泄漏 | C 泄漏 |
|---|---|---|---|---|
| debate_gemini_first (debate_brainstorm) | fresh_per_call | ✓ | ✓ | ✓ (`debate_conclusion`) |
| debate_claude_first (debate_design) | fresh_per_call | ✓ | ✓ | ✓ (`debate_conclusion`) |
| apex_coder | inherit | — | ✓ | ✓ (`apex_conclusion`) |
| tool_discovery | fresh_per_call | ✓ | ✓ | ✓ (`discovery_report`) |
| tool_evaluate | fresh_per_call | ✓ | ✓ | ✓ (`discovery_report`) |
| video_quality_loop | fresh_per_call | ✓ | ✓ | — |
| colony_coder（作为 Master）| fresh_per_call | ✓ | ✓ | — |

**受害主体**：

- **technical_architect 主图：重伤**。调用链 debate_brainstorm / debate_design / apex_coder / tool_discovery / tool_evaluate / video_quality_loop，全部都会泄漏。
- **ColonyCoder（作为独立主图）：间接轻伤**。plan/execute/qa 三个 fresh_per_call 子图会触发 A+B，但 ColonyCoder 内部 LLM 节点用自己的状态字段（`refined_plan`、`qa_analysis`、`e2e_plan` 等），不读 `debate_conclusion`/`apex_conclusion`，所以 C 类注入不直接增加 token。
- **ColonyCoder 内部 `colony_coder_planner → debate_claude_first`** 的路径会让 planner 的 state 带上 `debate_conclusion`，planner 若再次跑 LlmNode 会被 C 注入一次。
- ColonyCoder 2026-04-17 的 "session 重置 + 快照" context 防护遮住了部分 A/B 症状。

## 修复

三处改动（2026-04-20 已 apply）：

### 1. `framework/agent_loader.py` — 附带修 `Annotated is not defined`

Apex_coder / colony_coder 的 `state.py` 用了 `from __future__ import annotations`，`Annotated[dict, _merge_dict]` 被存为字符串。`get_type_hints()` 通过 `sys.modules[cls.__module__]` 找 globals 来 eval 这些字符串，但旧代码 `importlib.util.module_from_spec` + `exec_module` **没把 module 写进 `sys.modules`**，导致 globals 空 → `NameError: name 'Annotated' is not defined`。

```python
_sys.modules[_mod_name] = _mod   # ← 新增：必须在 exec_module 前注册
_spec.loader.exec_module(_mod)
```

### 2. `framework/nodes/subgraph_init_node.py::_subgraph_exit`

```python
result = {
    "messages": removals,
    "subgraph_topic": "",           # ← 新增，清 A
    "previous_node_output": "",     # ← 新增，清 B
}
```

所有 session_mode 的子图退出时一并清空，符合 `BaseAgentState` 注释里的承诺 "_subgraph_init 入口写入、出口清空"。

### 3. `framework/nodes/llm/llm_node.py` — 消费后清空 C

```python
result["debate_conclusion"] = ""
result["apex_conclusion"]   = ""
result["knowledge_result"]  = ""
result["discovery_report"]  = ""

if self._output_field and raw_output:
    result[self._output_field] = raw_output   # 覆盖：子图末尾生产者不受影响
```

`output_field` 写入在清空后，子图末尾生产这四个字段的节点仍然能把结论传回父图；父图 Claude Main 消费完下一轮自动清空，不再累积。

## 遗留（历史 checkpoint 数据）

代码修复只对新写入生效。现有 session checkpoint 里已泄漏的字段要等：

- `debate_conclusion` / `apex_conclusion` / 等：下一次父图 LlmNode 跑一轮自动清（**一次**额外的大 prompt）。
- `subgraph_topic` / `previous_node_output`：下一次子图调用才清（如果用户不再触发子图就一直留着）。

想立即清零：受影响频道执行 `!clear` 或 `!new`。

## 验证

- [framework/nodes/llm/llm_node.py](../../../framework/nodes/llm/llm_node.py)
- [framework/nodes/subgraph_init_node.py](../../../framework/nodes/subgraph_init_node.py)
- [framework/agent_loader.py](../../../framework/agent_loader.py)

手动 smoke：
```python
# 复现 A→NameError 已通过
# 复现 B→get_type_hints(ApexCoderState, include_extras=True) 正常返回
```

`systemctl --user restart technical_architect` → `[Discord] controller 已初始化（graph 已编译）`，无 Annotated 报错。

## 长期改进（与 subgraph-output-field-unification.md 关联）

C 类泄漏的根因之一是 `BaseAgentState` 硬编码 4 个输出字段，每次加新子图都要同步修改 `_build_gemini_section`、`_fresh_init` 等多处。若落地 `subgraph_outputs: Annotated[dict, _merge_dict]` 统一方案，消费-清空逻辑可以集中在一处，避免再次出现类似泄漏。
