# Subgraph Output Field Unification

> Date: 2026-04-12
> Status: Backlog
> Depends on: unified-subgraph-integration.md (完成后再做)

## 问题

`BaseAgentState` 上硬编码了 5 个子图输出字段：

```python
debate_conclusion: str   # debate 子图
apex_conclusion: str     # apex_coder 子图
knowledge_result: str    # knowledge_shelf 子图
discovery_report: str    # tool_discovery 子图
refined_plan: str        # colony_coder_planner
```

每新增一个子图需要：
1. `BaseAgentState` 加字段
2. `SubgraphInputState` 排除它
3. `LlmNode._build_gemini_section()` 加 `if` 分支读取
4. `_session_init` / `_fresh_init` 加清理行

## 提议方案

统一为字典字段：

```python
subgraph_outputs: Annotated[dict, _merge_dict]
# {"debate_conclusion": "...", "apex_conclusion": "...", ...}
```

### 好处

- 新子图零改动 `BaseAgentState`
- `_build_gemini_section()` 遍历 dict，不硬编码
- `_session_init` 只需 `"subgraph_outputs": {}` 一行清理
- `SubgraphInputState` 排除一个字段即可

### 影响范围

- `framework/schema/base.py` — 字段定义
- `framework/nodes/llm/llm_node.py` — `output_field` 写入逻辑 + `_build_gemini_section()` 读取逻辑
- `framework/nodes/session_init_node.py` — 清理逻辑
- `blueprints/functional_graphs/*/entity.json` — `output_field` 声明（语义不变，只是写入目标变了）
- `blueprints/functional_graphs/colony_coder_*/validators.py` — 读取 `refined_plan` 的地方
- 所有 schema 子类 — 去掉各自的 output 字段覆盖

### 注意

`refined_plan` 比较特殊 — 被 colony_coder_executor 和 colony_coder_qa 的 validators 直接读取（`state.get("refined_plan", "")`），需要改为从 `subgraph_outputs` 读取。

## 与 _build_gemini_section 的关系

当前 `_build_gemini_section()` 为每个字段注入不同的中文标签（`[辩论结论]`、`[ApexCoder 结论]` 等）。统一后可以改为：

```python
_LABELS = {
    "debate_conclusion": "辩论结论",
    "apex_conclusion": "ApexCoder 结论",
    "knowledge_result": "知识库查询结果",
    "discovery_report": "工具发现报告",
}
for key, value in state.get("subgraph_outputs", {}).items():
    if value:
        label = _LABELS.get(key, key)
        parts.append(f"\n[{label}]\n{value}\n")
```
