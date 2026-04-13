# Unified Subgraph Integration — Native Subgraph + Entry Cleanup Node

> Date: 2026-04-12
> Status: Proposed — 待实施
> Supersedes: `session-mode-design.md` 中的 wrapper 层设计（wrapper 层将被此方案替代）

## 问题

当前 `session_mode` 的三种非默认模式使用不同的接入方式：

| session_mode | 接入方式 | LangGraph 视角 |
|---|---|---|
| persistent | `builder.add_node(id, CompiledStateGraph)` | 原生子图 ✅ |
| fresh_per_call | `builder.add_node(id, async_wrapper)` | 普通函数 ❌ |
| isolated | `builder.add_node(id, async_wrapper)` | 普通函数 ❌ |

`fresh_per_call` 和 `isolated` 用 async wrapper 包裹子图来实现 state 清理。
这导致 LangGraph 失去子图可见性：

1. `astream(subgraphs=True)` 无法追踪内部节点
2. `get_graph(xray=True)` 看不到子图结构
3. Debug/tracing 工具失效

## 核心思路

**所有 session_mode 都使用 LangGraph 原生子图接入（`builder.add_node(id, CompiledStateGraph)`）。**

State 清理逻辑不再用外部 wrapper，而是作为 DETERMINISTIC 节点动态注入到子图的 entry 前面。

## 架构

### 子图声明（entity.json 不变）

每个子图在 entity.json 中声明 `entry` 和 `exit`（或通过 edges 隐式声明 exit）：

```json
{
  "name": "colony_coder_planner",
  "graph": {
    "entry": "design_debate",
    "nodes": [...],
    "edges": [
      {"from": "decomposition_validator", "to": "__end__", "type": "routing_to"}
    ]
  }
}
```

子图不声明 `__start__` / `__end__`。框架在构建时自动补 `START → entry` 和 `exit → END`。

### 动态注入清理节点

`_build_declarative()` 在构建子图时，根据父图传入的 `session_mode` 决定是否注入清理节点：

```
persistent（默认）:
  START → entry → ... → exit → END
  (不注入)

fresh_per_call:
  START → _session_init → entry → ... → exit → END
  (注入 _session_init: 清理 node_sessions + messages + output fields)

isolated:
  START → _session_init → entry → ... → exit → END
  (注入 _session_init: 清理 node_sessions; session_key 已在 build 时强制唯一)
```

### _session_init 节点

DETERMINISTIC 类型节点，由框架内置（不需要 validators.py）。
根据 session_mode 执行不同的清理逻辑：

```python
# framework/nodes/session_init_node.py

def make_session_init(session_mode: str):
    """返回一个 DETERMINISTIC 节点函数，根据 session_mode 清理 state。"""

    def _fresh_init(state: dict) -> dict:
        """fresh_per_call: 清理所有 session 和临时字段。"""
        msgs = state.get("messages", [])
        human_msgs = [m for m in reversed(msgs) if getattr(m, "type", "") == "human"]
        fresh_msgs = [human_msgs[0]] if human_msgs else (msgs[-1:] if msgs else [])
        _topic = state.get("routing_context", "") or state.get("subgraph_topic", "")
        return {
            "node_sessions": {},
            "messages": fresh_msgs,
            "routing_context": "",
            "debate_conclusion": "",
            "apex_conclusion": "",
            "knowledge_result": "",
            "discovery_report": "",
            "previous_node_output": "",
            "subgraph_topic": _topic,
        }

    def _isolated_init(state: dict) -> dict:
        """isolated: 只清理 node_sessions。"""
        return {"node_sessions": {}}

    if session_mode == "fresh_per_call":
        return _fresh_init
    elif session_mode == "isolated":
        return _isolated_init
    else:
        return None  # persistent: 不需要
```

### 父图侧接入（全部统一）

```python
# agent_loader.py, 外部子图分支

# 不再区分 session_mode 的接入方式
# 全部用原生子图
inner_graph = await inner_loader.build_graph(
    checkpointer=None,
    is_subgraph=True,
    session_mode=session_mode,          # 新参数：传入 session_mode
    force_unique_session_keys=(session_mode == "isolated"),
)
builder.add_node(node_id, inner_graph)  # 始终原生子图
```

### _build_declarative 内部变化

在补 `START → entry` 的逻辑处（line 1035-1047），根据 session_mode 决定是否注入：

```python
_graph_entry = graph_spec.get("entry")

if is_subgraph and session_mode in ("fresh_per_call", "isolated"):
    init_fn = make_session_init(session_mode)
    init_id = "_session_init"
    builder.add_node(init_id, init_fn)
    builder.add_edge(START, init_id)
    builder.add_edge(init_id, _graph_entry)
elif _graph_entry and not _has_start_edge:
    builder.add_edge(START, _graph_entry)
```

## 好处

1. **所有子图对 LangGraph 可见** — `astream(subgraphs=True)` 对所有 mode 都有效
2. **清理逻辑可观察** — `_session_init` 是普通节点，debug flow log 能看到
3. **清理逻辑可测试** — 可以单独测试 `make_session_init()` 函数
4. **干掉 async wrapper** — 删除 `_fresh_wrapper`、`_isolated_wrapper`
5. **SubgraphInputState 可能可以简化** — 清理逻辑统一在 `_session_init`，schema 层 filter 可以放松

## 对现有代码的影响

### 删除

- `agent_loader.py` 中的 `_fresh_wrapper` async function
- `agent_loader.py` 中的 `_isolated_wrapper` async function
- `_get_subgraph_session_keys()` helper（isolated wrapper 用的，现在 build 时处理）

### 修改

- `agent_loader.py: _build_declarative()` — 新增 `session_mode` 参数，注入逻辑
- `agent_loader.py` 外部子图分支 — 统一用 `builder.add_node(id, inner_graph)`

### 新增

- `framework/nodes/session_init_node.py` — `make_session_init()` 工厂函数

### 不变

- 所有 entity.json — 声明式配置不需要改
- `SubgraphInputState` — 暂时保留，后续可以评估是否简化
- `inherit` mode — 仍然 NotImplemented

## 与 ColonyCoder 的关系

ColonyCoder 全部使用 persistent 模式（默认），不受此重构影响。
但此重构完成后，如果 ColonyCoder 被外部以 `fresh_per_call` 引用，
内部子图也能正确被 `subgraphs=True` 追踪。

## 与 Debug 功能的关系

此重构是 Debug 功能的前置条件之一（确保所有子图对 LangGraph 可见）。
但 ColonyCoder 的 debug 可以先做（它全是 persistent，不需要等此重构）。

实施顺序建议：
1. 先做 ColonyCoder debug（DebugConsoleReporter + astream subgraphs=True）
2. 再做此重构（统一子图接入方式）
3. 最后验证 debate 等 fresh_per_call 子图的 debug 也正常

## 风险

1. **checkpoint 行为变化** — async wrapper 阻止了 LangGraph 为子图创建 checkpoint namespace。
   改为原生子图后，LangGraph 会为每个子图创建独立 namespace。
   对于 `checkpointer=None` 的场景（colony_coder）无影响。
   对于有 checkpointer 的 role agents，需要验证 checkpoint 恢复行为。

2. **`_session_init` 节点的 state schema 兼容性** — 清理函数返回的 dict
   必须与子图的 state_schema 兼容（字段名、reducer）。
   `base_schema` 和 `colony_coder_schema` 都有 `node_sessions` 的 `_merge_dict` reducer，
   返回 `{}` 应该安全。需要测试 `debate_schema` 的 `messages` 的 `add_messages` reducer
   对 `fresh_msgs` 截断的行为。

3. **`push_graph_scope` / `pop_graph_scope` 的时机** — 当前 `_fresh_wrapper` 里有
   手动的 scope push/pop。改为原生子图后，scope tracking 由 `subgraphs=True` 的
   namespace tuple 提供，框架侧的 scope 栈不再需要手动管理。
   需要确认 `log_graph_flow` 和 `log_state_snapshot` 的目录结构是否需要适配。
