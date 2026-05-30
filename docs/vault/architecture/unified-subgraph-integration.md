# Unified Subgraph Integration — Native Subgraph + Symmetric Init/Exit Nodes

> Date: 2026-04-12
> Status: Implemented
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

此外，`BaseAgentState` 使用自定义 `_keep_last_2` reducer 截断 messages，
而非 LangGraph 原生的 `add_messages`。这是因为子图内部 messages 会回流父图造成污染。

## 核心思路

1. **所有 session_mode 都使用 LangGraph 原生子图接入（`builder.add_node(id, CompiledStateGraph)`）。**
2. **State 清理逻辑不再用外部 wrapper，而是作为 DETERMINISTIC 节点对称注入到子图的 entry 前面和 exit 后面。**
3. **`_subgraph_exit` 统一清理所有子图的出口 messages，使父图可以安全使用 LangGraph 原生 `add_messages` reducer。**

## 架构

### 对称 Init/Exit 节点

所有子图（无论 session_mode）在构建时由 `_build_declarative()` 自动注入边界节点：

```
START → _subgraph_init → entry → ... → exit → _subgraph_exit → END
         (清理入口状态)                         (清理出口 messages)
```

- `_subgraph_init`：根据 session_mode 执行不同的入口清理
- `_subgraph_exit`：所有 mode 统一，RemoveMessage 清除子图内部所有 messages

### 四种 session_mode 规则

| session_mode | 首次 LLM 调用 | retry LLM 调用 | `_subgraph_exit` 行为 | 用途 |
|---|---|---|---|---|
| **persistent** | 新建 session | resume 自己的 session | RemoveMessage 所有 msgs | 子图跨调用保持 session |
| **inherit** | **fork 父 session**（`fork_session=True`） | resume 自己的 fork | RemoveMessage 所有 msgs + 清子图 session keys | 子图继承父图上下文，互不影响 |
| **fresh_per_call** | 新建 session | N/A（每次调用全新） | RemoveMessage 所有 msgs | 每次调用全新（debate 等） |
| **isolated** | 新建 session (unique key) | N/A | RemoveMessage 所有 msgs | 完全隔离 |

### persistent vs inherit 的区别

两者 init 都不清理 node_sessions，但 **session 创建方式**和**退出清理**完全不同：

- **persistent**：子图 LLM 节点创建全新 session（与父图无关）。checkpoint 保存子图的 session UUID。下次调用恢复 → 子图"记得"上次对话。退出时不清 session keys。

- **inherit**：子图 LLM 节点 **fork 父图的 session**（通过 Claude SDK `fork_session=True`）。子图看到父图完整对话历史，但后续对话不影响父 session。退出时**清掉子图的 session keys** → 下次调用重新 fork。

#### inherit 的 fork 语义

```
父图 session: uuid-A (technical_architect 与用户的完整对话)

子图 LLM 节点首次调用:
  node_sessions["apex_qa"] = ""（空）
  → sdk_query(resume=uuid-A, fork_session=True)
  → 得到 uuid-fork-qa（独立 session，起点 = uuid-A 的完整历史）
  → node_sessions["apex_qa"] = "uuid-fork-qa"

子图 LLM 节点 retry:
  node_sessions["apex_qa"] = "uuid-fork-qa"（非空）
  → sdk_query(resume=uuid-fork-qa, fork_session=False)
  → 继续自己的 fork session（记得上次做了什么，不会重复犯错）

子图退出 (_subgraph_exit):
  → 清掉 node_sessions["apex_qa"]、["apex_coder"]
  → 父图只保留 {"claude_main": "uuid-A"}
  → 下次调用子图时重新 fork
```

#### inherit 的独立性保证

同一子图内多个 LLM 节点（如 QA 和 Coder）各自 fork 父 session：

- QA fork → 看到父图对话 + QA 自己的对话
- Coder fork → 看到父图对话 + Coder 自己的对话（**看不到 QA 的 reasoning**）
- 通过 `reset_for_coder`（DETERMINISTIC 节点）清掉 QA 的 messages 保证 Coder 的 prompt 不含 QA 的思考

#### inherit_from 配置

子图 LLM 节点在 entity.json 声明要继承哪个父图 session：

```json
{
  "id": "claude_qa",
  "type": "CLAUDE_SDK",
  "session_key": "apex_qa",
  "inherit_from": "claude_main"
}
```

框架在 inherit 模式下：查找 `node_sessions[inherit_from]` 获取父 session UUID → 传给 call_llm 的 `inherit_from` 参数。

#### ClaudeSDKNode 实现

```python
async def call_llm(self, prompt, session_id="", inherit_from="", ...):
    if not session_id and inherit_from:
        # 首次 + inherit → fork 父 session
        options = ClaudeAgentOptions(resume=inherit_from, fork_session=True, ...)
    elif session_id:
        # 有自己的 session → resume（retry 场景）
        options = ClaudeAgentOptions(resume=session_id, ...)
    else:
        # 全新 session（独立运行场景）
        options = ClaudeAgentOptions(...)
```

### `_subgraph_exit` 按 mode 区分

| mode | _subgraph_exit 行为 |
|---|---|
| persistent | RemoveMessage 所有 msgs |
| **inherit** | RemoveMessage 所有 msgs + **清掉子图 session keys** |
| fresh_per_call | RemoveMessage 所有 msgs |
| isolated | RemoveMessage 所有 msgs |

inherit 模式下，`_subgraph_exit` 需要知道子图有哪些 session_key（构建时从 entity.json 提取）：

```python
def make_subgraph_exit(session_mode="persistent", subgraph_session_keys=None):
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

### `_subgraph_exit` 为何对 messages 统一清理

子图内部的 messages（辩论每轮发言、coder 每步输出等）对父图是噪声。父图只需要通过 state 字段（如 `debate_conclusion`）获取子图结论。

统一的 exit 清理使得：
1. 父图可以安全使用 LangGraph 原生 `add_messages` reducer（替换自定义 `_keep_last_2`）
2. 父图 messages 只包含父图自己节点的输出，不被子图内部 messages 污染
3. persistent 子图的 session 持久化不受影响（`node_sessions` 由 checkpoint 管理，与 messages 无关）

### `_subgraph_exit` 对 persistent checkpoint 的影响

```
第一次调用:
  子图执行 → node_sessions={"claude_propose": "uuid-1"}, messages=[m1, m2, m3]
  _subgraph_exit → messages=[]（RemoveMessage 清空）
  checkpoint 保存: node_sessions={"claude_propose": "uuid-1"}, messages=[]

第二次调用:
  checkpoint 恢复: node_sessions={"claude_propose": "uuid-1"}, messages=[]
  子图执行 → LLM 通过 uuid-1 恢复 session（对话历史在 SDK 内部管理）
```

messages 丢失不影响 LLM 对话连续性 — 对话历史由 LLM SDK session 管理（Claude SDK → `~/.claude/`，Gemini CLI → `~/.gemini/`），LangGraph messages 只是近期上下文窗口。

### `_subgraph_init` 实现

```python
# framework/nodes/subgraph_init_node.py

def make_subgraph_init(session_mode: str):
    """根据 session_mode 返回入口清理函数。persistent/inherit 返回 None（不注入）。"""

    if session_mode == "fresh_per_call":
        def _fresh_init(state: dict) -> dict:
            msgs = state.get("messages", [])
            removals = [RemoveMessage(id=m.id) for m in msgs]
            human_msgs = [m for m in reversed(msgs) if getattr(m, "type", "") == "human"]
            fresh = [HumanMessage(content=human_msgs[0].content)] if human_msgs else (
                [type(msgs[-1])(content=msgs[-1].content)] if msgs else []
            )
            _topic = state.get("routing_context", "") or state.get("subgraph_topic", "")
            return {
                "node_sessions": {},
                "messages": removals + fresh,
                "routing_context": "",
                "debate_conclusion": "",
                "apex_conclusion": "",
                "knowledge_result": "",
                "discovery_report": "",
                "previous_node_output": "",
                "subgraph_topic": _topic,
            }
        return _fresh_init

    elif session_mode == "isolated":
        def _isolated_init(state: dict) -> dict:
            return {"node_sessions": {}}
        return _isolated_init

    else:  # persistent, inherit
        return None
```

### `_subgraph_exit` 实现

```python
def make_subgraph_exit():
    """所有子图统一的出口清理：RemoveMessage 清除所有内部 messages。"""
    def _exit_cleanup(state: dict) -> dict:
        msgs = state.get("messages", [])
        return {"messages": [RemoveMessage(id=m.id) for m in msgs]}
    return _exit_cleanup
```

### `_build_declarative` 注入逻辑

```python
_needs_init = is_subgraph and session_mode in ("fresh_per_call", "isolated")
_needs_exit = is_subgraph  # 所有子图都注入 exit

# Entry side
if _needs_init and _graph_entry and not _has_start_edge:
    _init_fn = make_subgraph_init(session_mode)
    builder.add_node("_subgraph_init", _init_fn)
    builder.add_edge(START, "_subgraph_init")
    builder.add_edge("_subgraph_init", _graph_entry)
elif _graph_entry and not _has_start_edge:
    builder.add_edge(START, _graph_entry)

# Exit side
if _needs_exit and _graph_exit and not _has_end_edge:
    _exit_fn = make_subgraph_exit()
    builder.add_node("_subgraph_exit", _exit_fn)
    builder.add_edge(_graph_exit, "_subgraph_exit")
    builder.add_edge("_subgraph_exit", END)
elif _graph_exit and not _has_end_edge:
    builder.add_edge(_graph_exit, END)
```

### 父图侧接入（全部统一）

```python
# agent_loader.py, 外部子图分支 — 不再区分 session_mode 的接入方式
inner_graph = await inner_loader.build_graph(
    checkpointer=None,
    is_subgraph=True,
    session_mode=session_mode,
    force_unique_session_keys=(session_mode == "isolated"),
)
builder.add_node(node_id, inner_graph)  # 始终原生子图
```

## 附带变更

### 删除 `_keep_last_2` → 使用 `add_messages`

`BaseAgentState.messages` 从 `Annotated[list[BaseMessage], _keep_last_2]` 改为 `Annotated[list[BaseMessage], add_messages]`。

`_keep_last_2` 的存在是为了防止子图 messages 污染父图。现在由 `_subgraph_exit` 在源头解决，不再需要。

### 删除 `SubgraphMapperNode` 死代码

`SubgraphMapperNode`（`framework/nodes/subgraph/subgraph_mapper.py`）没有被任何 blueprint 引用，是死代码。其 `subgraph_topic` 管理职责已由 `_subgraph_init` 的 `fresh_per_call` 分支承担。

删除：
- `framework/nodes/subgraph/subgraph_mapper.py`
- `framework/builtins.py` 中的 `SUBGRAPH_MAPPER` 注册

### 删除 async wrappers

删除 `agent_loader.py` 中的：
- `_fresh_wrapper` async function
- `_isolated_wrapper` async function
- `_get_subgraph_session_keys()` helper
- `push_graph_scope` / `pop_graph_scope` 在 wrappers 中的调用

### 实现 `inherit` session_mode

`inherit` 不再是 `NotImplementedError`。语义：无 init 清理（继承父图 state），有 exit 清理（RemoveMessage）。与 persistent 的区别在于 checkpoint 行为。

## 好处

1. **所有子图对 LangGraph 可见** — `astream(subgraphs=True)` 对所有 mode 都有效
2. **清理逻辑可观察** — `_subgraph_init` / `_subgraph_exit` 是普通节点，debug flow log 能看到
3. **清理逻辑可测试** — 可以单独测试工厂函数
4. **干掉 async wrapper** — 删除 `_fresh_wrapper`、`_isolated_wrapper`
5. **使用 LangGraph 原生 `add_messages`** — 删除自定义 `_keep_last_2`
6. **对称设计** — init 处理入口，exit 处理出口，职责清晰
7. **inherit 可用** — 不再 NotImplementedError

## 对现有代码的影响

### 删除

- `agent_loader.py`: `_fresh_wrapper`、`_isolated_wrapper`、`_get_subgraph_session_keys()`
- `framework/schema/base.py`: `_keep_last_2` 函数
- `framework/nodes/subgraph/subgraph_mapper.py`: 整个文件
- `framework/builtins.py`: `SUBGRAPH_MAPPER` 注册

### 修改

- `agent_loader.py: _build_declarative()` — 新增 `session_mode` 参数，注入 init/exit 节点
- `agent_loader.py: build_graph()` — 新增 `session_mode` 参数
- `agent_loader.py` 外部子图分支 — 统一用 `builder.add_node(id, inner_graph)`
- `framework/schema/base.py` — `_keep_last_2` → `add_messages`
- `framework/builtins.py` — 删除 SUBGRAPH_MAPPER
- 注释更新：`llm_node.py`、`gemini.py`、`base.py` 中对 SubgraphMapperNode 的引用

### 新增

- `framework/nodes/subgraph_init_node.py` — `make_subgraph_init()` + `make_subgraph_exit()` 工厂函数

### 不变

- 所有 entity.json — 声明式配置不需要改
- `SubgraphInputState` — 暂时保留，后续可以评估是否简化

## 风险

1. **persistent checkpoint + exit 清理** — `_subgraph_exit` 清空 messages 后，checkpoint 保存的 messages 为空。
   对话历史由 LLM SDK session 管理，不依赖 LangGraph messages。但需要验证 persistent 子图（colony_coder）
   在 messages=[] 恢复后是否正常工作。

2. **`add_messages` reducer + RemoveMessage 兼容性** — 已验证：LangGraph 1.0.10 中
   RemoveMessage 只在所属 reducer scope 内生效，不传播到父图。子图内部 RemoveMessage 不影响父图 messages。
   注意：RemoveMessage 对不存在的 ID 会抛异常，`_subgraph_exit` 只删除实际存在的 messages，安全。

3. **`_subgraph_init` 的 state schema 兼容性** — 清理函数返回的 dict 必须与子图的 state_schema 兼容。
   `debate_schema` 的 `add_messages` reducer 接受 RemoveMessage + 新消息列表（已验证）。
   `base_schema` 改为 `add_messages` 后行为一致。
