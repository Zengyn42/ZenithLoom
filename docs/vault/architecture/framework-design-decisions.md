# 框架设计决策记录

> 最后更新: 2026-03-18

记录 BootstrapBuilder 框架在实际开发中形成的设计规则，每条都有对应的 bug 或需求作为来源。

---

## 决策 1: State 字段所有权

> 来源: colony_coder 392 次死循环

**规则**: 每个 state 字段有且仅有一种节点类型可以写入。

| 字段 | 所有者 | 禁止写入 |
|------|--------|----------|
| retry_count | DETERMINISTIC validator | LLM 节点 |
| routing_target | 所有节点（各自设置自己的路由） | — |
| tasks, execution_order | DETERMINISTIC validator (解析后填充) | LLM 节点 (输出在 message 里) |
| messages | 所有节点 | — |
| node_sessions | LLM 节点 (写自己的 session_key) | — |

**原因**: LlmNode 返回 `retry_count: 0` 覆盖了 validator 的递增值，导致无限循环。

---

## 决策 2: tools=[] 表示只读模式

> 来源: planner/debate 节点无视 system prompt 使用工具写文件

**规则**: node_config 中 `"tools": []` 表示"只读模式" — 禁止写入/执行类工具，保留只读工具。

**语义**:
| 配置 | 含义 |
|------|------|
| `"tools"` 不存在 | 使用全局工具集（config.tools） |
| `"tools": ["Read", "Grep"]` | 仅允许指定工具 |
| `"tools": []` | 只读模式：禁止 Write/Edit/Bash 等，保留 Read/Glob/Grep/WebSearch/WebFetch |

**实现（两层）**:

1. `llm_node.py _select_tools()` — 区分"未配置"和"配置为空列表"：
```python
_MISSING = object()
node_tools = self._cfg.get("tools", _MISSING)
if node_tools is _MISSING:
    tools = list(self.config.tools)  # 未配置 → 用全局
else:
    tools = list(node_tools or [])   # 显式配置 → 用配置值
```

2. `claude.py _make_options()` — 空列表时注入 disallowed_tools：
```python
_WRITE_TOOLS = ["Write", "Edit", "MultiEdit", "Bash", "TodoWrite", "NotebookEdit"]
if isinstance(_allowed, list) and len(_allowed) == 0:
    _disallowed = _WRITE_TOOLS  # 禁止写入，保留只读
```

**原因**:
- Claude SDK `allowed_tools=[]` 表示"未指定"（默认全部允许），不能用它禁工具
- 辩论节点需要搜索网络、读取文件来查资料，但绝不能写文件
- System prompt "禁止使用工具" 不可靠，Claude 会无视

---

## 决策 3: LLM 输出解析由 DETERMINISTIC 节点负责

> 来源: task_decompose 输出 JSON 但 validator 不识别

**规则**: LLM 节点只往 `messages` 写 AIMessage。结构化数据的提取（JSON 解析、字段填充）由下游 DETERMINISTIC 节点完成。

**原因**:
- LLM 输出格式不可靠（可能包含 markdown 围栏、前后文说明）
- 解析逻辑与验证逻辑天然耦合（解析失败 = 验证失败 = retry）
- DETERMINISTIC 节点是确定性的，便于测试和调试

---

## 决策 4: 辩论子图 session 是临时的

> 来源: 架构设计

**规则**: SUBGRAPH_REF 引用的辩论子图运行结束后，其内部 session 文件被清理。辩论结论通过 `messages` 和 `state_out` 映射传递给父图，不通过 session 继承。

**实现**: `SubgraphRefNode._cleanup_orphan_sessions()` 在子图结束后：
1. 找出 `node_sessions` 中新增的 key
2. 删除对应的 Gemini session JSON 文件
3. 删除对应的 Claude session 目录

**原因**: 辩论子图可能被多次调用（不同议题），旧 session 会污染新辩论的上下文。

---

## 决策 5: SubgraphRefNode routing_context fallback

> 来源: planner 中 debate 子图收不到任务内容

**规则**: 当 `state["routing_context"]` 为空时，SubgraphRefNode 自动 fallback 到父图 `messages[-1].content` 作为子图输入。

**原因**: `routing_context` 是主图路由信号专用字段。当子图作为固定管线的一环（非路由触发）时，任务内容在 messages 里，不在 routing_context 里。

---

## 决策 6: Token 安全阀是 node-level 的

> 来源: colony_coder 瞬间耗光 token

**规则**: 每个 LLM 节点在 `call_llm()` 之前检查累计 token。限额按 3 级优先级配置：

```
node_config["token_limit"]  >  按 type 默认值  >  BB_TOKEN_LIMIT 环境变量
```

**原因**:
- 云端 API (Claude/Gemini) 计费，限额需要紧（50k）
- 本地推理 (Ollama) 不计费，限额可以松（1M）
- 特殊节点可能需要更大/更小的限额，支持 node_config 覆盖

---

## 决策 7: Debug 日志按图层级目录存放

> 来源: 老板要求

**规则**: 日志存为 `.md` 文件，目录结构反映图的嵌套层级。

```
logs/2026-03-18/
  colony_coder/                    ← master 图
    flow.md                        ← 节点进出、路由
    thinking.md                    ← LLM 思考内容
    colony_coder_planner/          ← planner 子图
      flow.md
      thinking.md
      design_debate/               ← debate 子子图
        flow.md
        thinking.md
```

**实现**: `ContextVar _graph_scope` 维护当前图层级栈，`push_graph_scope(name)` / `pop_graph_scope()` 在 GraphController 和 SubgraphRefNode 中管理。

---

## 决策 8: 异构辩论优于自我辩论

> 来源: planner 测试对比

**规则**: 需要设计评审时，优先使用 `debate_claude_first`（Claude 提案 + Gemini 审查）而非多个 Claude 节点自我辩论。

**证据**:
- Claude 自我辩论时，3 个节点产出高度重复的内容，没有真正的对抗
- Claude + Gemini 辩论中，Gemini 从不同维度（跨平台兼容性、渲染性能、边界 case）提出了 Claude 未考虑的问题
- 最终 task_decompose 产出的任务更完整、更健壮
