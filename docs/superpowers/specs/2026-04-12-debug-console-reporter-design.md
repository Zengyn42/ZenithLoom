# DebugConsoleReporter — 通用子图 Debug 可视化

> Date: 2026-04-12
> Status: Approved

## 目标

为任意 LangGraph 图提供实时的树形缩进 debug 输出 + markdown 日志，
利用 LangGraph 原生 `astream(subgraphs=True)` 追踪所有子图层级。

前置条件：unified subgraph integration 重构已完成（所有 session_mode 使用原生子图）。

## 范围

### 新增

| 文件 | 职责 |
|------|------|
| `framework/debug_reporter.py` | `DebugConsoleReporter` — 通用 debug 输出类 |
| `run_colony_coder_debug.py` | ColonyCoder + snake 游戏的 debug runner |

### 不改

| 文件 | 原因 |
|------|------|
| `framework/agent_loader.py` | 子图接入已是原生方式，不需要改 |
| `framework/debug.py` | 保留现有 `_wrap_node_for_flow_log` 的 flow.md 日志 |
| `run_colony_coder_e2e.py` | 保留，新 runner 是独立脚本 |
| 所有 entity.json | 不改 |

---

## DebugConsoleReporter 设计

### 数据源

LangGraph `astream(stream_mode="updates", subgraphs=True)` 返回：

```python
async for namespace, event in graph.astream(state, stream_mode="updates", subgraphs=True):
    # namespace: tuple[str, ...]
    #   () = 顶层
    #   ("plan:abc123",) = plan 子图
    #   ("plan:abc123", "design_debate:def456") = 嵌套子图
    # event: dict[str, dict]
    #   {node_id: {state_changes...}}
```

### Console 输出格式

树形缩进 + 时间戳，全量 LLM 输出：

```
14:23:01   [colony_coder] ▶ plan
14:23:01     [plan] ▶ _subgraph_init
14:23:01     [plan] ◀ _subgraph_init
14:23:02       [design_debate] ▶ claude_propose
14:23:15       [design_debate] ◀ claude_propose (2,341 chars)
               │ 我建议使用 MVC 架构来实现蛇对战游戏...
               │ 核心类: Snake, Food, GameBoard, AIController
               │ ... (2,341 chars total)
14:23:15       [design_debate] ▶ gemini_critique_1
14:23:28       [design_debate] ◀ gemini_critique_1 (1,892 chars)
               │ 这个方案有几个问题需要考虑...
14:25:03     [plan] ◀ design_debate
14:25:03     [plan] ▶ claude_swarm
14:25:30     [plan] ◀ claude_swarm (3,201 chars)
               │ 综合三个角度的评审...
14:25:30     [plan] ▶ task_decompose
14:25:45     [plan] ◀ task_decompose
               │ {"tasks": [{"id": "t1", ...}], ...}
               State: tasks=4 items, execution_order=['t1','t2','t3','t4']
14:25:45     [plan] ▶ decomposition_validator
               Route: → __end__
14:25:45     [plan] ◀ _subgraph_exit
14:25:45   [colony_coder] ◀ plan

14:25:45   [colony_coder] ▶ execute
14:25:45     [execute] ▶ inject_task_context
               State: current_task_id='t1', retry_count=0
14:25:46     [execute] ▶ code_gen
14:27:45     [execute] ◀ code_gen (8,234 chars)
               │ I'll implement the Snake class first...
14:27:46     [execute] ▶ run_tests
               State: execution_returncode=0
               Route: → __end__
14:27:46     [execute] ◀ _subgraph_exit
14:27:46   [colony_coder] ◀ execute
```

### 输出规则

1. **AIMessage 全量输出**：从 `changes["messages"]` 中提取 AIMessage，全文打印，`│` 前缀
2. **routing_target**：非空时打印 `Route: → target`
3. **其他非空 state 变化**：`key=value` 格式，跳过 messages 和空值
4. **子图进入/退出**：通过 namespace 深度变化检测，打印 `▶ subgraph_name` / `◀ subgraph_name`
5. **跳过 `__start__` / `__end__` 节点**

### 值格式化规则

| 类型 | 格式 |
|------|------|
| str（短，≤80 chars） | `key='value'` |
| str（长，>80 chars） | `key=value (N chars)` |
| list | `key=N items` |
| dict | `key={N keys}` |
| int / float / bool / None | `key=value` |

### API

```python
class DebugConsoleReporter:
    def __init__(self, graph_name: str):
        """
        Args:
            graph_name: 顶层图名称，用于 console 输出和 markdown 日志目录
        """
    
    def on_event(self, namespace: tuple, update: dict) -> None:
        """处理一个 astream(subgraphs=True) 事件。
        
        调用方直接在 async for 循环中调用此方法。
        """
    
    def print_summary(self) -> None:
        """打印最终执行摘要：耗时、节点数、成功/失败。"""
```

### 内部方法

```python
def _scope_name(self, namespace: tuple) -> str:
    """从 namespace 提取 scope 名称。
    ("plan:abc123",) → "plan"
    ("plan:abc123", "design_debate:def456") → "design_debate"
    () → self._graph_name
    """

def _depth(self, namespace: tuple) -> int:
    """namespace 长度 = 缩进深度。"""

def _format_value(self, value) -> str:
    """通用值格式化。"""

def _print_node_event(self, namespace, node_id, changes):
    """打印单个节点的进入 + 输出 + state 变化。"""

def _detect_scope_change(self, prev_namespace, curr_namespace):
    """检测子图进入/退出事件。"""

def _write_markdown(self, namespace, node_id, changes):
    """写入 markdown 日志文件。"""
```

### Markdown 日志

写入 `logs/YYYY-MM-DD/<graph_name>/debug_report.md`，单文件，按时间顺序追加。

格式：

```markdown
# Debug Report — colony_coder

## 14:23:01 — plan / design_debate / claude_propose

**Output** (2,341 chars):

```
我建议使用 MVC 架构来实现蛇对战游戏...
```

**State changes:**
- `routing_target` = `__end__`
- `tasks` = 4 items

---
```

与现有的 `flow.md` / `thinking.md` / `state_snapshots.md` 独立，不冲突。

---

## run_colony_coder_debug.py

独立的 debug runner 脚本，用于跑 ColonyCoder + snake 游戏任务：

```python
#!/usr/bin/env python3
"""ColonyCoder debug runner — 带完整 debug 输出跑 snake 游戏。"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

import blueprints.functional_graphs.colony_coder.state  # noqa: F401
from framework.agent_loader import EntityLoader
from framework.debug import set_debug
from framework.debug_reporter import DebugConsoleReporter

SNAKE_TASK = """用 Python 写一个双蛇对战游戏（Snake Battle）。
... (沿用 run_colony_coder_e2e.py 的现有 prompt)
"""

async def main():
    set_debug(True)
    
    loader = EntityLoader(Path("blueprints/functional_graphs/colony_coder"))
    graph = await loader.build_graph(checkpointer=None)
    
    reporter = DebugConsoleReporter("colony_coder")
    
    async for namespace, event in graph.astream(
        {"messages": [HumanMessage(content=SNAKE_TASK)]},
        stream_mode="updates",
        subgraphs=True,
    ):
        reporter.on_event(namespace, event)
    
    reporter.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
```

### 脚本行为

1. 启用 debug 模式
2. 构建 colony_coder 图（无 checkpointer）
3. 用 `subgraphs=True` 流式执行
4. 每个事件实时打印到 console
5. 同时写入 markdown 日志
6. 结束后打印执行摘要

---

## 与现有 debug 系统的关系

| 系统 | 数据源 | 输出 | 状态 |
|------|--------|------|------|
| `_wrap_node_for_flow_log` | 节点 wrapper | `flow.md` + `state_snapshots.md` | 保留，继续工作 |
| `log_node_thinking` | LlmNode 内部 | `thinking.md` | 保留，继续工作 |
| `DebugConsoleReporter` (新) | `astream(subgraphs=True)` | console + `debug_report.md` | 新增 |

三者可以同时工作，输出到不同位置，互不冲突。
`DebugConsoleReporter` 的优势是能看到完整的子图层级（通过 namespace），
而 `_wrap_node_for_flow_log` 的 flow.md 没有子图层级信息（因为 `push_graph_scope` 已移除）。
