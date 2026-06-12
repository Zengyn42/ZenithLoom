# ZenithLoom Observability — 节点级实时可观测 + 执行回放 设计文档

> Status: APPROVED (经 debate_design 4 轮辩论收敛, 2026-06-11)
> Author: Hani · 无垠智穹
> Scope: P1–P4 全量设计；本文档同时是 ApexCoder 的实现规约（P1 为当前执行范围）

---

## 1. 目标与非目标

### 目标
- **实时可观测**：浏览器中以图形式实时查看每个 Agent 的 LangGraph 执行——当前节点高亮、节点 start/end、路由决策、（P3）LLM 流式输出与 token 消耗
- **执行回放**：基于 SQLite checkpoint 历史，时间轴拖动回放任意 thread 的执行过程，节点间 state diff 呈现
- **多 Agent 汇聚**：hani/asa/dan/jei 四个独立进程的事件汇聚到单一 Server、单一前端

### 非目标（明确排除）
- ❌ 可视化编辑器（只读，永不写回 blueprint）
- ❌ 自建事件持久化层（checkpoint + 内存 ring buffer 覆盖全部需求，YAGNI）
- ❌ 多用户/鉴权（单用户本地 WSL 环境）

### 硬约束
- **核心执行流永不因观测而阻塞或失败**：Server 不在线时 Agent 完全正常运行
- framework 侵入量最小化：P1 ≈ 15 行，P3 ≈ 10 行，P2/P4 零侵入

---

## 2. 已验证的代码事实（实现的接入点）

| 事实 | 位置 | 对设计的影响 |
|------|------|-------------|
| `GraphController._astream_graph` 当前用 `stream_mode="values", subgraphs=False`，收集最终 state，空帧时 fallback `ainvoke` | `framework/graph_controller.py` (~L76) | tap 需改为多模式 `["values", "updates", "debug"]` + `subgraphs=True`，**必须保留 values 最终态收集与 fallback 逻辑** |
| LLM 流式回调是 ContextVar 机制：`_stream_cb` / `_channel_send_cb`，签名 `(text: str) -> None` | `framework/nodes/llm/llm_node.py` (~L85) | P3 桥接点；注意 callback 可能是 sync 函数 |
| `AgentGraph` / `NodeSpec` / `EdgeSpec` dataclass | `framework/loader/graph_spec.py` | 前端拓扑数据源；用 `dataclasses.asdict` 序列化即可 |
| checkpointer 为 SQLite（`hani.db` 等，位于 `EdenGateway/agents/<name>/`），`get_state_history` 原生支持 | `framework/graph_controller.py` `_exec_on_checkpointer` | 回放唯一数据源；Server 以 **read-only** 模式打开 |
| `push_graph_scope(entity_name)` 在 `run()` 中包裹执行 | `framework/graph_controller.py` `run()` | agent_name 可从此获取或直接用 `self._entity_name` |
| 多模式 + subgraphs=True 时 astream 事件形状为 `(namespace_tuple, mode, event)` | LangGraph 行为 | tap 解包逻辑见 §4.2 |

---

## 3. 总体架构

```
┌─ agent 进程 (hani/asa/dan/jei, 各自独立) ──────────────────┐
│  GraphController._astream_graph                            │
│      │ 多模式事件 (ns, mode, event)                         │
│      ▼                                                     │
│  observability client (framework/observability_client.py)  │
│   - 白名单过滤（emit 之前）                                  │
│   - 有界 asyncio.Queue(4096)，满时丢最旧                     │
│   - 后台 Task → WebSocket 推送 JSON lines                   │
│   - Server 不在线：静默禁用 + 指数退避重连                    │
└──────────────┬──────────────────────────────────────────────┘
               │ ws://127.0.0.1:8765/ingest
               ▼
┌─ ObservabilityServer (独立进程, observability/) ────────────┐
│  FastAPI + uvicorn                                          │
│   /ingest        ← agent clients (WS, JSON lines)           │
│   /ws/events     → 前端订阅实时事件 (WS, 广播)                │
│   /api/agents    → 在线 agent 列表 + 最新 graph_snapshot     │
│   /api/history/{agent}/{thread_id}  → checkpoint 历史摘要    │
│   /api/diffs/{agent}/{thread_id}    → 预计算 JSON patch 序列 │
│   ProcessPoolExecutor(2) + LRU 缓存 → diff 计算              │
│   内存 ring buffer (P3, per-agent 最近 20 run 的 llm_chunk)  │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌─ Web 前端 (observability/frontend/, React Flow) ────────────┐
│   只读拓扑渲染 (NodeSpec→Node, EdgeSpec→Edge)                │
│   实时节点状态高亮 / (P2) 时间轴回放 + diff / (P3) 消息侧栏    │
└─────────────────────────────────────────────────────────────┘
```

部署：Server 由 systemd user unit `zl-observability.service` 管理，与各 agent unit 并列。前端为静态构建产物，由 FastAPI `StaticFiles` 直接托管（不需要独立 node server）。

---

## 4. 详细设计

### 4.1 事件 Schema

```python
# observability/schema.py（Server 与 client 共享；client 侧在 framework 内复制一份极简版以避免反向依赖）
@dataclass
class ObservEvent:
    v: int                  # schema 版本，从 1 开始
    agent_name: str         # "hani" 等
    thread_id: str          # checkpoint 溯源必需
    run_id: str             # 每次 controller.run() 生成 uuid4
    checkpoint_ns: str      # 取自 LangGraph 事件 namespace（实时↔回放 ID 缝合的桥梁）
    node_id: str            # 节点名；graph 级事件为 "__graph__"
    event_type: str         # graph_snapshot | run_start | run_end |
                            # node_start | node_end | state_update | llm_chunk
    payload: dict           # 类型相关数据（见下表）
    timestamp: float        # time.time()
    seq: int                # per-client 单调递增，前端排序 + 丢失检测
```

| event_type | payload | 来源 |
|------------|---------|------|
| `graph_snapshot` | `{"graph": asdict(AgentGraph)}` | client 启动时 + Server 重连成功时各发一次 |
| `run_start` / `run_end` | `{"input_preview": str(≤200字符)}` / `{}` | `run()` 包裹 |
| `node_start` / `node_end` | `{"ns": [...]}` | debug 模式 `on_chain_start/end`（白名单） |
| `state_update` | `{"node": ..., "keys_changed": [...]}` — **只发 key 列表与摘要，不发全量 state** | updates 模式 |
| `llm_chunk` (P3) | `{"text": chunk}` | stream callback 桥接 |

线缆格式：JSON lines over WebSocket text frame。payload 中任何值序列化失败时降级为 `repr()` 截断 1000 字符——**序列化错误绝不抛出到执行流**。

### 4.2 EventBus tap（P1，framework 唯一改动点）

`_astream_graph` 改造（保持原契约：返回最终 state；fallback 保留）：

```python
ALLOWED_DEBUG_TYPES = {"checkpoint", "task", "task_result"}  # LangGraph debug 事件类型
# 注：LangGraph debug 模式的事件 type 实际为 task/task_result/checkpoint；
# task=节点开始, task_result=节点结束。实现时以实际版本行为为准，
# 原则不变：白名单只放行拓扑级生命周期事件，其余在 emit 之前丢弃。

async for chunk in self._graph.astream(
    init_state, config=config,
    stream_mode=["values", "updates", "debug"],
    subgraphs=True,
):
    ns, mode, event = chunk          # 多模式+subgraphs=True 的事件形状
    if mode == "values":
        final_state = event          # 原有最终态收集逻辑不变
        continue
    obsv_client.tap(ns, mode, event,
                    agent_name=self._entity_name,
                    thread_id=self._active_thread_id,
                    run_id=run_id)   # 内部做白名单过滤 + emit_nowait，永不抛异常
```

`framework/observability_client.py`（新文件，~150 行，不算侵入）：
- `tap()`：mode == "debug" 且 type 不在白名单 → 直接 return（零成本，在入队之前）
- `emit_nowait()`：`queue.put_nowait`，`QueueFull` 时 `get_nowait()` 丢最旧再放入，并自增 `dropped` 计数器
- 后台 sender Task：连接 `ws://127.0.0.1:8765/ingest`；断线指数退避（1s→2s→…→60s 封顶）；未连接时事件仍入队（队列即天然缓冲）
- 开关：环境变量 `ZL_OBSERV_URL`（默认 `ws://127.0.0.1:8765/ingest`），设为空字符串则整个 client 为 no-op
- 生命周期：`GraphController.run()` 入口发 `run_start`，结束（finally）发 `run_end`

### 4.3 ObservabilityServer（P1 框架 + P2 回放）

目录：`ZenithLoom/observability/`

```
observability/
├── server.py            # FastAPI app：ingest/ws/api 路由
├── schema.py            # ObservEvent + 序列化
├── hub.py               # 内存事件枢纽：agent registry、前端广播、ring buffer(P3)
├── replay.py            # P2：checkpoint 读取 + diff 计算（进程池 worker 函数）
├── frontend/            # React + React Flow（Vite 构建，产物进 frontend/dist）
├── requirements.txt     # fastapi uvicorn websockets (P2: +langgraph-checkpoint-sqlite, deepdiff/jsonpatch)
└── zl-observability.service  # systemd user unit 模板
```

核心逻辑：
- `/ingest`：接收 agent client 事件 → 更新 agent registry（最新 graph_snapshot、在线状态、last_seq）→ 广播给所有 `/ws/events` 订阅者
- `/ws/events`：前端连接时先收到全量 `agents + graph_snapshot` 初始化包，之后收实时事件
- 前端断线不影响 ingest；ingest 断线将该 agent 标记 offline（前端灰显）

P2 回放：
- agent 的 db 路径在 Server 配置中静态声明（`config.toml`：`{name: db_path}` 映射，指向 `EdenGateway/agents/<name>/<name>.db`）
- `replay.py::compute_diffs(thread_id, db_path)`：read-only 打开 SQLite → `get_state_history` → 相邻 checkpoint 间生成 JSON patch 序列（jsonpatch 格式）→ 返回 `[{checkpoint_id, checkpoint_ns, ts, node, patch}, ...]`
- 必须在 `ProcessPoolExecutor(max_workers=2)` 中执行；结果 LRU 缓存，键 `(thread_id, checkpoint_count)`
- 前端时间轴消费 patch 序列增量应用，不拉全量 state
- 实时↔回放 ID 缝合：前端维护 `Map<checkpoint_ns, seq_range>`，从实时视图点击节点执行记录 → 定位对应 checkpoint diff

### 4.4 前端（P1 静态拓扑 + 状态高亮）

- React + Vite + React Flow + zustand（状态管理），TypeScript
- 映射：`NodeSpec → ReactFlow Node`（type 决定节点样式：llm/subgraph_ref/builtin…），`EdgeSpec → ReactFlow Edge`（conditional 边虚线）；layout 用 dagre 自动布局
- 节点状态机：`idle(灰) → running(蓝色脉动) → done(绿) → error(红)`；由 `node_start/node_end` 事件驱动，`run_end` 后 3s 全部归位 idle
- 子图（SUBGRAPH_REF）：P1 渲染为单个复合节点 + namespace 事件徽标计数；展开内部拓扑放 P2+
- Agent 切换：顶部 tab（hani/asa/dan/jei + 在线状态点）

### 4.5 LLM 流式桥接（P3，framework 第二处改动 ~10 行）

`LlmNode.__call__` 入口包装 ContextVar callback（注意现有 callback 是 **sync** 签名 `(text)->None`）：

```python
if obsv_client.enabled():
    _orig = get_stream_callback()
    def _observed(chunk: str):
        obsv_client.emit_llm_chunk(chunk, node_id=..., ...)  # nowait, 永不抛
        if _orig:
            _orig(chunk)
    set_stream_callback(_observed)
```

Server 端 ring buffer：per-agent 最近 20 个 run 的 llm_chunk，内存保存，不落盘。

---

## 5. 分阶段交付与验收标准

| 阶段 | 交付物 | framework 改动 | 验收标准 |
|------|--------|---------------|---------|
| **P1** | observability_client.py + tap 改造 + Server (ingest/ws/agents) + React Flow 静态拓扑 + 节点状态高亮 + systemd unit | `graph_controller.py` ~15 行 | ① Server 关闭时 4 agent 全部正常运行（回归 `test_cli.py` 8/8）② 浏览器打开能看到 hani 拓扑 ③ Discord 发一条消息，对应节点实时高亮流转 ④ kill Server 再重启，client 自动重连恢复 |
| **P2** | replay.py + diffs API + 前端时间轴 + state diff 渲染 + checkpoint_ns 缝合 | 零 | ① 选历史 thread 拖时间轴可逐步回放 ② diff 计算不阻塞实时事件广播（进程池验证） |
| **P3** | stream callback 桥接 + 消息侧栏 + token 角标 + ring buffer | `llm_node.py` ~10 行 | ① 点节点看到实时流式输出 ② 原 Discord 流式输出行为不变 |
| **P4** | 多 Agent 总览仪表盘 + 跨 agent 时间线 | 零 | 一屏看 4 agent 状态 |

## 6. 测试要求（P1）

- 单测：白名单过滤器（debug 垃圾事件不入队）、队列满丢最旧、序列化失败降级、`ZL_OBSERV_URL=""` 时 no-op
- 集成：mock astream 事件流 → client → 内存 Server → 断言前端收到的事件序列与 seq 连续性
- 回归：`tests/test_cli.py` 全过；Server 离线场景下 agent run 耗时无显著增加

## 7. 风险与备注

- LangGraph 多模式 astream 的事件元组形状随版本有差异（`(ns, mode, event)` vs `(mode, event)`），实现时先用本仓库实际安装版本写一个最小验证脚本确认形状再接入
- debug 模式事件的实际 type 名需对照本仓库 LangGraph 版本验证（见 §4.2 注）
- `state_update` 不发全量 state 是带宽与隐私的双重考虑；详情留给 P2 回放（数据源是 checkpoint，更完整）
