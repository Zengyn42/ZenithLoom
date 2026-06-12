# Observability v2 — JSONL 真相源 + Sprite 像素办公室

> 状态：**待老板审批**（设计先行，暂不实现）
> 日期：2026-06-11
> 来源：老板三点指令 + debate_design 四轮辩论收敛（方案 D v1.2）
> 前置：[observability-design.md](observability-design.md)（v1，将被本方案替代）

## 老板指令

1. 更新底层接口设计
2. 去掉现在的 local HTTP 底层接口（移除 FastAPI/HTTP server 层）
3. 前端升级为 sprite 像素小人版本（参考 pixel-agents），先设计后实现

## 裁决：方案 D — JSONL 真相源 + 可选极薄 Viewer

agent 写 JSONL 文件作为唯一真相源；一个无状态极薄 viewer 进程 tail 文件转 WS 给浏览器（可抛弃组件，非 server）。

**三条核心理由：**

1. **老板意图对齐**：pixel-agents 模式（JSONL + 零 server）的精确执行。去掉的是 FastAPI/HTTP REST 层，保留的 viewer 是无状态管道。
2. **No-op 安全最大化**：agent 侧 `emit()` 最终形态 = `json.dumps() + deque.append()`，两个纯内存操作，数学上不可能阻塞 event loop。`ZL_OBSERV_DIR=''` 时整个模块空壳。
3. **复杂度最低**：agent 侧仅重写 `observability_client.py` 一个文件；`graph_controller.py` 零改动（tap 点签名不变）。前端独立解耦。

## 一、Agent 侧写入端

```python
class ObservabilityClient:
    def __init__(self, agent_name: str, obs_dir: str | None = None):
        if not obs_dir:
            self._enabled = False; return
        self._enabled = True
        self._agent = agent_name
        self._dir = Path(obs_dir)
        self._buffer = collections.deque(maxlen=8192)  # 线程安全 ring buffer
        self._notify = threading.Event()
        self._active_sessions: dict[str, dict] = {}     # thread_id → last run_start event
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._current_file, self._current_size = self._open_file()
        self._write_restart_marker()                    # agent_restart 锚点（清僵尸会话）
        atexit.register(self._graceful_flush)           # 临终 500ms 强制 drain
        self._writer.start()

    def emit(self, event: dict) -> None:
        """纯内存操作，绝不阻塞 event loop"""
        if not self._enabled: return
        try:
            line = json.dumps({"ts": time.time(), "agent": self._agent, **event}) + '\n'
            self._buffer.append(line)
            if event.get("event_type") == "run_start":
                self._active_sessions[event["thread_id"]] = event
            elif event.get("event_type") == "run_end":
                self._active_sessions.pop(event.get("thread_id", ""), None)
            self._notify.set()
        except Exception:
            pass
```

**辩论定型的关键设计决策：**

| 设计点 | 决策 |
|--------|------|
| I/O 隔离 | `deque(maxlen=8192)` + daemon 写入线程，`emit()` 零磁盘 I/O |
| 临终数据保障 | `atexit.register(self._graceful_flush)` — 500ms 超时强制 drain + flush |
| 轮转时机 | 写入时持续检测 `_current_size >= 10MB` + 启动时兜底 |
| 轮转断层修复 | 轮转时对 `_active_sessions` 中每个未结束 session 写 `session_resume` 锚点到新文件头部 |
| 僵尸会话防御 | 进程启动写 `agent_restart` 事件，作废之前所有未闭环 session |

轮转实现：close → rename 到 `archive/{agent}.events.{ts}.jsonl` → 后台线程 gzip → 新文件头部重写活跃 session 的 `session_resume`。

## 二、JSONL 文件布局与事件 schema

```
~/Foundation/EdenGateway/observability/
├── hani.events.jsonl
├── asa.events.jsonl
├── jei.events.jsonl
└── archive/
    └── hani.events.1749600000.jsonl.gz
```

**v2 新增事件类型**（v1 的 run_start/run_end/node_start/node_end schema 不变）：

| event_type | 触发时机 | 用途 |
|---|---|---|
| `agent_restart` | 进程启动 | viewer 作废此前所有未闭环 session |
| `session_resume` | 文件轮转 | 单文件自包含，viewer 无需跨文件扫描 |

## 三、Viewer（无状态 + Initial Snapshot，< 200 行）

- 启动：`chokidar.watch(obs_dir + '/*.events.jsonl')`
- 新 WS 连接：每个文件从末尾反向扫描（≤2000 行）→ 按 `agent_restart`/`run_end` 闭环规则重建活跃 session → 发 `{"type":"snapshot","active_runs":[...]}` → 进入 tail→broadcast
- 轮转感知：chokidar `unlink`+`add` 自动切换 watcher
- **文件即状态**：viewer 无内存状态，随时 kill 重启零丢失
- 无 REST / 无路由 / 无数据库

## 四、前端：Sprite 像素办公室

**双消费路径**：Electron/VSCode webview 直接 `fs.watch`（零 viewer）；浏览器连 viewer WS。

**事件 → Sprite 状态机**（每个小人 = 一个 `{agent}:{thread_id}`）：

```
snapshot 恢复     → 直接就位（无过渡动画）
agent_restart    → 清空该 agent 所有 sprite
run_start        → ARRIVING（门口走向空工位）
session_resume   → 同 run_start
node_start       → WORKING(node_type)：
                    CLAUDE_SDK → 蓝光敲键盘 | GEMINI_API → 绿光敲键盘
                    SUBGRAPH_REF → 打电话 | HEARTBEAT → 巡逻扫描
node_end         → 200ms BRIEF_PAUSE
run_end          → DEPARTING → IDLE
```

**GC 回收**：30s 无事件 → IDLE（眨眼）；5min → LEAVING（收拾离场 + destroy）；硬上限 MAX_SPRITES=20，超出强制淘汰最久 IDLE。

**渲染**：PixiJS（2D WebGL）。工位网格 4×5=20。Agent 颜色：hani=蓝、asa=橙、jei=绿、dan=紫。素材复用 pixel-agents sprite sheet（**需先确认 license**，否则自制/购买）。

## 五、环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ZL_OBSERV_DIR` | `''`（禁用） | 设路径启用 v2；空 = 全 no-op |
| `ZL_OBSERV_URL` | 保留 | v1 兼容，迁移期 feature flag |
| `ZL_OBSERV_VERSION` | `1` | `2` = v2 client |

## 六、改动文件清单

| 文件 | 改动 |
|------|------|
| `framework/observability_client.py` | 重写：asyncio.Queue+WS → deque+daemon thread+JSONL |
| `framework/graph_controller.py` | **零改动**（`emit_run_start`/`tap`/`emit_run_end` 签名不变） |
| `observability/server.py` + `hub.py` | 迁移完成后删除 |
| `observability/viewer.py`（新） | 极薄 tail + WS 桥接 |
| `observability/frontend/` | 重写：React Flow → PixiJS sprite 办公室 |
| systemd units | 加 `ZL_OBSERV_DIR` / `ZL_OBSERV_VERSION` |

## 七、迁移步骤

| # | 步骤 | 风险 | 回退 |
|---|------|------|------|
| 1 | 新建 `observability_client_v2.py`，与 v1 并存 | 零 | 删文件 |
| 2 | `graph_controller.py` 加 `ZL_OBSERV_VERSION` flag | 低 | 改 env |
| 3 | viewer 搭建，验证 tail + snapshot + WS | 低 | 独立进程 |
| 4 | PixiJS sprite 前端，mock JSONL 调试 | 零 | 纯前端 |
| 5 | 三 agent 切 v2，观察 JSONL 输出 | 低 | env 回退 |
| 6 | 稳定一周后删 v1 client + FastAPI server | 末步 | git revert |

## 八、风险

1. **pixel-agents 素材 license** — 动工前必须确认（MIT/Apache 可用，否则自制/购买）
2. **LangGraph astream 多模式元组形状随版本变化** — 当前安装版本上跑验证脚本
3. **SIGKILL 不可捕获** — atexit 对 `kill -9` 无效，靠下次启动的 `agent_restart` 清僵尸（最优工程妥协）

## 点睛

本方案的核心价值：observability 的可靠性边界从「网络层」退回到「文件系统层」。v1 用 WebSocket+FastAPI+Hub 三层做到的事，v2 用 `open('a') + write + flush` 一层做到，其余全是可选装饰。

— Hani · 无垠智穹
