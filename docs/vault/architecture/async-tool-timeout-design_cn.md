# External Tool 超时异步化设计文档

> 最后更新: 2026-03-24
> 来源: 三轮 Claude-Gemini Design Review (debate_design)
> 状态: **已批准，待实现**

---

## 一、问题陈述

当前 LangGraph 主循环在执行 external tool（Claude CLI、Gemini CLI、shell 命令等子进程调用）时同步阻塞。工具执行时间不可控时，整个 graph 卡住，用户无法得到任何响应。

**现有超时机制的局限**：各节点有 hard timeout（30s~300s），超时直接 kill 子进程——工作成果全部丢失。

---

## 二、最终方案

**一句话概括：启动时 tee + soft_timeout 后不 kill 而是注册 Heartbeat 监控 + 结果写入 Task Vault + 响铃通知用户手动恢复。**

### 核心数据流

```
ExternalToolNode 启动子进程（tee 到 buffer + BoundedFile）
        │
        ├─ 子进程在 soft_timeout 内完成 → 正常返回 ToolMessage
        │
        └─ soft_timeout 到达 →
              ① 返回 ToolMessage(content="[PENDING]", metadata={task_id})
              ② AsyncTaskManager 向 Heartbeat 注册监控任务
              ③ Heartbeat 非阻塞检测子进程完成 → 结果写入 Task Vault
              ④ SSE push "task_completed" → BaseInterface 后台线程收到
              ⑤ 响铃 \a 提醒用户
              ⑥ 用户下次输入时，自动从 Vault 取结果，覆写 PENDING 消息
              ⑦ 发起新的 graph.invoke()，传入修正后的完整历史
```

---

## 三、V1 范围

| 做 | 不做（V2） |
|---|---|
| 纯子进程工具（shell、Gemini CLI）后台化 | Claude SDK 后台化（仅加进度提示） |
| 单任务后台限制 | 多任务并发 |
| 响铃 + 用户手动触发结果注入 | 自动 resume |
| PID Registry + atexit 清理 | process group 方案 |
| BoundedFileWriter 50MB 截断保护 | 动态调整截断阈值 |

---

## 四、关键设计决策

| # | 决策 | 选择 | 核心理由 |
|---|------|------|---------|
| 1 | IO 策略 | 启动时 tee，非中途切换 | 中途重定向 pipe 有数据丢失和 broken pipe 风险；tee 方案简单可靠 |
| 2 | Graph 挂起机制 | `[PENDING]` 消息覆写 + 新 invocation | 不依赖 LangGraph 特定版本；与同步 CLI 模型兼容；调试直观 |
| 3 | 结果回注触发 | 响铃 + 用户手动触发（非自动 resume） | BaseInterface CLI 侧是同步阻塞模型，改造为事件驱动成本过高；V1 低风险妥协 |
| 4 | Claude SDK 处理 | V1 不后台化，仅进度提示 | 嵌套 tool use 的后台自主执行涉及安全性问题，复杂度不可控 |
| 5 | 并发后台任务 | V1 单任务限制 | 避免乱序注入和 state 一致性问题；实际场景中并发长任务极罕见 |
| 6 | 超时管理 | `call_later` 回调，非 Min-Heap | 后台任务数量 ≤3，简单回调足够；Min-Heap 属过度设计 |
| 7 | 进程清理 | PID Registry + atexit 逐个清理 | 避免 `os.setsid` 的 macOS 兼容性问题；跨平台安全 |
| 8 | 结果归档 | Task Vault 持久化（diskcache/jsonl） | 用户 rollback 后结果不丢失；支持 `/task result <id>` 事后查询 |
| 9 | 磁盘防爆 | BoundedFileWriter（50MB 截断） | 防止死循环输出撑满磁盘；截断不 kill 进程，kill 由 hard_timeout 负责 |
| 10 | soft_timeout 值 | 按节点类型区分，Blueprint 可配置 | Shell 30s / Gemini CLI 60s / Claude SDK 90s（仅提示用） |

---

## 五、风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| 孤儿进程泄漏（主进程崩溃后子进程残留） | 高 | PID Registry + `atexit` 钩子；Heartbeat 启动时扫描残留 PID 文件并清理 |
| Checkpoint 断层（PENDING 覆写破坏 append-only 语义） | 中 | 使用同 message ID 覆写而非追加；router 节点显式处理 PENDING 状态 |
| 用户 rollback 导致结果注入上下文不一致 | 中 | 注入前校验 task_id 对应的 conversation_turn；不匹配则存档不注入 |
| 磁盘写满 | 中 | BoundedFileWriter 50MB 硬上限 + 截断标记 |
| Heartbeat 自身重启丢失监控状态 | 中 | 任务注册信息持久化到 Task Vault；启动时 reconciliation 扫描 |
| Asyncio 跨线程通信死锁 | 中 | 严格使用 `run_coroutine_threadsafe` + Heartbeat 暴露 `get_loop()` |
| soft_timeout 假阳性 | 低 | 按节点类型差异化配置；Blueprint 支持运行时覆盖 |

---

## 六、文件结构与接口

### 改动范围（6 个文件）

```
# 新增
framework/async_task_manager.py      # 后台任务生命周期管理
framework/bounded_file_writer.py     # 带截断保护的文件写入器

# 修改（核心）
framework/nodes/external_tool_node.py     # 基类加 soft_timeout + tee 逻辑
framework/heartbeat.py                    # 注册 TASK_MONITOR + 完成检测
framework/base_interface.py               # SSE 监听 + 响铃 + PENDING 消费
mcp_servers/heartbeat/server.py           # 新增 task_completed SSE event type

# 最小化改动（仅配置/注册）
framework/builtins.py                     # 注册 TASK_MONITOR 节点类型
```

### 关键接口

```python
# === AsyncTaskManager (新增) ===
class AsyncTaskManager:
    register_task(task_id, pid, output_path, hard_timeout) -> None
    query_task(task_id) -> TaskStatus  # RUNNING | COMPLETED | FAILED | TIMEOUT
    get_result(task_id) -> str | None
    cancel_task(task_id) -> bool
    cleanup_all() -> None              # atexit 调用


# === BoundedFileWriter (新增) ===
class BoundedFileWriter:
    __init__(path, max_bytes=50_000_000)  # 50MB
    write(data: bytes) -> int             # 超限后静默丢弃，写截断标记
    close() -> None
    path -> str


# === ExternalToolNode 新增钩子 (修改) ===
class ExternalToolNode:
    soft_timeout: int       # 子类可覆写，或从 Blueprint 读取
    hard_timeout: int

    on_soft_timeout(proc, task_id, output_path) -> ToolMessage
        # 默认：注册 Heartbeat 任务，返回 PENDING 消息
        # Claude 子类覆写：仅打印进度提示，不中断

    _tee_subprocess(cmd) -> (proc, buffer, BoundedFileWriter)
        # 启动子进程 + tee 线程


# === BaseInterface 新增 (修改) ===
class BaseInterface:
    _completed_tasks_queue: queue.Queue     # 线程安全队列
    _sse_listener_thread: Thread            # 后台监听 Heartbeat SSE

    _on_task_completed(task_id) -> None     # 入队 + 响铃
    _consume_pending_tasks(state) -> state  # invoke 前调用，覆写 PENDING 消息


# === Heartbeat 新增 (修改) ===
# heartbeat.py
get_loop() -> asyncio.AbstractEventLoop     # 线程安全 loop accessor

# server.py — 新增 SSE event
# event: task_completed
# data: { "task_id": "xxx", "status": "completed|failed|timeout" }
```

---

## 七、实现顺序

1. `bounded_file_writer.py` + `async_task_manager.py`（基础设施，可独立测试）
2. `external_tool_node.py` 改造（核心逻辑：tee + soft_timeout 分支）
3. Heartbeat 侧（`heartbeat.py` + `server.py`，TASK_MONITOR 注册与检测）
4. `base_interface.py`（结果消费 + PENDING 覆写 + 响铃通知）

---

## 八、V2 展望（不在 V1 范围内）

- BaseInterface CLI 侧引入 `prompt_toolkit`，实现事件驱动的自动 resume
- LangGraph `interrupt/Command(resume=...)` 原生挂起恢复
- Claude SDK 后台化：分离式消费者 + safe/unsafe tool whitelist
- 多任务并发：引入 sequence number + 有序结果队列
- Discord Interface 侧的全自动 resume（已有 async event loop，改造成本低）

---

— Hani · 无垠智穹
