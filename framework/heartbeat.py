"""
框架级心跳调度器 — framework/heartbeat.py

HeartbeatManager: 主图可感知、可控制的并行任务调度器。
每个 task 独立 asyncio 协程，有自己的 interval、lock、退避策略。
通过 HeartbeatManager 共享对象与主图桥接（查询 + 控制 API）。

TASK_MONITOR 任务类型：短 interval 轮询 PID 是否存活，
检测后台子进程完成后触发 on_complete 回调。
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from mcp_servers.heartbeat.task_vault import pid_alive

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 23  # hours
_MONITOR_POLL_INTERVAL = 60  # seconds — TASK_MONITOR 的轮询间隔（1 分钟）


# ---------------------------------------------------------------------------
# TaskEntry — 单个定时任务的运行时状态
# ---------------------------------------------------------------------------

@dataclass
class TaskEntry:
    id: str                           # "probe_ollama"
    type: str                         # "PROBE" | "HEARTBEAT"
    config: dict                      # 原始 blueprint 配置
    node: object                      # ProbeNode 或 HeartbeatNode 实例
    interval_hours: float             # 当前频率（可被用户修改）
    status: str = "idle"              # "idle" | "running" | "OK" | "FAILED: xxx"
    last_run: datetime | None = None
    next_run: datetime | None = None  # 下次计划运行时间（由 _task_loop/run_now 维护）
    last_result: str = ""             # 最近一次执行的返回内容
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------------------------------------------------------------------
# HeartbeatManager — 并行任务调度器
# ---------------------------------------------------------------------------

class HeartbeatManager:
    """
    从 heartbeat blueprint (heartbeat.json) 加载 tasks，
    为每个 task 启动独立 asyncio 后台循环，
    提供查询 / 控制 API 供 LLM 工具调用。

    失败告警机制：
      task 执行失败时，alert 自动入队（_alerts 列表）。
      Agent 通过 get_alerts() 拉取未确认告警，拉取即清空。
    """

    def __init__(self, blueprint_path: Path, state_path: Path,
                 on_failure=None, on_complete=None,
                 task_overrides: dict | None = None,
                 session_registry: set | None = None):
        """
        on_failure:  async callable(event_dict) — 失败时立即回调（level=error）。
        on_complete: async callable(event_dict) — 每次执行完成后回调（含成功/告警）。
                     event_dict 含 level 字段：info / warning / error。
                     由 MCP Server 设置为 send_log_message 推送。
        task_overrides: {task_id: {key: value}} — entity 级参数覆写。
                        支持 interval_hours, timeout 等任意字段。
        session_registry: 活跃 session 集合引用（用于检测 agent 断连）。
        """
        self._blueprint_path = Path(blueprint_path)
        self._state_path = Path(state_path)
        self._tasks: dict[str, TaskEntry] = {}
        self._loops: dict[str, asyncio.Task] = {}
        self._alerts: list[dict] = []  # 未确认的失败告警（兜底，供 heartbeat_alerts 工具拉取）
        self._on_failure = on_failure  # async callable(event_dict) | None
        self._on_complete = on_complete  # async callable(event_dict) | None
        self._task_overrides = task_overrides or {}  # entity 级参数覆写
        self._session_registry: set | None = session_registry  # 活跃 session 集合

    # ── 生命周期 ──────────────────────────────────────────────────────────

    async def start(self):
        """
        1. 读 heartbeat blueprint → tasks 列表
        2. 读 state_path → merge 用户修改过的 interval
        3. 为每个 task 实例化节点（复用 registry.get_node_factory()）
        4. 为每个 task 启动独立 asyncio.create_task(_task_loop)
        """
        import framework.builtins  # noqa: F401 — 确保内置类型已注册
        from framework.registry import get_node_factory

        blueprint = json.loads(self._blueprint_path.read_text(encoding="utf-8"))
        saved_state = self._load_state()

        for task_def in blueprint.get("tasks", []):
            task_id = task_def["id"]

            # merge entity 级覆写（优先级: entity overrides > blueprint 默认）
            if task_id in self._task_overrides:
                task_def = {**task_def, **self._task_overrides[task_id]}

            task_type = task_def.get("type", "PROBE")
            interval = task_def.get("interval_hours", _DEFAULT_INTERVAL)

            # merge 用户运行时覆盖的 interval（优先级最高）
            if task_id in saved_state:
                interval = saved_state[task_id].get("interval_hours", interval)

            # 实例化节点（复用 registry 工厂）
            factory = get_node_factory(task_type)
            # 工厂签名：(config, node_config) → node
            # heartbeat tasks 不需要 AgentConfig，传 None；
            # 但 PROBE 工厂只用 node_config，HEARTBEAT 工厂也只用 node_config
            from framework.config import AgentConfig
            dummy_config = AgentConfig()
            node = factory(dummy_config, task_def)

            entry = TaskEntry(
                id=task_id,
                type=task_type,
                config=task_def,
                node=node,
                interval_hours=interval,
            )
            self._tasks[task_id] = entry

        # 启动所有 task loops
        for task_id in self._tasks:
            self._loops[task_id] = asyncio.create_task(
                self._task_loop(task_id),
                name=f"heartbeat:{task_id}",
            )

        task_ids = list(self._tasks.keys())
        logger.info(f"[heartbeat] HeartbeatManager started: {task_ids}")

    async def stop(self):
        """取消所有后台 loop（graceful CancelledError handling），保存状态。"""
        for task_id, task in self._loops.items():
            if not task.done():
                task.cancel()
        # 等待所有 task 完成取消
        if self._loops:
            await asyncio.gather(*self._loops.values(), return_exceptions=True)
        self._loops.clear()
        self._save_state()
        logger.info("[heartbeat] HeartbeatManager stopped")

    # ── 查询 API（给 LLM 工具调用）────────────────────────────────────────

    def list_tasks(self) -> str:
        """返回格式化表格：ID / 状态 / 上次执行 / 下次执行 / 频率"""
        if not self._tasks:
            return "No heartbeat tasks configured."

        lines = ["ID | Type | Status | Last Run | Interval(h)", "---|------|--------|----------|------------"]
        for entry in self._tasks.values():
            last = entry.last_run.strftime("%Y-%m-%d %H:%M") if entry.last_run else "never"
            lines.append(
                f"{entry.id} | {entry.type} | {entry.status} | {last} | {entry.interval_hours}"
            )
        return "\n".join(lines)

    def get_status(self, task_id: str) -> str:
        """单个任务详情 + 最近执行结果"""
        entry = self._tasks.get(task_id)
        if entry is None:
            return f"Unknown task: {task_id!r}. Known: {list(self._tasks.keys())}"

        last = entry.last_run.strftime("%Y-%m-%d %H:%M:%S") if entry.last_run else "never"
        return (
            f"Task: {entry.id}\n"
            f"Type: {entry.type}\n"
            f"Status: {entry.status}\n"
            f"Interval: {entry.interval_hours}h\n"
            f"Last Run: {last}\n"
            f"Last Result: {entry.last_result or '(none)'}"
        )

    def get_alerts(self) -> list[dict]:
        """
        拉取并清空未确认的失败告警。
        返回 list[dict]，每项含 task_id, type, error, consecutive_failures, time。
        空列表表示无告警。
        """
        alerts = list(self._alerts)
        self._alerts.clear()
        return alerts

    # ── 控制 API（给 LLM 工具调用）────────────────────────────────────────

    async def run_now(self, task_id: str) -> str:
        """立即执行指定 task。如果正在执行中则返回提示。"""
        entry = self._tasks.get(task_id)
        if entry is None:
            return f"Unknown task: {task_id!r}. Known: {list(self._tasks.keys())}"

        if entry._lock.locked():
            return f"Task {task_id!r} is already running."

        async with entry._lock:
            failed_exc = None
            try:
                entry.status = "running"
                result = await entry.node(state={})
                msgs = result.get("messages", [])
                content = msgs[0].content if msgs else str(result)
                entry.status = "OK"
                entry.last_result = content
            except Exception as e:
                entry.status = f"FAILED: {e}"
                entry.last_result = str(e)
                content = f"FAILED: {e}"
                failed_exc = e
            finally:
                entry.last_run = datetime.now()
                entry.next_run = entry.last_run + timedelta(hours=entry.interval_hours)
                self._save_state()

        next_dt = entry.next_run
        next_run_str = next_dt.isoformat(timespec="seconds")
        if failed_exc is not None:
            alert = {
                "task_id": task_id,
                "type": entry.type,
                "level": "error",
                "error": str(failed_exc),
                "consecutive_failures": 1,
                "time": datetime.now().isoformat(timespec="seconds"),
                "next_run": next_run_str,
            }
            if self._on_failure:
                try:
                    await self._on_failure(alert)
                except Exception as cb_err:
                    logger.warning(f"[heartbeat] run_now on_failure callback error: {cb_err}")
        else:
            event = {
                "task_id": task_id,
                "type": entry.type,
                "level": "info",
                "content": content,
                "time": datetime.now().isoformat(timespec="seconds"),
                "next_run": next_run_str,
            }
            if self._on_complete:
                try:
                    await self._on_complete(event)
                except Exception as cb_err:
                    logger.warning(f"[heartbeat] run_now on_complete callback error: {cb_err}")

        return content

    def set_interval(self, task_id: str, hours: float) -> str:
        """更新 interval，cancel 旧 loop，启动新 loop，save state。"""
        entry = self._tasks.get(task_id)
        if entry is None:
            return f"Unknown task: {task_id!r}. Known: {list(self._tasks.keys())}"

        if hours <= 0:
            return f"Invalid interval: {hours}. Must be > 0."

        old_interval = entry.interval_hours
        entry.interval_hours = hours

        # cancel 旧 loop，启动新 loop
        old_task = self._loops.get(task_id)
        if old_task and not old_task.done():
            old_task.cancel()
        self._loops[task_id] = asyncio.create_task(
            self._task_loop(task_id),
            name=f"heartbeat:{task_id}",
        )

        self._save_state()
        return f"Task {task_id!r} interval changed: {old_interval}h → {hours}h"

    # ── TASK_MONITOR 动态注册 ─────────────────────────────────────────────

    async def register_monitor(
        self,
        task_id: str,
        pid: int,
        output_path: str,
        hard_timeout: float,
        agent_id: str = "",
        agent_session: object | None = None,
    ) -> str:
        """动态注册一个 TASK_MONITOR 任务（后台子进程监控）。

        短 interval 轮询 PID 存活状态，完成后触发 on_complete 回调。

        Args:
            task_id: 唯一任务标识。
            pid: 子进程 PID。
            output_path: 子进程输出文件路径。
            hard_timeout: 最大允许运行时间（秒）。
            agent_id: 关联的 agent 标识。
            agent_session: MCP SSE session 引用（用于定向推送）。

        Returns:
            状态消息。
        """
        if task_id in self._tasks:
            return f"Task {task_id!r} already registered."

        entry = TaskEntry(
            id=task_id,
            type="TASK_MONITOR",
            config={
                "pid": pid,
                "output_path": output_path,
                "hard_timeout": hard_timeout,
                "agent_id": agent_id,
                "agent_session": agent_session,
            },
            node=None,  # TASK_MONITOR 不需要 node 实例
            interval_hours=_MONITOR_POLL_INTERVAL / 3600,  # 转换为小时
        )
        self._tasks[task_id] = entry

        self._loops[task_id] = asyncio.create_task(
            self._monitor_loop(task_id, pid, output_path, hard_timeout),
            name=f"heartbeat:monitor:{task_id}",
        )
        logger.info(
            f"[heartbeat] TASK_MONITOR registered: {task_id} pid={pid} "
            f"agent_id={agent_id!r} hard_timeout={hard_timeout}s"
        )
        return f"Monitor registered: {task_id}"

    def list_monitors(self, agent_id: str) -> list[dict]:
        """返回指定 agent 关联的所有 TASK_MONITOR 任务列表。

        Args:
            agent_id: 要查询的 agent 标识。

        Returns:
            匹配的 monitor 任务信息列表。
        """
        results: list[dict] = []
        for entry in self._tasks.values():
            if entry.type != "TASK_MONITOR":
                continue
            if entry.config.get("agent_id") != agent_id:
                continue
            results.append({
                "task_id": entry.id,
                "status": entry.status,
                "pid": entry.config.get("pid"),
                "output_path": entry.config.get("output_path"),
                "hard_timeout": entry.config.get("hard_timeout"),
                "last_run": entry.last_run.isoformat(timespec="seconds") if entry.last_run else None,
            })
        return results

    async def _send_to_session(self, session: object, event: dict, level: str = "info") -> bool:
        """尝试向指定 session 发送 SSE 事件。

        Returns:
            True 发送成功；False 发送失败（session 可能已断开）。
        """
        try:
            await session.send_log_message(level=level, data=event, logger="heartbeat")
            return True
        except Exception as e:
            logger.warning(f"[heartbeat] failed to send to agent session: {e}")
            return False

    async def _monitor_loop(
        self,
        task_id: str,
        pid: int,
        output_path: str,
        hard_timeout: float,
    ) -> None:
        """TASK_MONITOR 专用循环：短 interval 轮询 PID 存活状态。

        每次 poll 循环：
        1. 检查 agent_session 是否仍活跃（断连则 kill 子进程）
        2. 向关联 agent 推送 monitor_heartbeat SSE 事件
        3. 检查 hard_timeout
        4. 检查 PID 存活
        """
        entry = self._tasks.get(task_id)
        if entry is None:
            return

        start_time = asyncio.get_event_loop().time()
        entry.status = "monitoring"

        # 从 config 中获取 agent_session
        agent_session: object | None = entry.config.get("agent_session")

        while True:
            await asyncio.sleep(_MONITOR_POLL_INTERVAL)

            elapsed = asyncio.get_event_loop().time() - start_time

            # ── 需求 5：检查 agent_session 是否仍在 session_registry 中 ──
            if (
                agent_session is not None
                and self._session_registry is not None
                and agent_session not in self._session_registry
            ):
                logger.warning(
                    f"[heartbeat] TASK_MONITOR: agent disconnected for {task_id}, killing pid={pid}"
                )
                try:
                    os.kill(pid, 9)  # SIGKILL
                except (ProcessLookupError, OSError):
                    pass
                entry.status = "FAILED"
                entry.last_result = "agent disconnected"
                entry.last_run = datetime.now()

                try:
                    from mcp_servers.heartbeat.task_vault import TaskVault, TaskStatus
                    TaskVault.get_instance().mark_completed(task_id, TaskStatus.FAILED)
                except Exception as e:
                    logger.warning(f"[heartbeat] failed to mark task failed: {e}")

                event = {
                    "task_id": task_id,
                    "type": "TASK_MONITOR",
                    "level": "error",
                    "error": "agent disconnected",
                    "status": "failed",
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
                if self._on_failure:
                    try:
                        await self._on_failure(event)
                    except Exception as cb_err:
                        logger.warning(f"[heartbeat] monitor on_failure error: {cb_err}")
                break

            # ── 需求 4：向关联 Agent 推送 monitor_heartbeat SSE 事件 ──
            heartbeat_event: dict = {
                "task_id": task_id,
                "type": "TASK_MONITOR",
                "level": "info",
                "status": "monitoring",
                "elapsed_seconds": int(elapsed),
                "pid": pid,
                "pid_alive": pid_alive(pid),
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            if agent_session is not None:
                sent = await self._send_to_session(agent_session, heartbeat_event, level="info")
                if not sent and self._on_complete:
                    # session 无效，走广播
                    try:
                        await self._on_complete(heartbeat_event)
                    except Exception as cb_err:
                        logger.warning(f"[heartbeat] monitor heartbeat broadcast error: {cb_err}")
            elif self._on_complete:
                try:
                    await self._on_complete(heartbeat_event)
                except Exception as cb_err:
                    logger.warning(f"[heartbeat] monitor heartbeat broadcast error: {cb_err}")

            # hard_timeout 检查
            if elapsed > hard_timeout:
                logger.warning(
                    f"[heartbeat] TASK_MONITOR hard_timeout ({hard_timeout}s): {task_id}"
                )
                try:
                    os.kill(pid, 9)  # SIGKILL
                except (ProcessLookupError, OSError):
                    pass
                entry.status = "TIMEOUT"
                entry.last_result = f"hard_timeout after {hard_timeout}s"
                entry.last_run = datetime.now()

                # 标记 TaskVault 中的任务
                try:
                    from mcp_servers.heartbeat.task_vault import TaskVault, TaskStatus
                    TaskVault.get_instance().mark_completed(task_id, TaskStatus.TIMEOUT)
                except Exception as e:
                    logger.warning(f"[heartbeat] failed to mark task timeout: {e}")

                event = {
                    "task_id": task_id,
                    "type": "TASK_MONITOR",
                    "level": "error",
                    "error": f"hard_timeout after {hard_timeout}s",
                    "status": "timeout",
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
                if self._on_failure:
                    try:
                        await self._on_failure(event)
                    except Exception as cb_err:
                        logger.warning(f"[heartbeat] monitor on_failure error: {cb_err}")
                break

            # PID 存活检查
            if not pid_alive(pid):
                logger.info(f"[heartbeat] TASK_MONITOR: pid {pid} exited for {task_id}")

                # 读取输出
                result_text = ""
                try:
                    p = Path(output_path)
                    if p.exists():
                        result_text = p.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    result_text = f"(failed to read output: {e})"

                entry.status = "OK"
                entry.last_result = result_text[:500]  # 截取预览
                entry.last_run = datetime.now()

                # 标记 TaskVault 中的任务
                try:
                    from mcp_servers.heartbeat.task_vault import TaskVault, TaskStatus
                    TaskVault.get_instance().mark_completed(task_id, TaskStatus.COMPLETED)
                except Exception as e:
                    logger.warning(f"[heartbeat] failed to mark task completed: {e}")

                event = {
                    "task_id": task_id,
                    "type": "TASK_MONITOR",
                    "level": "info",
                    "status": "completed",
                    "content": result_text[:500],
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
                if self._on_complete:
                    try:
                        await self._on_complete(event)
                    except Exception as cb_err:
                        logger.warning(f"[heartbeat] monitor on_complete error: {cb_err}")
                break

        # 清理 — 从 tasks 字典移除（monitor 是一次性的）
        self._tasks.pop(task_id, None)
        self._save_state()

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """线程安全的 event loop accessor。"""
        return asyncio.get_event_loop()

    # ── 内部 ──────────────────────────────────────────────────────────────

    async def _task_loop(self, task_id: str):
        """单个 task 的后台无限循环。首次立即执行，之后按 interval 休眠。"""
        entry = self._tasks[task_id]
        consecutive_failures = 0
        first_run = True

        while True:
            # 首次立即执行，之后按 interval 休眠
            if not first_run:
                if consecutive_failures >= 3:
                    sleep_hours = min(entry.interval_hours * 2, 48)
                else:
                    sleep_hours = entry.interval_hours
                await asyncio.sleep(sleep_hours * 3600)
            first_run = False

            async with entry._lock:
                try:
                    entry.status = "running"
                    result = await entry.node(state={})
                    msgs = result.get("messages", [])
                    content = msgs[0].content if msgs else str(result)
                    entry.status = "OK"
                    entry.last_result = content
                    consecutive_failures = 0

                    # 判定完成事件级别（节点可在输出中标注 level=warning）
                    if "level=warning" in content:
                        level = "warning"
                    else:
                        level = "info"

                    # 构建完成事件
                    next_dt = datetime.now() + timedelta(hours=entry.interval_hours)
                    event = {
                        "task_id": task_id,
                        "type": entry.type,
                        "level": level,
                        "content": content,
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "next_run": next_dt.isoformat(timespec="seconds"),
                    }
                    # 推送完成回调（SSE push to Agent）
                    if self._on_complete:
                        try:
                            await self._on_complete(event)
                        except Exception as cb_err:
                            logger.warning(f"[heartbeat] on_complete callback error: {cb_err}")

                except asyncio.CancelledError:
                    raise  # 让 cancel 正常传播
                except Exception as e:
                    entry.status = f"FAILED: {e}"
                    entry.last_result = str(e)
                    consecutive_failures += 1
                    logger.warning(
                        f"[heartbeat] task {task_id!r} failed "
                        f"(consecutive={consecutive_failures}): {e}"
                    )
                    # 构建告警
                    next_hours = min(entry.interval_hours * 2, 48) if consecutive_failures >= 3 else entry.interval_hours
                    next_dt = datetime.now() + timedelta(hours=next_hours)
                    alert = {
                        "task_id": task_id,
                        "type": entry.type,
                        "level": "error",
                        "error": str(e),
                        "consecutive_failures": consecutive_failures,
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "next_run": next_dt.isoformat(timespec="seconds"),
                    }
                    # 告警入队（兜底，供 heartbeat_alerts 工具拉取）
                    self._alerts.append(alert)
                    # 主动推送回调（SSE push to Agent）
                    if self._on_failure:
                        try:
                            await self._on_failure(alert)
                        except Exception as cb_err:
                            logger.warning(f"[heartbeat] on_failure callback error: {cb_err}")
                finally:
                    entry.last_run = datetime.now()
                    next_hours = min(entry.interval_hours * 2, 48) if consecutive_failures >= 3 else entry.interval_hours
                    entry.next_run = entry.last_run + timedelta(hours=next_hours)
                    self._save_state()

            if consecutive_failures >= 3:
                logger.warning(
                    f"[heartbeat] task {task_id!r} backing off "
                    f"(consecutive_failures={consecutive_failures})"
                )

    def _save_state(self):
        """写 heartbeat_state.json — 只存用户修改和运行时数据，blueprint 默认值不重复存。"""
        state = {}
        for entry in self._tasks.values():
            state[entry.id] = {
                "interval_hours": entry.interval_hours,
                "status": entry.status,
                "last_run": entry.last_run.isoformat() if entry.last_run else None,
                "last_result": entry.last_result,
            }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"[heartbeat] failed to save state: {e}")

    def _load_state(self) -> dict:
        """读 heartbeat_state.json，返回 {task_id: {interval_hours, status, ...}}。"""
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[heartbeat] failed to load state: {e}")
            return {}
