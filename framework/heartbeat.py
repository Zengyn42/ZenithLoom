"""
框架级心跳调度器 — framework/heartbeat.py

HeartbeatManager: 主图可感知、可控制的并行任务调度器。
每个 task 独立 asyncio 协程，有自己的 interval、lock、退避策略。
通过 HeartbeatManager 共享对象与主图桥接（查询 + 控制 API）。
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 23  # hours


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
    """

    def __init__(self, blueprint_path: Path, state_path: Path):
        self._blueprint_path = Path(blueprint_path)
        self._state_path = Path(state_path)
        self._tasks: dict[str, TaskEntry] = {}
        self._loops: dict[str, asyncio.Task] = {}

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
            task_type = task_def.get("type", "PROBE")
            interval = task_def.get("interval_hours", _DEFAULT_INTERVAL)

            # merge 用户运行时覆盖的 interval
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

    # ── 控制 API（给 LLM 工具调用）────────────────────────────────────────

    async def run_now(self, task_id: str) -> str:
        """立即执行指定 task。如果正在执行中则返回提示。"""
        entry = self._tasks.get(task_id)
        if entry is None:
            return f"Unknown task: {task_id!r}. Known: {list(self._tasks.keys())}"

        if entry._lock.locked():
            return f"Task {task_id!r} is already running."

        async with entry._lock:
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
            finally:
                entry.last_run = datetime.now()
                self._save_state()

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
                finally:
                    entry.last_run = datetime.now()
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
