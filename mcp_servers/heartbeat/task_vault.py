"""
TaskVault — 后台任务注册表 + 结果归档

Heartbeat MCP 的内部模块，管理 soft_timeout 后被放入后台的子进程任务。
提供注册、查询、取消、结果获取和清理接口。

支持多个后台任务并行运行。

持久化：
  - PID Registry: data/heartbeat/monitors/<task_id>.pid.json
  - Task Vault:   data/heartbeat/monitors/vault.jsonl（追加写入）
"""

import atexit
import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MONITORS_DIR = _PROJECT_ROOT / "data" / "heartbeat" / "monitors"
_VAULT_PATH = _MONITORS_DIR / "vault.jsonl"


class TaskStatus(str, Enum):
    """后台任务状态。"""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass
class TaskRecord:
    """单个后台任务的运行时记录。"""
    task_id: str
    pid: int
    output_path: str
    hard_timeout: float
    registered_at: float  # time.time()
    status: TaskStatus = TaskStatus.RUNNING


class TaskVault:
    """后台任务注册表 + 结果归档。

    支持多个后台任务并行运行。
    """

    _instance: Optional["TaskVault"] = None

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        _MONITORS_DIR.mkdir(parents=True, exist_ok=True)
        self._reconcile_stale_pids()
        atexit.register(self.cleanup_all)

    @classmethod
    def get_instance(cls) -> "TaskVault":
        """全局单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── 公开 API ──────────────────────────────────────────────────────────

    def register_task(
        self,
        task_id: str,
        pid: int,
        output_path: str,
        hard_timeout: float,
    ) -> bool:
        """注册后台任务。

        支持多个后台任务并行运行。

        Returns:
            True 注册成功。
        """
        record = TaskRecord(
            task_id=task_id,
            pid=pid,
            output_path=output_path,
            hard_timeout=hard_timeout,
            registered_at=time.time(),
        )
        self._tasks[task_id] = record
        self._write_pid_file(record)
        logger.info(f"[task_vault] registered: {task_id} pid={pid} hard_timeout={hard_timeout}s")
        return True

    def query_task(self, task_id: str) -> TaskStatus | None:
        """查询任务状态。未知 task_id 返回 None。"""
        record = self._tasks.get(task_id)
        if record is None:
            vault_entry = self._find_in_vault(task_id)
            if vault_entry:
                return TaskStatus(vault_entry["status"])
            return None

        if record.status == TaskStatus.RUNNING:
            self._refresh_status(record)

        return record.status

    def get_result(self, task_id: str) -> str | None:
        """获取已完成任务的输出。RUNNING 状态或未知任务返回 None。"""
        record = self._tasks.get(task_id)
        if record is not None and record.status == TaskStatus.RUNNING:
            self._refresh_status(record)

        if record is not None and record.status != TaskStatus.RUNNING:
            return self._read_output(record.output_path)

        vault_entry = self._find_in_vault(task_id)
        if vault_entry and vault_entry.get("output_path"):
            return self._read_output(vault_entry["output_path"])

        return None

    def cancel_task(self, task_id: str) -> bool:
        """取消（kill）后台任务。

        Returns:
            True 成功发送 SIGTERM；False 任务不存在或已结束。
        """
        record = self._tasks.get(task_id)
        if record is None or record.status != TaskStatus.RUNNING:
            return False

        try:
            os.kill(record.pid, signal.SIGTERM)
            logger.info(f"[task_vault] killed: {task_id} pid={record.pid}")
        except ProcessLookupError:
            logger.info(f"[task_vault] pid {record.pid} already gone")
        except OSError as e:
            logger.warning(f"[task_vault] kill failed for {task_id}: {e}")
            return False

        record.status = TaskStatus.FAILED
        self._write_vault_entry(record)
        self._remove_pid_file(task_id)
        return True

    def cleanup_all(self) -> None:
        """清理所有后台任务（atexit 调用）。"""
        for task_id, record in list(self._tasks.items()):
            if record.status == TaskStatus.RUNNING:
                try:
                    os.kill(record.pid, signal.SIGTERM)
                    logger.info(f"[task_vault] cleanup: killed {task_id} pid={record.pid}")
                except (ProcessLookupError, OSError):
                    pass
                record.status = TaskStatus.FAILED
                self._write_vault_entry(record)
            self._remove_pid_file(task_id)

    def running_task_count(self) -> int:
        """返回当前正在运行的后台任务数量。"""
        count = 0
        for rec in self._tasks.values():
            if rec.status == TaskStatus.RUNNING:
                self._refresh_status(rec)
                if rec.status == TaskStatus.RUNNING:
                    count += 1
        return count

    def mark_completed(self, task_id: str, status: TaskStatus = TaskStatus.COMPLETED) -> None:
        """标记任务完成（由 HeartbeatManager._monitor_loop 调用）。"""
        record = self._tasks.get(task_id)
        if record is None:
            return
        record.status = status
        self._write_vault_entry(record)
        self._remove_pid_file(task_id)
        logger.info(f"[task_vault] marked {status.value}: {task_id}")

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _refresh_status(self, record: TaskRecord) -> None:
        """检查 PID 是否仍存活，更新状态。"""
        if record.status != TaskStatus.RUNNING:
            return

        elapsed = time.time() - record.registered_at
        if elapsed > record.hard_timeout:
            try:
                os.kill(record.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            record.status = TaskStatus.TIMEOUT
            self._write_vault_entry(record)
            self._remove_pid_file(record.task_id)
            logger.warning(
                f"[task_vault] hard_timeout ({record.hard_timeout}s): {record.task_id}"
            )
            return

        if not pid_alive(record.pid):
            record.status = TaskStatus.COMPLETED
            self._write_vault_entry(record)
            self._remove_pid_file(record.task_id)
            logger.info(f"[task_vault] completed (pid gone): {record.task_id}")

    def _write_pid_file(self, record: TaskRecord) -> None:
        """持久化 PID 信息。"""
        pid_path = _MONITORS_DIR / f"{record.task_id}.pid.json"
        try:
            pid_path.write_text(
                json.dumps({
                    "task_id": record.task_id,
                    "pid": record.pid,
                    "output_path": record.output_path,
                    "hard_timeout": record.hard_timeout,
                    "registered_at": record.registered_at,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"[task_vault] failed to write pid file: {e}")

    @staticmethod
    def _remove_pid_file(task_id: str) -> None:
        """删除 PID 文件。"""
        pid_path = _MONITORS_DIR / f"{task_id}.pid.json"
        pid_path.unlink(missing_ok=True)

    def _write_vault_entry(self, record: TaskRecord) -> None:
        """追加任务完成记录到 vault.jsonl。"""
        entry = {
            "task_id": record.task_id,
            "pid": record.pid,
            "output_path": record.output_path,
            "status": record.status.value,
            "registered_at": record.registered_at,
            "completed_at": time.time(),
        }
        try:
            with open(_VAULT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[task_vault] failed to write vault entry: {e}")

    @staticmethod
    def _find_in_vault(task_id: str) -> dict | None:
        """从 vault.jsonl 查找任务记录（最后一条匹配）。"""
        if not _VAULT_PATH.exists():
            return None
        result = None
        try:
            with open(_VAULT_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("task_id") == task_id:
                            result = entry
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return result

    @staticmethod
    def _read_output(output_path: str) -> str | None:
        """读取输出文件内容。"""
        try:
            p = Path(output_path)
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"[task_vault] failed to read output: {e}")
        return None

    def _reconcile_stale_pids(self) -> None:
        """启动时扫描残留 PID 文件，清理已死进程。"""
        for pid_file in _MONITORS_DIR.glob("*.pid.json"):
            try:
                data = json.loads(pid_file.read_text(encoding="utf-8"))
                _pid = data["pid"]
                _task_id = data["task_id"]
                if pid_alive(_pid):
                    record = TaskRecord(
                        task_id=_task_id,
                        pid=_pid,
                        output_path=data["output_path"],
                        hard_timeout=data["hard_timeout"],
                        registered_at=data["registered_at"],
                    )
                    self._tasks[_task_id] = record
                    logger.info(f"[task_vault] reconciled running task: {_task_id} pid={_pid}")
                else:
                    logger.info(f"[task_vault] reconcile: stale pid {_pid} for {_task_id}")
                    pid_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"[task_vault] reconcile error for {pid_file}: {e}")
                pid_file.unlink(missing_ok=True)


# ── 公共工具函数 ────────────────────────────────────────────────────────────

def pid_alive(pid: int) -> bool:
    """检查 PID 是否存活。"""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
