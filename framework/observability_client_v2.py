"""
ZenithLoom Observability Client v2 — framework/observability_client_v2.py

Architecture: JSONL 真相源（文件即状态，零网络依赖）

Design decisions (from debate_design, 方案 D v1.2):
  - emit() = json.dumps() + deque.append() — 纯内存，数学上不阻塞 event loop
  - daemon 写入线程：唯一磁盘 I/O 隔离点
  - ZL_OBSERV_DIR='' → 整个模块空壳 (enabled=False)
  - agent_restart 锚点：进程启动时写入，viewer 据此作废僵尸 session
  - session_resume 锚点：文件轮转时对每个活跃 session 写入新文件头部，单文件自包含
  - atexit graceful_flush：500ms 超时强制 drain，SIGKILL 不可捕获靠下次 agent_restart 清理
  - 轮转：10MB → rename to archive/{agent}.events.{ts}.jsonl → 后台 gzip → 新文件

Public interface (完全兼容 v1 签名，graph_controller.py tap 点零改动):
  get_client() -> ObservClientV2
  client.start(agent_name: str) -> Coroutine (awaitable no-op for asyncio compat)
  client.emit_run_start(run_id, thread_id, input_preview)
  client.tap(ns, mode, event, *, run_id, thread_id)
  client.emit_run_end(run_id, thread_id)
  client.enabled() -> bool
"""

from __future__ import annotations

import atexit
import collections
import gzip
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FILE_BYTES = 10 * 1024 * 1024       # 10 MB rotation threshold
_BUFFER_MAXLEN = 8192                    # deque ring buffer
_GRACEFUL_FLUSH_TIMEOUT_S = 0.5         # atexit drain budget
_WRITER_WAKE_INTERVAL_S = 0.05          # wake interval when no notify
_ALLOWED_DEBUG_TYPES: frozenset[str] = frozenset({"task", "task_result"})


# ---------------------------------------------------------------------------
# ObservClientV2
# ---------------------------------------------------------------------------

class ObservClientV2:
    """
    JSONL-file-backed observability client.
    Thread-safe: emit() from any thread/coroutine; writer thread owns all I/O.
    """

    def __init__(self, agent_name: str, obs_dir: str, defer: bool = False) -> None:
        """
        defer=True: postpone file/writer initialization until start(agent_name)
        delivers the real agent name (used by the module singleton, whose name
        is unknown at import time).
        """
        if not obs_dir:
            self._enabled = False
            return

        self._enabled = True
        self._agent = agent_name
        self._dir = Path(obs_dir)
        self._archive_dir = self._dir / "archive"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(exist_ok=True)

        # Thread-safe ring buffer: producer=emit(), consumer=writer thread
        self._buffer: collections.deque[str] = collections.deque(maxlen=_BUFFER_MAXLEN)
        self._notify = threading.Event()
        self._stop = threading.Event()

        # Active sessions: thread_id → run_start event dict (for session_resume on rotate)
        self._active_sessions: dict[str, dict] = {}
        self._sessions_lock = threading.Lock()

        # Current output file state (owned by writer thread after start)
        self._current_file: Any = None   # file object
        self._current_path: Path | None = None
        self._current_size: int = 0

        self._started = False
        self._init_lock = threading.Lock()
        if not defer:
            self._finish_init()

    def _finish_init(self) -> None:
        """Open output file, write agent_restart marker, start writer thread."""
        # Open the file (may rotate if existing file already > limit)
        self._current_file, self._current_path, self._current_size = self._open_file()

        # Write agent_restart marker immediately (synchronous — before writer starts)
        self._write_sync(self._make_event("agent_restart", {}))

        # Register graceful flush on exit
        atexit.register(self._graceful_flush)

        # Start daemon writer thread
        self._writer = threading.Thread(
            target=self._write_loop, name=f"obsv-writer-{self._agent}", daemon=True
        )
        self._writer.start()
        self._started = True

    # ── Public interface (v1-compatible) ─────────────────────────────────────

    def enabled(self) -> bool:
        return self._enabled

    async def start(self, agent_name: str) -> None:
        """
        Inject the real agent name and finish deferred initialization
        (v1-compatible signature). Idempotent; safe to call multiple times.
        For non-deferred clients (tests / direct construction) this is a no-op.
        """
        if not self._enabled or self._started:
            return
        with self._init_lock:
            if self._started:
                return
            if agent_name:
                self._agent = agent_name
            self._finish_init()

    def emit_run_start(self, run_id: str, thread_id: str, input_preview: str = "") -> None:
        """Emit run_start event. Never raises."""
        if not self._enabled:
            return
        try:
            session_data = {
                "run_id": run_id,
                "thread_id": thread_id,
                "input_preview": input_preview[:200],
            }
            # Store as dict (for session_resume reconstruction on rotation)
            with self._sessions_lock:
                self._active_sessions[thread_id] = session_data
            self._enqueue(self._make_event("run_start", session_data))
        except Exception as exc:
            logger.debug(f"[obsv_v2] emit_run_start error (non-fatal): {exc}")

    def emit_run_end(self, run_id: str, thread_id: str) -> None:
        """Emit run_end event. Never raises."""
        if not self._enabled:
            return
        try:
            with self._sessions_lock:
                self._active_sessions.pop(thread_id, None)
            self._enqueue(self._make_event("run_end", {
                "run_id": run_id,
                "thread_id": thread_id,
            }))
        except Exception as exc:
            logger.debug(f"[obsv_v2] emit_run_end error (non-fatal): {exc}")

    def tap(
        self,
        ns: Any,
        mode: str,
        event: Any,
        *,
        run_id: str,
        thread_id: str,
    ) -> None:
        """
        Process one astream chunk. Identical whitelist + payload logic to v1.
        Never raises.
        """
        if not self._enabled:
            return
        try:
            if mode == "values":
                return

            checkpoint_ns = ":".join(str(x) for x in ns) if ns else ""

            if mode == "debug":
                evt_type = event.get("type", "") if isinstance(event, dict) else ""
                if evt_type not in _ALLOWED_DEBUG_TYPES:
                    return
                payload_raw = event.get("payload", {}) if isinstance(event, dict) else {}
                node_id = payload_raw.get("name", "__graph__") if isinstance(payload_raw, dict) else "__graph__"
                zl_event_type = "node_start" if evt_type == "task" else "node_end"
                payload = {
                    "ns": list(ns) if ns else [],
                    "node": node_id,
                    "triggers": payload_raw.get("triggers", []) if isinstance(payload_raw, dict) else [],
                }
                if evt_type == "task_result":
                    payload["error"] = payload_raw.get("error") if isinstance(payload_raw, dict) else None

            elif mode == "updates":
                if not isinstance(event, dict):
                    return
                node_id = next(iter(event), "__graph__")
                node_updates = event.get(node_id, {})
                keys_changed = list(node_updates.keys()) if isinstance(node_updates, dict) else []
                zl_event_type = "state_update"

                # Build updates_preview (same logic as v1)
                updates_preview: dict = {}
                try:
                    if isinstance(node_updates, dict):
                        for k, v in node_updates.items():
                            try:
                                if isinstance(v, list) and v:
                                    last = v[-1]
                                    text = getattr(last, "content", None) or repr(last)
                                    updates_preview[k] = str(text)[:500]
                                else:
                                    updates_preview[k] = repr(v)[:500]
                            except Exception:
                                updates_preview[k] = "<unserializable>"
                    try:
                        preview_json = json.dumps(updates_preview, default=str)
                        if len(preview_json) > 4096:
                            updates_preview = {}
                    except Exception:
                        updates_preview = {}
                except Exception:
                    updates_preview = {}

                payload = {
                    "node": node_id,
                    "keys_changed": keys_changed,
                    "checkpoint_ns": checkpoint_ns,
                    "updates_preview": updates_preview,
                }
            else:
                return

            self._enqueue(self._make_event(zl_event_type, {
                "run_id": run_id,
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "node_id": node_id,
                **payload,
            }))
        except Exception as exc:
            logger.debug(f"[obsv_v2] tap error (non-fatal): {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_event(self, event_type: str, data: dict) -> str:
        """Serialize one event to a JSON line."""
        try:
            return json.dumps({"ts": time.time(), "agent": self._agent,
                               "event_type": event_type, **data},
                              default=str) + "\n"
        except Exception:
            return json.dumps({"ts": time.time(), "agent": self._agent,
                               "event_type": event_type, "_err": "serialize_failed"}) + "\n"

    def _enqueue(self, line: str) -> None:
        """Push to ring buffer and wake writer. Never raises."""
        try:
            self._buffer.append(line)
            self._notify.set()
        except Exception:
            pass

    def _open_file(self):
        """Open (or create) the current events file. Returns (file, path, size)."""
        path = self._dir / f"{self._agent}.events.jsonl"
        size = path.stat().st_size if path.exists() else 0
        # If existing file already over limit, rotate it first
        if size >= _MAX_FILE_BYTES:
            self._rotate_file_sync(path, size)
            size = 0
        f = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
        return f, path, size

    def _rotate_file_sync(self, path: Path, current_size: int) -> None:
        """
        Synchronous rotation (called from __init__ before writer starts).
        Renames current file to archive, gzips in background.
        """
        try:
            ts = int(time.time())
            archive_path = self._archive_dir / f"{self._agent}.events.{ts}.jsonl"
            path.rename(archive_path)
            # Kick off background gzip (daemon thread, fire-and-forget)
            threading.Thread(
                target=_gzip_file, args=(archive_path,), daemon=True
            ).start()
        except Exception as exc:
            logger.debug(f"[obsv_v2] rotate_sync error: {exc}")

    def _write_sync(self, line: str) -> None:
        """Write one line synchronously (called from __init__ before writer thread)."""
        try:
            if self._current_file and not self._current_file.closed:
                self._current_file.write(line)
                self._current_file.flush()
                self._current_size += len(line.encode("utf-8"))
        except Exception as exc:
            logger.debug(f"[obsv_v2] write_sync error: {exc}")

    # ── Writer thread ─────────────────────────────────────────────────────────

    def _write_loop(self) -> None:
        """Background daemon thread: drain buffer → write → rotate when needed."""
        while not self._stop.is_set():
            self._notify.wait(timeout=_WRITER_WAKE_INTERVAL_S)
            self._notify.clear()
            self._drain()

    def _drain(self) -> None:
        """Flush all pending lines from buffer to file."""
        while self._buffer:
            try:
                line = self._buffer.popleft()
            except IndexError:
                break
            try:
                if self._current_file is None or self._current_file.closed:
                    self._current_file, self._current_path, self._current_size = self._open_file()
                self._current_file.write(line)
                self._current_size += len(line.encode("utf-8"))
                # Check rotation threshold
                if self._current_size >= _MAX_FILE_BYTES:
                    self._rotate()
            except Exception as exc:
                logger.debug(f"[obsv_v2] drain write error: {exc}")

        try:
            if self._current_file and not self._current_file.closed:
                self._current_file.flush()
        except Exception:
            pass

    def _rotate(self) -> None:
        """
        Rotate current file:
          1. Close current file
          2. Rename to archive/{agent}.events.{ts}.jsonl
          3. Start background gzip
          4. Open new file
          5. Write session_resume markers for all active sessions
        """
        try:
            # Capture active sessions under lock before closing
            with self._sessions_lock:
                active = dict(self._active_sessions)

            # Close current file
            try:
                self._current_file.flush()
                self._current_file.close()
            except Exception:
                pass

            # Rename to archive
            ts = int(time.time())
            if self._current_path and self._current_path.exists():
                archive_path = self._archive_dir / f"{self._agent}.events.{ts}.jsonl"
                try:
                    self._current_path.rename(archive_path)
                    threading.Thread(
                        target=_gzip_file, args=(archive_path,), daemon=True
                    ).start()
                except Exception as exc:
                    logger.debug(f"[obsv_v2] rotate rename error: {exc}")

            # Open new file
            new_path = self._dir / f"{self._agent}.events.jsonl"
            self._current_file = open(new_path, "a", encoding="utf-8", buffering=1)
            self._current_path = new_path
            self._current_size = 0

            # Write session_resume markers for every active (unclosed) session
            for thread_id, run_start_evt in active.items():
                try:
                    resume_line = self._make_event("session_resume", {
                        "thread_id": thread_id,
                        "run_id": run_start_evt.get("run_id", ""),
                        "input_preview": run_start_evt.get("input_preview", ""),
                    })
                    self._current_file.write(resume_line)
                    self._current_size += len(resume_line.encode("utf-8"))
                except Exception:
                    pass

            self._current_file.flush()
        except Exception as exc:
            logger.debug(f"[obsv_v2] rotate error: {exc}")

    def _graceful_flush(self) -> None:
        """
        atexit handler: drain remaining buffer within 500ms, then close file.
        Called on clean exit (not SIGKILL — use agent_restart for that).
        """
        if not self._enabled:
            return
        try:
            self._stop.set()
            deadline = time.monotonic() + _GRACEFUL_FLUSH_TIMEOUT_S
            while self._buffer and time.monotonic() < deadline:
                self._drain()
                time.sleep(0.01)
            # Final drain pass
            self._drain()
            if self._current_file and not self._current_file.closed:
                self._current_file.flush()
                self._current_file.close()
        except Exception as exc:
            logger.debug(f"[obsv_v2] graceful_flush error: {exc}")


# ---------------------------------------------------------------------------
# Background gzip helper (fire-and-forget)
# ---------------------------------------------------------------------------

def _gzip_file(src: Path) -> None:
    """Compress src to src.gz then remove src. Runs in daemon thread."""
    try:
        gz_path = Path(str(src) + ".gz")
        with open(src, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        src.unlink()
    except Exception as exc:
        logger.debug(f"[obsv_v2] gzip error for {src}: {exc}")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: ObservClientV2 | None = None


def get_client() -> ObservClientV2:
    """
    Return the module-level singleton ObservClientV2.
    agent_name is set lazily via start(); obs_dir read from ZL_OBSERV_DIR env var.
    """
    global _client
    if _client is None:
        obs_dir = os.environ.get("ZL_OBSERV_DIR", "")
        _client = ObservClientV2(agent_name="unknown", obs_dir=obs_dir, defer=True)
    return _client
