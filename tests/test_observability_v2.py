"""
Unit tests for observability v2 components:
  - framework/observability_client_v2.py
  - observability/viewer.py (snapshot builder + tail reader)

Tests:
  1.  No-op when ZL_OBSERV_DIR='' (enabled=False, no file created)
  2.  emit_run_start / emit_run_end round-trip in JSONL file
  3.  agent_restart anchor written at client init
  4.  tap() whitelist: debug events not in {"task","task_result"} discarded
  5.  tap() whitelist: "task" → node_start, "task_result" → node_end
  6.  tap() updates mode → state_update with updates_preview
  7.  graceful_flush drains buffer before file close
  8.  Rotation: file renamed to archive, new file created, session_resume written
  9.  viewer build_snapshot: run_start → active_runs
  10. viewer build_snapshot: run_end closes matching run
  11. viewer build_snapshot: agent_restart wipes all prior sessions
  12. viewer build_snapshot: session_resume overrides run_start
  13. viewer build_snapshot: empty dir → empty snapshot
  14. viewer _read_tail_lines: returns last N lines
  15. viewer FileTailer: detects new lines on poll
"""

import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.observability_client_v2 import ObservClientV2, _gzip_file
from observability.viewer import build_snapshot, _read_tail_lines, FileTailer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(obs_dir: str = "") -> ObservClientV2:
    """Create a fresh client. Wraps atexit so tests don't leak."""
    return ObservClientV2(agent_name="hani", obs_dir=obs_dir)


def _read_jsonl(path: Path) -> list[dict]:
    """Parse all valid JSON lines from a file."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


# ---------------------------------------------------------------------------
# 1. No-op when obs_dir is empty
# ---------------------------------------------------------------------------

def test_noop_when_obs_dir_empty():
    client = _make_client("")
    assert not client.enabled()
    # None of these should raise or create files
    client.emit_run_start("r", "t", "hi")
    client.emit_run_end("r", "t")
    client.tap((), "debug", {"type": "task", "payload": {"name": "n"}},
               run_id="r", thread_id="t")


# ---------------------------------------------------------------------------
# 2. emit_run_start / emit_run_end appear in JSONL file
# ---------------------------------------------------------------------------

def test_run_start_end_written_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)
        client.emit_run_start("run-1", "thread-1", "hello world")
        client.emit_run_end("run-1", "thread-1")

        # Flush via graceful_flush
        client._graceful_flush()

        events_path = Path(tmpdir) / "hani.events.jsonl"
        assert events_path.exists()

        events = _read_jsonl(events_path)
        types = [e["event_type"] for e in events]
        assert "agent_restart" in types
        assert "run_start" in types
        assert "run_end" in types

        run_start = next(e for e in events if e["event_type"] == "run_start")
        assert run_start["run_id"] == "run-1"
        assert run_start["thread_id"] == "thread-1"
        assert "hello world" in run_start["input_preview"]


# ---------------------------------------------------------------------------
# 3. agent_restart anchor written at init
# ---------------------------------------------------------------------------

def test_agent_restart_anchor_at_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)
        client._graceful_flush()

        events_path = Path(tmpdir) / "hani.events.jsonl"
        events = _read_jsonl(events_path)
        assert events[0]["event_type"] == "agent_restart"
        assert events[0]["agent"] == "hani"


# ---------------------------------------------------------------------------
# 4. tap() whitelist: non-whitelisted debug events discarded
# ---------------------------------------------------------------------------

def test_tap_whitelist_discards_non_task():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)
        # Clear buffer from init (agent_restart)
        client._graceful_flush()
        events_before = _read_jsonl(Path(tmpdir) / "hani.events.jsonl")

        for bad_type in ("checkpoint", "metadata", "on_chain_start", "foo"):
            client.tap(
                ns=(), mode="debug",
                event={"type": bad_type, "payload": {"name": "n"}},
                run_id="r", thread_id="t",
            )

        client._graceful_flush()
        events_after = _read_jsonl(Path(tmpdir) / "hani.events.jsonl")
        # No new events beyond what was there before
        assert len(events_after) == len(events_before)


# ---------------------------------------------------------------------------
# 5. tap() task → node_start, task_result → node_end
# ---------------------------------------------------------------------------

def test_tap_task_events_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)

        client.tap(
            ns=(), mode="debug",
            event={"type": "task", "payload": {"name": "claude_main", "triggers": ["start"]}},
            run_id="r1", thread_id="t1",
        )
        client.tap(
            ns=(), mode="debug",
            event={"type": "task_result", "payload": {"name": "claude_main", "error": None}},
            run_id="r1", thread_id="t1",
        )
        client._graceful_flush()

        events = _read_jsonl(Path(tmpdir) / "hani.events.jsonl")
        types = [e["event_type"] for e in events]
        assert "node_start" in types
        assert "node_end" in types

        node_start = next(e for e in events if e["event_type"] == "node_start")
        assert node_start["node_id"] == "claude_main"


# ---------------------------------------------------------------------------
# 6. tap() updates mode → state_update + updates_preview
# ---------------------------------------------------------------------------

def test_tap_updates_mode_preview():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)

        client.tap(
            ns=(), mode="updates",
            event={"my_node": {"answer": 42, "status": "ok"}},
            run_id="r", thread_id="t",
        )
        client._graceful_flush()

        events = _read_jsonl(Path(tmpdir) / "hani.events.jsonl")
        upd = next((e for e in events if e["event_type"] == "state_update"), None)
        assert upd is not None
        assert "answer" in upd.get("updates_preview", {})
        assert "status" in upd.get("updates_preview", {})


# ---------------------------------------------------------------------------
# 7. graceful_flush drains all pending lines
# ---------------------------------------------------------------------------

def test_graceful_flush_drains_buffer():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)

        # Emit many events into buffer
        for i in range(50):
            client.emit_run_start(f"r{i}", f"t{i}", f"msg {i}")

        # Flush
        client._graceful_flush()

        events = _read_jsonl(Path(tmpdir) / "hani.events.jsonl")
        run_starts = [e for e in events if e["event_type"] == "run_start"]
        assert len(run_starts) == 50


# ---------------------------------------------------------------------------
# 8. Rotation: file rotated when >= 10MB, session_resume written to new file
# ---------------------------------------------------------------------------

def test_rotation_creates_archive_and_session_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        client = _make_client(tmpdir)

        # Simulate active session
        client.emit_run_start("run-active", "thread-active", "active session")
        client._graceful_flush()  # ensure run_start is in _active_sessions

        # Force rotation by calling _rotate() directly
        # (avoids needing to write 10MB of actual data)
        client._drain()  # flush buffer first
        client._rotate()

        client._graceful_flush()  # flush new file too

        # Old file should be in archive (possibly gzip'd, wait briefly)
        archive_dir = Path(tmpdir) / "archive"
        assert archive_dir.exists()

        # New file should have session_resume
        new_file = Path(tmpdir) / "hani.events.jsonl"
        assert new_file.exists()
        events = _read_jsonl(new_file)
        resumes = [e for e in events if e["event_type"] == "session_resume"]
        assert len(resumes) >= 1
        assert resumes[0]["thread_id"] == "thread-active"
        assert resumes[0]["run_id"] == "run-active"


# ---------------------------------------------------------------------------
# Viewer tests — build_snapshot
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, events: list[dict]) -> None:
    """Write events to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps({"ts": time.time(), **evt}) + "\n")


# ---------------------------------------------------------------------------
# 9. build_snapshot: run_start → active_runs
# ---------------------------------------------------------------------------

def test_snapshot_run_start_active():
    with tempfile.TemporaryDirectory() as tmpdir:
        obs_dir = Path(tmpdir)
        _write_jsonl(obs_dir / "hani.events.jsonl", [
            {"agent": "hani", "event_type": "agent_restart"},
            {"agent": "hani", "event_type": "run_start",
             "run_id": "r1", "thread_id": "t1", "input_preview": "hello"},
        ])

        snap = build_snapshot(obs_dir)
        assert snap["type"] == "snapshot"
        assert len(snap["active_runs"]) == 1
        assert snap["active_runs"][0]["run_id"] == "r1"


# ---------------------------------------------------------------------------
# 10. build_snapshot: run_end closes matching run
# ---------------------------------------------------------------------------

def test_snapshot_run_end_closes():
    with tempfile.TemporaryDirectory() as tmpdir:
        obs_dir = Path(tmpdir)
        _write_jsonl(obs_dir / "hani.events.jsonl", [
            {"agent": "hani", "event_type": "agent_restart"},
            {"agent": "hani", "event_type": "run_start",
             "run_id": "r1", "thread_id": "t1"},
            {"agent": "hani", "event_type": "run_end",
             "run_id": "r1", "thread_id": "t1"},
        ])

        snap = build_snapshot(obs_dir)
        assert snap["active_runs"] == []


# ---------------------------------------------------------------------------
# 11. build_snapshot: agent_restart wipes all prior sessions
# ---------------------------------------------------------------------------

def test_snapshot_agent_restart_wipes():
    with tempfile.TemporaryDirectory() as tmpdir:
        obs_dir = Path(tmpdir)
        _write_jsonl(obs_dir / "hani.events.jsonl", [
            {"agent": "hani", "event_type": "run_start",
             "run_id": "r-old", "thread_id": "t1"},
            {"agent": "hani", "event_type": "agent_restart"},
            {"agent": "hani", "event_type": "run_start",
             "run_id": "r-new", "thread_id": "t2"},
        ])

        snap = build_snapshot(obs_dir)
        active_ids = [r["run_id"] for r in snap["active_runs"]]
        assert "r-old" not in active_ids
        assert "r-new" in active_ids


# ---------------------------------------------------------------------------
# 12. build_snapshot: session_resume overrides prior run_start for same thread
# ---------------------------------------------------------------------------

def test_snapshot_session_resume_overrides():
    with tempfile.TemporaryDirectory() as tmpdir:
        obs_dir = Path(tmpdir)
        _write_jsonl(obs_dir / "hani.events.jsonl", [
            {"agent": "hani", "event_type": "agent_restart"},
            {"agent": "hani", "event_type": "run_start",
             "run_id": "r-original", "thread_id": "t1"},
            {"agent": "hani", "event_type": "session_resume",
             "run_id": "r-resumed", "thread_id": "t1",
             "input_preview": "resumed session"},
        ])

        snap = build_snapshot(obs_dir)
        assert len(snap["active_runs"]) == 1
        assert snap["active_runs"][0]["run_id"] == "r-resumed"
        assert snap["active_runs"][0]["event_type"] == "session_resume"


# ---------------------------------------------------------------------------
# 13. build_snapshot: empty dir → empty snapshot
# ---------------------------------------------------------------------------

def test_snapshot_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        snap = build_snapshot(Path(tmpdir))
        assert snap["type"] == "snapshot"
        assert snap["active_runs"] == []


# ---------------------------------------------------------------------------
# 14. _read_tail_lines returns last N lines
# ---------------------------------------------------------------------------

def test_read_tail_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for i in range(100):
            f.write(f'{{"seq": {i}}}\n')
        tmp_path = Path(f.name)

    try:
        lines = _read_tail_lines(tmp_path, 10)
        assert len(lines) == 10
        # Last line should have seq=99
        last = json.loads(lines[-1])
        assert last["seq"] == 99
        # First of the 10 should be seq=90
        first = json.loads(lines[0])
        assert first["seq"] == 90
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 15. FileTailer detects new lines
# ---------------------------------------------------------------------------

def test_file_tailer_detects_new_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "hani.events.jsonl"

        # Create file with initial content
        with open(fpath, "w", encoding="utf-8") as f:
            f.write('{"event_type": "agent_restart"}\n')

        tailer = FileTailer(fpath)
        # First poll: starts from end → no lines (we don't replay history)
        lines = tailer.poll()
        assert lines == []

        # Append new content
        with open(fpath, "a", encoding="utf-8") as f:
            f.write('{"event_type": "run_start", "run_id": "r1"}\n')
            f.write('{"event_type": "node_start", "node_id": "n1"}\n')

        lines = tailer.poll()
        assert len(lines) == 2
        parsed = [json.loads(l) for l in lines]
        assert parsed[0]["event_type"] == "run_start"
        assert parsed[1]["event_type"] == "node_start"

        # Third poll: no new content → empty
        assert tailer.poll() == []
