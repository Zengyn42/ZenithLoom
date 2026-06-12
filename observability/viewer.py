"""
ZenithLoom Observability Viewer v2 — observability/viewer.py

Stateless JSONL-tail → WebSocket bridge.
No REST / no routes / no database / no persistent state.

Architecture:
  - Watches ZL_OBSERV_DIR/*.events.jsonl via stdlib polling (no chokidar/watchfiles dep)
  - New WS connection:
      1. Scan each file from the tail (≤2000 lines) to rebuild active_runs snapshot
      2. Send {"type": "snapshot", "active_runs": [...]} to the new client
      3. Enter tail → broadcast loop
  - File rotation awareness: detects file disappear+recreate (inode change), re-opens
  - Port 8766, single endpoint /ws

Snapshot rebuild rules (from design doc):
  - agent_restart  → invalidate all sessions seen before it for that agent
  - run_end        → close the matching run_id
  - session_resume → treat as active session marker (overrides prior run_start for same thread)
  - run_start      → open session; if no matching run_end in scan window → active

Dependencies: asyncio, websockets (already in requirements.txt), stdlib only.

Run:
    ZL_OBSERV_DIR=~/Foundation/EdenGateway/observability python3 -m observability.viewer
or:
    python3 observability/viewer.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WS_PORT = int(os.environ.get("ZL_VIEWER_PORT", "8766"))
SCAN_TAIL_LINES = 2000          # max lines to scan from tail for snapshot rebuild
POLL_INTERVAL_S = 0.25          # file poll frequency (seconds)
KEEPALIVE_INTERVAL_S = 25.0     # WS ping interval


# ---------------------------------------------------------------------------
# Snapshot builder — pure function, no side effects
# ---------------------------------------------------------------------------

def build_snapshot(obs_dir: Path) -> dict:
    """
    Scan all *.events.jsonl files in obs_dir (≤SCAN_TAIL_LINES each).
    Apply agent_restart / run_end / session_resume / run_start rules.
    Returns {"type": "snapshot", "active_runs": [run_start_or_resume_event, ...]}.

    Rules (applied per-agent within each file's tail window):
      - agent_restart  → wipe all runs accumulated so far for that agent
      - run_end        → remove matching run_id from active set
      - session_resume → add/update active set for (agent, thread_id)
      - run_start      → add to active set for (agent, thread_id) if not yet seen
    """
    # active: (agent, thread_id) → event dict
    active: dict[tuple[str, str], dict] = {}

    jsonl_files = sorted(obs_dir.glob("*.events.jsonl")) if obs_dir.exists() else []

    for fpath in jsonl_files:
        lines = _read_tail_lines(fpath, SCAN_TAIL_LINES)
        agent_wiped: set[str] = set()

        for line in lines:
            try:
                evt = json.loads(line)
            except Exception:
                continue
            agent = evt.get("agent", "")
            event_type = evt.get("event_type", "")
            thread_id = evt.get("thread_id", "")
            run_id = evt.get("run_id", "")

            if event_type == "agent_restart":
                # Wipe all active sessions for this agent accumulated so far
                to_remove = [k for k in active if k[0] == agent]
                for k in to_remove:
                    del active[k]
                agent_wiped.add(agent)

            elif event_type == "run_end":
                # Close the run by run_id
                to_remove = [
                    k for k, v in active.items()
                    if k[0] == agent and v.get("run_id") == run_id
                ]
                for k in to_remove:
                    del active[k]

            elif event_type in ("run_start", "session_resume"):
                # Mark as active — session_resume takes precedence (written at rotation)
                if thread_id:
                    key = (agent, thread_id)
                    # session_resume overwrites; run_start only sets if not present
                    if event_type == "session_resume" or key not in active:
                        active[key] = evt

    return {
        "type": "snapshot",
        "active_runs": list(active.values()),
    }


def _read_tail_lines(path: Path, n: int) -> list[str]:
    """
    Read up to n lines from the end of path.
    Returns lines in chronological order (oldest first).
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return []

            # Read a chunk from the end — 200 bytes/line estimate
            chunk_size = min(size, n * 200)
            f.seek(max(0, size - chunk_size))
            raw = f.read()

        lines = raw.decode("utf-8", errors="replace").splitlines()
        # If we didn't read from the start, the first line may be partial — drop it
        if size > chunk_size:
            lines = lines[1:]

        return lines[-n:]
    except Exception as exc:
        logger.debug(f"[viewer] tail read error for {path}: {exc}")
        return []


# ---------------------------------------------------------------------------
# File tail iterator — yields new lines as they appear
# ---------------------------------------------------------------------------

class FileTailer:
    """
    Polls a single JSONL file and yields new lines.
    Detects file rotation (inode change / file disappear + recreate).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._pos: int = 0
        self._inode: int | None = None
        self._init()

    def _init(self) -> None:
        try:
            if self._path.exists():
                stat = self._path.stat()
                self._inode = stat.st_ino
                self._pos = stat.st_size   # start from end (don't replay history)
            else:
                self._inode = None
                self._pos = 0
        except Exception:
            self._inode = None
            self._pos = 0

    def poll(self) -> list[str]:
        """Return any new lines since last call. Handles rotation silently."""
        lines: list[str] = []
        try:
            if not self._path.exists():
                # File removed (rotation in progress) — reset
                self._inode = None
                self._pos = 0
                return lines

            stat = self._path.stat()
            # Detect rotation: inode changed or file shrank
            if self._inode is not None and (stat.st_ino != self._inode or stat.st_size < self._pos):
                logger.debug(f"[viewer] rotation detected on {self._path.name}")
                self._inode = stat.st_ino
                self._pos = 0   # read new file from beginning (it has session_resume markers)

            self._inode = stat.st_ino

            if stat.st_size <= self._pos:
                return lines

            with open(self._path, "rb") as f:
                f.seek(self._pos)
                raw = f.read(stat.st_size - self._pos)
                self._pos = f.tell()

            text = raw.decode("utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
        except Exception as exc:
            logger.debug(f"[viewer] poll error for {self._path.name}: {exc}")
        return lines


# ---------------------------------------------------------------------------
# Viewer WS Server
# ---------------------------------------------------------------------------

async def _handle_client(websocket: Any, obs_dir: Path) -> None:
    """Handle one WebSocket connection: send snapshot, then tail → broadcast."""
    try:
        # Step 1: Build and send initial snapshot
        snapshot = build_snapshot(obs_dir)
        await websocket.send(json.dumps(snapshot))
        logger.info(f"[viewer] client connected; sent snapshot ({len(snapshot['active_runs'])} active runs)")
    except Exception as exc:
        logger.warning(f"[viewer] snapshot send failed: {exc}")
        return

    # Step 2: Open tailers for all current JSONL files
    tailers: dict[str, FileTailer] = {}

    def _refresh_tailers() -> None:
        """Add tailers for newly created JSONL files."""
        for fpath in obs_dir.glob("*.events.jsonl"):
            key = fpath.name
            if key not in tailers:
                tailers[key] = FileTailer(fpath)

    _refresh_tailers()

    # Step 3: Tail loop
    try:
        last_keepalive = time.monotonic()
        while True:
            _refresh_tailers()

            any_new = False
            for tailer in list(tailers.values()):
                for line in tailer.poll():
                    try:
                        # Validate JSON before forwarding
                        json.loads(line)
                        await websocket.send(line)
                        any_new = True
                    except json.JSONDecodeError:
                        pass  # skip malformed lines
                    except Exception:
                        raise  # WS send failure → reconnect

            # Keepalive
            now = time.monotonic()
            if now - last_keepalive >= KEEPALIVE_INTERVAL_S:
                await websocket.send(json.dumps({"type": "ping"}))
                last_keepalive = now

            if not any_new:
                await asyncio.sleep(POLL_INTERVAL_S)

    except Exception as exc:
        logger.debug(f"[viewer] client disconnected: {exc}")


async def _serve(obs_dir: Path) -> None:
    """Start the WebSocket server on port WS_PORT."""
    try:
        import websockets  # type: ignore
    except ImportError:
        logger.error("[viewer] 'websockets' package not found. Install: pip install websockets")
        sys.exit(1)

    handler = lambda ws: _handle_client(ws, obs_dir)

    async with websockets.serve(handler, "127.0.0.1", WS_PORT):
        logger.info(f"[viewer] ZenithLoom Observability Viewer v2 listening on ws://127.0.0.1:{WS_PORT}/ws")
        logger.info(f"[viewer] Watching: {obs_dir}")
        await asyncio.Future()  # run forever


def main() -> None:
    obs_dir_str = os.environ.get("ZL_OBSERV_DIR", "")
    if not obs_dir_str:
        print("Error: ZL_OBSERV_DIR environment variable not set.", file=sys.stderr)
        print("Usage: ZL_OBSERV_DIR=/path/to/obs_dir python3 -m observability.viewer", file=sys.stderr)
        sys.exit(1)

    obs_dir = Path(obs_dir_str).expanduser().resolve()
    obs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    asyncio.run(_serve(obs_dir))


if __name__ == "__main__":
    main()
