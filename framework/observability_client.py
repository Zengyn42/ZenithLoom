"""
ZenithLoom Observability Client — framework/observability_client.py

Lightweight asyncio client that taps LangGraph astream events and pushes
them to the ObservabilityServer over WebSocket.

Design principles
-----------------
- The core execution flow MUST NEVER be blocked or raised by this module.
  Every public method wraps its body in try/except and swallows errors silently.
- ZL_OBSERV_URL="" → entire client is a no-op (disabled at import time).
- Bounded asyncio.Queue(4096) — when full, OLDEST item is dropped.
- Exponential back-off reconnect: 1 s → 2 s → … → 60 s cap.
- Events are serialised to JSON lines before sending.
  Serialisation failures degrade to repr() truncated to 1000 chars.

Usage (called by GraphController)
----------------------------------
    from framework.observability_client import get_client

    client = get_client()
    await client.start(loop, agent_name="hani")       # once at GraphController init
    client.emit_run_start(run_id, thread_id, preview) # at run() entry
    client.tap(ns, mode, event, run_id, thread_id)    # inside _astream_graph loop
    client.emit_run_end(run_id, thread_id)            # at run() finally

Event types emitted in P1
--------------------------
    run_start    — start of GraphController.run()
    run_end      — end of GraphController.run() (always, even on error)
    node_start   — debug event type "task"
    node_end     — debug event type "task_result"
    state_update — updates mode event (only key names, no full state)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed debug event types (whitelist — everything else is discarded BEFORE
# entering the queue, zero cost)
# ---------------------------------------------------------------------------
_ALLOWED_DEBUG_TYPES: frozenset[str] = frozenset({"task", "task_result"})

# ---------------------------------------------------------------------------
# Event dataclass (client-side slim version; schema.py in server is canonical)
# ---------------------------------------------------------------------------

@dataclass
class _Event:
    v: int
    agent_name: str
    thread_id: str
    run_id: str
    checkpoint_ns: str
    node_id: str
    event_type: str
    payload: dict
    timestamp: float
    seq: int


def _safe_json(obj: Any) -> str:
    """Serialise obj to JSON; on failure degrade to repr() truncated to 1000 chars."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return json.dumps({"_repr": repr(obj)[:1000]})


def _safe_payload(raw: Any) -> dict:
    """Turn an arbitrary event payload into a JSON-safe dict."""
    try:
        if isinstance(raw, dict):
            # Attempt round-trip through JSON to strip non-serialisable values
            return json.loads(json.dumps(raw, default=str))
        return {"raw": repr(raw)[:500]}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# ObservClient
# ---------------------------------------------------------------------------

class ObservClient:
    """
    Per-process singleton client.
    Thread-safe: all mutation happens in the asyncio event loop that called start().
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._enabled = bool(url)
        self._agent_name: str = "unknown"
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4096)
        self._seq: int = 0
        self._dropped: int = 0
        self._sender_task: asyncio.Task | None = None
        self._started: bool = False

    # ── Public ────────────────────────────────────────────────────────────────

    def enabled(self) -> bool:
        return self._enabled

    async def start(self, agent_name: str) -> None:
        """
        Start background sender task.
        Safe to call multiple times; subsequent calls are no-ops.
        Must be called from within a running asyncio event loop.
        """
        if not self._enabled or self._started:
            return
        try:
            self._agent_name = agent_name
            self._started = True
            self._sender_task = asyncio.ensure_future(self._sender_loop())
        except Exception as exc:
            logger.debug(f"[obsv] start failed (non-fatal): {exc}")

    def tap(
        self,
        ns: tuple,
        mode: str,
        event: Any,
        *,
        run_id: str,
        thread_id: str,
    ) -> None:
        """
        Process one astream chunk.
        Called from _astream_graph — MUST NEVER raise.

        ns    — namespace tuple from LangGraph (subgraphs=True)
        mode  — "values" | "updates" | "debug"
        event — the event dict/payload from LangGraph
        """
        if not self._enabled:
            return
        try:
            if mode == "values":
                return  # GraphController handles values for final state; we ignore

            checkpoint_ns = ":".join(str(x) for x in ns) if ns else ""

            if mode == "debug":
                evt_type = event.get("type", "") if isinstance(event, dict) else ""
                if evt_type not in _ALLOWED_DEBUG_TYPES:
                    return  # whitelist filter — discard before queue
                payload_raw = event.get("payload", {}) if isinstance(event, dict) else {}
                node_id = payload_raw.get("name", "__graph__") if isinstance(payload_raw, dict) else "__graph__"
                zl_event_type = "node_start" if evt_type == "task" else "node_end"
                payload = {
                    "ns": list(ns),
                    "node": node_id,
                    "triggers": payload_raw.get("triggers", []) if isinstance(payload_raw, dict) else [],
                }
                if evt_type == "task_result":
                    payload["error"] = payload_raw.get("error") if isinstance(payload_raw, dict) else None

            elif mode == "updates":
                # event is {node_id: state_updates_dict}
                if not isinstance(event, dict):
                    return
                node_id = next(iter(event), "__graph__")
                keys_changed = list(event.get(node_id, {}).keys()) if isinstance(event.get(node_id), dict) else []
                zl_event_type = "state_update"

                # Build updates_preview: truncated string representation of each changed value
                updates_preview: dict = {}
                try:
                    if isinstance(event, dict):
                        node_updates = event.get(node_id, {})
                        if isinstance(node_updates, dict):
                            for k, v in node_updates.items():
                                try:
                                    if isinstance(v, list) and v:
                                        last = v[-1]
                                        text = getattr(last, 'content', None) or repr(last)
                                        updates_preview[k] = str(text)[:500]
                                    else:
                                        updates_preview[k] = repr(v)[:500]
                                except Exception:
                                    updates_preview[k] = '<unserializable>'
                    # Check total size — if preview dict exceeds 4KB, drop it
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
                return  # unknown mode

            self._enqueue(_Event(
                v=1,
                agent_name=self._agent_name,
                thread_id=thread_id,
                run_id=run_id,
                checkpoint_ns=checkpoint_ns,
                node_id=node_id,
                event_type=zl_event_type,
                payload=payload,
                timestamp=time.time(),
                seq=self._next_seq(),
            ))
        except Exception as exc:
            logger.debug(f"[obsv] tap error (non-fatal): {exc}")

    def emit_run_start(self, run_id: str, thread_id: str, input_preview: str = "") -> None:
        """Emit run_start event. MUST NEVER raise."""
        if not self._enabled:
            return
        try:
            self._enqueue(_Event(
                v=1,
                agent_name=self._agent_name,
                thread_id=thread_id,
                run_id=run_id,
                checkpoint_ns="",
                node_id="__graph__",
                event_type="run_start",
                payload={"input_preview": input_preview[:200]},
                timestamp=time.time(),
                seq=self._next_seq(),
            ))
        except Exception as exc:
            logger.debug(f"[obsv] emit_run_start error (non-fatal): {exc}")

    def emit_run_end(self, run_id: str, thread_id: str) -> None:
        """Emit run_end event. MUST NEVER raise."""
        if not self._enabled:
            return
        try:
            self._enqueue(_Event(
                v=1,
                agent_name=self._agent_name,
                thread_id=thread_id,
                run_id=run_id,
                checkpoint_ns="",
                node_id="__graph__",
                event_type="run_end",
                payload={},
                timestamp=time.time(),
                seq=self._next_seq(),
            ))
        except Exception as exc:
            logger.debug(f"[obsv] emit_run_end error (non-fatal): {exc}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _enqueue(self, evt: _Event) -> None:
        """Put JSON-serialised event in queue; drop OLDEST if full."""
        try:
            line = _safe_json(asdict(evt)) + "\n"
            try:
                self._queue.put_nowait(line)
            except asyncio.QueueFull:
                # Drop oldest, then insert new
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self._dropped += 1
                try:
                    self._queue.put_nowait(line)
                except asyncio.QueueFull:
                    pass
        except Exception as exc:
            logger.debug(f"[obsv] _enqueue error (non-fatal): {exc}")

    async def _sender_loop(self) -> None:
        """Background task: connect to server, drain queue, reconnect on failure."""
        backoff = 1.0
        while True:
            try:
                import websockets
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=5,
                ) as ws:
                    logger.info(f"[obsv] connected to {self._url}")
                    backoff = 1.0  # reset on successful connect
                    while True:
                        line = await asyncio.wait_for(self._queue.get(), timeout=30)
                        try:
                            await ws.send(line)
                        except Exception:
                            # Put back and reconnect
                            try:
                                self._queue.put_nowait(line)
                            except asyncio.QueueFull:
                                pass
                            raise
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(f"[obsv] sender disconnected ({exc}), retry in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: ObservClient | None = None


def get_client() -> ObservClient:
    """Return the module-level singleton ObservClient (created on first call)."""
    global _client
    if _client is None:
        url = os.environ.get("ZL_OBSERV_URL", "ws://127.0.0.1:8765/ingest")
        _client = ObservClient(url)
    return _client
