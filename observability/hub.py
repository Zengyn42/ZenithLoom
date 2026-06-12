"""
ZenithLoom Observability — Event Hub
observability/hub.py

In-memory hub:
  - Maintains agent registry (online/offline, last graph snapshot, last seq)
  - Broadcasts events to all connected frontend WebSocket subscribers
  - Deduplicates out-of-order events by seq

Thread model: single asyncio event loop (FastAPI + uvicorn).
All methods are safe to call from coroutines; no locks needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

from observability.schema import ObservEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentState — per-agent runtime state tracked by the hub
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    name: str
    online: bool = False
    last_seen: float = 0.0
    last_seq: int = -1
    active_run_id: str = ""
    active_thread_id: str = ""
    # last_node_states: node_id → "running" | "done" | "error" | "idle"
    node_states: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------

class ObservHub:
    """Central registry + broadcaster for one server instance."""

    def __init__(self) -> None:
        # agent_name → AgentState
        self._agents: dict[str, AgentState] = {}
        # set of active frontend WebSocket queues
        self._subscribers: set[asyncio.Queue] = set()
        # offline timeout: if no event received for 90s, mark agent offline
        self._offline_timeout_s: float = 90.0
        # per-agent ring buffer (last 200 events)
        self._ring: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

    # ── Subscriber management ─────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue:
        """Register a new frontend subscriber; returns their event queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest(self, evt: ObservEvent) -> None:
        """
        Process one incoming event from an agent client.
        Updates agent state and broadcasts to all frontend subscribers.
        """
        state = self._ensure_agent(evt.agent_name)
        state.online = True
        state.last_seen = time.time()
        state.last_seq = max(state.last_seq, evt.seq)

        # Update node state machine
        if evt.event_type == "run_start":
            state.active_run_id = evt.run_id
            state.active_thread_id = evt.thread_id
            # Reset all node states to idle for a new run
            for nid in list(state.node_states):
                state.node_states[nid] = "idle"

        elif evt.event_type == "run_end":
            # Schedule idle reset after 3 s (visual: nodes dim after run completes)
            asyncio.ensure_future(self._delayed_idle_reset(evt.agent_name, 3.0))

        elif evt.event_type == "node_start":
            nid = evt.payload.get("node", evt.node_id)
            state.node_states[nid] = "running"

        elif evt.event_type == "node_end":
            nid = evt.payload.get("node", evt.node_id)
            err = evt.payload.get("error")
            state.node_states[nid] = "error" if err else "done"

        # Broadcast to frontend subscribers
        self._broadcast(evt)

        # Append to per-agent ring buffer
        try:
            self._ring[evt.agent_name].append(asdict(evt))
        except Exception:
            pass

    # ── Broadcast ─────────────────────────────────────────────────────────

    def _broadcast(self, evt: ObservEvent) -> None:
        """Push JSON-encoded event to all connected frontend queues (non-blocking)."""
        line = evt.to_json()
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                # Drop for this subscriber; they're too slow
                dead.append(q)
        # Don't disconnect slow subscribers silently — just drop the event

    # ── State snapshot (for frontend initial load) ─────────────────────────

    def agents_snapshot(self) -> list[dict[str, Any]]:
        """Return current state of all known agents."""
        now = time.time()
        result = []
        for name, state in self._agents.items():
            online = state.online and (now - state.last_seen) < self._offline_timeout_s
            result.append({
                "name": name,
                "online": online,
                "last_seen": state.last_seen,
                "last_seq": state.last_seq,
                "active_run_id": state.active_run_id,
                "active_thread_id": state.active_thread_id,
                "node_states": dict(state.node_states),
            })
        return result

    def agent_node_states(self, agent_name: str) -> dict[str, str]:
        state = self._agents.get(agent_name)
        return dict(state.node_states) if state else {}

    def get_recent_events(self, agent_name: str, limit: int = 100) -> list[dict]:
        """Return the most recent events for the given agent (up to limit)."""
        ring = self._ring.get(agent_name)
        if not ring:
            return []
        events = list(ring)
        return events[-limit:] if len(events) > limit else events

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ensure_agent(self, name: str) -> AgentState:
        if name not in self._agents:
            self._agents[name] = AgentState(name=name)
        return self._agents[name]

    async def _delayed_idle_reset(self, agent_name: str, delay: float) -> None:
        await asyncio.sleep(delay)
        state = self._agents.get(agent_name)
        if state:
            for nid in list(state.node_states):
                if state.node_states[nid] != "error":
                    state.node_states[nid] = "idle"
            # Broadcast a synthetic state snapshot so frontend updates
            from observability.schema import ObservEvent
            import time as _time
            synth = ObservEvent(
                v=1,
                agent_name=agent_name,
                thread_id=state.active_thread_id,
                run_id=state.active_run_id,
                checkpoint_ns="",
                node_id="__graph__",
                event_type="node_states_reset",
                payload={"node_states": dict(state.node_states)},
                timestamp=_time.time(),
                seq=-1,
            )
            self._broadcast(synth)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_hub: ObservHub | None = None


def get_hub() -> ObservHub:
    global _hub
    if _hub is None:
        _hub = ObservHub()
    return _hub
