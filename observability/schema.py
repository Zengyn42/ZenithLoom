"""
ZenithLoom Observability — Event Schema
observability/schema.py

Canonical definition of ObservEvent for the server side.
The client-side (_Event in framework/observability_client.py) is a slim
copy that avoids a reverse dependency on the observability package.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ObservEvent:
    """
    Single observability event transmitted from an agent process to the server.

    Wire format: JSON line (newline-terminated) over WebSocket text frame.

    event_type values (P1)
    ----------------------
    run_start    — GraphController.run() / invoke() entry
    run_end      — GraphController.run() / invoke() exit (always, even on error)
    node_start   — LangGraph debug event type "task" (node begins executing)
    node_end     — LangGraph debug event type "task_result" (node finished)
    state_update — LangGraph updates mode event (key names only, no full state)
    """

    v: int                  # schema version; always 1 for P1
    agent_name: str         # "hani" | "asa" | "dan" | "jei"
    thread_id: str          # LangGraph thread_id for checkpoint lookup
    run_id: str             # uuid4, one per run() / invoke() call
    checkpoint_ns: str      # LangGraph namespace path (real-time ↔ replay bridge)
    node_id: str            # node name; "__graph__" for graph-level events
    event_type: str
    payload: dict[str, Any]
    timestamp: float        # time.time()
    seq: int                # per-agent monotonically increasing; used for ordering + loss detection

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "ObservEvent":
        d = json.loads(line)
        return cls(**d)

    @classmethod
    def from_dict(cls, d: dict) -> "ObservEvent":
        return cls(**d)
