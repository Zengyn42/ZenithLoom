"""
Integration tests for observability server (hub + schema).

Tests:
  1. ObservHub.ingest() updates agent state correctly
  2. node_start → running, node_end → done/error
  3. run_start resets node states to idle
  4. hub.agents_snapshot() returns correct structure
  5. hub.subscribe() / unsubscribe() queue lifecycle
  6. Broadcast: ingest() puts event in all subscriber queues
  7. hub handles node_states_reset synthetic event
  8. ObservEvent serialise / deserialise round-trip
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from observability.schema import ObservEvent
from observability.hub import ObservHub


def _make_event(**kwargs) -> ObservEvent:
    defaults = dict(
        v=1,
        agent_name="hani",
        thread_id="thread-1",
        run_id="run-1",
        checkpoint_ns="",
        node_id="__graph__",
        event_type="run_start",
        payload={},
        timestamp=time.time(),
        seq=1,
    )
    defaults.update(kwargs)
    return ObservEvent(**defaults)


# ---------------------------------------------------------------------------
# 1. ingest() creates agent entry on first event
# ---------------------------------------------------------------------------

def test_ingest_creates_agent():
    hub = ObservHub()
    evt = _make_event(agent_name="hani", event_type="run_start", seq=1)
    hub.ingest(evt)
    snapshot = hub.agents_snapshot()
    names = [a["name"] for a in snapshot]
    assert "hani" in names


# ---------------------------------------------------------------------------
# 2. node_start → running; node_end → done
# ---------------------------------------------------------------------------

def test_node_lifecycle_states():
    hub = ObservHub()
    hub.ingest(_make_event(event_type="run_start", seq=1))
    hub.ingest(_make_event(
        event_type="node_start",
        node_id="claude_main",
        payload={"node": "claude_main", "ns": []},
        seq=2,
    ))

    state = hub.agent_node_states("hani")
    assert state["claude_main"] == "running"

    hub.ingest(_make_event(
        event_type="node_end",
        node_id="claude_main",
        payload={"node": "claude_main", "error": None},
        seq=3,
    ))
    state = hub.agent_node_states("hani")
    assert state["claude_main"] == "done"


# ---------------------------------------------------------------------------
# 3. node_end with error → "error" state
# ---------------------------------------------------------------------------

def test_node_end_error_state():
    hub = ObservHub()
    hub.ingest(_make_event(event_type="run_start", seq=1))
    hub.ingest(_make_event(
        event_type="node_start",
        node_id="api_node",
        payload={"node": "api_node", "ns": []},
        seq=2,
    ))
    hub.ingest(_make_event(
        event_type="node_end",
        node_id="api_node",
        payload={"node": "api_node", "error": "TimeoutError"},
        seq=3,
    ))
    state = hub.agent_node_states("hani")
    assert state["api_node"] == "error"


# ---------------------------------------------------------------------------
# 4. run_start resets node states to idle
# ---------------------------------------------------------------------------

def test_run_start_resets_states():
    hub = ObservHub()
    hub.ingest(_make_event(event_type="run_start", seq=1))
    hub.ingest(_make_event(
        event_type="node_end",
        node_id="claude_main",
        payload={"node": "claude_main", "error": None},
        seq=2,
    ))
    # Verify done state
    assert hub.agent_node_states("hani").get("claude_main") == "done"

    # Second run_start should reset
    hub.ingest(_make_event(event_type="run_start", run_id="run-2", seq=3))
    assert hub.agent_node_states("hani").get("claude_main") == "idle"


# ---------------------------------------------------------------------------
# 5. agents_snapshot() structure
# ---------------------------------------------------------------------------

def test_agents_snapshot_structure():
    hub = ObservHub()
    hub.ingest(_make_event(agent_name="hani", event_type="run_start", seq=1))
    hub.ingest(_make_event(agent_name="asa", event_type="run_start", seq=1))

    snapshot = hub.agents_snapshot()
    assert len(snapshot) == 2
    for item in snapshot:
        assert "name" in item
        assert "online" in item
        assert "node_states" in item
        assert "active_run_id" in item


# ---------------------------------------------------------------------------
# 6. subscribe / unsubscribe + broadcast
# ---------------------------------------------------------------------------

def test_subscribe_receives_events():
    hub = ObservHub()
    q = hub.subscribe()

    evt = _make_event(event_type="node_start", payload={"node": "n1", "ns": []}, seq=5)
    hub.ingest(evt)

    assert not q.empty()
    line = q.get_nowait()
    import json
    parsed = json.loads(line)
    assert parsed["event_type"] == "node_start"

    hub.unsubscribe(q)


def test_unsubscribe_stops_receiving():
    hub = ObservHub()
    q = hub.subscribe()
    hub.unsubscribe(q)

    hub.ingest(_make_event(event_type="run_start", seq=1))
    # After unsubscribe, queue should be empty (no more broadcasts)
    assert q.empty()


# ---------------------------------------------------------------------------
# 7. ObservEvent round-trip serialisation
# ---------------------------------------------------------------------------

def test_observevent_json_roundtrip():
    evt = _make_event(
        event_type="node_start",
        node_id="debate_brainstorm",
        payload={"node": "debate_brainstorm", "ns": ["subgraph:123"]},
        seq=42,
    )
    line = evt.to_json()
    restored = ObservEvent.from_json(line)
    assert restored.event_type == "node_start"
    assert restored.node_id == "debate_brainstorm"
    assert restored.seq == 42
    assert restored.payload["node"] == "debate_brainstorm"


# ---------------------------------------------------------------------------
# 8. Multiple agents are tracked independently
# ---------------------------------------------------------------------------

def test_multiple_agents_independent():
    hub = ObservHub()

    hub.ingest(_make_event(agent_name="hani", event_type="run_start", run_id="r-hani", seq=1))
    hub.ingest(_make_event(agent_name="asa", event_type="run_start", run_id="r-asa", seq=1))

    hub.ingest(_make_event(
        agent_name="hani",
        event_type="node_start",
        node_id="hani_node",
        payload={"node": "hani_node", "ns": []},
        seq=2,
    ))

    hani_states = hub.agent_node_states("hani")
    asa_states = hub.agent_node_states("asa")

    assert hani_states.get("hani_node") == "running"
    assert "hani_node" not in asa_states


# ---------------------------------------------------------------------------
# 9. get_recent_events returns correct events
# ---------------------------------------------------------------------------

def test_get_recent_events_returns_correct_events():
    hub = ObservHub()
    hub.ingest(_make_event(agent_name="hani", event_type="run_start", seq=1))
    hub.ingest(_make_event(agent_name="hani", event_type="node_start", payload={"node": "n1", "ns": []}, seq=2))
    hub.ingest(_make_event(agent_name="hani", event_type="node_end", payload={"node": "n1", "error": None}, seq=3))

    events = hub.get_recent_events("hani")
    assert len(events) == 3
    assert events[0]["event_type"] == "run_start"
    assert events[1]["event_type"] == "node_start"
    assert events[2]["event_type"] == "node_end"


# ---------------------------------------------------------------------------
# 10. get_recent_events returns empty list for unknown agent
# ---------------------------------------------------------------------------

def test_get_recent_events_unknown_agent():
    hub = ObservHub()
    events = hub.get_recent_events("nonexistent_agent")
    assert events == []


# ---------------------------------------------------------------------------
# 11. get_recent_events limit parameter
# ---------------------------------------------------------------------------

def test_get_recent_events_limit():
    hub = ObservHub()
    for i in range(10):
        hub.ingest(_make_event(agent_name="hani", event_type="run_start", seq=i + 1))

    events = hub.get_recent_events("hani", limit=3)
    assert len(events) == 3
    # Should be the last 3
    assert events[-1]["seq"] == 10
    assert events[-2]["seq"] == 9
    assert events[-3]["seq"] == 8


# ---------------------------------------------------------------------------
# 12. Ring buffer max 200 events
# ---------------------------------------------------------------------------

def test_ring_buffer_max_200():
    hub = ObservHub()
    for i in range(250):
        hub.ingest(_make_event(agent_name="hani", event_type="run_start", seq=i + 1))

    events = hub.get_recent_events("hani", limit=300)
    assert len(events) == 200
    # Last event should have seq=250
    assert events[-1]["seq"] == 250
    # First event should have seq=51 (250 - 200 + 1)
    assert events[0]["seq"] == 51


# ---------------------------------------------------------------------------
# 13. /api/events/{agent} endpoint structure
# ---------------------------------------------------------------------------

def test_api_events_endpoint():
    """Test the /api/events/{agent_name} endpoint via direct hub calls."""
    hub = ObservHub()
    hub.ingest(_make_event(agent_name="hani", event_type="run_start", seq=1))
    hub.ingest(_make_event(agent_name="hani", event_type="node_start", payload={"node": "n1", "ns": []}, seq=2))

    # Simulate what the endpoint does
    events = hub.get_recent_events("hani", limit=100)
    result = {"agent": "hani", "events": events}

    assert result["agent"] == "hani"
    assert isinstance(result["events"], list)
    assert len(result["events"]) == 2
    for evt in result["events"]:
        assert "event_type" in evt
        assert "agent_name" in evt
        assert "seq" in evt


# ---------------------------------------------------------------------------
# 14. Subgraph expansion in server (mock entity.json reads)
# ---------------------------------------------------------------------------

def test_subgraph_expansion():
    """Test _expand_subgraph_nodes attaches subgraph data to SUBGRAPH_REF nodes."""
    import json
    import tempfile
    import os
    from pathlib import Path

    # Create a temporary directory structure for the child subgraph
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create child subgraph entity.json
        child_dir = Path(tmpdir) / "my_subgraph"
        child_dir.mkdir()
        child_entity = {
            "graph": {
                "nodes": [{"id": "child_node_a", "type": "CLAUDE_SDK"}],
                "edges": [{"from": "child_node_a", "to": "__end__"}],
            }
        }
        (child_dir / "entity.json").write_text(json.dumps(child_entity))

        # Parent graph with a SUBGRAPH_REF node pointing to child_dir
        parent_graph = {
            "nodes": [
                {"id": "normal_node", "type": "CLAUDE_SDK"},
                {"id": "subgraph_node", "type": "SUBGRAPH_REF", "agent_dir": "my_subgraph"},
            ],
            "edges": [{"from": "normal_node", "to": "subgraph_node"}],
        }

        from observability.server import _expand_subgraph_nodes
        expanded = _expand_subgraph_nodes(parent_graph, Path(tmpdir))

        nodes_by_id = {n["id"]: n for n in expanded["nodes"]}

        # normal_node should not have subgraph
        assert "subgraph" not in nodes_by_id["normal_node"]

        # subgraph_node should have subgraph expanded
        assert "subgraph" in nodes_by_id["subgraph_node"]
        sub = nodes_by_id["subgraph_node"]["subgraph"]
        assert len(sub["nodes"]) == 1
        assert sub["nodes"][0]["id"] == "child_node_a"
        assert len(sub["edges"]) == 1


def test_subgraph_expansion_missing_dir_leaves_node_unchanged():
    """If agent_dir doesn't exist, node should be unchanged (no subgraph key)."""
    from pathlib import Path
    from observability.server import _expand_subgraph_nodes

    parent_graph = {
        "nodes": [
            {"id": "missing_sub", "type": "SUBGRAPH_REF", "agent_dir": "nonexistent_dir"},
        ],
        "edges": [],
    }

    expanded = _expand_subgraph_nodes(parent_graph, Path("/tmp"))
    node = expanded["nodes"][0]
    assert "subgraph" not in node
