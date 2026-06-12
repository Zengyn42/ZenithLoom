"""
Unit tests for framework/observability_client.py (P1)

Tests:
  1. ZL_OBSERV_URL="" → client is no-op (enabled=False)
  2. Whitelist filter: debug events NOT in {"task","task_result"} are discarded
  3. Whitelist filter: "task" and "task_result" ARE passed through
  4. Queue full → oldest item is dropped, new item inserted
  5. Serialisation failure → event is degraded (no raise)
  6. run_start / run_end events are enqueued correctly
  7. state_update events (updates mode) are enqueued with keys_changed
  8. tap() never raises regardless of bad input
"""

import asyncio
import importlib
import os
import sys
import time
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_client(url: str = "ws://127.0.0.1:9999/ingest"):
    """Create a fresh ObservClient with the given URL (no background task started)."""
    # Re-import to get a fresh module state (reset singleton)
    from framework.observability_client import ObservClient
    return ObservClient(url)


# ---------------------------------------------------------------------------
# 1. No-op when ZL_OBSERV_URL is empty
# ---------------------------------------------------------------------------

def test_noop_when_url_empty():
    client = _make_client("")
    assert not client.enabled()
    # emit_run_start / emit_run_end should not raise
    client.emit_run_start("rid", "tid", "hello")
    client.emit_run_end("rid", "tid")
    # Queue should be empty
    assert client._queue.qsize() == 0


# ---------------------------------------------------------------------------
# 2. Debug event NOT in whitelist is discarded before queue
# ---------------------------------------------------------------------------

def test_whitelist_blocks_non_task_debug_events():
    client = _make_client()
    client._agent_name = "hani"

    # These debug types should be discarded
    for bad_type in ("checkpoint", "metadata", "on_chain_start", "on_chain_end", "unknown"):
        client.tap(
            ns=(),
            mode="debug",
            event={"type": bad_type, "payload": {"name": "some_node"}},
            run_id="r1",
            thread_id="t1",
        )

    assert client._queue.qsize() == 0, "Non-whitelisted debug events should not enter queue"


# ---------------------------------------------------------------------------
# 3. Whitelist allows "task" and "task_result"
# ---------------------------------------------------------------------------

def test_whitelist_allows_task_events():
    client = _make_client()
    client._agent_name = "hani"

    client.tap(
        ns=(),
        mode="debug",
        event={"type": "task", "payload": {"name": "claude_main", "triggers": ["start"]}},
        run_id="r1",
        thread_id="t1",
    )
    client.tap(
        ns=(),
        mode="debug",
        event={"type": "task_result", "payload": {"name": "claude_main", "error": None}},
        run_id="r1",
        thread_id="t1",
    )

    assert client._queue.qsize() == 2, "task and task_result should be enqueued"

    import json
    first_line = client._queue.get_nowait()
    evt = json.loads(first_line)
    assert evt["event_type"] == "node_start"
    assert evt["node_id"] == "claude_main"

    second_line = client._queue.get_nowait()
    evt2 = json.loads(second_line)
    assert evt2["event_type"] == "node_end"


# ---------------------------------------------------------------------------
# 4. Queue full → drop oldest, insert new
# ---------------------------------------------------------------------------

def test_queue_full_drops_oldest():
    from framework.observability_client import ObservClient
    import asyncio

    client = ObservClient("ws://127.0.0.1:9999/ingest")
    client._agent_name = "hani"
    # Fill the queue to capacity with sentinel values
    for i in range(4096):
        client._queue.put_nowait(f"item_{i}\n")

    assert client._queue.qsize() == 4096
    assert client._dropped == 0

    # One more tap should drop oldest and insert new
    client.tap(
        ns=(),
        mode="debug",
        event={"type": "task", "payload": {"name": "overflow_node", "triggers": []}},
        run_id="r2",
        thread_id="t2",
    )

    assert client._dropped == 1
    assert client._queue.qsize() == 4096

    # The last item should be the new event, not item_0
    # Drain and check last item
    last = None
    while not client._queue.empty():
        last = client._queue.get_nowait()
    assert last is not None
    import json
    evt = json.loads(last)
    assert evt["event_type"] == "node_start"
    assert evt["node_id"] == "overflow_node"


# ---------------------------------------------------------------------------
# 5. Serialisation failure → no raise
# ---------------------------------------------------------------------------

def test_serialisation_failure_no_raise():
    """
    If an event contains an unserializable object, _enqueue should degrade
    gracefully and not raise.
    """
    from framework.observability_client import _safe_json, _safe_payload

    # _safe_json must not raise
    class Unserializable:
        pass

    result = _safe_json({"key": Unserializable()})
    # Should produce a JSON string (using default=str fallback)
    import json
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# 6. run_start / run_end events
# ---------------------------------------------------------------------------

def test_run_start_end_enqueue():
    import json

    client = _make_client()
    client._agent_name = "hani"

    client.emit_run_start("run-abc", "thread-xyz", "hello world")
    client.emit_run_end("run-abc", "thread-xyz")

    assert client._queue.qsize() == 2

    start_evt = json.loads(client._queue.get_nowait())
    assert start_evt["event_type"] == "run_start"
    assert start_evt["run_id"] == "run-abc"
    assert start_evt["thread_id"] == "thread-xyz"
    assert start_evt["payload"]["input_preview"] == "hello world"

    end_evt = json.loads(client._queue.get_nowait())
    assert end_evt["event_type"] == "run_end"
    assert end_evt["run_id"] == "run-abc"


# ---------------------------------------------------------------------------
# 7. state_update (updates mode) event
# ---------------------------------------------------------------------------

def test_state_update_event():
    import json

    client = _make_client()
    client._agent_name = "hani"

    client.tap(
        ns=(),
        mode="updates",
        event={"claude_main": {"messages": ["msg1"], "routing_target": "debate"}},
        run_id="r3",
        thread_id="t3",
    )

    assert client._queue.qsize() == 1
    evt = json.loads(client._queue.get_nowait())
    assert evt["event_type"] == "state_update"
    assert evt["node_id"] == "claude_main"
    assert set(evt["payload"]["keys_changed"]) == {"messages", "routing_target"}


# ---------------------------------------------------------------------------
# 8. tap() never raises on garbage input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ns,mode,event", [
    (None, "debug", None),
    ((), "debug", "not_a_dict"),
    ((), "updates", None),
    ((), "???", {}),
    ((1, 2, 3), "debug", {"type": "task", "payload": None}),
])
def test_tap_never_raises(ns, mode, event):
    client = _make_client()
    client._agent_name = "hani"
    # Must not raise regardless of input
    try:
        client.tap(ns=ns, mode=mode, event=event, run_id="r", thread_id="t")
    except Exception as exc:
        pytest.fail(f"tap() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 9. Monotonic seq counter
# ---------------------------------------------------------------------------

def test_seq_is_monotonic():
    import json

    client = _make_client()
    client._agent_name = "hani"

    for _ in range(5):
        client.emit_run_start("r", "t", "x")

    seqs = []
    while not client._queue.empty():
        evt = json.loads(client._queue.get_nowait())
        seqs.append(evt["seq"])

    assert seqs == sorted(seqs), "seq must be monotonically increasing"
    assert len(set(seqs)) == len(seqs), "seq must be unique"


# ---------------------------------------------------------------------------
# 10. updates mode includes updates_preview with truncated values
# ---------------------------------------------------------------------------

def test_updates_preview_included():
    import json

    client = _make_client()
    client._agent_name = "hani"

    client.tap(
        ns=(),
        mode="updates",
        event={"some_node": {"key1": "hello world", "key2": 42}},
        run_id="r1",
        thread_id="t1",
    )

    assert client._queue.qsize() == 1
    evt = json.loads(client._queue.get_nowait())
    assert evt["event_type"] == "state_update"
    preview = evt["payload"]["updates_preview"]
    assert "key1" in preview
    assert "key2" in preview
    # key1 is a plain string — repr() wraps in quotes, or may be truncated
    assert "hello world" in preview["key1"]


# ---------------------------------------------------------------------------
# 11. updates mode — messages list → last content extracted
# ---------------------------------------------------------------------------

def test_updates_preview_messages_list_last_content():
    import json

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    client = _make_client()
    client._agent_name = "hani"

    msg1 = FakeMessage("first message")
    msg2 = FakeMessage("last message content")

    client.tap(
        ns=(),
        mode="updates",
        event={"chat_node": {"messages": [msg1, msg2]}},
        run_id="r1",
        thread_id="t1",
    )

    assert client._queue.qsize() == 1
    evt = json.loads(client._queue.get_nowait())
    preview = evt["payload"]["updates_preview"]
    assert "messages" in preview
    assert "last message content" in preview["messages"]


# ---------------------------------------------------------------------------
# 12. updates_preview dropped when > 4KB
# ---------------------------------------------------------------------------

def test_updates_preview_dropped_when_too_large():
    import json

    client = _make_client()
    client._agent_name = "hani"

    # Create many keys each with 500-char values so total JSON exceeds 4096 bytes.
    # Each key entry is ~520 chars, so 9 keys = ~4680 chars > 4096 threshold.
    large_updates = {f"key_{i}": "x" * 5000 for i in range(9)}

    client.tap(
        ns=(),
        mode="updates",
        event={"big_node": large_updates},
        run_id="r1",
        thread_id="t1",
    )

    assert client._queue.qsize() == 1
    evt = json.loads(client._queue.get_nowait())
    # updates_preview should be empty dict (dropped due to size)
    preview = evt["payload"]["updates_preview"]
    assert preview == {}, f"Expected empty dict, got keys: {list(preview.keys())}"
