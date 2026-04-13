"""
Unit tests for make_subgraph_init / make_subgraph_exit — factories that produce
deterministic cleanup nodes for subgraph boundaries.
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage


# ── make_subgraph_init tests ──────────────────────────────────────────────


def test_init_persistent_returns_none():
    """persistent needs no init cleanup."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    assert make_subgraph_init("persistent") is None


def test_init_inherit_returns_none():
    """inherit needs no init cleanup (inherits parent state as-is)."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    assert make_subgraph_init("inherit") is None


def test_init_fresh_per_call_clears_sessions_and_messages():
    """fresh_per_call must clear node_sessions, trim messages to last human, clear output fields."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("fresh_per_call")
    assert fn is not None

    state = {
        "messages": [
            HumanMessage(content="old question", id="h1"),
            AIMessage(content="old answer", id="a1"),
            HumanMessage(content="current topic", id="h2"),
        ],
        "node_sessions": {"claude_main": "uuid-old"},
        "routing_context": "some routing signal",
        "subgraph_topic": "",
        "debate_conclusion": "stale conclusion",
        "apex_conclusion": "stale apex",
        "knowledge_result": "stale knowledge",
        "discovery_report": "stale discovery",
        "previous_node_output": "stale output",
    }
    result = fn(state)

    assert result["node_sessions"] == {}
    assert result["debate_conclusion"] == ""
    assert result["apex_conclusion"] == ""
    assert result["knowledge_result"] == ""
    assert result["discovery_report"] == ""
    assert result["previous_node_output"] == ""
    assert result["routing_context"] == ""
    assert result["subgraph_topic"] == "some routing signal"

    msgs = result["messages"]
    removes = [m for m in msgs if isinstance(m, RemoveMessage)]
    humans = [m for m in msgs if isinstance(m, HumanMessage)]
    assert len(removes) == 3
    assert len(humans) == 1
    assert humans[0].content == "current topic"


def test_init_fresh_per_call_empty_messages():
    """fresh_per_call with no messages should not crash."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("fresh_per_call")
    state = {
        "messages": [],
        "node_sessions": {"x": "y"},
        "routing_context": "",
        "subgraph_topic": "topic",
        "debate_conclusion": "",
        "apex_conclusion": "",
        "knowledge_result": "",
        "discovery_report": "",
        "previous_node_output": "",
    }
    result = fn(state)
    assert len(result["messages"]) == 0
    assert result["subgraph_topic"] == "topic"


def test_init_fresh_per_call_no_human_message_keeps_last():
    """If no HumanMessage exists, keep the last message (whatever type)."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("fresh_per_call")
    state = {
        "messages": [AIMessage(content="only ai msg", id="a1")],
        "node_sessions": {},
        "routing_context": "",
        "subgraph_topic": "",
        "debate_conclusion": "",
        "apex_conclusion": "",
        "knowledge_result": "",
        "discovery_report": "",
        "previous_node_output": "",
    }
    result = fn(state)
    msgs = result["messages"]
    removes = [m for m in msgs if isinstance(m, RemoveMessage)]
    non_removes = [m for m in msgs if not isinstance(m, RemoveMessage)]
    assert len(removes) == 1
    assert len(non_removes) == 1
    assert non_removes[0].content == "only ai msg"


def test_init_fresh_per_call_topic_fallback():
    """If routing_context is empty, preserve existing subgraph_topic."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("fresh_per_call")
    state = {
        "messages": [],
        "node_sessions": {},
        "routing_context": "",
        "subgraph_topic": "existing topic",
        "debate_conclusion": "",
        "apex_conclusion": "",
        "knowledge_result": "",
        "discovery_report": "",
        "previous_node_output": "",
    }
    result = fn(state)
    assert result["subgraph_topic"] == "existing topic"


def test_init_isolated_clears_only_sessions():
    """isolated mode only clears node_sessions."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    fn = make_subgraph_init("isolated")
    assert fn is not None
    state = {
        "node_sessions": {"claude_main": "uuid-123", "gemini": "uuid-456"},
        "messages": [HumanMessage(content="hello", id="h1")],
    }
    result = fn(state)
    assert result == {"node_sessions": {}}


def test_init_unknown_mode_returns_none():
    """Unknown session_mode returns None."""
    from framework.nodes.subgraph_init_node import make_subgraph_init
    assert make_subgraph_init("some_future_mode") is None


# ── make_subgraph_exit tests ──────────────────────────────────────────────


def test_exit_removes_all_messages():
    """Exit must RemoveMessage all internal messages."""
    from framework.nodes.subgraph_init_node import make_subgraph_exit
    fn = make_subgraph_exit()
    assert fn is not None
    state = {
        "messages": [
            HumanMessage(content="topic", id="h1"),
            AIMessage(content="propose", id="a1"),
            AIMessage(content="critique", id="a2"),
            AIMessage(content="revise", id="a3"),
            AIMessage(content="conclusion", id="a4"),
        ],
    }
    result = fn(state)
    msgs = result["messages"]
    removes = [m for m in msgs if isinstance(m, RemoveMessage)]
    non_removes = [m for m in msgs if not isinstance(m, RemoveMessage)]
    assert len(removes) == 5
    assert len(non_removes) == 0
    assert {m.id for m in removes} == {"h1", "a1", "a2", "a3", "a4"}


def test_exit_empty_messages():
    """Exit with no messages returns empty list."""
    from framework.nodes.subgraph_init_node import make_subgraph_exit
    fn = make_subgraph_exit()
    result = fn({"messages": []})
    assert result == {"messages": []}


def test_base_agent_state_uses_add_messages():
    """BaseAgentState must use add_messages reducer, not _keep_last_2."""
    from framework.schema.base import BaseAgentState
    from typing import get_type_hints
    hints = get_type_hints(BaseAgentState, include_extras=True)
    messages_hint = hints["messages"]
    metadata = getattr(messages_hint, "__metadata__", ())
    from langgraph.graph.message import add_messages
    assert add_messages in metadata, (
        f"BaseAgentState.messages should use add_messages reducer, got {metadata}"
    )
