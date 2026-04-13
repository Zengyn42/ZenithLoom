"""
subgraph_init_node — symmetric init/exit cleanup nodes for subgraph boundaries.

Injected by _build_declarative() at subgraph build time:

    START → [_subgraph_init] → entry → ... → exit → _subgraph_exit → END

_subgraph_init:  entry cleanup, only for fresh_per_call / isolated (persistent/inherit = None)
_subgraph_exit:  exit cleanup, for ALL subgraphs — RemoveMessage all internal messages

See docs/vault/architecture/unified-subgraph-integration.md for design.
"""

import logging
from langchain_core.messages import HumanMessage, RemoveMessage
from framework.schema.reducers import CLEAR_DICT

logger = logging.getLogger(__name__)


def make_subgraph_init(session_mode: str, keep_fields: list[str] | None = None):
    """Return entry cleanup function per session_mode.

    Args:
        session_mode: "fresh_per_call", "isolated", etc.
        keep_fields: field names to preserve (not clear) during fresh_per_call init.
                     Declared via "fresh_keep_fields" in entity.json subgraph node def.

    Returns None for persistent, inherit, and unknown modes (no init needed).
    """
    _keep = frozenset(keep_fields or [])

    if session_mode == "fresh_per_call":

        def _fresh_init(state: dict) -> dict:
            msgs = state.get("messages", [])
            removals = [RemoveMessage(id=m.id) for m in msgs]
            human_msgs = [m for m in reversed(msgs) if getattr(m, "type", "") == "human"]
            if human_msgs:
                fresh = [HumanMessage(content=human_msgs[0].content)]
            elif msgs:
                last = msgs[-1]
                fresh = [type(last)(content=last.content)]
            else:
                fresh = []
            _topic = state.get("routing_context", "") or state.get("subgraph_topic", "")
            logger.debug(
                "[subgraph_init:fresh_per_call] clearing sessions + output fields + "
                "trimming messages %d → %d (keep_fields=%s)", len(msgs), len(fresh), _keep or "none",
            )
            result = {
                "node_sessions": CLEAR_DICT.copy(),
                "messages": removals + fresh,
                "routing_context": "",
                "debate_conclusion": "",
                "apex_conclusion": "",
                "knowledge_result": "",
                "discovery_report": "",
                "previous_node_output": "",
                "subgraph_topic": _topic,
            }
            # Preserve fields specified by fresh_keep_fields
            for field in _keep:
                result.pop(field, None)
            return result

        return _fresh_init

    elif session_mode == "isolated":

        def _isolated_init(state: dict) -> dict:
            logger.debug("[subgraph_init:isolated] clearing node_sessions")
            return {"node_sessions": CLEAR_DICT.copy()}

        return _isolated_init

    else:  # persistent, inherit, unknown
        return None


def make_subgraph_exit():
    """Return exit cleanup function — uniform for ALL subgraphs.

    Removes all internal messages via RemoveMessage so they don't
    pollute the parent graph's message list.
    """

    def _exit_cleanup(state: dict) -> dict:
        msgs = state.get("messages", [])
        removals = [RemoveMessage(id=m.id) for m in msgs]
        logger.debug("[subgraph_exit] removing %d internal messages", len(msgs))
        return {"messages": removals}

    return _exit_cleanup
