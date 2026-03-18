"""Deterministic validator nodes for colony_coder_planner.

Node: decomposition_validator
  Routes to: execute (pass) | task_decompose (retry) | __end__ (abort)
"""

RETRY_CAP = 2


def decomposition_validator(state: dict) -> dict:
    """Validate task decomposition output from task_decompose.

    Valid if: tasks non-empty, execution_order non-empty, all order IDs exist in tasks.
    """
    tasks = state.get("tasks") or []
    execution_order = state.get("execution_order") or []
    retry_count = state.get("retry_count", 0)

    def _is_valid() -> bool:
        if not tasks or not execution_order:
            return False
        task_ids = {t["id"] for t in tasks if isinstance(t, dict) and "id" in t}
        return bool(task_ids) and all(oid in task_ids for oid in execution_order)

    if _is_valid():
        # "__end__" exits the planner subgraph; master graph continues to executor via fixed edge
        return {"routing_target": "__end__"}

    if retry_count >= RETRY_CAP:
        return {
            "routing_target": "__end__",
            "success": False,
            "abort_reason": "decomposition_failed_after_retries",
        }

    return {"routing_target": "task_decompose", "retry_count": retry_count + 1}
