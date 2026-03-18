"""Deterministic validator nodes for colony_coder_integrator.

Node: integration_route
  Routes to: __end__ (pass/abort) | integration_rescue (retry)
"""

RETRY_CAP = 2


def integration_route(state: dict) -> dict:
    """Route integration test results."""
    vo = state.get("validation_output") or {}
    status = vo.get("status", "fail")
    retry_count = state.get("retry_count", 0)

    if status == "pass":
        return {"routing_target": "__end__", "success": True}

    if status == "abort" or retry_count >= RETRY_CAP:
        return {
            "routing_target": "__end__",
            "success": False,
            "abort_reason": vo.get("rationale", "integration_abort"),
        }

    return {"routing_target": "integration_rescue", "retry_count": retry_count + 1}
