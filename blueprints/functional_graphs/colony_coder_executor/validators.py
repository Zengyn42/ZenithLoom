"""Deterministic validator nodes for colony_coder_executor.

Nodes (each function = one DETERMINISTIC node):
  hard_validate     — routes pass/fail/abort based on validation_output
  error_classifier  — classifies error: self_fix vs claude_rescue
  rescue_router     — cross-task issues: dual-write to cross_task_issues + routing_target
  rollback_state    — cascade-rollback affected tasks, route to claude_rescue

Helper (not a node):
  cascade_rollback  — transitively find all tasks dependent on affected_task_ids
"""

RETRY_CAP = 3
TRANSIENT_RETRY_CAP = 2
TRANSIENT_CATEGORIES = {"syntax_error", "import_error", "test_failure", "lint_error"}
CROSS_TASK_CATEGORIES = {"cross_task", "interface_mismatch", "dependency_break"}


def hard_validate(state: dict) -> dict:
    """Route based on validation_output.status."""
    vo = state.get("validation_output") or {}
    status = vo.get("status", "fail")
    retry_count = state.get("retry_count", 0)

    if status == "pass":
        return {"routing_target": "execute"}

    if status == "abort" or retry_count >= RETRY_CAP:
        return {
            "routing_target": "__end__",
            "success": False,
            "abort_reason": vo.get("rationale", "hard_validate_abort"),
        }

    return {"routing_target": "error_classifier"}


def error_classifier(state: dict) -> dict:
    """Classify error type and route to self_fix or claude_rescue."""
    vo = state.get("validation_output") or {}
    category = vo.get("category", "unknown")
    severity = vo.get("severity", "medium")
    transient_retry = state.get("transient_retry_count", 0)

    if category in CROSS_TASK_CATEGORIES:
        return {"routing_target": "rescue_router"}

    if category in TRANSIENT_CATEGORIES and transient_retry < TRANSIENT_RETRY_CAP:
        return {"routing_target": "self_fix", "transient_retry_count": transient_retry + 1}

    return {"routing_target": "claude_rescue"}


def rescue_router(state: dict) -> dict:
    """Handle cross-task failures.

    Dual-write: appends to cross_task_issues (accumulator) AND sets routing_target.
    """
    vo = state.get("validation_output") or {}
    cross_task_issues = list(state.get("cross_task_issues") or [])
    current_task_id = state.get("current_task_id", "")

    issue_record = {
        "task_id": current_task_id,
        "category": vo.get("category"),
        "severity": vo.get("severity"),
        "rationale": vo.get("rationale"),
        "affected_scope": vo.get("affected_scope", ""),
    }
    cross_task_issues.append(issue_record)

    raw_scope = vo.get("affected_scope", "")
    affected_task_ids = [s.strip() for s in raw_scope.split(",") if s.strip()]

    return {
        "routing_target": "rollback_state",
        "cross_task_issues": cross_task_issues,
        "affected_task_ids": affected_task_ids,
        "rescue_scope": "cross_task",
        "rescue_rationale": vo.get("rationale", ""),
    }


def cascade_rollback(tasks: list, affected_task_ids: list) -> set:
    """Helper (not a graph node): transitively find all tasks dependent on affected_task_ids."""
    affected = set(affected_task_ids)
    changed = True
    while changed:
        changed = False
        for task in tasks:
            tid = task.get("id")
            deps = task.get("dependencies") or []
            if tid not in affected and any(d in affected for d in deps):
                affected.add(tid)
                changed = True
    return affected


def rollback_state(state: dict) -> dict:
    """Mark affected tasks for re-execution; route to claude_rescue."""
    tasks = state.get("tasks") or []
    affected_task_ids = state.get("affected_task_ids") or []
    all_affected = cascade_rollback(tasks, affected_task_ids)
    completed_tasks = [t for t in (state.get("completed_tasks") or []) if t not in all_affected]
    return {
        "completed_tasks": completed_tasks,
        "current_task_index": 0,
        "routing_target": "claude_rescue",
    }
