"""Deterministic validator nodes for colony_coder_executor.

Nodes (each function = one DETERMINISTIC node):
  extract_test_files      — scan test_tool/ for test files
  validate_router         — single router: pass/self_fix/rescue/abort
  rescue_router           — cross-task issues: dual-write to cross_task_issues
  rollback_state          — cascade-rollback affected tasks
  inject_rescue_context   — inject test_tool + plan context for claude_rescue

Helper (not a node):
  cascade_rollback   — transitively find all tasks dependent on affected_task_ids
"""

import logging

logger = logging.getLogger(__name__)

def extract_test_files(state: dict) -> dict:
    """Scan working_directory/test_tool/ for test_*.py files written by test_designer.

    test_designer writes files to {working_directory}/test_tool/.
    This node scans that subfolder to populate state["test_files"].
    Paths are relative to working_directory, e.g. "test_tool/test_snake.py".
    """
    import glob
    import os

    working_dir = state.get("working_directory", "")
    test_files = []

    if working_dir:
        test_tool_dir = os.path.join(working_dir, "test_tool")
        found = sorted(glob.glob(f"{test_tool_dir}/test_*.py"))
        test_files = [f"test_tool/{os.path.basename(f)}" for f in found]

    if test_files:
        logger.info(f"[extract_test_files] found: {test_files}")
    else:
        logger.warning(f"[extract_test_files] no test_*.py in {working_dir}/test_tool/")

    return {"test_files": test_files}


SELF_FIX_CAP = 5
CROSS_TASK_CATEGORIES = {"cross_task", "interface_mismatch", "dependency_break"}


def validate_router(state: dict) -> dict:
    """Single router: read validation_output, decide next step.

    pass                    → __end__ (success)
    abort                   → __end__ (fail)
    fail + ≤5 retries       → self_fix (Ollama, same session, any category)
    fail + 5 retries used:
      cross_task category   → rescue_router (record issue) → rollback → claude_rescue
      other category        → rollback_state → inject_rescue_context → claude_rescue
    """
    vo = state.get("validation_output") or {}
    status = vo.get("status", "fail")
    category = vo.get("category", "unknown")
    self_fix_count = state.get("transient_retry_count", 0)

    # pass → done
    if status == "pass":
        return {"routing_target": "__end__", "success": True}

    # abort → give up
    if status == "abort":
        return {
            "routing_target": "__end__",
            "success": False,
            "abort_reason": vo.get("rationale", "validate_router_abort"),
        }

    # any failure + under cap → self_fix
    if self_fix_count < SELF_FIX_CAP:
        return {"routing_target": "self_fix", "transient_retry_count": self_fix_count + 1}

    # 5 retries exhausted → rollback → claude_rescue
    if category in CROSS_TASK_CATEGORIES:
        return {"routing_target": "rescue_router"}

    return {"routing_target": "rollback_state"}


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
    """Mark affected tasks for re-execution; route to inject_rescue_context."""
    tasks = state.get("tasks") or []
    affected_task_ids = state.get("affected_task_ids") or []
    all_affected = cascade_rollback(tasks, affected_task_ids)
    completed_tasks = [t for t in (state.get("completed_tasks") or []) if t not in all_affected]
    return {
        "completed_tasks": completed_tasks,
        "current_task_index": 0,
    }


def inject_rescue_context(state: dict) -> dict:
    """Build a HumanMessage with full context for claude_rescue.

    claude_rescue runs in a separate Claude SDK session, has no access to
    the Ollama executor_session history. This node injects everything it needs:
      - working_directory and file layout
      - test_tool files (paths + contents)
      - planner's refined_plan
      - validation errors and error_history
      - what self_fix attempted (retry count)
    """
    import os

    working_dir = state.get("working_directory", "")
    test_files = state.get("test_files") or []
    refined_plan = state.get("refined_plan", "")
    vo = state.get("validation_output") or {}
    error_history = state.get("error_history") or []
    retry_count = state.get("transient_retry_count", 0)
    cross_task_issues = state.get("cross_task_issues") or []

    lines = ["## Rescue Mission — Full Context\n"]
    lines.append(f"**Working directory**: `{working_dir}`")
    lines.append(f"**self_fix attempts exhausted**: {retry_count} retries failed\n")

    # Planner context
    if refined_plan:
        lines.append("### Planner's Design")
        lines.append(f"```\n{refined_plan[:1000]}\n```\n")

    # Test tool files — read contents so claude_rescue knows the spec
    if test_files:
        lines.append("### Test Tool Files (these are the SPECIFICATION — do NOT modify)")
        for tf in test_files:
            full_path = os.path.join(working_dir, tf)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                lines.append(f"\n**`{tf}`** ({len(content.splitlines())} lines):")
                lines.append(f"```python\n{content}\n```")
            except OSError:
                lines.append(f"\n**`{tf}`**: ⚠️ could not read file")

    # Source files — read current state
    if working_dir and os.path.isdir(working_dir):
        source_files = [
            f for f in os.listdir(working_dir)
            if f.endswith(".py") and not f.startswith("test_") and f != "__init__.py"
        ]
        if source_files:
            lines.append("\n### Current Source Files (these need fixing)")
            for sf in sorted(source_files):
                full_path = os.path.join(working_dir, sf)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    lines.append(f"\n**`{sf}`** ({len(content.splitlines())} lines):")
                    lines.append(f"```python\n{content}\n```")
                except OSError:
                    lines.append(f"\n**`{sf}`**: ⚠️ could not read file")

    # Error details
    lines.append("\n### Latest Validation Error")
    lines.append(f"- status: `{vo.get('status', 'unknown')}`")
    lines.append(f"- category: `{vo.get('category', 'unknown')}`")
    lines.append(f"- rationale: {vo.get('rationale', 'N/A')}")

    if error_history:
        lines.append(f"\n### Error History ({len(error_history)} entries)")
        for i, err in enumerate(error_history[-5:], 1):
            lines.append(f"{i}. {err}")

    if cross_task_issues:
        lines.append(f"\n### Cross-task Issues")
        for issue in cross_task_issues:
            lines.append(f"- task `{issue.get('task_id')}`: {issue.get('rationale', '')}")

    lines.append("\n### Your Mission")
    lines.append("1. Read the test files in test_tool/ — they are the SPECIFICATION.")
    lines.append("2. Fix or rewrite the source code so ALL tests pass.")
    lines.append(f"3. Run: `cd {working_dir} && python3 <test_file>` to verify.")
    lines.append("4. Do NOT modify test_tool/ files.")

    from langchain_core.messages import HumanMessage
    return {"messages": [HumanMessage(content="\n".join(lines))]}
