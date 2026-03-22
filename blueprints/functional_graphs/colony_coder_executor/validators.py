"""Deterministic validator nodes for colony_coder_executor.

Nodes (each function = one DETERMINISTIC node):
  inject_task_context  — build prompt with task info + qa_analysis feedback (if any)
  run_tests            — mechanically run test_tool/run_tests.sh
  test_route           — route pass/fail, retry up to TEST_RETRY_CAP
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

TEST_RETRY_CAP = 5


def inject_task_context(state: dict) -> dict:
    """Build a HumanMessage with task context for code_gen.

    On first entry: injects refined_plan + task descriptions.
    On re-entry from QA: also injects qa_analysis feedback.
    On re-entry from test_route: injects test failure output.
    """
    from langchain_core.messages import HumanMessage

    refined_plan = state.get("refined_plan", "")
    tasks = state.get("tasks") or []
    execution_order = state.get("execution_order") or []
    working_dir = state.get("working_directory", "")
    qa_analysis = state.get("qa_analysis", "")
    qa_fail_count = state.get("qa_fail_count", 0)

    logger.info(
        f"[inject_task_context] working_dir={working_dir!r} "
        f"tasks={len(tasks)} qa_fail_count={qa_fail_count} "
        f"has_qa_analysis={bool(qa_analysis)}"
    )

    lines = ["## Coding Task\n"]
    lines.append(f"**Working directory**: `{working_dir}`\n")

    # QA feedback from previous cycle (if any)
    if qa_analysis:
        logger.info(
            f"[inject_task_context] injecting QA feedback "
            f"({len(qa_analysis)} chars, qa_fail_count={qa_fail_count})"
        )
        lines.append("### ⚠️ QA Feedback (from previous attempt)")
        lines.append("The QA engineer tested your code and found issues:")
        lines.append(f"```\n{qa_analysis[-3000:]}\n```\n")
        lines.append("Fix these issues in your implementation.\n")

    # Plan
    if refined_plan:
        lines.append("### Design Plan")
        lines.append(f"```\n{refined_plan[:2000]}\n```\n")

    # Tasks
    if tasks:
        lines.append("### Tasks to Implement")
        for tid in execution_order:
            task = next((t for t in tasks if t.get("id") == tid), None)
            if task:
                deps = task.get("dependencies") or []
                dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
                lines.append(f"- **{tid}**: {task.get('description', '')}{dep_str}")

    lines.append(f"\n### Instructions")
    lines.append("1. Create all source files in the working directory.")
    lines.append("2. Write unit tests in test_tool/unit_tests/.")
    lines.append("3. Write test_tool/run_tests.sh to run all tests.")
    lines.append("4. Run tests and fix until all pass.")

    return {
        "messages": [HumanMessage(content="\n".join(lines))],
        "retry_count": 0,  # reset executor-internal retry count
    }


def run_tests(state: dict) -> dict:
    """Run test_tool/run_tests.sh mechanically. No LLM involved."""
    working_dir = state.get("working_directory", "")
    runner = os.path.join(working_dir, "test_tool", "run_tests.sh")

    logger.info(f"[run_tests] runner={runner} exists={os.path.isfile(runner)}")

    if not os.path.isfile(runner):
        logger.warning(f"[run_tests] runner not found: {runner}")
        return {
            "execution_stdout": "",
            "execution_stderr": f"test_tool/run_tests.sh not found in {working_dir}",
            "execution_returncode": 1,
        }

    try:
        r = subprocess.run(
            ["bash", runner],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        logger.info(
            f"[run_tests] exit_code={r.returncode} "
            f"stdout_len={len(r.stdout)} stderr_len={len(r.stderr)}"
        )
        if r.returncode != 0:
            logger.debug(f"[run_tests] stderr={r.stderr[-500:]}")
        return {
            "execution_stdout": r.stdout[-3000:] if r.stdout else "",
            "execution_stderr": r.stderr[-3000:] if r.stderr else "",
            "execution_returncode": r.returncode,
        }
    except subprocess.TimeoutExpired:
        logger.error("[run_tests] TIMEOUT (120s)")
        return {
            "execution_stdout": "",
            "execution_stderr": "run_tests.sh exceeded 120s timeout",
            "execution_returncode": -1,
        }
    except OSError as e:
        logger.error(f"[run_tests] OSError: {e}")
        return {
            "execution_stdout": "",
            "execution_stderr": str(e),
            "execution_returncode": -1,
        }


def test_route(state: dict) -> dict:
    """Route based on test results. Pass → __end__, Fail → code_gen (retry)."""
    from langchain_core.messages import HumanMessage

    rc = state.get("execution_returncode")
    retry_count = state.get("retry_count", 0)

    if rc == 0:
        logger.info("[test_route] ✅ tests PASSED → __end__")
        return {"routing_target": "__end__"}

    logger.info(
        f"[test_route] ❌ tests FAILED rc={rc} "
        f"(attempt {retry_count + 1}/{TEST_RETRY_CAP})"
    )

    if retry_count >= TEST_RETRY_CAP:
        logger.warning(
            f"[test_route] retry cap ({TEST_RETRY_CAP}) exhausted → __end__ "
            f"(QA will catch remaining issues)"
        )
        return {"routing_target": "__end__"}

    logger.info(f"[test_route] → code_gen (retry {retry_count + 1})")

    # Inject failure details back to code_gen
    stdout = state.get("execution_stdout", "")
    stderr = state.get("execution_stderr", "")
    output = (stdout + "\n" + stderr).strip()

    lines = [f"## Tests FAILED (attempt {retry_count + 1}/{TEST_RETRY_CAP})\n"]
    lines.append("### Test Output")
    lines.append(f"```\n{output[-3000:]}\n```\n")
    lines.append("### Your Task")
    lines.append("1. Read the test output above.")
    lines.append("2. Read your source files using read_file.")
    lines.append("3. Fix the issues.")
    lines.append("4. Write the fixed files back using write_file.")
    lines.append("5. The tests will be re-run automatically after you finish.")

    return {
        "routing_target": "code_gen",
        "retry_count": retry_count + 1,
        "messages": [HumanMessage(content="\n".join(lines))],
    }
