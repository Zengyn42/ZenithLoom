"""Deterministic validator nodes for colony_coder_qa.

Nodes (each function = one DETERMINISTIC node):
  inject_e2e_context     — build prompt with e2e_plan + working_dir file listing
  run_e2e                — mechanically run test_tool/run_e2e.sh
  e2e_route              — route: pass→__end__, fail→execute (via parent), rescue escalation
  inject_rescue_context  — build full context prompt for qa_rescue
  run_e2e_rescue         — mechanically run test_tool/run_e2e.sh (same as run_e2e)
  rescue_route           — route: pass→__end__, fail→qa_rescue retry, abort
"""

import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

QA_FAIL_CAP = 5
RESCUE_FAIL_CAP = 5


# ---------------------------------------------------------------------------
# inject_e2e_context
# ---------------------------------------------------------------------------

def inject_e2e_context(state: dict) -> dict:
    """Build a HumanMessage with e2e_plan + working_dir context for generate_e2e."""
    from langchain_core.messages import HumanMessage

    e2e_plan = state.get("e2e_plan") or {}
    qa_plan = state.get("qa_plan", "")  # fallback to legacy field
    working_dir = state.get("working_directory", "")

    logger.info(
        f"[inject_e2e_context] working_dir={working_dir!r} "
        f"has_e2e_plan={bool(e2e_plan)} has_qa_plan={bool(qa_plan)}"
    )

    lines = ["## E2E Test Generation Task\n"]
    lines.append(f"**Working directory**: `{working_dir}`\n")

    # E2E plan from planner
    if e2e_plan:
        lines.append("### E2E Test Plan (from Planner)")
        lines.append(f"```json\n{json.dumps(e2e_plan, indent=2, ensure_ascii=False)}\n```\n")

        criteria = e2e_plan.get("acceptance_criteria") or []
        if criteria:
            lines.append("### Acceptance Criteria")
            for i, c in enumerate(criteria, 1):
                lines.append(f"{i}. {c}")
            lines.append("")

        scenarios = e2e_plan.get("test_scenarios") or []
        if scenarios:
            lines.append("### Test Scenarios")
            for s in scenarios:
                lines.append(f"- {s}")
            lines.append("")

        run_cmd = e2e_plan.get("run_command", "")
        if run_cmd:
            lines.append(f"### Run Command: `{run_cmd}`\n")

        headless = e2e_plan.get("headless_notes", "")
        if headless:
            lines.append(f"### Headless Testing Notes\n{headless}\n")
    elif qa_plan:
        # Fallback to legacy qa_plan (plain string)
        lines.append("### QA Plan (legacy format)")
        lines.append(f"```\n{qa_plan}\n```\n")

    # File listing
    if working_dir and os.path.isdir(working_dir):
        all_files = []
        for root, dirs, files in os.walk(working_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), working_dir)
                all_files.append(rel)
        logger.info(f"[inject_e2e_context] found {len(all_files)} files in working_dir")
        if all_files:
            lines.append("### Files in Working Directory")
            for f in sorted(all_files):
                lines.append(f"- `{f}`")
        else:
            lines.append("### Files in Working Directory\n(empty)")
    else:
        logger.warning(f"[inject_e2e_context] working_dir not found or empty: {working_dir!r}")

    lines.append("\n### Your Task")
    lines.append("1. Read the acceptance criteria and test scenarios above.")
    lines.append("2. Examine the source files using Read.")
    lines.append("3. Write E2E test scripts to test_tool/e2e_tests/.")
    lines.append("4. Write test_tool/run_e2e.sh to run all E2E tests.")
    lines.append("5. Run: bash test_tool/run_e2e.sh")
    lines.append("6. Report E2E_VERDICT: PASS or E2E_VERDICT: FAIL.")

    prompt_len = sum(len(l) for l in lines)
    logger.info(f"[inject_e2e_context] built prompt ({prompt_len} chars)")
    return {"messages": [HumanMessage(content="\n".join(lines))]}


# ---------------------------------------------------------------------------
# run_e2e / run_e2e_rescue
# ---------------------------------------------------------------------------

def _run_e2e_tests(state: dict, caller: str = "run_e2e") -> dict:
    """Shared implementation: run test_tool/run_e2e.sh mechanically."""
    working_dir = state.get("working_directory", "")
    runner = os.path.join(working_dir, "test_tool", "run_e2e.sh")

    logger.info(f"[{caller}] runner={runner} exists={os.path.isfile(runner)}")

    if not os.path.isfile(runner):
        logger.warning(f"[{caller}] runner not found: {runner}")
        return {
            "execution_stdout": "",
            "execution_stderr": f"test_tool/run_e2e.sh not found in {working_dir}",
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
            f"[{caller}] exit_code={r.returncode} "
            f"stdout_len={len(r.stdout)} stderr_len={len(r.stderr)}"
        )
        if r.returncode != 0:
            logger.debug(f"[{caller}] stderr={r.stderr[-500:]}")
        return {
            "execution_stdout": r.stdout[-3000:] if r.stdout else "",
            "execution_stderr": r.stderr[-3000:] if r.stderr else "",
            "execution_returncode": r.returncode,
        }
    except subprocess.TimeoutExpired:
        logger.error(f"[{caller}] TIMEOUT (120s)")
        return {
            "execution_stdout": "",
            "execution_stderr": "run_e2e.sh exceeded 120s timeout",
            "execution_returncode": -1,
        }
    except OSError as e:
        logger.error(f"[{caller}] OSError: {e}")
        return {
            "execution_stdout": "",
            "execution_stderr": str(e),
            "execution_returncode": -1,
        }


def run_e2e(state: dict) -> dict:
    """Run E2E tests after generate_e2e."""
    logger.info("[run_e2e] → running E2E tests (post generate_e2e)")
    return _run_e2e_tests(state, caller="run_e2e")


def run_e2e_rescue(state: dict) -> dict:
    """Run E2E tests after qa_rescue."""
    logger.info("[run_e2e_rescue] → running E2E tests (post qa_rescue)")
    return _run_e2e_tests(state, caller="run_e2e_rescue")


# ---------------------------------------------------------------------------
# e2e_route
# ---------------------------------------------------------------------------

def e2e_route(state: dict) -> dict:
    """Route E2E test results.

    Pass (rc==0)                      → __end__ (routing_target="__end__", success=true)
    Fail, qa_fail_count < QA_FAIL_CAP → __end__ (routing_target="execute", qa_analysis set)
                                        Parent graph sees routing_target="execute" and loops back.
    Fail, qa_fail_count >= QA_FAIL_CAP→ inject_rescue_context (escalate to rescue)

    Internal E2E script errors (generate_e2e wrote buggy tests):
      Detected by checking if generate_e2e's verdict was FAIL vs test runner exit code.
      For simplicity, we route to generate_e2e for internal retry on first failure within a QA cycle.
    """
    rc = state.get("execution_returncode")
    qa_fail_count = state.get("qa_fail_count", 0)

    logger.info(
        f"[e2e_route] rc={rc} qa_fail_count={qa_fail_count}/{QA_FAIL_CAP}"
    )

    if rc == 0:
        logger.info("[e2e_route] ✅ E2E tests PASSED → __end__ (success)")
        return {
            "routing_target": "__end__",
            "success": True,
        }

    # E2E failed
    logger.info(
        f"[e2e_route] ❌ E2E tests FAILED "
        f"(qa_fail_count will be {qa_fail_count + 1}/{QA_FAIL_CAP})"
    )

    if qa_fail_count + 1 >= QA_FAIL_CAP:
        # Exhausted → escalate to rescue
        logger.warning(
            f"[e2e_route] QA fail cap ({QA_FAIL_CAP}) reached "
            f"→ inject_rescue_context (escalate to rescue)"
        )
        return {"routing_target": "inject_rescue_context"}

    # Send back to executor via parent graph
    stdout = state.get("execution_stdout", "")
    stderr = state.get("execution_stderr", "")
    output = (stdout + "\n" + stderr).strip()

    qa_analysis = (
        f"E2E test failed (attempt {qa_fail_count + 1}/{QA_FAIL_CAP}).\n\n"
        f"Test output:\n{output[-3000:]}"
    )

    logger.info(
        f"[e2e_route] → routing_target='execute' "
        f"(qa_analysis={len(qa_analysis)} chars, qa_fail_count→{qa_fail_count + 1})"
    )

    return {
        "routing_target": "execute",  # exits QA subgraph; parent routes to execute
        "qa_fail_count": qa_fail_count + 1,
        "qa_analysis": qa_analysis,
    }


# ---------------------------------------------------------------------------
# inject_rescue_context
# ---------------------------------------------------------------------------

def inject_rescue_context(state: dict) -> dict:
    """Build full context for qa_rescue: e2e_plan + failure details + source code."""
    from langchain_core.messages import HumanMessage

    working_dir = state.get("working_directory", "")
    e2e_plan = state.get("e2e_plan") or {}
    qa_plan = state.get("qa_plan", "")
    refined_plan = state.get("refined_plan", "")
    qa_fail_count = state.get("qa_fail_count", 0)

    logger.info(
        f"[inject_rescue_context] working_dir={working_dir!r} "
        f"qa_fail_count={qa_fail_count} has_e2e_plan={bool(e2e_plan)} "
        f"has_refined_plan={bool(refined_plan)}"
    )

    stdout = state.get("execution_stdout", "")
    stderr = state.get("execution_stderr", "")
    test_output = (stdout + "\n" + stderr).strip()

    lines = ["## Rescue Mission — Full Context\n"]
    lines.append(f"**Working directory**: `{working_dir}`")
    lines.append(f"**Previous attempts exhausted**: {qa_fail_count} QA cycles failed\n")

    # E2E plan
    if e2e_plan:
        lines.append("### E2E Test Plan (Acceptance Criteria)")
        lines.append(f"```json\n{json.dumps(e2e_plan, indent=2, ensure_ascii=False)}\n```\n")
    elif qa_plan:
        lines.append("### QA Plan (legacy)")
        lines.append(f"```\n{qa_plan}\n```\n")

    # Design
    if refined_plan:
        lines.append("### Design Plan")
        lines.append(f"```\n{refined_plan[:2000]}\n```\n")

    # E2E failure output
    if test_output:
        lines.append("### E2E Test Failure Output")
        lines.append(f"```\n{test_output[-3000:]}\n```\n")

    # Source code
    if working_dir and os.path.isdir(working_dir):
        source_files = []
        for root, dirs, files in os.walk(working_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "test_tool")]
            for f in files:
                if f.endswith(".py"):
                    source_files.append(os.path.join(root, f))

        if source_files:
            lines.append("### Current Source Files (fix these)")
            for sf in sorted(source_files):
                rel = os.path.relpath(sf, working_dir)
                try:
                    with open(sf, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    lines.append(f"\n**`{rel}`** ({len(content.splitlines())} lines):")
                    lines.append(f"```python\n{content}\n```")
                except OSError:
                    lines.append(f"\n**`{rel}`**: could not read file")

    lines.append("\n### Your Mission")
    lines.append("1. Study the E2E test plan — it defines what 'correct' means.")
    lines.append("2. Read the E2E test failure output — it shows what's broken.")
    lines.append("3. Fix or rewrite the SOURCE CODE (do NOT modify E2E tests).")
    lines.append("4. Run: bash test_tool/run_e2e.sh to verify your fix.")
    lines.append("5. Report RESCUE_VERDICT: PASS or RESCUE_VERDICT: FAIL.")

    prompt_len = sum(len(l) for l in lines)
    logger.info(f"[inject_rescue_context] built rescue prompt ({prompt_len} chars)")
    return {"messages": [HumanMessage(content="\n".join(lines))]}


# ---------------------------------------------------------------------------
# rescue_route
# ---------------------------------------------------------------------------

def rescue_route(state: dict) -> dict:
    """Route rescue results.

    Pass (rc==0)                             → __end__ (success)
    Fail, rescue_fail_count < RESCUE_FAIL_CAP → qa_rescue (retry)
    Fail, rescue_fail_count >= RESCUE_FAIL_CAP → __end__ (abort)
    """
    rc = state.get("execution_returncode")
    rescue_fail_count = state.get("rescue_fail_count", 0)

    logger.info(
        f"[rescue_route] rc={rc} rescue_fail_count={rescue_fail_count}/{RESCUE_FAIL_CAP}"
    )

    if rc == 0:
        logger.info("[rescue_route] ✅ rescue PASSED → __end__ (success)")
        return {
            "routing_target": "__end__",
            "success": True,
        }

    logger.info(
        f"[rescue_route] ❌ rescue FAILED "
        f"(attempt {rescue_fail_count + 1}/{RESCUE_FAIL_CAP})"
    )

    if rescue_fail_count + 1 >= RESCUE_FAIL_CAP:
        logger.warning(
            f"[rescue_route] rescue cap ({RESCUE_FAIL_CAP}) exhausted → __end__ (abort)"
        )
        return {
            "routing_target": "__end__",
            "success": False,
            "abort_reason": "qa_rescue_exhausted",
        }

    logger.info(
        f"[rescue_route] → qa_rescue (retry {rescue_fail_count + 1})"
    )
    # Retry rescue
    return {
        "routing_target": "qa_rescue",
        "rescue_fail_count": rescue_fail_count + 1,
    }
