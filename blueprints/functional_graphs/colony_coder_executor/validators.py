"""Deterministic validator nodes for colony_coder_executor.

Nodes (each function = one DETERMINISTIC node):
  inject_task_context  — build prompt with task info, clear session, inject dependencies
  run_tests            — git stash snapshot + run test_tool/run_tests.sh
  test_route           — compound regression check, deterministic summary, session reset

Context Explosion Fix (2026-04-17):
  Session reset + deterministic summary + git stash anti-regression + replace_lines.
  See Vault: Colony Coder Context Explosion — Final Implementation Plan.md
"""

import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

TEST_RETRY_CAP = 5


# ── Git Stash Snapshot Helpers ───────────────────────────────────────


def _ensure_git_repo(working_dir: str) -> bool:
    """Ensure working_dir is a git repo with at least one commit."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=working_dir, capture_output=True, timeout=5,
        )
        if r.returncode != 0:
            subprocess.run(
                ["git", "init"], cwd=working_dir,
                capture_output=True, timeout=5, check=True,
            )
            # Set local identity so commits work in any environment
            subprocess.run(
                ["git", "config", "user.email", "colony-coder@zengyn.local"],
                cwd=working_dir, capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "config", "user.name", "Colony Coder"],
                cwd=working_dir, capture_output=True, timeout=5,
            )
            logger.info(f"[git] initialized repo in {working_dir}")

        # Ensure at least one commit (stash requires it)
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_dir, capture_output=True, timeout=5,
        )
        if r.returncode != 0:
            # Set identity if repo existed but has no commits
            subprocess.run(
                ["git", "config", "user.email", "colony-coder@zengyn.local"],
                cwd=working_dir, capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "config", "user.name", "Colony Coder"],
                cwd=working_dir, capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "add", "-A"], cwd=working_dir,
                capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "[colony] init"],
                cwd=working_dir, capture_output=True, text=True, timeout=10,
            )
        return True
    except Exception as e:
        logger.warning(f"[git] ensure repo failed: {e}")
        return False


def _stash_snapshot(working_dir: str, label: str) -> str | None:
    """Save current state via git stash, then re-apply it.

    Returns the stash commit hash for later rollback/diff.
    Working directory is left unchanged after this call.
    """
    if not _ensure_git_repo(working_dir):
        return None
    try:
        # Stage everything (including untracked files)
        subprocess.run(
            ["git", "add", "-A"], cwd=working_dir,
            capture_output=True, timeout=10,
        )
        # Stash with label
        r = subprocess.run(
            ["git", "stash", "push", "-m", f"[colony] {label}"],
            cwd=working_dir, capture_output=True, text=True, timeout=10,
        )
        if "No local changes" in (r.stdout + r.stderr):
            # Nothing to stash — return HEAD as the snapshot
            r2 = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=working_dir, capture_output=True, text=True, timeout=5,
            )
            logger.info(f"[stash] no changes to stash for '{label}'")
            return r2.stdout.strip() if r2.returncode == 0 else None

        # Get stash commit hash (stable, unlike stash@{{N}} indices which shift)
        r2 = subprocess.run(
            ["git", "rev-parse", "stash@{0}"],
            cwd=working_dir, capture_output=True, text=True, timeout=5,
        )
        stash_hash = r2.stdout.strip() if r2.returncode == 0 else None

        # Re-apply immediately so working directory stays unchanged
        subprocess.run(
            ["git", "stash", "apply"],
            cwd=working_dir, capture_output=True, timeout=10,
        )

        if stash_hash:
            logger.info(f"[stash] snapshot '{label}' → {stash_hash[:8]}")
        return stash_hash
    except Exception as e:
        logger.warning(f"[stash] snapshot failed: {e}")
        return None


def _stash_diff_summary(working_dir: str, stash_hash: str | None) -> str:
    """Compact diff summary between stash snapshot and current working tree."""
    if not stash_hash:
        return ""
    try:
        r = subprocess.run(
            ["git", "diff", "--stat", stash_hash],
            cwd=working_dir, capture_output=True, text=True, timeout=10,
        )
        return r.stdout[-1000:] if r.stdout else ""
    except Exception:
        return ""


def _stash_restore(working_dir: str, stash_hash: str) -> bool:
    """Restore working directory to a stash snapshot.

    Discards all current changes, then re-applies the stash.
    """
    try:
        # Discard all current modifications and untracked files
        subprocess.run(
            ["git", "checkout", "."],
            cwd=working_dir, capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=working_dir, capture_output=True, timeout=10,
        )
        # Re-apply the snapshot
        r = subprocess.run(
            ["git", "stash", "apply", stash_hash],
            cwd=working_dir, capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            logger.info(f"[stash] restored to {stash_hash[:8]}")
            return True
        else:
            logger.error(f"[stash] apply failed: {r.stderr[:200]}")
            return False
    except Exception as e:
        logger.error(f"[stash] restore failed: {e}")
        return False


# ── Test Result Parsing ──────────────────────────────────────────────


def _parse_pytest_results(stdout: str, stderr: str) -> dict:
    """Parse pytest output for counts and test names."""
    output = stdout + "\n" + stderr

    counts = {"passed": 0, "failed": 0, "errors": 0, "total": 0}
    m = re.search(r"(\d+) passed", output)
    if m:
        counts["passed"] = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        counts["failed"] = int(m.group(1))
    m = re.search(r"(\d+) error", output)
    if m:
        counts["errors"] = int(m.group(1))
    counts["total"] = counts["passed"] + counts["failed"] + counts["errors"]

    # Extract test names from PASSED/FAILED lines
    failed_tests = []
    for m in re.finditer(r"FAILED\s+([\w/.:]+)", output):
        failed_tests.append(m.group(1))
    passed_tests = []
    for m in re.finditer(r"PASSED\s+([\w/.:]+)", output):
        passed_tests.append(m.group(1))

    return {
        **counts,
        "failed_tests": failed_tests,
        "passed_tests": passed_tests,
    }


def _check_regression(prev_results: dict | None, curr_results: dict) -> bool:
    """Compound regression detection: 3 conditions must ALL be true.

    1. passed_now < passed_prev
    2. A previously-passing test now fails (true regression)
    3. total_collected didn't increase (rules out 'unlocked more tests')
    """
    if not prev_results:
        return False

    cond1 = curr_results["passed"] < prev_results["passed"]
    prev_passed = set(prev_results.get("passed_tests", []))
    curr_failed = set(curr_results.get("failed_tests", []))
    cond2 = bool(prev_passed & curr_failed)
    cond3 = curr_results["total"] <= prev_results["total"]

    if cond1 and cond2 and cond3:
        regressed = prev_passed & curr_failed
        logger.warning(
            f"[regression] DETECTED: passed {prev_results['passed']}→{curr_results['passed']}, "
            f"regressed tests: {regressed}"
        )
    return cond1 and cond2 and cond3


# ── Session & Summary Helpers ────────────────────────────────────────


def _extract_intent_snippet(state: dict) -> str:
    """Extract the first assistant message from current session (model's initial plan)."""
    node_sessions = state.get("node_sessions") or {}
    session_uuid = node_sessions.get("executor_session")
    if not session_uuid:
        return ""
    ollama_sessions = state.get("ollama_sessions") or {}
    messages = ollama_sessions.get(session_uuid, [])
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            # Take first ~200 tokens (rough: 4 chars/token = 800 chars)
            return msg["content"][:800]
    return ""


def _clear_executor_session(state: dict) -> dict:
    """Return state update dict that clears the executor's Ollama session."""
    node_sessions = state.get("node_sessions") or {}
    session_uuid = node_sessions.get("executor_session")
    if session_uuid:
        ollama_sessions = dict(state.get("ollama_sessions") or {})
        ollama_sessions[session_uuid] = []
        return {"ollama_sessions": ollama_sessions}
    return {}


def _build_deterministic_summary(
    diff_summary: str,
    test_results: dict,
    test_output: str,
    intent_snippet: str = "",
    attempt: int = 0,
) -> str:
    """Build deterministic context summary for session-reset injection."""
    lines = [f"## Iteration Summary (attempt {attempt})\n"]

    if intent_snippet:
        lines.append("### Previous Approach")
        lines.append(f"{intent_snippet}\n")

    if diff_summary:
        lines.append("### Changes Made (diff --stat)")
        lines.append(f"```\n{diff_summary}\n```\n")

    lines.append("### Test Results")
    lines.append(f"- Passed: {test_results.get('passed', 0)}")
    lines.append(f"- Failed: {test_results.get('failed', 0)}")
    lines.append(f"- Errors: {test_results.get('errors', 0)}")

    if test_results.get("failed", 0) > 0 or test_results.get("errors", 0) > 0:
        lines.append("\n### Error Output (truncated)")
        lines.append(f"```\n{test_output[-1500:]}\n```")

    return "\n".join(lines)


# ── Dependency Extraction ────────────────────────────────────────────


def _extract_dependencies(state: dict, working_dir: str) -> str:
    """Extract dependency signatures from Planner-declared dependencies."""
    tasks = state.get("tasks") or []
    dep_lines = []
    seen_files: set[str] = set()

    for task in tasks:
        for dep in task.get("dependencies", []):
            # Skip task-ID dependencies (strings like "t1", "t2")
            if not isinstance(dep, dict):
                continue
            dep_file = dep.get("file", "")
            if not dep_file or dep_file in seen_files:
                continue
            seen_files.add(dep_file)
            full_path = os.path.join(working_dir, dep_file)
            if not os.path.isfile(full_path):
                continue
            try:
                content = open(full_path, encoding="utf-8").read()
                symbols = dep.get("symbols", [])
                if symbols:
                    dep_lines.append(f"#### {dep_file}")
                    for sym in symbols:
                        # Extract symbol name (handle "Class.method() -> Type" format)
                        sym_name = sym.split("(")[0].split(".")[-1].strip()
                        pattern = rf"^((?:class|def|async\s+def)\s+{re.escape(sym_name)}\b.*)"
                        match = re.search(pattern, content, re.MULTILINE)
                        if match:
                            dep_lines.append(f"```python\n{match.group(1)}\n```")
                        else:
                            dep_lines.append(f"- `{sym}` (signature not found)")
            except Exception:
                pass

    return "\n".join(dep_lines) if dep_lines else ""


# ── Node: inject_task_context ────────────────────────────────────────


def inject_task_context(state: dict) -> dict:
    """Build initial context for code_gen. Clears session for fresh start.

    Runs on:
      - First entry (plan -> execute)
      - QA re-entry (qa -> execute)
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

    # Ensure git repo for stash snapshots
    if working_dir:
        _ensure_git_repo(working_dir)

    lines = ["## Coding Task\n"]
    lines.append(f"**Working directory**: `{working_dir}`\n")

    # ── Acceptance Criteria from Planner (tells Qwen what QA will test) ──
    e2e_plan = state.get("e2e_plan") or {}
    criteria = e2e_plan.get("acceptance_criteria") or []
    if criteria:
        lines.append("### Acceptance Criteria (your code MUST pass ALL of these)")
        lines.append("The QA engineer will test your code against these criteria.")
        lines.append("Your unit tests should cover each one.\n")
        for i, c in enumerate(criteria, 1):
            lines.append(f"  AC{i}: {c}")
        lines.append("")

    # QA feedback from previous cycle
    if qa_analysis:
        logger.info(
            f"[inject_task_context] injecting QA feedback "
            f"({len(qa_analysis)} chars, qa_fail_count={qa_fail_count})"
        )
        lines.append("### QA Feedback (from previous attempt)")
        lines.append("The QA engineer tested your code and found issues:")
        lines.append(f"```\n{qa_analysis[-3000:]}\n```\n")
        lines.append("Fix these issues in your implementation.\n")

    # Planner-declared dependencies
    deps = _extract_dependencies(state, working_dir)
    if deps:
        lines.append("### External Dependencies (interfaces you must use)")
        lines.append(deps + "\n")

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
                task_deps = task.get("dependencies") or []
                dep_str = f" (depends on: {', '.join(task_deps)})" if task_deps else ""
                lines.append(
                    f"- **{tid}**: {task.get('description', '')}{dep_str}"
                )

    lines.append("\n### Instructions")
    lines.append("⚠️ **STEP 1 IS CRITICAL — DO NOT SKIP IT.**")
    lines.append("1. **FIRST: Write the PRODUCT CODE** — create the main source files described in the task. This is your PRIMARY job. Do NOT skip to writing tests.")
    lines.append("2. Use `read_file` to examine existing files (output includes line numbers).")
    lines.append("3. Use `write_file` for NEW files, `replace_lines` for EDITING existing files.")
    lines.append("4. AFTER product code is complete, write unit tests in test_tool/unit_tests/.")
    lines.append("5. Write test_tool/run_tests.sh to run all tests.")
    lines.append("6. Run tests via `bash_exec` and fix until all pass.")
    if criteria:
        lines.append(
            f"\n**REMINDER**: Your code will be tested against {len(criteria)} "
            f"acceptance criteria listed above. Make sure every AC is satisfied."
        )
    lines.append(
        "\n**IMPORTANT**: Always `read_file` before `replace_lines` to get current line numbers."
    )

    # State updates: clear session, reset counters
    updates: dict = {
        "messages": [HumanMessage(content="\n".join(lines))],
        "retry_count": 0,
        "prev_test_results": None,
        "prev_snapshot_hash": None,
        "intent_snippet": "",
    }
    updates.update(_clear_executor_session(state))
    return updates


# ── Node: run_tests ──────────────────────────────────────────────────


def run_tests(state: dict) -> dict:
    """Git stash snapshot current code, then run test_tool/run_tests.sh."""
    working_dir = state.get("working_directory", "")
    runner = os.path.join(working_dir, "test_tool", "run_tests.sh")
    retry_count = state.get("retry_count", 0)

    logger.info(f"[run_tests] runner={runner} exists={os.path.isfile(runner)}")

    # Stash snapshot before testing
    stash_hash = _stash_snapshot(working_dir, f"pre-test-attempt-{retry_count}")

    if not os.path.isfile(runner):
        logger.warning(f"[run_tests] runner not found: {runner}")
        return {
            "execution_stdout": "",
            "execution_stderr": f"test_tool/run_tests.sh not found in {working_dir}",
            "execution_returncode": 1,
            "prev_snapshot_hash": stash_hash,
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
            "prev_snapshot_hash": stash_hash,
        }
    except subprocess.TimeoutExpired:
        logger.error("[run_tests] TIMEOUT (120s)")
        return {
            "execution_stdout": "",
            "execution_stderr": "run_tests.sh exceeded 120s timeout",
            "execution_returncode": -1,
            "prev_snapshot_hash": stash_hash,
        }
    except OSError as e:
        logger.error(f"[run_tests] OSError: {e}")
        return {
            "execution_stdout": "",
            "execution_stderr": str(e),
            "execution_returncode": -1,
            "prev_snapshot_hash": stash_hash,
        }


# ── Node: test_route ─────────────────────────────────────────────────


def test_route(state: dict) -> dict:
    """Route based on test results. Session reset + deterministic summary on retry.

    Pass  -> __end__
    Fail  -> compound regression check -> rollback if needed -> code_gen (retry)
    Cap   -> __end__ (QA will catch remaining issues)
    """
    from langchain_core.messages import HumanMessage

    rc = state.get("execution_returncode")
    retry_count = state.get("retry_count", 0)
    working_dir = state.get("working_directory", "")
    stdout = state.get("execution_stdout", "")
    stderr = state.get("execution_stderr", "")
    output = (stdout + "\n" + stderr).strip()

    # Parse test results
    curr_results = _parse_pytest_results(stdout, stderr)

    if rc == 0:
        logger.info(
            f"[test_route] tests PASSED "
            f"({curr_results['passed']} passed) -> __end__"
        )
        return {"routing_target": "__end__", "prev_test_results": curr_results}

    logger.info(
        f"[test_route] tests FAILED rc={rc} "
        f"(attempt {retry_count + 1}/{TEST_RETRY_CAP}) "
        f"passed={curr_results['passed']} failed={curr_results['failed']}"
    )

    if retry_count >= TEST_RETRY_CAP:
        logger.warning(
            f"[test_route] retry cap ({TEST_RETRY_CAP}) exhausted -> __end__"
        )
        return {"routing_target": "__end__", "prev_test_results": curr_results}

    # ── Compound regression check ──
    prev_results = state.get("prev_test_results")
    prev_hash = state.get("prev_snapshot_hash")
    is_regression = _check_regression(prev_results, curr_results)

    if is_regression and prev_hash and working_dir:
        logger.warning(f"[test_route] REGRESSION detected, restoring stash {prev_hash[:8]}")
        _stash_restore(working_dir, prev_hash)

    # ── Capture intent snippet before clearing session ──
    intent_snippet = _extract_intent_snippet(state)

    # ── Build deterministic summary ──
    diff_sum = _stash_diff_summary(working_dir, prev_hash) if not is_regression else ""
    summary = _build_deterministic_summary(
        diff_summary=diff_sum,
        test_results=curr_results,
        test_output=output,
        intent_snippet=intent_snippet,
        attempt=retry_count + 1,
    )

    # ── Build fresh context message ──
    msg_lines = [summary]
    msg_lines.append("\n## Your Task")

    if is_regression:
        msg_lines.append(
            "**WARNING: Your last change caused a REGRESSION (previously-passing tests now fail). "
            "The code has been rolled back to the last good state.**"
        )
        msg_lines.append("Try a different approach this time.")
    else:
        msg_lines.append("1. Use `read_file` to examine relevant source files (note line numbers).")
        msg_lines.append("2. Fix the failing tests using `replace_lines` or `write_file`.")
        msg_lines.append("3. Tests will be re-run automatically after you finish.")

    # ── Session reset + state updates ──
    updates: dict = {
        "routing_target": "code_gen",
        "retry_count": retry_count + 1,
        "prev_test_results": curr_results,
        "intent_snippet": intent_snippet,
        "messages": [HumanMessage(content="\n".join(msg_lines))],
    }
    updates.update(_clear_executor_session(state))
    return updates
