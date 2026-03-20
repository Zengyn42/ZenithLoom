"""Deterministic validator nodes for colony_coder_integrator.

Nodes:
  inject_test_context — build a HumanMessage with test_files + working_directory for integration_test
  integration_route   — routes to __end__ (pass/abort) | integration_rescue (retry)
"""

from langchain_core.messages import HumanMessage

RETRY_CAP = 2


def inject_test_context(state: dict) -> dict:
    """Inject test context into messages so integration_test (Ollama) knows what to run.

    test_files are in {working_directory}/test_tool/ and test the source code
    in {working_directory}/. The integration_test runs test_tool against code_gen output.
    """
    working_dir = state.get("working_directory", "")
    test_files = state.get("test_files") or []
    final_files = state.get("final_files") or []
    refined_plan = state.get("refined_plan", "")

    lines = ["## Integration Test Instructions\n"]
    lines.append(f"**Working directory**: `{working_dir}`\n")

    if test_files:
        lines.append("**Test tool files** (pre-written tests in `test_tool/` subfolder):")
        for tf in test_files:
            lines.append(f"  - `{tf}`")
        lines.append(f"\n**How to run**: `cd {working_dir} && python3 <test_file>`")
        lines.append("Each test file imports source code from the parent directory (working_directory).")
    else:
        lines.append("⚠️ No test files found in test_tool/. Submit validation with status='fail', category='missing_tests'.")

    if final_files:
        lines.append(f"\n**Source files to verify** (written by code_gen): {', '.join(f'`{f}`' for f in final_files)}")
        lines.append("The test_tool tests should exercise these source files.")

    if refined_plan:
        lines.append(f"\n**Plan summary**: {refined_plan[:300]}")

    lines.append("\n**Your job**:")
    lines.append("1. Run each test file from test_tool/.")
    lines.append("2. If ALL tests pass (exit 0): submit_validation with status='pass'.")
    lines.append("3. If ANY test fails: submit_validation with status='fail', include the full error output in rationale, set category='test_failure'.")
    lines.append("4. Do NOT modify any files. Just run and report.")

    content = "\n".join(lines)
    return {"messages": [HumanMessage(content=content)]}


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
