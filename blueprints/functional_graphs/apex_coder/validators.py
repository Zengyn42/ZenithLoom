"""DETERMINISTIC nodes for apex_coder TDD pipeline.

Nodes:
  splitter          — extract user_requirements, create working_directory
  reset_for_coder   — clear QA messages, build clean prompt for Coder
"""

import os
import re
import uuid
from pathlib import Path


def splitter(state: dict) -> dict:
    from langchain_core.messages import HumanMessage

    msg = state["messages"][0].content.strip()

    if msg.startswith("/") and Path(msg).is_file():
        user_requirements = Path(msg).read_text(encoding="utf-8")
    else:
        user_requirements = msg

    wd_match = re.search(
        r"[#]*\s*(?:工作目录|working.?dir(?:ectory)?)[:：]\s*(\S+)",
        user_requirements,
        re.IGNORECASE,
    )
    if wd_match:
        working_directory = wd_match.group(1)
    else:
        working_directory = f"/tmp/apex_{uuid.uuid4().hex[:8]}"

    Path(working_directory).mkdir(parents=True, exist_ok=True)
    Path(working_directory, "test_tool", "qa_tests").mkdir(parents=True, exist_ok=True)

    return {
        "user_requirements": user_requirements,
        "working_directory": working_directory,
        "messages": [HumanMessage(content=user_requirements)],
    }


def reset_for_coder(state: dict) -> dict:
    from langchain_core.messages import HumanMessage, RemoveMessage

    user_req = state.get("user_requirements", "")
    working_dir = state.get("working_directory", "")
    qa_bypass = state.get("qa_bypass", False)
    run_qa_script = state.get("run_qa_script", "")

    msgs = state.get("messages", [])
    removals = [RemoveMessage(id=m.id) for m in msgs]

    lines = [f"## User Requirements\n\n{user_req}"]
    lines.append(f"\n## Working Directory: `{working_dir}`")

    if qa_bypass:
        lines.append("\n## QA: BYPASSED (no tests to pass)")
    else:
        lines.append(f"\n## QA Tests")
        lines.append(f"- Runner script: `{run_qa_script}`")
        lines.append(f"- **Read the QA tests FIRST to understand what's expected.**")
        lines.append(f"- **Run `bash {run_qa_script}` and ensure ALL tests pass before finishing.**")
        lines.append(f"- **DO NOT modify any files in `test_tool/qa_tests/`.**")

        qa_dir = os.path.join(working_dir, "test_tool", "qa_tests")
        if os.path.isdir(qa_dir):
            test_files = [f for f in os.listdir(qa_dir) if f.endswith(".py") and f != "__init__.py"]
            if test_files:
                lines.append(f"\nQA test files ({len(test_files)}):")
                for f in sorted(test_files):
                    lines.append(f"  - `{qa_dir}/{f}`")

    prompt = "\n".join(lines)
    return {
        "messages": removals + [HumanMessage(content=prompt)],
    }
