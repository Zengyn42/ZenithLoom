"""ColonyCoderState — colony_coder 体系共用的 LangGraph state schema.

planner / executor / qa / master 三个子图共享此 schema，
包含任务管理、验证输出、QA / rescue 上下文、Ollama session 等字段。

Auto-registers as "colony_coder_schema" on import.
"""

from __future__ import annotations

from typing import Annotated, Optional

from framework.schema.base import BaseAgentState
from framework.schema.reducers import _merge_dict
from framework.registry import register_schema


class ColonyCoderState(BaseAgentState):
    # Task management
    tasks: list                        # list of {"id", "description", "dependencies"}
    execution_order: list              # ordered list of task ids
    refined_plan: str
    qa_plan: str                       # (legacy) QA test design — kept for backward compat
    e2e_plan: dict                     # E2E test plan from planner: {acceptance_criteria, test_scenarios}
    working_directory: str
    current_task_index: int
    current_task_id: str
    retry_count: int
    transient_retry_count: int
    error_history: list
    completed_tasks: list

    # Cross-task issues accumulate across tasks
    cross_task_issues: list

    # QA feedback
    qa_analysis: str                   # QA failure analysis sent back to executor
    qa_fail_count: int                 # execute↔qa loop counter (cap: 5)
    rescue_fail_count: int             # qa_rescue loop counter (cap: 5)

    # Validation output from deterministic routers
    validation_output: Optional[dict]

    # Routing (last-write-wins to survive concurrent updates from fan-in races)
    routing_target: Annotated[str, lambda a, b: b]

    # Rescue context
    rescue_scope: str
    rescue_rationale: str
    affected_task_ids: list

    # Code execution results
    execution_command: str
    execution_stdout: str
    execution_stderr: str
    execution_returncode: Optional[int]

    # E2E test artifacts
    e2e_test_dir: str                  # path to E2E test directory (written by QA)

    # Override node_sessions with merge reducer for parallel fan-out writes
    # (test_designer + code_gen both write node_sessions simultaneously)
    node_sessions: Annotated[dict, _merge_dict]

    # Ollama sessions stored in state (not files) for LangGraph checkpoint compatibility.
    # merge_dict reducer prevents parallel node writes from clobbering each other.
    ollama_sessions: Annotated[dict, _merge_dict]

    # Test files written by test_designer (ApexCoder) for integration verification
    test_files: list                   # list of test file paths relative to working_directory

    # Final output
    final_files: list
    abort_reason: Optional[str]
    success: bool


# Auto-register on import
register_schema("colony_coder_schema", ColonyCoderState)
