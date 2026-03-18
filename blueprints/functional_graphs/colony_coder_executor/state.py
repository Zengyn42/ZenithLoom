"""ColonyCoderExecutorState — LangGraph state schema for colony_coder_executor.

Auto-registers as "colony_executor" schema on import.
"""

from __future__ import annotations

from typing import Annotated, Optional

from framework.state import BaseAgentState
from framework.agent_loader import register_state_schema


def _merge_dict(a: dict, b: dict) -> dict:
    """Merge reducer: b's values overwrite a's for shared keys. Safe for parallel node writes."""
    return {**a, **b}


class ColonyCoderExecutorState(BaseAgentState):
    # Task management
    tasks: list                        # list of {"id", "description", "dependencies"}
    execution_order: list              # ordered list of task ids
    refined_plan: str
    working_directory: str
    current_task_index: int
    current_task_id: str
    retry_count: int
    transient_retry_count: int
    error_history: list
    completed_tasks: list

    # Cross-task issues accumulate across tasks
    cross_task_issues: list

    # Validation output from soft_validate / hard_validate
    validation_output: Optional[dict]

    # Rescue context
    rescue_scope: str
    rescue_rationale: str
    affected_task_ids: list

    # Ollama sessions stored in state (not files) for LangGraph checkpoint compatibility.
    # merge_dict reducer prevents parallel node writes from clobbering each other.
    ollama_sessions: Annotated[dict, _merge_dict]

    # Final output
    final_files: list
    abort_reason: Optional[str]
    success: bool


# Auto-register on import
register_state_schema("colony_executor", ColonyCoderExecutorState)
