"""ApexCoderState — apex_coder TDD pipeline state schema.

splitter / ClaudeQA / reset_for_coder / ClaudeCoder 共享此 schema。

Auto-registers as "apex_coder_schema" on import.
"""

from __future__ import annotations

from typing import Annotated, Optional

from framework.schema.base import BaseAgentState
from framework.schema.reducers import _merge_dict
from framework.registry import register_schema


class ApexCoderState(BaseAgentState):
    user_requirements: str
    working_directory: str
    qa_bypass: bool
    qa_tests_dir: str
    run_qa_script: str
    qa_summary: str
    apex_conclusion: str
    node_sessions: Annotated[dict, _merge_dict]


register_schema("apex_coder_schema", ApexCoderState)
