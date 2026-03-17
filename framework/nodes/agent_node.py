# backward-compat shim — real code moved to framework/nodes/llm/llm_node.py
from framework.nodes.llm.llm_node import (
    LlmNode,
    LlmNode as AgentNode,
    LlmNode as AgentClaudeNode,
    _build_project_section,
    _read_project_file,
    _extract_json,
)
__all__ = ["AgentNode", "AgentClaudeNode", "LlmNode"]
