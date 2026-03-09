"""
Asa Agent — LangGraph 状态机工厂（6行）

所有行为由 agents/asa/agent.json 驱动。
使用 Llama 作为主 LLM（LlamaNode）。
"""

from pathlib import Path

from framework.agent_loader import AgentLoader
from framework.graph import get_config, new_session, switch_session  # noqa: F401

_loader = AgentLoader(Path(__file__).parent)
get_engine = _loader.get_engine
invalidate_engine = _loader.invalidate_engine
