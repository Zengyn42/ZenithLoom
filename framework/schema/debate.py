"""
debate_schema — 辩论子图专用 state schema

DebateState: 继承 BaseAgentState，覆盖 messages 使用 add_messages reducer
保留全部消息历史（每轮发言都可见）。
debate_claude_first / debate_gemini_first 子图使用此 schema。
"""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from framework.schema.base import BaseAgentState


class DebateState(BaseAgentState):
    """辩论子图专用 state：继承 BaseAgentState，覆盖 messages 保留全部历史。"""
    messages: Annotated[list[BaseMessage], add_messages]


# Auto-register on import
from framework.registry import register_schema

register_schema("debate_schema", DebateState)
