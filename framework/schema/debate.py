"""
debate_schema — 辩论子图专用 state schema

DebateState: 保留全部消息历史（每轮发言都可见），使用 add_messages reducer。
debate_claude_first / debate_gemini_first 子图使用此 schema。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from framework.schema.reducers import _merge_dict


class DebateState(TypedDict):
    """辩论子图专用 state：保留全部消息历史，供每个发言节点看到完整辩论记录。"""
    messages: Annotated[list[BaseMessage], add_messages]
    knowledge_vault: str   # 知识库路径（从父图透传）
    project_docs: str      # 子项目文档路径（从父图透传）
    # 以下字段供 GeminiNode / ClaudeNode __call__ 正常读写（不会真正用于辩论逻辑）
    node_sessions: Annotated[dict, _merge_dict]  # 节点 session UUID 映射; merge reducer prevents parallel node writes from clobbering each other
    routing_target: str      # 路由信号（辩论内不使用，但节点 __call__ 会清零写回）
    routing_context: str     # 路由上下文（首轮 Gemini 读取辩题）
    consult_count: int       # 咨询计数
    workspace: str           # 工作目录（ClaudeNode cwd）
    project_root: str        # 项目根目录
    rollback_reason: str     # 回退原因（辩论内始终为空）
    retry_count: int         # 重试计数


# Auto-register on import
from framework.registry import register_schema

register_schema("debate_schema", DebateState)
