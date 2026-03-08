"""
框架级基础状态 — BaseAgentState

对话历史由 Claude SDK session 管理，LangGraph state 只保留最近 2 条消息用于路由决策。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


def _keep_last_2(existing: list, new) -> list:
    """只保留最近 2 条消息。对话历史交给 SDK session 管理。"""
    combined = existing + (new if isinstance(new, list) else [new])
    return combined[-2:]


class BaseAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], _keep_last_2]
    gemini_context: str  # 路由信号：Gemini 建议 或 __PENDING__topic|ctx
    project_root: str  # 当前工作目录（!setproject 设置）
    project_meta: dict  # {"plan": "path", "tasks": "path"}
    consult_count: int  # 当轮已咨询 Gemini 次数
    last_stable_commit: str  # git 快照 hash
    retry_count: int  # 当轮回退重试次数
    rollback_reason: str  # 触发回退的原因（非空 = 需要回退）
    claude_session_id: str  # SDK resume 用的 session UUID
