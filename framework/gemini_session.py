"""
Gemini CLI 兼容 Session 存储 — 向后兼容导出

实现已移至 framework/gemini/gemini_session.py
"""
from framework.gemini.gemini_session import (  # noqa: F401
    ConversationRecord,
    MessageRecord,
    TokensSummary,
    ToolCallRecord,
    append_history,
    get_project_id,
    list_sessions,
    load_session,
    new_session,
    save_session,
    to_api_history,
)

__all__ = [
    "ConversationRecord",
    "MessageRecord",
    "TokensSummary",
    "ToolCallRecord",
    "append_history",
    "get_project_id",
    "list_sessions",
    "load_session",
    "new_session",
    "save_session",
    "to_api_history",
]
