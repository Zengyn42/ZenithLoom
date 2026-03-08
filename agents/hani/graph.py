"""
Hani Agent — LangGraph 状态机组装

build_hani_graph() 返回编译后的 CompiledStateGraph。
"""

import logging
import os

import aiosqlite
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from agents.hani.config import load_hani_config, load_hani_system_prompt
from agents.hani.hani_claude_node import HaniClaudeNode
from framework.nodes.claude_node import ClaudeNode
from framework.nodes.gemini_node import GeminiNode
from framework.nodes.git_nodes import GitRollbackNode, GitSnapshotNode
from framework.nodes.validate_node import ValidateNode
from framework.nodes.vram_flush_node import VramFlushNode
from framework.state import BaseAgentState

logger = logging.getLogger(__name__)

_MAX_CONSULTS_PER_TURN = 1


def _gemini_node_wrapper(gemini: GeminiNode):
    """把 GeminiNode.consult 包装成 LangGraph 节点函数。"""

    async def _node(state: BaseAgentState) -> dict:
        pending = state.get("gemini_context", "")
        if pending.startswith("__PENDING__"):
            rest = pending[len("__PENDING__"):]
            parts = rest.split("|", 1)
            topic = parts[0]
            context = parts[1] if len(parts) > 1 else ""
        else:
            topic = pending
            context = ""

        session_id = state.get("claude_session_id", "")
        result = await gemini.consult(topic, context, session_id)

        return {
            "gemini_context": result,
            "consult_count": state.get("consult_count", 0) + 1,
        }

    return _node


def _validate_route(state: BaseAgentState) -> str:
    """validate 后的路由。"""
    if state.get("rollback_reason"):
        return "rollback"
    ctx = state.get("gemini_context", "")
    count = state.get("consult_count", 0)
    if ctx.startswith("__PENDING__"):
        if count >= _MAX_CONSULTS_PER_TURN:
            logger.warning(
                f"[route] consult_count={count} 已达上限，强制结束"
            )
            return "end"
        return "consult_gemini"
    return "end"


async def build_hani_graph(config=None, checkpointer=None):
    """
    构建 Hani agent 的 StateGraph。

    Args:
        config: AgentConfig，None 则自动加载
        checkpointer: LangGraph checkpointer，None 则自动创建 AsyncSqliteSaver
    """
    if config is None:
        config = load_hani_config()

    system_prompt = load_hani_system_prompt()

    # 节点实例化
    claude = ClaudeNode(config, system_prompt)
    gemini = GeminiNode(config, claude)
    git_snapshot = GitSnapshotNode()
    git_rollback = GitRollbackNode()
    validate = ValidateNode(config)
    vram_flush = VramFlushNode()
    hani_claude = HaniClaudeNode(claude, gemini)

    # 图构建
    builder = StateGraph(BaseAgentState)
    builder.add_node("git_snapshot", git_snapshot)
    builder.add_node("claude_agent", hani_claude)
    builder.add_node("validate", validate)
    builder.add_node("gemini_advisor", _gemini_node_wrapper(gemini))
    builder.add_node("git_rollback", git_rollback)
    builder.add_node("vram_flush", vram_flush)

    builder.set_entry_point("git_snapshot")
    builder.add_edge("git_snapshot", "claude_agent")
    builder.add_edge("claude_agent", "validate")
    builder.add_conditional_edges(
        "validate",
        _validate_route,
        {
            "rollback": "git_rollback",
            "consult_gemini": "gemini_advisor",
            "end": "vram_flush",
        },
    )
    builder.add_edge("git_rollback", "claude_agent")
    builder.add_edge("gemini_advisor", "claude_agent")
    builder.add_edge("vram_flush", END)

    # Checkpointer
    if checkpointer is None:
        db_path = os.path.abspath(config.db_path)
        conn = await aiosqlite.connect(db_path)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

    return builder.compile(checkpointer=checkpointer)


# 单例引擎
_engine = None


async def get_engine():
    global _engine
    if _engine is None:
        _engine = await build_hani_graph()
    return _engine


def get_config() -> dict:
    config = load_hani_config()
    return {"configurable": {"thread_id": config.session_thread_id}}


def invalidate_engine() -> None:
    """引擎缓存失效（compact/reset 后调用）。"""
    global _engine
    _engine = None
