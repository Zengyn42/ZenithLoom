"""
框架级通用 Agent 图构建器 — framework/graph.py

build_agent_graph() 构建标准 LangGraph 状态机：
  git_snapshot → agent_node → validate → [rollback|gemini_advisor|vram_flush] → END

任何 agent 都可以用自己的 agent_node 调用此函数，不需要重复写图结构。

Session 管理（模块级）：
  _active         — 当前激活的命名 session
  get_config()    — 动态返回 thread_id，供 engine.astream() 使用
  switch_session()— 切换到已有命名 session
  new_session()   — 创建并切换到新命名 session
  get_engine()    — 单例引擎，懒加载
  invalidate_engine() — 使引擎缓存失效（compact/reset 后调用）
"""

import logging
import os
from dataclasses import dataclass

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.gemini.node import GeminiNode, GeminiQuotaError
from framework.nodes.git_nodes import GitRollbackNode, GitSnapshotNode
from framework.nodes.validate_node import ValidateNode
from framework.nodes.vram_flush_node import VramFlushNode
from framework.state import BaseAgentState

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# 模块级 session 状态（每个进程共享一个 active session）
# ---------------------------------------------------------------------------

@dataclass
class _ActiveSession:
    name: str        # 人类可读名称
    thread_id: str   # LangGraph thread_id


_active: _ActiveSession | None = None


def get_config(default_thread_id: str = "default_session") -> dict:
    """动态返回当前 thread_id（随 _active 变化）。"""
    if _active:
        if is_debug():
            logger.debug(f"[graph] get_config: active={_active.name} tid={_active.thread_id}")
        return {"configurable": {"thread_id": _active.thread_id}}
    return {"configurable": {"thread_id": default_thread_id}}


# ---------------------------------------------------------------------------
# Session 切换
# ---------------------------------------------------------------------------

async def switch_session(name: str, session_mgr) -> str:
    """
    切换到已有命名 session，返回新 thread_id。
    GeminiNode 的 _records/_clients 缓存保留（按 session_id UUID 索引，无需清空）。
    """
    global _active

    env = session_mgr.get_envelope(name)
    if env is None:
        raise ValueError(
            f"Session {name!r} 不存在。用 `!new {name}` 创建新 session。"
        )

    _active = _ActiveSession(name=name, thread_id=env.thread_id)
    logger.info(f"[graph] switched to session {name!r} (thread_id={env.thread_id})")
    if is_debug():
        logger.debug(f"[graph] node_sessions={env.node_sessions}")
    return env.thread_id


async def new_session(name: str, session_mgr) -> str:
    """
    创建并切换到新命名 session，返回新 thread_id。
    """
    global _active

    env = session_mgr.create_session(name)
    _active = _ActiveSession(name=name, thread_id=env.thread_id)
    logger.info(f"[graph] new session {name!r} created (thread_id={env.thread_id})")
    return env.thread_id


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

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

        if is_debug():
            logger.debug(f"[graph.gemini_wrapper] topic={topic!r} context_len={len(context)}")

        # 从 node_sessions 读取 Gemini session UUID
        ns = dict(state.get("node_sessions") or {})
        gemini_session_id = ns.get("gemini_main", "")

        try:
            result, new_gemini_sid = await gemini.consult(
                topic, context, session_id=gemini_session_id
            )
        except GeminiQuotaError as e:
            logger.error(f"[graph.gemini_wrapper] 配额错误，图终止: {e}")
            return {
                "gemini_context": f"[Gemini 配额错误，本轮跳过: {e}]",
                "consult_count": state.get("consult_count", 0) + 1,
            }

        # 写回 Gemini session UUID
        ns["gemini_main"] = new_gemini_sid
        return {
            "gemini_context": result,
            "consult_count": state.get("consult_count", 0) + 1,
            "node_sessions": ns,
        }

    return _node


def _make_validate_route(max_consults: int):
    """工厂函数：生成 validate 后的路由函数，使用 config.max_gemini_consults。"""
    def _validate_route(state: BaseAgentState) -> str:
        if state.get("rollback_reason"):
            return "rollback"
        ctx = state.get("gemini_context", "")
        count = state.get("consult_count", 0)
        if ctx.startswith("__PENDING__"):
            if count >= max_consults:
                logger.warning(f"[graph] consult_count={count} >= max={max_consults}，强制结束")
                return "end"
            return "consult_gemini"
        return "end"
    return _validate_route


# ---------------------------------------------------------------------------
# Generic graph builder
# ---------------------------------------------------------------------------

async def build_agent_graph(
    config: AgentConfig,
    agent_node,
    gemini: GeminiNode,
    checkpointer=None,
    use_vram_flush: bool = False,
):
    """
    构建标准 Agent StateGraph。

    任何 agent 传入自己的 agent_node（实现了 __call__(state) → dict 的对象），
    其余节点（git_snapshot、validate、gemini_advisor、git_rollback）
    由框架统一提供。vram_flush 节点为可选，由 agent.json["vram_flush"] 控制。

    Args:
        config:          AgentConfig
        agent_node:      实现了 async __call__(state: BaseAgentState) -> dict 的节点对象
        gemini:          GeminiNode 实例（已初始化）
        checkpointer:    LangGraph checkpointer，None 则自动创建 AsyncSqliteSaver
        use_vram_flush:  是否在图末尾插入 GPU 清洗节点（默认 False）
    """
    git_snapshot = GitSnapshotNode()
    git_rollback = GitRollbackNode()
    validate = ValidateNode(config)

    builder = StateGraph(BaseAgentState)
    builder.add_node("git_snapshot", git_snapshot)
    builder.add_node("claude_agent", agent_node)
    builder.add_node("validate", validate)
    builder.add_node("gemini_advisor", _gemini_node_wrapper(gemini))
    builder.add_node("git_rollback", git_rollback)

    terminal = "vram_flush" if use_vram_flush else END
    if use_vram_flush:
        builder.add_node("vram_flush", VramFlushNode())
        builder.add_edge("vram_flush", END)

    builder.set_entry_point("git_snapshot")
    builder.add_edge("git_snapshot", "claude_agent")
    builder.add_edge("claude_agent", "validate")
    builder.add_conditional_edges(
        "validate",
        _make_validate_route(config.max_gemini_consults),
        {
            "rollback": "git_rollback",
            "consult_gemini": "gemini_advisor",
            "end": terminal,
        },
    )
    builder.add_edge("git_rollback", "claude_agent")
    builder.add_edge("gemini_advisor", "claude_agent")

    if checkpointer is None:
        db_path = os.path.abspath(config.db_path)
        conn = await aiosqlite.connect(db_path)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

    logger.info(f"[graph] graph built for config.name={config.name!r}")
    return builder.compile(checkpointer=checkpointer)
