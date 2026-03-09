"""
框架级通用 Agent 图构建器 — framework/graph.py

build_agent_graph() 构建可配置的 LangGraph 状态机。

LangGraph 设计哲学：StateGraph 是纯 builder——节点是任意 callable，
拓扑由代码显式声明。本模块把节点工厂与拓扑组装分开，
通过 GraphSpec 让 agent.json 声明自己需要哪些组件，无需写 Python。

GraphSpec（agent.json["graph"] 字段）：
  use_git      bool  git_snapshot + git_rollback（默认 true，适合写代码的 agent）
  use_validate bool  validate 节点，错误时触发 rollback（默认 true）
  use_gemini   bool  gemini_advisor 节点（默认 true）
  use_vram_flush bool GPU 清洗节点（默认 false）

等效拓扑：
  full（默认）: git_snapshot → agent → validate → [rollback|gemini|END]
  chat:        agent → END          （use_git=false, use_validate=false）
  worker:      agent → validate → END（use_git=false, use_gemini=false）

需要完全自定义拓扑：在 agents/<name>/graph.py 定义 build_graph(loader, checkpointer)。

Session 管理（模块级）：
  _active         — 当前激活的命名 session
  get_config()    — 动态返回 thread_id，供 engine.astream() 使用
  switch_session()— 切换到已有命名 session
  new_session()   — 创建并切换到新命名 session
"""

import logging
import os
from dataclasses import dataclass, field

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
# GraphSpec — agent.json["graph"] 驱动的拓扑配置
# ---------------------------------------------------------------------------

@dataclass
class GraphSpec:
    """
    声明式图拓扑配置，从 agent.json["graph"] 加载。

    use_git=True  → 插入 git_snapshot（入口）和 git_rollback（错误恢复）
    use_validate  → 插入 validate 节点，验证失败触发 rollback / gemini 咨询
    use_gemini    → validate 后可路由到 gemini_advisor
    use_vram_flush→ 终止前清洗 GPU 显存
    """
    use_git: bool = True
    use_validate: bool = True
    use_gemini: bool = True
    use_vram_flush: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "GraphSpec":
        return cls(
            use_git=d.get("use_git", True),
            use_validate=d.get("use_validate", True),
            use_gemini=d.get("use_gemini", True),
            use_vram_flush=d.get("use_vram_flush", False),
        )


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
    spec: "GraphSpec | None" = None,
    # 向后兼容旧参数
    use_vram_flush: bool = False,
):
    """
    构建可配置的 Agent StateGraph。

    拓扑由 GraphSpec 控制（来自 agent.json["graph"]），默认等价于原来的完整图。
    LangGraph 节点 = callable(state) → dict，边 = 显式声明，条件路由 = 函数。

    Args:
        config:       AgentConfig
        agent_node:   async __call__(state) → dict
        gemini:       GeminiNode 实例
        checkpointer: LangGraph checkpointer，None 自动创建 AsyncSqliteSaver
        spec:         GraphSpec，None 时使用默认（全功能）
        use_vram_flush: 向后兼容参数，优先级低于 spec.use_vram_flush
    """
    if spec is None:
        spec = GraphSpec(use_vram_flush=use_vram_flush)

    if is_debug():
        logger.debug(
            f"[graph] spec: git={spec.use_git} validate={spec.use_validate} "
            f"gemini={spec.use_gemini} vram={spec.use_vram_flush}"
        )

    builder = StateGraph(BaseAgentState)
    builder.add_node("claude_agent", agent_node)

    # ── 终止节点 ──────────────────────────────────────────────────────────
    terminal = END
    if spec.use_vram_flush:
        builder.add_node("vram_flush", VramFlushNode())
        builder.add_edge("vram_flush", END)
        terminal = "vram_flush"

    # ── validate + 条件路由 ───────────────────────────────────────────────
    if spec.use_validate:
        validate = ValidateNode(config)
        builder.add_node("validate", validate)
        builder.add_edge("claude_agent", "validate")

        route_map: dict[str, str] = {"end": terminal}

        if spec.use_git:
            git_rollback = GitRollbackNode()
            builder.add_node("git_rollback", git_rollback)
            builder.add_edge("git_rollback", "claude_agent")
            route_map["rollback"] = "git_rollback"
        else:
            # 没有 git rollback，验证失败也走 end
            route_map["rollback"] = terminal

        if spec.use_gemini:
            builder.add_node("gemini_advisor", _gemini_node_wrapper(gemini))
            builder.add_edge("gemini_advisor", "claude_agent")
            route_map["consult_gemini"] = "gemini_advisor"
        else:
            route_map["consult_gemini"] = terminal

        builder.add_conditional_edges(
            "validate",
            _make_validate_route(config.max_gemini_consults),
            route_map,
        )
    else:
        builder.add_edge("claude_agent", terminal)

    # ── 入口 ──────────────────────────────────────────────────────────────
    if spec.use_git:
        git_snapshot = GitSnapshotNode()
        builder.add_node("git_snapshot", git_snapshot)
        builder.add_edge("git_snapshot", "claude_agent")
        builder.set_entry_point("git_snapshot")
    else:
        builder.set_entry_point("claude_agent")

    # ── Checkpointer ──────────────────────────────────────────────────────
    if checkpointer is None:
        db_path = os.path.abspath(config.db_path)
        conn = await aiosqlite.connect(db_path)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

    logger.info(f"[graph] graph built for config.name={config.name!r}")
    return builder.compile(checkpointer=checkpointer)
