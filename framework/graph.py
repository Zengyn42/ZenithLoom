"""
框架级通用 Agent 图构建器 — framework/graph.py

build_agent_graph() 构建可配置的 LangGraph 状态机（Priority 3 默认图）。

GraphSpec（entity.json["graph"] 字段）：
  use_git      bool  git_snapshot + git_rollback（默认 true）
  use_validate bool  validate 节点（默认 true）
  use_vram_flush bool GPU 清洗节点（默认 false）

路由机制：
  validate 后检查 state["routing_target"]（由 AgentNode 写入）：
    非空 → 结束（consult 路由仅 Priority 2 声明式图支持）
    空   → 结束（END）
    有错误 → git_rollback

推荐使用 Priority 2（entity.json 声明式图）替代此模块。
Priority 1（graph.py）和 Priority 2（"nodes"+"edges"）均绕过本模块。
"""

import logging
import os
from dataclasses import dataclass

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.git_nodes import GitRollbackNode, GitSnapshotNode
from framework.nodes.validate_node import ValidateNode
from framework.nodes.vram_flush_node import VramFlushNode
from framework.state import BaseAgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GraphSpec — entity.json["graph"] 驱动的拓扑配置
# ---------------------------------------------------------------------------

@dataclass
class GraphSpec:
    """
    声明式图拓扑配置，从 entity.json["graph"] 加载。

    use_git=True  → 插入 git_snapshot（入口）和 git_rollback（错误恢复）
    use_validate  → 插入 validate 节点，验证失败触发 rollback
    use_vram_flush→ 终止前清洗 GPU 显存
    """
    use_git: bool = True
    use_validate: bool = True
    use_vram_flush: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "GraphSpec":
        return cls(
            use_git=d.get("use_git", True),
            use_validate=d.get("use_validate", True),
            use_vram_flush=d.get("use_vram_flush", False),
        )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _make_validate_route():
    """工厂函数：生成 validate 后的路由函数。使用 routing_target 字段。"""
    def _validate_route(state: BaseAgentState) -> str:
        if state.get("rollback_reason"):
            return "rollback"
        routing_target = state.get("routing_target", "")
        if routing_target:
            return "consult"
        return "end"
    return _validate_route


# ---------------------------------------------------------------------------
# Generic graph builder (Priority 3)
# ---------------------------------------------------------------------------

async def build_agent_graph(
    config: AgentConfig,
    agent_node,
    checkpointer=None,
    spec: "GraphSpec | None" = None,
    use_vram_flush: bool = False,
):
    """
    构建可配置的 Agent StateGraph（Priority 3 默认图）。

    Args:
        config:       AgentConfig
        agent_node:   async __call__(state) → dict（通常为 ClaudeNode）
        checkpointer: LangGraph checkpointer，None 自动创建 AsyncSqliteSaver
        spec:         GraphSpec，None 时使用默认（全功能）
        use_vram_flush: 向后兼容参数，优先级低于 spec.use_vram_flush
    """
    if spec is None:
        spec = GraphSpec(use_vram_flush=use_vram_flush)

    if is_debug():
        logger.debug(
            f"[graph] spec: git={spec.use_git} validate={spec.use_validate} "
            f"vram={spec.use_vram_flush}"
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
            route_map["rollback"] = terminal

        route_map["consult"] = terminal

        builder.add_conditional_edges(
            "validate",
            _make_validate_route(),
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
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=10000")
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

    logger.info(f"[graph] graph built for config.name={config.name!r}")
    return builder.compile(checkpointer=checkpointer)
