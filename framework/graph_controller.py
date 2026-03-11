"""
框架级图控制器 — framework/graph_controller.py

GraphController 是图的统一入口：
  - 管理活跃 thread_id（LangGraph checkpoint 键）
  - 封装 graph.ainvoke()，自动更新 sessions.json 中的 node_sessions
  - 提供命名 session 切换接口（new_session / switch_session）
  - 提供 checkpoint 快照查询（get_history，P2 断点恢复）

AgentLoader.get_controller() 替代 get_engine()；
interfaces/cli.py 和 interfaces/discord_bot.py 统一通过 controller.run() 执行图。

Session 生命周期（重启恢复流程）：
  controller 启动 → _init_session() 从 sessions.json 读最近 thread_id
  → controller.run(user_input) → graph.ainvoke(config={thread_id})
  → LangGraph checkpointer 自动 restore BaseAgentState（含 node_sessions）
  → 各节点从 state["node_sessions"][node_key] 读 UUID，resume 上次会话
"""

import logging

from langchain_core.messages import HumanMessage

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.session_mgr import SessionManager

logger = logging.getLogger(__name__)


class GraphController:
    """
    图级统一控制器。

    用法：
        controller = await loader.get_controller()
        response = await controller.run("用户输入")
        await controller.new_session("session-b")
        await controller.switch_session("session-a")
    """

    def __init__(self, graph, session_mgr: SessionManager, config: AgentConfig):
        self._graph = graph
        self._session_mgr = session_mgr
        self._config = config
        self._active_thread_id: str = self._init_session()

    def _init_session(self) -> str:
        """
        启动时选择活跃 session：
        - sessions.json 非空 → 使用第一个（最近创建）
        - 为空 → 创建 "default" session
        """
        sessions = self._session_mgr.list_all()
        if sessions:
            first_env = next(iter(sessions.values()))
            tid = first_env.thread_id
            if is_debug():
                logger.debug(f"[controller] init session: thread_id={tid}")
            return tid
        env = self._session_mgr.create_session("default")
        logger.info(f"[controller] created default session: thread_id={env.thread_id}")
        return env.thread_id

    def get_config(self) -> dict:
        """返回 LangGraph 调用配置（thread_id）。"""
        return {"configurable": {"thread_id": self._active_thread_id}}

    async def run(self, user_input: str) -> str:
        """
        执行图，返回最终 AI 消息内容。
        LangGraph checkpointer 自动 restore/save 含 node_sessions 的 BaseAgentState。
        执行后将更新的 node_sessions 写回 sessions.json。
        """
        if is_debug():
            logger.debug(
                f"[controller] run: thread_id={self._active_thread_id} "
                f"input_len={len(user_input)}"
            )

        # 注入 per-session workspace（从 SessionEnvelope 读取）
        name = self._session_mgr.find_name_by_thread_id(self._active_thread_id)
        env = self._session_mgr.get_envelope(name) if name else None
        workspace = env.workspace if env else ""

        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)], "workspace": workspace},
            config=self.get_config(),
        )

        # 将最新 node_sessions 写回 sessions.json
        ns = result.get("node_sessions") or {}
        if ns:
            name = self._session_mgr.find_name_by_thread_id(self._active_thread_id)
            if name:
                for node_key, uuid in ns.items():
                    if uuid:
                        try:
                            self._session_mgr.update_node_session(name, node_key, uuid)
                        except Exception as e:
                            logger.warning(
                                f"[controller] update_node_session failed: {e}"
                            )

        messages = result.get("messages", [])
        return messages[-1].content if messages else ""

    async def new_session(self, name: str, workspace: str = "") -> None:
        """
        创建新命名 session，UUID 为空，各节点首次运行时自动建立新上下文。
        workspace 为该 session 的工作目录，注入到图状态。
        name 已存在时抛 ValueError（来自 SessionManager）。
        """
        env = self._session_mgr.create_session(name, workspace=workspace)
        self._active_thread_id = env.thread_id
        logger.info(
            f"[controller] new session {name!r}: thread_id={env.thread_id} "
            f"workspace={workspace!r}"
        )

    async def switch_session(self, name: str) -> None:
        """
        切换到已有命名 session。
        LangGraph checkpointer 自动 restore 该 thread_id 的完整 BaseAgentState。
        """
        env = self._session_mgr.get_envelope(name)
        if env is None:
            raise ValueError(
                f"Session {name!r} 不存在。用 !new {name} 创建新 session。"
            )
        self._active_thread_id = env.thread_id
        logger.info(
            f"[controller] switched to session {name!r}: thread_id={env.thread_id}"
        )

    async def get_history(self) -> dict:
        """
        读取当前 thread 的 checkpoint 快照，用于断点现场还原（P2）。

        返回：
          thread_id  — 当前 thread_id
          values     — 完整 BaseAgentState 快照
          next_nodes — 下一步将执行的节点列表
          created_at — 快照时间戳
        """
        snapshot = await self._graph.aget_state(self.get_config())
        return {
            "thread_id": self._active_thread_id,
            "values": snapshot.values,
            "next_nodes": list(snapshot.next),
            "created_at": getattr(snapshot, "created_at", None),
        }

    @property
    def active_thread_id(self) -> str:
        """当前活跃 thread_id（只读）。"""
        return self._active_thread_id

    @property
    def session_mgr(self) -> SessionManager:
        """SessionManager 引用（供接口层调用）。"""
        return self._session_mgr
