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
from framework.rollback_log import RollbackLog
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
        self._rollback_log = RollbackLog(config.db_path)
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

        # 将最新 node_sessions 从 checkpoint 同步到 sessions.json（单向：checkpoint → sessions.json）
        # sessions.json 中的 node_sessions 是冗余副本，LangGraph 不会读取它；
        # 实际运行时各节点从 checkpoint 恢复的 state["node_sessions"] 获取 UUID。
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

    async def log_snapshot(self) -> None:
        """
        读取当前 checkpoint state，若有 last_stable_commit 则写入 rollback_log。
        CLI 在每轮流式输出结束后调用，记录 (commit_hash, node_sessions, project_root)。
        """
        snapshot = await self._graph.aget_state(self.get_config())
        commit = snapshot.values.get("last_stable_commit") or ""
        if not commit:
            return
        ns = snapshot.values.get("node_sessions") or {}
        project_root = snapshot.values.get("project_root") or ""
        self._rollback_log.log_turn(
            self._active_thread_id, commit, ns, project_root=project_root
        )

    async def rollback_to_turn(self, n: int, reason: str = "") -> dict:
        """
        三层回退到第 N 条快照（N=1=最近一次，N=2=倒数第二次...）：
          1. git reset --hard <commit_hash>（仅当 project_root 非空）
          2. LangGraph aupdate_state → node_sessions 恢复为旧 UUID
          3. 写入 .DO_NOT_REPEAT.md tombstone（仅当 reason 非空）

        返回 {"ok": bool, "msg": str, "commit": str, "node_sessions": dict}
        """
        from framework.nodes.git_ops import rollback as git_rollback
        from framework.nodes.git_nodes import _write_tombstone

        record = self._rollback_log.get_nth_ago(self._active_thread_id, n)
        if not record:
            return {"ok": False, "msg": f"没有找到第 {n} 条历史快照（当前 thread 共记录了多少条？用 !snapshots 查看）"}

        commit_hash = record["commit_hash"]
        old_node_sessions = record["node_sessions"]
        project_root = record["project_root"]

        # 1. git reset --hard
        if project_root and commit_hash:
            ok = git_rollback(project_root, commit_hash)
            if not ok:
                return {"ok": False, "msg": f"git reset --hard {commit_hash[:8]} 失败，请检查 project_root={project_root!r}"}
        else:
            logger.warning("[controller] rollback: project_root 为空，跳过 git reset")

        # 2. 更新 LangGraph checkpoint 的 node_sessions
        await self._graph.aupdate_state(
            self.get_config(),
            {"node_sessions": old_node_sessions},
        )

        # 3. 同步 sessions.json 中的 node_sessions
        name = self._session_mgr.find_name_by_thread_id(self._active_thread_id)
        if name:
            for node_key, uuid in old_node_sessions.items():
                if uuid:
                    try:
                        self._session_mgr.update_node_session(name, node_key, uuid)
                    except Exception:
                        pass

        # 4. tombstone
        if project_root and reason:
            _write_tombstone(
                project_root,
                reason,
                f"[!rollback {n}] 手动回退到 commit {commit_hash[:8]}",
            )

        logger.info(
            f"[controller] rollback N={n} → commit={commit_hash[:8]} "
            f"ns_keys={list(old_node_sessions.keys())}"
        )
        return {
            "ok": True,
            "msg": f"已回退到 {commit_hash[:8]}，node_sessions 已恢复",
            "commit": commit_hash,
            "node_sessions": old_node_sessions,
        }

    @property
    def rollback_log(self) -> RollbackLog:
        """RollbackLog 引用（供接口层查询历史）。"""
        return self._rollback_log

    @property
    def active_thread_id(self) -> str:
        """当前活跃 thread_id（只读）。"""
        return self._active_thread_id

    @property
    def session_mgr(self) -> SessionManager:
        """SessionManager 引用（供接口层调用）。"""
        return self._session_mgr
