"""
外部 Agent 引用节点 — framework/nodes/agent_ref_node.py

AgentRefNode 将外部 agent 目录编译为 LangGraph 子图，
并在调用时处理父图 ↔ 子图之间的 state 字段映射。

node_config 字段：
  agent_dir  str   外部 agent 目录路径（相对于当前工作目录或绝对路径）
  state_in   dict  {子图字段: 父图字段}  调用前从父图 state 注入
  state_out  dict  {父图字段: 子图字段 | "last_message"}  调用后写回父图 state
               "last_message" 特殊值：取子图 messages[-1].content

用法示例（agent.json）：
  {"id": "debate_brainstorm", "type": "AGENT_REF",
   "agent_dir": "agents/debate_gemini_first",
   "state_in":  {"task": "routing_context", "knowledge_vault": "knowledge_vault"},
   "state_out": {"debate_conclusion": "last_message"}}
"""

import logging
import shutil
from pathlib import Path

from langchain_core.messages import AIMessage
from framework.config import AgentConfig

logger = logging.getLogger(__name__)


class AgentRefNode:
    """
    外部 Agent 引用节点。

    编译时懒加载外部 agent 目录的图（首次 __call__ 时触发）。
    不使用独立 checkpointer（父图负责 checkpointing）。
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        agent_dir = Path(node_config["agent_dir"]).resolve()
        if not agent_dir.exists():
            raise ValueError(f"AgentRefNode: agent_dir not found: {agent_dir}")

        from framework.agent_loader import AgentLoader
        self._loader = AgentLoader(agent_dir)
        self._state_in: dict[str, str] = node_config.get("state_in", {})
        self._state_out: dict[str, str] = node_config.get("state_out", {})
        self._graph = None
        logger.info(f"[agent_ref] registered ref → {agent_dir.name!r}")

    async def _ensure_graph(self) -> None:
        if self._graph is None:
            logger.info(f"[agent_ref] compiling {self._loader.name!r}...")
            self._graph = await self._loader.build_graph(checkpointer=None)
            logger.info(f"[agent_ref] {self._loader.name!r} compiled")

    async def __call__(self, state: dict) -> dict:
        await self._ensure_graph()

        # 记录进入前的 node_sessions key（用于事后清理子图产生的孤儿 session）
        original_ns_keys = set((state.get("node_sessions") or {}).keys())

        # 构建子图入口 state：只传子图 schema 需要的字段，避免 LangGraph 拒绝未知 key
        task = state.get("routing_context", "")
        from langchain_core.messages import HumanMessage
        sub_state: dict = {
            "messages": [HumanMessage(content=task)] if task else [],
            "knowledge_vault": state.get("knowledge_vault", ""),
            "project_docs": state.get("project_docs", ""),
            "routing_context": task,
            "workspace": state.get("workspace", ""),
            "project_root": state.get("project_root", ""),
        }
        # state_in 映射：父图字段 → 子图字段
        for sub_key, parent_key in self._state_in.items():
            sub_state[sub_key] = state.get(parent_key, "")

        logger.info(f"[agent_ref] invoking {self._loader.name!r}")

        # 使用 astream(updates) 实时输出辩论过程，显示每个节点的身份
        last_state: dict = {}
        graph_name = self._loader.name
        print(f"\n{'─' * 60}", flush=True)
        print(f"  [{graph_name}] 辩论开始", flush=True)
        print(f"{'─' * 60}", flush=True)

        async for event in self._graph.astream(sub_state, stream_mode="updates"):
            for node_id, update in event.items():
                if node_id in ("__start__", "__end__"):
                    continue
                # 跟踪最新 state 用于最终结果映射
                last_state.update(update)
                msgs = update.get("messages", [])
                for msg in msgs:
                    content = getattr(msg, "content", "")
                    if not content:
                        continue
                    msg_type = getattr(msg, "type", "ai")
                    if msg_type == "human":
                        label = "议题"
                    else:
                        label = node_id
                    print(f"\n  ┌─ [{label}]", flush=True)
                    for line in content.split("\n"):
                        print(f"  │ {line}", flush=True)
                    print(f"  └─", flush=True)

        print(f"\n{'─' * 60}", flush=True)
        print(f"  [{graph_name}] 辩论结束", flush=True)
        print(f"{'─' * 60}\n", flush=True)

        # 从子图结果映射回父图 state
        out: dict = {}
        for parent_key, src in self._state_out.items():
            if src == "last_message":
                msgs = last_state.get("messages", [])
                if msgs:
                    last_msg = msgs[-1] if isinstance(msgs[-1], str) else msgs[-1].content
                    out[parent_key] = last_msg
                else:
                    out[parent_key] = ""
            else:
                out[parent_key] = last_state.get(src, "")

        # 将辩论结论作为 AIMessage 写入 messages，使 claude_main 能直接读到
        conclusion = out.get("debate_conclusion", "")
        if conclusion:
            out["messages"] = [AIMessage(
                content=f"[辩论结论]\n\n{conclusion}"
            )]

        # 清理子图产生的孤儿 session（磁盘文件）
        self._cleanup_orphan_sessions(last_state, original_ns_keys)

        # 辩论完成后才计入 consult_count，确保 max_retry 在重入时生效
        out["consult_count"] = state.get("consult_count", 0) + 1

        logger.info(
            f"[agent_ref] {self._loader.name!r} done, "
            f"out_keys={list(out.keys())}"
        )
        return out

    def _cleanup_orphan_sessions(
        self, result: dict, original_keys: set[str]
    ) -> None:
        """
        子图运行后，找出新增的 node_sessions key（辩论节点产生的），
        删除对应的 Gemini 和 Claude 磁盘 session，防止孤儿积累。
        """
        result_ns = result.get("node_sessions") or {}
        new_keys = set(result_ns.keys()) - original_keys
        if not new_keys:
            return

        cleaned = 0
        for key in new_keys:
            sid = result_ns[key]
            if not sid:
                continue

            # Gemini session 文件：~/.gemini/tmp/{project}/chats/session-*-{uuid[:8]}.json
            try:
                import framework.gemini.gemini_session as gem_sess
                if gem_sess.delete_session(sid):
                    cleaned += 1
            except Exception as e:
                logger.debug(f"[agent_ref] gemini session cleanup failed for {sid[:8]}: {e}")

            # Claude session 目录：~/.claude/session-env/{uuid}/
            claude_dir = Path.home() / ".claude" / "session-env" / sid
            if claude_dir.exists():
                try:
                    shutil.rmtree(claude_dir)
                    cleaned += 1
                    logger.debug(f"[agent_ref] deleted claude session {sid[:8]}")
                except Exception as e:
                    logger.debug(f"[agent_ref] claude session cleanup failed for {sid[:8]}: {e}")

        if cleaned:
            logger.info(
                f"[agent_ref] cleaned {cleaned} orphan session(s) "
                f"from {len(new_keys)} subgraph node(s)"
            )
