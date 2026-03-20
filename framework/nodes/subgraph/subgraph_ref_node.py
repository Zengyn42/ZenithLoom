"""
外部 Agent 引用节点 — framework/nodes/subgraph/subgraph_ref_node.py

SubgraphRefNode 将外部 agent 目录编译为 LangGraph 子图，
并在调用时处理父图 ↔ 子图之间的 state 字段映射。

node_config 字段：
  agent_dir   str   外部 agent 目录路径（相对于当前工作目录或绝对路径）
  state_in    dict  {子图字段: 父图字段}  调用前从父图 state 注入
  state_out   dict  {父图字段: 子图字段 | "last_message"}  调用后写回父图 state
                "last_message" 特殊值：取子图 messages[-1].content
  max_retry   int|null  该子图在一轮对话中最多被调用的次数（默认 null = 无限制）
                超限时不执行子图，返回 AIMessage 告知主图原因。

用法示例（agent.json）：
  {"id": "debate_brainstorm", "type": "SUBGRAPH_REF",
   "agent_dir": "agents/debate_gemini_first",
   "max_retry": 1,
   "state_in":  {"task": "routing_context", "knowledge_vault": "knowledge_vault"},
   "state_out": {"debate_conclusion": "last_message"}}
"""

import logging
import shutil
from pathlib import Path

from langchain_core.messages import AIMessage
from framework.config import AgentConfig
from framework.debug import is_debug, push_graph_scope, pop_graph_scope

logger = logging.getLogger(__name__)


class SubgraphRefNode:
    """
    外部 Agent 引用节点。

    编译时懒加载外部 agent 目录的图（首次 __call__ 时触发）。
    不使用独立 checkpointer（父图负责 checkpointing）。
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        agent_dir = Path(node_config["agent_dir"]).resolve()
        if not agent_dir.exists():
            raise ValueError(f"SubgraphRefNode: agent_dir not found: {agent_dir}")

        from framework.agent_loader import EntityLoader
        self._loader = EntityLoader(agent_dir)
        self._node_id: str = node_config.get("id", agent_dir.name)
        self._state_in: dict[str, str] = node_config.get("state_in", {})
        self._state_out: dict[str, str] = node_config.get("state_out", {})
        self._max_retry: int | None = node_config.get("max_retry")  # None = 无限制
        self._graph = None
        logger.info(
            f"[subgraph_ref] registered ref → {agent_dir.name!r} "
            f"(max_retry={self._max_retry})"
        )

    async def _ensure_graph(self) -> None:
        if self._graph is None:
            logger.info(f"[subgraph_ref] compiling {self._loader.name!r}...")
            self._graph = await self._loader.build_graph(checkpointer=None)
            logger.info(f"[subgraph_ref] {self._loader.name!r} compiled")

    async def __call__(self, state: dict) -> dict:
        # ── max_retry 限速检查 ──────────────────────────────────────────────
        call_counts = dict(state.get("subgraph_call_counts") or {})
        my_count = call_counts.get(self._node_id, 0)

        if self._max_retry is not None and my_count >= self._max_retry:
            reason = (
                f"子图 {self._node_id} 本轮已被调用 {my_count} 次，"
                f"达到上限 max_retry={self._max_retry}，跳过执行。"
            )
            logger.warning(f"[subgraph_ref] {reason}")
            return {
                "messages": [AIMessage(
                    content=f"[子图限速] {reason}\n请换一种方式处理，或直接回复用户。"
                )],
                "subgraph_call_counts": call_counts,
            }

        # ── 正常执行子图 ────────────────────────────────────────────────────
        await self._ensure_graph()

        # 记录进入前的 node_sessions key（用于事后清理子图产生的孤儿 session）
        original_ns_keys = set((state.get("node_sessions") or {}).keys())

        # 构建子图入口 state：只传子图 schema 需要的字段，避免 LangGraph 拒绝未知 key
        task = state.get("routing_context", "")
        # routing_context 为空时，fallback 到父图最后一条消息内容
        if not task:
            parent_msgs = state.get("messages") or []
            if parent_msgs:
                task = parent_msgs[-1].content
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

        logger.info(f"[subgraph_ref] invoking {self._loader.name!r} (call #{my_count + 1})")
        if is_debug():
            logger.debug(
                f"[subgraph_ref/{self._node_id}] state_in={self._state_in} "
                f"state_out={self._state_out} sub_state_keys={list(sub_state.keys())}"
            )

        # 使用 astream(updates) 实时输出过程，显示每个节点的身份
        last_state: dict = {}
        graph_name = self._loader.name
        print(f"\n{'─' * 60}", flush=True)
        print(f"  [{graph_name}] 子图开始", flush=True)
        print(f"{'─' * 60}", flush=True)

        # 进入子图 scope（日志按层级目录存放）
        push_graph_scope(self._node_id)
        try:
            async for event in self._graph.astream(sub_state, stream_mode="updates"):
                for node_id, update in event.items():
                    if node_id in ("__start__", "__end__"):
                        continue
                    if not update:
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
        finally:
            pop_graph_scope()

        print(f"\n{'─' * 60}", flush=True)
        print(f"  [{graph_name}] 子图结束", flush=True)
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

        # 将子图结论作为 AIMessage 写入 messages，使父图主节点能直接读到
        # 遍历所有 state_out 映射结果，取第一个非空值作为结论
        conclusion = ""
        for parent_key in self._state_out:
            val = out.get(parent_key, "")
            if val:
                conclusion = val
                break

        if conclusion:
            out["messages"] = [AIMessage(
                content=f"[子图结论]\n\n{conclusion}"
            )]

        # 清理子图产生的孤儿 session（磁盘文件）
        self._cleanup_orphan_sessions(last_state, original_ns_keys)

        # 更新按子图计数
        call_counts[self._node_id] = my_count + 1
        out["subgraph_call_counts"] = call_counts
        out["consult_count"] = state.get("consult_count", 0) + 1

        logger.info(
            f"[subgraph_ref] {self._loader.name!r} done, "
            f"call_count={my_count + 1}, out_keys={list(out.keys())}"
        )
        if is_debug():
            for pk, src in self._state_out.items():
                val = out.get(pk, "")
                preview = str(val)[:100] if val else "(empty)"
                logger.debug(f"[subgraph_ref/{self._node_id}] state_out: {pk}←{src} = {preview!r}")
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
                import framework.nodes.llm.gemini_session as gem_sess
                if gem_sess.delete_session(sid):
                    cleaned += 1
            except Exception as e:
                logger.debug(f"[subgraph_ref] gemini session cleanup failed for {sid[:8]}: {e}")

            # Claude session 目录：~/.claude/session-env/{uuid}/
            claude_dir = Path.home() / ".claude" / "session-env" / sid
            if claude_dir.exists():
                try:
                    shutil.rmtree(claude_dir)
                    cleaned += 1
                    logger.debug(f"[subgraph_ref] deleted claude session {sid[:8]}")
                except Exception as e:
                    logger.debug(f"[subgraph_ref] claude session cleanup failed for {sid[:8]}: {e}")

        if cleaned:
            logger.info(
                f"[subgraph_ref] cleaned {cleaned} orphan session(s) "
                f"from {len(new_keys)} subgraph node(s)"
            )


AgentRefNode = SubgraphRefNode
