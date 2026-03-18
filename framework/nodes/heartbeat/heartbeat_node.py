"""
HEARTBEAT 节点 — 将一个完整主图作为心跳任务定期调用。

每个 HEARTBEAT 节点持有独立的 EntityLoader，使用固定 thread_id "__heartbeat__"
让 LangGraph checkpointer 自动维持跨 invocation 的完整 state（含 node_sessions、
messages 等），从而实现 LLM session 记忆连续性。

node_config 字段：
  agent_dir  str  目标 agent 目录（必填，相对于项目根）
  prompt     str  每次调用注入的触发 HumanMessage（必填）
"""

import logging
from pathlib import Path

from langchain_core.messages import AIMessage

from framework.debug import is_debug

logger = logging.getLogger(__name__)

_HEARTBEAT_THREAD = "__heartbeat__"


class HeartbeatNode:
    """
    调度主图节点。

    持有独立 EntityLoader（有自己的 SQLite DB 和 sessions.json），
    每次 __call__ 向目标主图发一条 HumanMessage，由 LangGraph checkpointer
    自动 restore/save 完整 state，维持跨 invocation 记忆。
    """

    def __init__(self, node_config: dict):
        self._agent_dir = Path(node_config["agent_dir"]).resolve()
        self._prompt = node_config["prompt"]
        self._controller = None  # lazily built

    async def _get_controller(self):
        if self._controller is None:
            from framework.agent_loader import EntityLoader
            loader = EntityLoader(self._agent_dir)
            self._controller = await loader.get_controller()
        return self._controller

    async def __call__(self, state: dict) -> dict:
        controller = await self._get_controller()
        logger.info(
            f"[heartbeat_run] invoking {self._agent_dir.name!r} "
            f"thread={_HEARTBEAT_THREAD!r} prompt={self._prompt[:60]!r}"
        )

        reply = await controller.invoke(self._prompt, thread_id=_HEARTBEAT_THREAD)

        logger.info(f"[heartbeat_run] {self._agent_dir.name!r} reply_len={len(reply)}")
        if is_debug():
            logger.debug(f"[heartbeat_run] reply_preview={reply[:200]!r}")
        return {"messages": [AIMessage(content=reply)]}


AgentRunNode = HeartbeatNode
