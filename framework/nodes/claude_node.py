"""
框架级 Claude SDK 节点 — ClaudeNode

可复用的 Claude Code CLI SDK 包装。
用法：node = ClaudeNode(config, system_prompt); 调用 await node.call_claude(...)
"""

import logging

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    get_session_messages,
    list_sessions as sdk_list_sessions,
)

from framework.config import AgentConfig
from framework.token_tracker import update_token_stats

logger = logging.getLogger(__name__)


class ClaudeNode:
    """
    可复用的 Claude SDK 节点。

    核心方法：
      call_claude(prompt, session_id, tools, cwd) → (text, new_session_id)
      get_recent_history(session_id, limit) → list[SessionMessage]
    """

    def __init__(self, config: AgentConfig, system_prompt: str = ""):
        self.config = config
        self.system_prompt = system_prompt

    async def call_claude(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> tuple[str, str]:
        """
        调用 Claude SDK，返回 (text, new_session_id)。

        - session_id 非空时用 resume 续接已有 session
        - tools 可覆盖 config 默认工具列表
        - cwd 可覆盖 config 默认工作目录
        """
        options = ClaudeAgentOptions(
            system_prompt=self.system_prompt or None,
            cwd=cwd or self.config.workspace or None,
            allowed_tools=tools or self.config.tools,
            permission_mode=self.config.permission_mode,
            resume=session_id or None,
            env={"CLAUDECODE": "", "CLAUDE_CODE_SESSION": ""},
        )

        result_text = ""
        new_session_id = session_id

        client = ClaudeSDKClient(options)
        try:
            await client.connect()
            await client.query(prompt, session_id=session_id or "default")

            async for msg in client.receive_messages():
                if isinstance(msg, ResultMessage):
                    new_session_id = msg.session_id or session_id
                    if msg.usage:
                        update_token_stats(msg.usage)
                    # ResultMessage.result 是权威最终文本
                    if msg.result:
                        result_text = msg.result.strip()
                    break  # ResultMessage 是最后一条
        finally:
            await client.disconnect()
        sid_short = new_session_id[:8] if new_session_id else "new"
        logger.info(
            f"[claude_node] sid={sid_short} output_len={len(result_text)}"
        )
        return result_text, new_session_id

    def get_recent_history(
        self, session_id: str, limit: int = 10
    ) -> list:
        """
        获取 Claude session 的近期消息（同步，因为 SDK 函数是同步的）。
        供 Gemini 等无状态节点获取上下文。
        """
        if not session_id:
            return []
        try:
            return get_session_messages(
                session_id,
                directory=self.config.workspace or None,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"[claude_node] 获取 session 历史失败: {e}")
            return []

    @staticmethod
    def list_sessions(directory: str | None = None, limit: int = 20) -> list:
        """列出可用的 Claude sessions。"""
        try:
            return sdk_list_sessions(directory=directory, limit=limit)
        except Exception as e:
            logger.warning(f"[claude_node] 列出 sessions 失败: {e}")
            return []
