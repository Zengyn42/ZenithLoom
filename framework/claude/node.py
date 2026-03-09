"""
框架级 Claude SDK 节点 — framework/claude/node.py

可复用的 Claude Code CLI SDK 包装。任何 LangGraph 图都可直接使用。
用法：node = ClaudeNode(config, system_prompt); await node.call_claude(...)
"""

import json
import logging

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    get_session_messages,
    list_sessions as sdk_list_sessions,
)
from claude_agent_sdk._errors import ProcessError

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.token_tracker import update_token_stats

logger = logging.getLogger(__name__)


class ClaudeNode:
    """
    可复用的 Claude SDK 节点（自包含 session-aware）。

    接口：
      call_claude(prompt, session_id, tools, cwd) → (text, new_session_id)
        - session_id 空 → 新建 session
        - session_id 非空 → resume 已有 session（~/.claude/ 本地存储）
      get_recent_history(session_id, limit) → list
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
        """
        model = self.config.claude_model or "default"
        sid_short = session_id[:8] if session_id else "new"
        logger.info(f"[claude] model={model} sid={sid_short}")
        if is_debug():
            logger.debug(f"[claude] prompt_len={len(prompt)} cwd={cwd!r}")

        # stderr 回调：确保 CLI 错误细节出现在日志里
        stderr_lines: list[str] = []

        def _on_stderr(line: str) -> None:
            stderr_lines.append(line)
            logger.debug(f"[claude/stderr] {line.rstrip()}")

        settings_val = (
            json.dumps(self.config.settings_override)
            if self.config.settings_override
            else None
        )
        options = ClaudeAgentOptions(
            system_prompt=self.system_prompt or None,
            cwd=cwd or self.config.workspace or None,
            allowed_tools=tools or self.config.tools,
            permission_mode=self.config.permission_mode,
            resume=session_id or None,
            model=self.config.claude_model or None,
            env={"CLAUDECODE": "", "CLAUDE_CODE_SESSION": "", "CLAUDE_AGENT_SDK": "1"},
            stderr=_on_stderr,
            setting_sources=self.config.setting_sources,
            settings=settings_val,
        )

        result_text = ""
        new_session_id = session_id

        async def _run_with_options(opts: ClaudeAgentOptions, sid: str) -> tuple[str, str]:
            _result = ""
            _new_sid = sid
            _client = ClaudeSDKClient(opts)
            try:
                await _client.connect()
                await _client.query(prompt, session_id=sid or "default")
                async for msg in _client.receive_messages():
                    if isinstance(msg, ResultMessage):
                        _new_sid = msg.session_id or sid
                        if msg.usage:
                            update_token_stats(msg.usage)
                        if msg.result:
                            _result = msg.result.strip()
                        break
            finally:
                await _client.disconnect()
            return _result, _new_sid

        try:
            result_text, new_session_id = await _run_with_options(options, session_id)
        except ProcessError as e:
            if stderr_lines:
                logger.error(
                    f"[claude] CLI stderr ({len(stderr_lines)} lines):\n"
                    + "\n".join(stderr_lines[-20:])
                )
            # resume 失败（session 文件不在当前 cwd 对应的目录下，或已被删除）
            if session_id:
                logger.warning(
                    f"[claude] resume sid={session_id[:8]} 失败（cwd 可能已变更），"
                    "以新 session 重试..."
                )
                fresh_options = ClaudeAgentOptions(
                    system_prompt=options.system_prompt,
                    cwd=options.cwd,
                    allowed_tools=options.allowed_tools,
                    permission_mode=options.permission_mode,
                    resume=None,
                    model=options.model,
                    env=options.env,
                    stderr=_on_stderr,
                )
                result_text, new_session_id = await _run_with_options(fresh_options, "")
            else:
                raise
        except Exception as e:
            if stderr_lines:
                logger.error(
                    f"[claude] CLI stderr ({len(stderr_lines)} lines):\n"
                    + "\n".join(stderr_lines[-20:])
                )
            raise

        new_sid_short = new_session_id[:8] if new_session_id else "new"
        logger.info(f"[claude] done sid={new_sid_short} output_len={len(result_text)}")
        if is_debug():
            logger.debug(f"[claude] output_preview={result_text[:200]!r}")
        return result_text, new_session_id

    # 标准 LLM 节点接口（与 LlamaNode 保持一致）
    call_llm = call_claude

    def get_recent_history(self, session_id: str, limit: int = 10) -> list:
        """获取 Claude session 近期消息（供 Gemini 等节点获取上下文）。"""
        if not session_id:
            return []
        try:
            return get_session_messages(
                session_id,
                directory=self.config.workspace or None,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"[claude] get_recent_history failed: {e}")
            return []

    @staticmethod
    def list_sessions(directory: str | None = None, limit: int = 20) -> list:
        """列出可用的 Claude sessions。"""
        try:
            return sdk_list_sessions(directory=directory, limit=limit)
        except Exception as e:
            logger.warning(f"[claude] list_sessions failed: {e}")
            return []
