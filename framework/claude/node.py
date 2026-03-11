"""
框架级 Claude SDK 节点 — framework/claude/node.py

ClaudeNode 继承 AgentNode，实现 call_llm() 接口：
  call_llm(prompt, session_id, tools, cwd) → (text, new_session_id)
    - session_id 空 → 新建 session
    - session_id 非空 → resume 已有 session（~/.claude/ 本地存储）

基类 AgentNode.__call__() 处理所有图协议逻辑（路由、注入、信号检测）；
ClaudeNode 只负责 Claude CLI SDK 调用。

构造器：
  ClaudeNode(config, node_config, system_prompt="")
    config       — AgentConfig（图级共享配置）
    node_config  — dict（节点级配置，来自 agent.json）
    system_prompt— Claude system prompt（由 AgentLoader.load_system_prompt() 提供）
"""

import json
import logging

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    get_session_messages,
    list_sessions as sdk_list_sessions,
    query as sdk_query,
)
from claude_agent_sdk._errors import ProcessError

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.agent_node import AgentNode
from framework.token_tracker import update_token_stats

logger = logging.getLogger(__name__)


class ClaudeNode(AgentNode):
    """
    Claude CLI SDK LLM 节点。

    继承 AgentNode，实现 call_llm()。
    基类处理路由信号检测、资源锁、session UUID 路由等框架逻辑。
    """

    def __init__(
        self,
        config: AgentConfig,
        node_config: dict,
        system_prompt: str = "",
    ):
        super().__init__(config, node_config)
        self.system_prompt = system_prompt

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> tuple[str, str]:
        """
        调用 Claude SDK，返回 (text, new_session_id)。
        """
        model = self.node_config.get("model") or self.node_config.get("claude_model") or "default"
        sid_short = session_id[:8] if session_id else "new"
        logger.info(f"[claude] model={model} sid={sid_short}")
        if is_debug():
            logger.debug(f"[claude] prompt_len={len(prompt)} cwd={cwd!r}")
            _stdin_msg = {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
                "session_id": session_id or "default",
            }
            logger.debug(f"[claude/stdin-json] {json.dumps(_stdin_msg, ensure_ascii=False)}")

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
            cwd=cwd or None,
            allowed_tools=tools or self.config.tools,
            permission_mode=self.config.permission_mode,
            resume=session_id or None,
            model=self.node_config.get("model") or self.node_config.get("claude_model") or None,
            env={"CLAUDECODE": "", "CLAUDE_CODE_SESSION": "", "CLAUDE_AGENT_SDK": "1"},
            stderr=_on_stderr,
            setting_sources=self.config.setting_sources,
            settings=settings_val,
        )

        result_text = ""
        new_session_id = session_id
        is_error = False

        async def _run_once(opts: ClaudeAgentOptions, sid: str, msg_text: str) -> tuple[str, str, bool]:
            """返回 (result_text, new_session_id, is_error)。

            使用 sdk_query()（而非 ClaudeSDKClient）：query() 通过关闭 stdin 让
            子进程优雅退出，确保会话历史（user/assistant 条目）写入 JSONL 文件，
            下一轮 --resume 才能成功。ClaudeSDKClient.disconnect() 直接 terminate()
            子进程，导致历史未落盘，resume 失败。
            """
            _result = ""
            _new_sid = sid
            _is_error = False
            async for msg in sdk_query(prompt=msg_text, options=opts):
                if isinstance(msg, ResultMessage):
                    _new_sid = msg.session_id or sid
                    _is_error = msg.is_error
                    if msg.usage:
                        update_token_stats(msg.usage)
                    if msg.result:
                        _result = msg.result.strip()
            return _result, _new_sid, _is_error

        try:
            result_text, new_session_id, is_error = await _run_once(options, session_id, prompt)
        except ProcessError as e:
            if stderr_lines:
                logger.error(
                    f"[claude] CLI stderr ({len(stderr_lines)} lines):\n"
                    + "\n".join(stderr_lines[-20:])
                )
            # resume 失败（ProcessError：session 文件不存在或已过期）→ 以新 session 重试
            if session_id:
                logger.warning(
                    f"[claude] resume sid={session_id[:8]} 失败（ProcessError），以新 session 重试..."
                )
                options_fresh = ClaudeAgentOptions(
                    system_prompt=options.system_prompt,
                    cwd=options.cwd,
                    allowed_tools=options.allowed_tools,
                    permission_mode=options.permission_mode,
                    resume=None,
                    model=options.model,
                    env=options.env,
                    stderr=_on_stderr,
                    setting_sources=options.setting_sources,
                    settings=settings_val,
                )
                result_text, new_session_id, is_error = await _run_once(options_fresh, "", prompt)
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

    # 向后兼容别名（旧代码调用 call_claude()）
    call_claude = call_llm

    def get_recent_history(self, session_id: str, limit: int = 10) -> list:
        """获取 Claude session 近期消息（供 Gemini 等节点获取上下文）。"""
        if not session_id:
            return []
        try:
            return get_session_messages(
                session_id,
                directory=None,
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
