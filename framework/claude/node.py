"""
框架级 Claude CLI 节点 — framework/claude/node.py

ClaudeNode 继承 AgentNode，实现 call_llm() 接口：
  call_llm(prompt, session_id, tools, cwd) -> (text, new_session_id)
    - session_id 空 -> 新建 session
    - session_id 非空 -> resume 已有 session（~/.claude/ 本地存储）

实现方式：直接调用 claude -p <prompt> --output-format json（非 SDK 流式协议）。
进程自然退出确保会话历史落盘，下一轮 --resume 可正常加载。

基类 AgentNode.__call__() 处理所有图协议逻辑（路由、注入、信号检测）；
ClaudeNode 只负责 Claude CLI 调用。
"""

import asyncio
import json
import logging
import os

from claude_agent_sdk import (
    get_session_messages,
    list_sessions as sdk_list_sessions,
)
from claude_agent_sdk._errors import ProcessError

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.agent_node import AgentNode
from framework.token_tracker import update_token_stats

logger = logging.getLogger(__name__)


class ClaudeNode(AgentNode):
    """
    Claude CLI LLM 节点。

    使用 asyncio.create_subprocess_exec(['claude', '-p', ...]) 直接调用，
    进程通过 communicate() 自然退出，确保会话历史写入 JSONL 文件。
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
        """调用 Claude CLI，返回 (text, new_session_id)。"""
        model = self.node_config.get("model") or self.node_config.get("claude_model") or "default"
        sid_short = session_id[:8] if session_id else "new"
        logger.info(f"[claude] model={model} sid={sid_short}")
        if is_debug():
            logger.debug(f"[claude] prompt_len={len(prompt)} cwd={cwd!r}")

        stderr_lines: list[str] = []

        def _on_stderr(line: str) -> None:
            stderr_lines.append(line)
            logger.debug(f"[claude/stderr] {line.rstrip()}")

        env = {
            **os.environ,
            "CLAUDECODE": "",
            "CLAUDE_CODE_SESSION": "",
            "CLAUDE_AGENT_SDK": "1",
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
        }

        def _build_cmd(sid: str, msg: str) -> list[str]:
            cmd = ["claude", "-p", msg, "--output-format", "json"]
            sp = self.node_config.get("system_prompt") or self.system_prompt
            if sp:
                cmd += ["--system-prompt", sp]
            if sid:
                cmd += ["--resume", sid]
            tool_list = tools or self.config.tools
            if tool_list:
                cmd += ["--allowedTools", ",".join(tool_list)]
            if self.config.permission_mode:
                cmd += ["--permission-mode", self.config.permission_mode]
            m = self.node_config.get("model") or self.node_config.get("claude_model")
            if m:
                cmd += ["--model", m]
            if self.config.setting_sources is not None:
                sources = ",".join(self.config.setting_sources) if self.config.setting_sources else ""
                cmd += ["--setting-sources", sources]
            if self.config.settings_override:
                cmd += ["--settings", json.dumps(self.config.settings_override)]
            return cmd

        async def _run_once(sid: str, msg_text: str) -> tuple[str, str, bool]:
            """
            运行 claude -p，返回 (result_text, new_session_id, is_error)。
            communicate() 等待进程自然退出，确保会话历史落盘。
            """
            cmd = _build_cmd(sid, msg_text)
            if is_debug():
                logger.debug(f"[claude/cmd] {cmd[:8]}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()

            for line in stderr_bytes.decode(errors="replace").splitlines():
                _on_stderr(line)

            if proc.returncode != 0:
                raise ProcessError(
                    f"Command failed with exit code {proc.returncode}",
                    exit_code=proc.returncode,
                    stderr=stderr_bytes.decode(errors="replace") or "Check stderr output for details",
                )

            raw = stdout_bytes.decode(errors="replace").strip()
            if is_debug():
                logger.debug(f"[claude/stdout-raw] {raw[:300]!r}")

            try:
                data = json.loads(raw)
                result = (data.get("result") or "").strip()
                new_sid = data.get("session_id") or sid
                is_err = bool(data.get("is_error", False))
                if data.get("usage"):
                    update_token_stats(data["usage"])
                return result, new_sid, is_err
            except json.JSONDecodeError:
                return raw, sid, False

        result_text = ""
        new_session_id = session_id
        is_error = False

        try:
            result_text, new_session_id, is_error = await _run_once(session_id, prompt)
        except ProcessError as e:
            if stderr_lines:
                logger.error(
                    f"[claude] CLI stderr ({len(stderr_lines)} lines):\n"
                    + "\n".join(stderr_lines[-20:])
                )
            # resume 失败 -> 以新 session 重试
            if session_id:
                logger.warning(
                    f"[claude] resume sid={session_id[:8]} 失败（ProcessError），以新 session 重试..."
                )
                result_text, new_session_id, is_error = await _run_once("", prompt)
            else:
                raise

        new_sid_short = new_session_id[:8] if new_session_id else "new"
        logger.info(f"[claude] done sid={new_sid_short} output_len={len(result_text)}")
        if is_debug():
            logger.debug(f"[claude] output_preview={result_text[:200]!r}")
        return result_text, new_session_id

    # 向后兼容别名
    call_claude = call_llm

    def get_recent_history(self, session_id: str, limit: int = 10) -> list:
        """获取 Claude session 近期消息。"""
        if not session_id:
            return []
        try:
            return get_session_messages(session_id, directory=None, limit=limit)
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
