"""
框架级 Claude 节点 — framework/nodes/llm/claude.py

两种实现：
  ClaudeSDKNode(LlmNode)  — Agent SDK（通过 claude_agent_sdk.query()）
  ClaudeCLINode(LlmNode)  — CLI subprocess（通过 asyncio.create_subprocess_exec）

共同接口：
  call_llm(prompt, session_id, tools, cwd) -> (text, new_session_id)
    - session_id 空 -> 新建 session
    - session_id 非空 -> resume 已有 session（~/.claude/ 本地存储）

ClaudeSDKNode：
  sdk_query() 通过 wait_for_result_and_end_input() 关闭 stdin，
  让子进程优雅退出，确保会话历史写入 JSONL 文件。
  SDK 内部 ProcessError 被包装为普通 Exception，需检测消息内容。

ClaudeCLINode：
  通过 `claude -p --output-format stream-json --verbose --include-partial-messages`
  子进程调用，逐行解析 JSON streaming events，实时回调 _stream_cb。
  不依赖 claude_agent_sdk Python 包。

基类 LlmNode.__call__() 处理所有图协议逻辑（路由、注入、信号检测）；
两个子类只负责 call_llm() 实现。

permission_mode 实现：
  两个子类均原生支持全部模式（default / plan / acceptEdits / bypassPermissions）。
  ClaudeSDKNode：直传 ClaudeAgentOptions(permission_mode=...)
  ClaudeCLINode：直传 --permission-mode 标志
  基类 _get_disallowed_tools() 自动处理 plan 模式的工具禁用。
"""

import asyncio
import json
import logging
import os

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    get_session_messages,
    list_sessions as sdk_list_sessions,
    query as sdk_query,
)
from claude_agent_sdk.types import StreamEvent
from claude_agent_sdk._errors import ProcessError

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.llm.llm_node import LlmNode as AgentNode
from framework.nodes.llm.llm_node import set_stream_callback, get_stream_callback, _stream_cb
from framework.token_tracker import update_token_stats

logger = logging.getLogger(__name__)


class ClaudeSDKNode(AgentNode):
    """
    Claude SDK LLM 节点。

    继承 AgentNode，实现 call_llm()。
    使用 sdk_query() 流式调用，每次调用 spawn 新 CLI 子进程。

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
        history: list | None = None,
    ) -> tuple[str, str]:
        """调用 Claude SDK，返回 (text, new_session_id)。history 由 SDK session 管理，忽略。"""
        model = self.node_config.get("model") or self.node_config.get("claude_model") or "default"
        sid_short = session_id[:8] if session_id else "new"
        logger.info(f"[claude] model={model} sid={sid_short}")
        if is_debug():
            logger.debug(f"[claude] prompt_len={len(prompt)} cwd={cwd!r}")

        stderr_lines: list[str] = []

        def _on_stderr(line: str) -> None:
            stderr_lines.append(line)
            logger.debug(f"[claude/stderr] {line.rstrip()}")

        # setting_sources / settings_override：node_config 优先，退回到顶层 AgentConfig
        _MISSING = object()
        _node_sources = self.node_config.get("setting_sources", _MISSING)
        setting_sources = (
            _node_sources if _node_sources is not _MISSING else self.config.setting_sources
        )
        settings_override = (
            self.node_config.get("settings_override") or self.config.settings_override
        )
        settings_val = json.dumps(settings_override) if settings_override else None

        # Optional extended thinking: node_config["thinking"] = "adaptive" | "disabled" | {"type":..., "budget_tokens":...}
        _thinking_raw = self.node_config.get("thinking")
        _thinking_cfg = None
        if _thinking_raw:
            if isinstance(_thinking_raw, str):
                _thinking_cfg = {"type": _thinking_raw}
            elif isinstance(_thinking_raw, dict):
                _thinking_cfg = _thinking_raw

        # permission_mode + disallowed_tools 统一由基类 LlmNode 管理
        # Claude SDK 原生支持 permission_mode，直接传入
        # disallowed_tools 由基类 _get_disallowed_tools() 根据 mode 自动计算
        _disallowed = self._get_disallowed_tools()

        def _make_options(sid: str) -> ClaudeAgentOptions:
            sp = self.node_config.get("system_prompt") or self.system_prompt or None
            node_tools = self.node_config.get("tools")
            if tools is not None:
                _allowed = tools
            elif node_tools is not None:
                _allowed = node_tools
            else:
                _allowed = self.config.tools

            # max_buffer_size: node_config 可配置, 默认 10MB (SDK 默认仅 1MB,
            # ColonyCoder QA/rescue 注入大量源码+测试输出时容易超限)
            _max_buf = self.node_config.get("max_buffer_size", 10 * 1024 * 1024)

            # MCP servers: 从 MCPManager 获取当前运行中的 server SSE 配置
            # 如果 node_config 指定 "mcp_names" 列表，只取指定的 server；
            # 否则取 MCPManager 中所有当前运行的 server（agent 已 acquire 的）。
            _mcp_servers: dict = {}
            try:
                from framework.mcp_manager import MCPManager
                _mgr = MCPManager.get_instance()
                _mcp_names = self.node_config.get("mcp_names")
                if _mcp_names is not None:
                    _mcp_servers = _mgr.get_sse_configs(_mcp_names)
                else:
                    _mcp_servers = _mgr.get_all_configs()
            except Exception as _mcp_err:
                logger.debug(f"[claude] mcp_manager lookup failed: {_mcp_err}")

            return ClaudeAgentOptions(
                system_prompt=sp,
                cwd=cwd or None,
                allowed_tools=_allowed,
                disallowed_tools=_disallowed,
                permission_mode=self._permission_mode,
                resume=sid or None,
                model=self.node_config.get("model") or self.node_config.get("claude_model") or None,
                env={"CLAUDECODE": "", "CLAUDE_CODE_SESSION": "", "CLAUDE_AGENT_SDK": "1"},
                stderr=_on_stderr,
                setting_sources=setting_sources,
                settings=settings_val,
                include_partial_messages=True,
                thinking=_thinking_cfg,
                add_dirs=self._add_dirs,
                max_buffer_size=_max_buf,
                mcp_servers=_mcp_servers or {},
            )

        async def _run_once(sid: str, msg_text: str) -> tuple[str, str, bool, dict]:
            """
            调用 sdk_query()，返回 (result_text, new_session_id, is_error, last_msg_usage)。

            sdk_query() 内部通过 wait_for_result_and_end_input() 关闭 stdin，
            子进程优雅退出后会话历史写入磁盘。

            注意：SDK 将传输层的 ProcessError 包装为普通 Exception 推入消息流，
            receive_messages() 收到 {"type":"error"} 时 raise Exception(msg)。
            该异常在此函数内透传，由外层统一处理。
            """
            _result = ""
            _new_sid = sid
            _is_error = False
            _in_thinking = False  # track current block type for ANSI styling
            _text_chunks: list[str] = []  # fallback when ResultMessage.result is empty
            # 追踪最后一次 API 调用的完整 usage dict（每次 tool use 循环都有独立的 message_start）。
            # 不能用 ResultMessage.usage — 那是累计值，复杂 tool use 会远超真实 context。
            # 这份 dict 会随返回值向外透出，供 call_llm 末尾的内联 token 行显示。
            _last_msg_usage: dict = {}

            async for msg in sdk_query(prompt=msg_text, options=_make_options(sid)):
                if isinstance(msg, StreamEvent):
                    cb = _stream_cb.get()
                    ev = msg.event
                    etype = ev.get("type")
                    if etype == "message_start":
                        # 每次 API 调用开始，捕获本次调用的 usage dict（浅拷贝）。
                        # 最终保留最后一次，反映本轮最终 API 调用的真实 context 占用。
                        _usage = ev.get("message", {}).get("usage", {})
                        if _usage:
                            _last_msg_usage = dict(_usage)
                    elif etype == "content_block_start":
                        btype = ev.get("content_block", {}).get("type")
                        if btype == "thinking":
                            _in_thinking = True
                        elif btype == "text" and _in_thinking:
                            _in_thinking = False
                    elif etype == "content_block_stop" and _in_thinking:
                        _in_thinking = False
                    elif etype == "content_block_delta":
                        delta = ev.get("delta", {})
                        dtype = delta.get("type")
                        if dtype == "text_delta":
                            text = delta.get("text", "")
                            _text_chunks.append(text)  # always accumulate
                            if cb:
                                cb(text, False)
                        elif dtype == "thinking_delta":
                            thinking_text = delta.get("thinking", "")
                            if cb and thinking_text:
                                cb(thinking_text, True)
                elif isinstance(msg, ResultMessage):
                    _new_sid = msg.session_id or sid
                    _is_error = msg.is_error
                    if msg.usage:
                        update_token_stats(msg.usage)
                    if msg.result:
                        _result = msg.result.strip()

            # include_partial_messages=True may leave ResultMessage.result empty;
            # fall back to text accumulated from stream events
            if not _result and _text_chunks:
                _result = "".join(_text_chunks).strip()

            return _result, _new_sid, _is_error, _last_msg_usage

        def _is_cli_exit_error(e: Exception) -> bool:
            """
            判断异常是否来自 CLI 子进程非零退出。
            ProcessError 在 SDK 内部被包装为 Exception(str(e))，
            因此同时检测类型和消息内容。
            """
            return isinstance(e, ProcessError) or "exit code" in str(e).lower()

        result_text = ""
        new_session_id = ""
        is_error = False
        last_msg_usage: dict = {}

        try:
            result_text, new_session_id, is_error, last_msg_usage = await _run_once(session_id, prompt)
        except Exception as e:
            if stderr_lines:
                logger.error(
                    f"[claude] CLI stderr ({len(stderr_lines)} lines):\n"
                    + "\n".join(stderr_lines[-20:])
                )
            # resume 失败 -> 以新 session 重试
            if _is_cli_exit_error(e) and session_id:
                logger.warning(
                    f"[claude] resume sid={session_id[:8]} 失败，以新 session 重试..."
                )
                try:
                    result_text, new_session_id, is_error, last_msg_usage = await _run_once("", prompt)
                except Exception as retry_err:
                    # 重试也失败：返回错误文本 + 空 session ID，
                    # 确保 llm_node.__call__ 正常写入 state 清掉坏 session，
                    # 避免 checkpoint 保留坏 session ID 导致无限 resume 失败循环。
                    logger.error(f"[claude] 新 session 重试也失败: {retry_err}")
                    result_text = f"[Claude 暂时不可用] {retry_err}"
                    new_session_id = ""
                    is_error = True
                    last_msg_usage = {}
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


# 向后兼容别名
ClaudeNode = ClaudeSDKNode


class ClaudeCLINode(AgentNode):
    """
    Claude CLI subprocess 节点（CLAUDE_CLI 节点类型）。

    通过 `claude -p --output-format stream-json --verbose --include-partial-messages`
    子进程调用，逐行解析 JSON streaming events。

    功能对齐 ClaudeSDKNode：
      - 实时 streaming（text_delta / thinking_delta → _stream_cb）
      - Session resume（--resume session_id）
      - Resume 失败自动重试（新 session）
      - Permission mode（--permission-mode 直传）
      - Token usage 统计（从 result event 提取）

    与 ClaudeSDKNode 的区别：
      - 不依赖 claude_agent_sdk Python 包
      - 直接管理子进程 stdout/stderr
      - 无 get_recent_history / list_sessions 方法
    """

    _DEFAULT_TIMEOUT = 120
    _MAX_TIMEOUT = 600
    _TIMEOUT_CHARS_PER_SEC = 200

    def __init__(
        self,
        config: AgentConfig,
        node_config: dict,
        system_prompt: str = "",
    ):
        super().__init__(config, node_config)
        self.system_prompt = system_prompt

    def _build_cmd(self, session_id: str = "") -> list[str]:
        """构建 claude CLI 命令行参数列表。"""
        cmd = [
            "claude", "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]

        # model
        model = (
            self.node_config.get("model")
            or self.node_config.get("claude_model")
            or None
        )
        if model:
            cmd.extend(["--model", model])

        # permission_mode
        cmd.extend(["--permission-mode", self._permission_mode])

        # system_prompt
        sp = self.node_config.get("system_prompt") or self.system_prompt or None
        if sp:
            cmd.extend(["--system-prompt", sp])

        # allowed_tools
        node_tools = self.node_config.get("tools")
        if node_tools is not None:
            _allowed = node_tools
        else:
            _allowed = self.config.tools
        if _allowed:
            cmd.extend(["--allowedTools"] + list(_allowed))

        # disallowed_tools
        _disallowed = self._get_disallowed_tools()
        if _disallowed:
            cmd.extend(["--disallowedTools"] + list(_disallowed))

        # resume
        if session_id:
            cmd.extend(["--resume", session_id])

        # add_dirs
        for d in self._add_dirs:
            cmd.extend(["--add-dir", str(d)])

        # setting_sources
        _MISSING = object()
        _node_sources = self.node_config.get("setting_sources", _MISSING)
        setting_sources = (
            _node_sources if _node_sources is not _MISSING else self.config.setting_sources
        )
        if setting_sources is not None:
            if isinstance(setting_sources, list):
                cmd.extend(["--setting-sources", ",".join(setting_sources)])
            else:
                cmd.extend(["--setting-sources", str(setting_sources)])

        # settings override
        settings_override = (
            self.node_config.get("settings_override") or self.config.settings_override
        )
        if settings_override:
            cmd.extend(["--settings", json.dumps(settings_override)])

        # MCP servers
        try:
            from framework.mcp_manager import MCPManager
            _mgr = MCPManager.get_instance()
            _mcp_names = self.node_config.get("mcp_names")
            if _mcp_names is not None:
                _mcp_configs = _mgr.get_sse_configs(_mcp_names)
            else:
                _mcp_configs = _mgr.get_all_configs()
            if _mcp_configs:
                cmd.extend(["--mcp-config", json.dumps({"mcpServers": _mcp_configs})])
        except Exception as _mcp_err:
            logger.debug(f"[claude-cli] mcp_manager lookup failed: {_mcp_err}")

        return cmd

    async def _run_cli(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> tuple[str, str, bool]:
        """
        执行单次 Claude CLI 调用，流式解析 stdout。

        返回 (result_text, session_id, is_error)。
        """
        cmd = self._build_cmd(session_id=session_id)

        # tools 参数覆盖 _build_cmd 中的默认值
        if tools is not None:
            # 移除 _build_cmd 已添加的 --allowedTools 段
            try:
                idx = cmd.index("--allowedTools")
                end = idx + 1
                while end < len(cmd) and not cmd[end].startswith("--"):
                    end += 1
                cmd[idx:end] = (["--allowedTools"] + list(tools)) if tools else []
            except ValueError:
                if tools:
                    cmd.extend(["--allowedTools"] + list(tools))

        effective_timeout = min(
            self._MAX_TIMEOUT,
            max(self._DEFAULT_TIMEOUT, len(prompt) // self._TIMEOUT_CHARS_PER_SEC),
        )

        env = dict(os.environ)
        env["CLAUDE_AGENT_SDK"] = "1"

        model = self.node_config.get("model") or self.node_config.get("claude_model") or "default"
        sid_short = session_id[:8] if session_id else "new"
        logger.info(f"[claude-cli] model={model} sid={sid_short} timeout={effective_timeout}s")
        if is_debug():
            logger.debug(f"[claude-cli] cmd={cmd}")
            logger.debug(f"[claude-cli] prompt_len={len(prompt)} cwd={cwd!r}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or None,
            env=env,
        )

        # Write prompt to stdin and close
        proc.stdin.write(prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()

        result_text = ""
        new_session_id = session_id
        is_error = False
        text_chunks: list[str] = []
        stderr_lines: list[str] = []

        cb = _stream_cb.get()

        try:
            async def _read_stderr():
                async for raw in proc.stderr:
                    line = raw.decode(errors="replace").rstrip()
                    stderr_lines.append(line)
                    logger.debug(f"[claude-cli/stderr] {line}")

            stderr_task = asyncio.create_task(_read_stderr())

            async def _read_stdout():
                nonlocal result_text, new_session_id, is_error
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace").rstrip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug(f"[claude-cli] skip non-JSON: {line[:100]}")
                        continue

                    msg_type = data.get("type")

                    if msg_type == "stream_event":
                        ev = data.get("event", {})
                        etype = ev.get("type")
                        if etype == "content_block_delta":
                            delta = ev.get("delta", {})
                            dtype = delta.get("type")
                            if dtype == "text_delta":
                                text = delta.get("text", "")
                                text_chunks.append(text)
                                if cb:
                                    cb(text, False)
                            elif dtype == "thinking_delta":
                                thinking_text = delta.get("thinking", "")
                                if cb and thinking_text:
                                    cb(thinking_text, True)

                    elif msg_type == "result":
                        result_text = data.get("result", "").strip()
                        new_session_id = data.get("session_id", session_id) or ""
                        is_error = data.get("is_error", False)
                        usage = data.get("usage")
                        if usage:
                            update_token_stats(usage)

            stdout_task = asyncio.create_task(_read_stdout())

            await asyncio.wait_for(proc.wait(), timeout=effective_timeout)
            await stdout_task
            await stderr_task

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"Claude CLI 超时 ({effective_timeout}s, prompt_len={len(prompt)})"
            )

        if proc.returncode != 0 and not result_text:
            stderr_text = "\n".join(stderr_lines[-20:]) if stderr_lines else ""
            raise RuntimeError(
                f"Claude CLI 退出码 {proc.returncode}: {stderr_text[:500]}"
            )

        # Fallback: if result event was missing, use accumulated text
        if not result_text and text_chunks:
            result_text = "".join(text_chunks).strip()

        return result_text, new_session_id, is_error

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
        history: list | None = None,
    ) -> tuple[str, str]:
        """调用 Claude CLI subprocess，返回 (text, new_session_id)。history 由 CLI session 管理，忽略。"""
        result_text = ""
        new_session_id = ""

        try:
            result_text, new_session_id, is_error = await self._run_cli(
                prompt, session_id=session_id, tools=tools, cwd=cwd,
            )
        except Exception as e:
            # resume 失败 → 以新 session 重试
            if session_id:
                logger.warning(
                    f"[claude-cli] resume sid={session_id[:8]} 失败，以新 session 重试: {e}"
                )
                try:
                    result_text, new_session_id, is_error = await self._run_cli(
                        prompt, session_id="", tools=tools, cwd=cwd,
                    )
                except Exception as retry_err:
                    logger.error(f"[claude-cli] 新 session 重试也失败: {retry_err}")
                    result_text = f"[Claude CLI 暂时不可用] {retry_err}"
                    new_session_id = ""
            else:
                raise

        new_sid_short = new_session_id[:8] if new_session_id else "none"
        logger.info(f"[claude-cli] done sid={new_sid_short} output_len={len(result_text)}")
        if is_debug():
            logger.debug(f"[claude-cli] output_preview={result_text[:200]!r}")
        return result_text, new_session_id

    # 向后兼容别名
    call_claude = call_llm
