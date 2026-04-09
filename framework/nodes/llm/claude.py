"""
框架级 Claude 节点 — framework/nodes/llm/claude.py

两种实现：
  ClaudeSDKNode(LlmNode)  — Agent SDK（通过 ClaudeSDKClient 持久连接）
  ClaudeCLINode(LlmNode)  — CLI subprocess（通过 asyncio.create_subprocess_exec）

共同接口：
  call_llm(prompt, session_id, tools, cwd) -> (text, new_session_id)
    - session_id 空 -> 新建 session
    - session_id 非空 -> resume 已有 session（~/.claude/ 本地存储）

ClaudeSDKNode：
  使用 ClaudeSDKClient 维持一个持久化 CLI 子进程。
  client.query(prompt, session_id=...) 按频道 session_id 路由。
  CLI 内部自动管理 context（auto-compact），避免 session 历史无限膨胀。
  asyncio.Lock 保证并发 query 串行执行（CLI 子进程单线程）。

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
from uuid import uuid4

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    get_session_messages,
    list_sessions as sdk_list_sessions,
)
from claude_agent_sdk.types import StreamEvent
from claude_agent_sdk._errors import ProcessError, CLIConnectionError

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.llm.llm_node import LlmNode as AgentNode
from framework.nodes.llm.llm_node import set_stream_callback, get_stream_callback, _stream_cb
from framework.token_tracker import update_token_stats

logger = logging.getLogger(__name__)


class ClaudeSDKNode(AgentNode):
    """
    Claude SDK LLM 节点（持久连接模式）。

    继承 AgentNode，实现 call_llm()。
    通过 ClaudeSDKClient 维持一个持久化 CLI 子进程，避免每次调用 spawn 新进程。
    CLI 内部自动管理 context（auto-compact），防止 session 历史无限膨胀。
    asyncio.Lock 保证并发频道的 query 串行执行。
    """

    # 类级别：跟踪所有活跃实例，供 reset_all_clients() 断开全部持久连接
    _instances: list["ClaudeSDKNode"] = []

    def __init__(
        self,
        config: AgentConfig,
        node_config: dict,
        system_prompt: str = "",
    ):
        super().__init__(config, node_config)
        self.system_prompt = system_prompt
        self._client: ClaudeSDKClient | None = None
        ClaudeSDKNode._instances.append(self)
        self._query_lock = asyncio.Lock()
        self._client_cwd: str | None = None
        self._stderr_lines: list[str] = []

    def _build_client_options(self, cwd: str | None = None) -> ClaudeAgentOptions:
        """构建 ClaudeSDKClient 选项（一次性，不含 per-query 字段）。"""
        sp = self.node_config.get("system_prompt") or self.system_prompt or None
        node_tools = self.node_config.get("tools")
        _allowed = node_tools if node_tools is not None else self.config.tools
        _disallowed = self._get_disallowed_tools()

        _MISSING = object()
        _node_sources = self.node_config.get("setting_sources", _MISSING)
        setting_sources = (
            _node_sources if _node_sources is not _MISSING else self.config.setting_sources
        )
        settings_override = (
            self.node_config.get("settings_override") or self.config.settings_override
        )
        settings_val = json.dumps(settings_override) if settings_override else None

        _thinking_raw = self.node_config.get("thinking")
        _thinking_cfg = None
        if _thinking_raw:
            if isinstance(_thinking_raw, str):
                _thinking_cfg = {"type": _thinking_raw}
            elif isinstance(_thinking_raw, dict):
                _thinking_cfg = _thinking_raw

        _max_buf = self.node_config.get("max_buffer_size", 10 * 1024 * 1024)

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

        def _on_stderr(line: str) -> None:
            self._stderr_lines.append(line)
            logger.debug(f"[claude/stderr] {line.rstrip()}")

        return ClaudeAgentOptions(
            system_prompt=sp,
            cwd=cwd or None,
            allowed_tools=_allowed,
            disallowed_tools=_disallowed,
            permission_mode=self._permission_mode,
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

    async def _ensure_client(self, cwd: str | None = None) -> ClaudeSDKClient:
        """懒创建并连接持久化 ClaudeSDKClient。cwd 变化时自动重连。"""
        if self._client is not None:
            if cwd and cwd != self._client_cwd:
                logger.info(f"[claude] cwd changed ({self._client_cwd} → {cwd}), reconnecting")
                await self._disconnect_client()
            else:
                return self._client

        options = self._build_client_options(cwd=cwd)
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._client_cwd = cwd
        logger.info(f"[claude] persistent client connected (cwd={cwd})")
        return self._client

    async def _disconnect_client(self) -> None:
        """断开并重置持久化 client。"""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.warning(f"[claude] disconnect error: {e}")
            self._client = None
            self._client_cwd = None

    @classmethod
    async def reset_all_clients(cls) -> int:
        """断开所有活跃实例的持久连接（供 !reset 调用）。

        下一次 call_llm 会自动重建 client，CLI 以全新状态启动。
        返回断开的 client 数量。
        """
        count = 0
        for inst in cls._instances:
            if inst._client is not None:
                await inst._disconnect_client()
                count += 1
        if count:
            logger.info(f"[claude] reset_all_clients: disconnected {count} client(s)")
        return count

    async def _run_query(self, session_id: str, prompt: str, cwd: str | None) -> tuple[str, str, bool]:
        """
        通过持久化 client 发送一次 query 并收集响应。

        返回 (result_text, new_session_id, is_error)。
        """
        client = await self._ensure_client(cwd=cwd)
        self._stderr_lines.clear()

        _result = ""
        _new_sid = session_id
        _is_error = False
        _in_thinking = False
        _text_chunks: list[str] = []

        # session_id 为空时生成唯一 UUID，确保每个 channel/调用获得独立 session。
        # 不能用 "default" — 持久连接模式下所有 channel 共享同一个 CLI 子进程，
        # "default" 会让不同 channel 意外 resume 同一个 session 导致内容泄漏。
        _effective_sid = session_id or str(uuid4())
        await client.query(prompt=prompt, session_id=_effective_sid)

        async for msg in client.receive_response():
            if isinstance(msg, StreamEvent):
                cb = _stream_cb.get()
                ev = msg.event
                etype = ev.get("type")
                if etype == "content_block_start":
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
                        _text_chunks.append(text)
                        if cb:
                            cb(text, False)
                    elif dtype == "thinking_delta":
                        thinking_text = delta.get("thinking", "")
                        if cb and thinking_text:
                            cb(thinking_text, True)
            elif isinstance(msg, ResultMessage):
                _new_sid = msg.session_id or session_id
                _is_error = msg.is_error
                if msg.usage:
                    update_token_stats(msg.usage)
                if msg.result:
                    _result = msg.result.strip()

        if not _result and _text_chunks:
            _result = "".join(_text_chunks).strip()

        return _result, _new_sid, _is_error

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
        history: list | None = None,
    ) -> tuple[str, str]:
        """调用 Claude SDK（持久连接），返回 (text, new_session_id)。history 由 SDK session 管理，忽略。"""
        model = self.node_config.get("model") or self.node_config.get("claude_model") or "default"
        sid_short = session_id[:8] if session_id else "new"
        logger.info(f"[claude] model={model} sid={sid_short}")
        if is_debug():
            logger.debug(f"[claude] prompt_len={len(prompt)} cwd={cwd!r}")

        result_text = ""
        new_session_id = ""
        is_error = False

        async with self._query_lock:
            try:
                result_text, new_session_id, is_error = await self._run_query(
                    session_id, prompt, cwd,
                )
            except (CLIConnectionError, OSError) as e:
                # CLI 子进程死了 → 重连一次
                logger.warning(f"[claude] client connection lost, reconnecting: {e}")
                await self._disconnect_client()
                try:
                    result_text, new_session_id, is_error = await self._run_query(
                        session_id, prompt, cwd,
                    )
                except Exception as retry_err:
                    logger.error(f"[claude] reconnect retry failed: {retry_err}")
                    result_text = f"[Claude 暂时不可用] {retry_err}"
                    new_session_id = ""
                    is_error = True
            except Exception as e:
                if self._stderr_lines:
                    logger.error(
                        f"[claude] CLI stderr ({len(self._stderr_lines)} lines):\n"
                        + "\n".join(self._stderr_lines[-20:])
                    )
                # resume 失败 → 以新 session 重试
                _is_exit = isinstance(e, ProcessError) or "exit code" in str(e).lower()
                if _is_exit and session_id:
                    logger.warning(
                        f"[claude] resume sid={session_id[:8]} 失败，以新 session 重试..."
                    )
                    try:
                        result_text, new_session_id, is_error = await self._run_query(
                            "", prompt, cwd,
                        )
                    except Exception as retry_err:
                        logger.error(f"[claude] 新 session 重试也失败: {retry_err}")
                        result_text = f"[Claude 暂时不可用] {retry_err}"
                        new_session_id = ""
                        is_error = True
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
