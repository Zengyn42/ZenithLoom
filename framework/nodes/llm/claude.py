"""
框架级 Claude SDK 节点 — framework/nodes/llm/claude.py

ClaudeSDKNode 继承 LlmNode，实现 call_llm() 接口：
  call_llm(prompt, session_id, tools, cwd) -> (text, new_session_id)
    - session_id 空 -> 新建 session
    - session_id 非空 -> resume 已有 session（~/.claude/ 本地存储）

实现方式：claude_agent_sdk.query()（SDK 流式协议）。
sdk_query() 通过 wait_for_result_and_end_input() 关闭 stdin，
让子进程优雅退出，确保会话历史（user/assistant 条目）写入 JSONL 文件。

注意：SDK 内部 Query._read_messages_task 会将 ProcessError 包装成
普通 Exception 再推入消息流，因此异常捕获需检测消息内容而非类型。

基类 LlmNode.__call__() 处理所有图协议逻辑（路由、注入、信号检测）；
ClaudeSDKNode 只负责 Claude SDK 调用。

permission_mode 实现：
  Claude SDK 原生支持全部四种模式（default / plan / acceptEdits / bypassPermissions）。
  - self._permission_mode  → 直传给 ClaudeAgentOptions(permission_mode=...)
  - self._get_disallowed_tools() → 直传给 ClaudeAgentOptions(disallowed_tools=...)
    plan 模式时，基类自动将 _WRITE_TOOLS 合并到 disallowed_tools 列表中。
  两个值均来自 LlmNode 基类，ClaudeSDKNode 无需自行处理 permission_mode 逻辑。
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
    使用 sdk_query() 流式调用，stdin 关闭后进程优雅退出，会话历史落盘。
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

        async def _run_once(sid: str, msg_text: str) -> tuple[str, str, bool]:
            """
            调用 sdk_query()，返回 (result_text, new_session_id, is_error)。

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

            async for msg in sdk_query(prompt=msg_text, options=_make_options(sid)):
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

            return _result, _new_sid, _is_error

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

        try:
            result_text, new_session_id, is_error = await _run_once(session_id, prompt)
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
                    result_text, new_session_id, is_error = await _run_once("", prompt)
                except Exception:
                    # Retry also failed; new_session_id stays ""
                    # llm_node will write "" which clears the session → fresh start next turn
                    raise
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
