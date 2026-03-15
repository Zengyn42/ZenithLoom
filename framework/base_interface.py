"""
BaseInterface — 共享基类，CLI / Discord / GChat 均继承。

提取三个接口公共逻辑：
  invoke_agent()      调用 LangGraph graph，返回完整回复字符串
                      若 _streaming=True，逐 token 调用 _on_stream_chunk()
  handle_command()    处理通用 ! 命令，返回回复字符串（None = 未识别，交子类处理）
  split_fence_aware() Markdown fence-aware 分块（Discord / GChat 发消息限制）
  extract_attachments() 解析 [SEND_FILE:...] 标记

子类专属（不在此实现）：
  CLI     — stdin/stdout 流式输出、!topology、!debug、!snapshots、!rollback
  Discord — discord.py 事件、白名单、typing indicator、文件发送、!memory / !compact 等
            Discord 因需 async 编辑消息，重写整个 invoke_agent()
  GChat   — gws 子进程流、space 管理
"""

import re as _re
from langchain_core.messages import HumanMessage

_SEND_FILE_RE = _re.compile(r"\[SEND_FILE:\s*([^\]]+)\]")


class BaseInterface:
    def __init__(self, loader) -> None:
        self._loader = loader
        self._controller = None
        self._session_mgr = None
        self._streaming: bool = True  # toggleable via !stream
        self._last_stream_chunk_count: int = 0  # chunks emitted in last invoke

    async def setup(self) -> None:
        """初始化 controller 和 session_mgr（所有子类应在 run() 开头调用）。"""
        self._controller = await self._loader.get_controller()
        self._session_mgr = self._controller.session_mgr

    # ------------------------------------------------------------------
    # Agent 调用
    # ------------------------------------------------------------------

    async def invoke_agent(
        self,
        user_input: str,
        extra_state: dict | None = None,
    ) -> str:
        """
        调用 LangGraph graph，返回最终回复字符串。

        若 _streaming=True，在生成过程中逐 token 调用 _on_stream_chunk()。
        Discord 因需 async 编辑消息，通常整体重写此方法，而非仅覆盖 _on_stream_chunk()。
        """
        from framework.claude.node import set_stream_callback

        engine = self._controller._graph
        config = self._controller.get_config()
        init_state: dict = {"messages": [HumanMessage(content=user_input)]}
        if extra_state:
            init_state.update(extra_state)

        self._last_stream_chunk_count = 0
        self._on_stream_reset()
        if self._streaming:
            set_stream_callback(self._on_stream_chunk)
        try:
            result_state = await engine.ainvoke(init_state, config=config)
        finally:
            if self._streaming:
                set_stream_callback(None)

        return self._extract_response(result_state)

    def _on_stream_chunk(self, text: str, is_thinking: bool = False) -> None:
        """
        每个流式 token 到来时调用（同步）。
        is_thinking=True 表示模型内部推理文本（thinking block）。
        CLI 子类覆盖此方法打印到 stdout；
        Discord 因需异步操作，重写整个 invoke_agent() 而非此方法。
        """
        pass

    def _on_stream_reset(self) -> None:
        """每次 invoke_agent() 开始前调用，供子类重置流式状态（如 _in_thinking 标志）。"""
        pass

    @staticmethod
    def _extract_response(result_state: dict) -> str:
        """从 ainvoke() 返回的最终 state 提取最后一条 AI 消息内容。"""
        msgs = result_state.get("messages", [])
        for msg in reversed(msgs):
            if getattr(msg, "type", "") == "ai" and getattr(msg, "content", ""):
                return msg.content
        return ""

    # ------------------------------------------------------------------
    # 通用 ! 命令处理
    # ------------------------------------------------------------------

    async def handle_command(self, cmd: str, arg: str) -> str | None:
        """
        处理通用 session 命令。
        返回回复字符串；返回 None 表示该命令不被本基类识别，交子类处理。

        覆盖范围：!new  !switch  !sessions  !session  !clear  !tokens  !resources  !stream
        """
        controller = self._controller
        session_mgr = self._session_mgr

        if cmd == "!new":
            if not arg:
                return "用法：!new <session名称> [工作目录]"
            parts = arg.split(maxsplit=1)
            new_name = parts[0]
            new_workspace = parts[1].strip() if len(parts) > 1 else ""
            try:
                await controller.new_session(new_name, workspace=new_workspace)
                ws_hint = f" workspace={new_workspace!r}" if new_workspace else ""
                return (
                    f"✅ 新 session '{new_name}' 已创建并激活"
                    f" (thread: {controller.active_thread_id}{ws_hint})"
                )
            except ValueError as e:
                return f"❌ {e}"
            except Exception as e:
                return f"创建失败: {e}"

        if cmd == "!switch":
            if not arg:
                return "用法：!switch <session名称>"
            try:
                await controller.switch_session(arg)
                return f"✅ 已切换到 session '{arg}' (thread: {controller.active_thread_id})"
            except ValueError as e:
                return f"❌ {e}"
            except Exception as e:
                return f"切换失败: {e}"

        if cmd == "!sessions":
            all_sessions = session_mgr.list_all()
            if not all_sessions:
                return "还没有任何命名 session。用 !new <名称> 创建第一个。"
            cur_tid = controller.active_thread_id
            lines = []
            for sname, env in all_sessions.items():
                marker = " ◀" if env.thread_id == cur_tid else ""
                lines.append(f"  {sname} → {env.thread_id}{marker}")
            return "\n".join(lines)

        if cmd == "!session":
            cur_tid = controller.active_thread_id
            cur_name = session_mgr.find_name_by_thread_id(cur_tid) or "（默认）"
            return f"当前 session: {cur_name} | thread_id: {cur_tid}"

        if cmd == "!resources":
            from framework.resource_lock import format_resource_status
            return format_resource_status()

        if cmd == "!tokens":
            from framework.token_tracker import get_token_stats, reset_token_stats
            if arg == "reset":
                reset_token_stats()
                return "Token 计数已重置。"
            s = get_token_stats()
            inp = s["input_tokens"]
            out = s["output_tokens"]
            cr = s["cache_read_input_tokens"]
            cc = s["cache_creation_input_tokens"]
            calls = s["calls"]
            cost_usd = (inp * 3 + out * 15 + cr * 0.3 + cc * 3.75) / 1_000_000
            saved_usd = cr * (3 - 0.3) / 1_000_000
            return (
                f"调用次数      : {calls}\n"
                f"Input tokens  : {inp:,}\n"
                f"Output tokens : {out:,}\n"
                f"Cache read    : {cr:,}  (省了 ${saved_usd:.4f})\n"
                f"Cache create  : {cc:,}\n"
                f"估算费用      : ~${cost_usd:.4f} USD"
            )

        if cmd == "!clear":
            cur_tid = controller.active_thread_id
            cur_name = session_mgr.find_name_by_thread_id(cur_tid) or "default"
            old_env = session_mgr.get_envelope(cur_name)
            workspace = old_env.workspace if old_env else ""
            session_mgr.delete(cur_name)
            new_env = session_mgr.create_session(cur_name, workspace=workspace)
            controller._active_thread_id = new_env.thread_id
            return f"Session '{cur_name}' 已重置。(new thread: {new_env.thread_id[:8]})"

        if cmd == "!stream":
            self._streaming = not self._streaming
            state = "ON" if self._streaming else "OFF"
            return f"Streaming: {state}"

        return None  # 未识别 → 子类处理

    # ------------------------------------------------------------------
    # 静态工具（Discord / GChat 共用）
    # ------------------------------------------------------------------

    @staticmethod
    def split_fence_aware(text: str, max_chars: int = 1900) -> list[str]:
        """Markdown fence-aware 文本分块，用于 Discord / GChat 消息长度限制。"""
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        remaining = text
        in_fence = False
        fence_lang = ""

        while len(remaining) > max_chars:
            candidate = remaining[:max_chars]
            cur_in_fence = in_fence
            cur_lang = fence_lang

            for line in candidate.split("\n"):
                stripped = line.strip()
                if stripped.startswith("```"):
                    if cur_in_fence:
                        cur_in_fence = False
                        cur_lang = ""
                    else:
                        cur_in_fence = True
                        cur_lang = stripped[3:].strip()

            if cur_in_fence:
                chunk = candidate + "\n```"
                remaining = f"```{cur_lang}\n" + remaining[max_chars:]
            else:
                chunk = candidate
                remaining = remaining[max_chars:]

            in_fence = cur_in_fence
            fence_lang = cur_lang
            chunks.append(chunk)

        if remaining:
            chunks.append(remaining)

        return chunks

    @staticmethod
    def extract_attachments(text: str) -> tuple[str, list[str]]:
        """
        从 agent 输出中提取所有 [SEND_FILE: /path/to/file] 标记。
        返回 (清理后文字, [文件路径列表])。
        """
        paths = [m.group(1).strip() for m in _SEND_FILE_RE.finditer(text)]
        clean_text = _SEND_FILE_RE.sub("", text).strip()
        return clean_text, paths
