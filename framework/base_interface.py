"""
BaseInterface — 共享基类，CLI / Discord / GChat 均继承。

通用命令（所有 connector 共享，在此实现）：
  !help / !topology / !debug / !stream / !tokens / !resources
  !new / !switch / !sessions / !session / !clear
  !memory / !compact / !reset / !setproject / !project
  !snapshots / !rollback

子类专属（不在此实现）：
  CLI     — stdin/stdout 流式输出，_prompt_rollback_reason() 交互式覆写
  Discord — discord.py 事件、per-channel session、!stop / !channels / !whoami
  GChat   — gws 子进程流、space 管理

Connector 注册：
  子类设置 _connector = Connector.CLI / Connector.DISCORD（来自 command_registry）。
  !help 据此过滤并生成对应 connector 的命令列表。
"""

import os
import re as _re
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage

_SEND_FILE_RE = _re.compile(r"\[SEND_FILE:\s*([^\]]+)\]")


class BaseInterface:
    # 子类设置，用于 !help 过滤；None = 显示全部
    _connector = None  # type: Connector | None

    def __init__(self, loader) -> None:
        self._loader = loader
        self._controller = None
        self._session_mgr = None
        self._streaming: bool = True
        self._last_stream_chunk_count: int = 0

    async def setup(self) -> None:
        """初始化 controller 和 session_mgr（所有子类应在 run() 开头调用）。"""
        self._controller = await self._loader.get_controller()
        self._session_mgr = self._controller.session_mgr

    # ------------------------------------------------------------------
    # Session 上下文解析（Discord 子类按频道覆写）
    # ------------------------------------------------------------------

    def _resolve_thread_id(self) -> str:
        """当前活跃 thread_id。Discord 子类按 channel_id 覆写。"""
        return self._controller.active_thread_id

    def _resolve_session_name(self) -> str | None:
        """当前 thread_id 对应的 session 名称。"""
        return self._session_mgr.find_name_by_thread_id(self._resolve_thread_id())

    def _resolve_workspace(self) -> str:
        """当前 session 的工作目录。"""
        name = self._resolve_session_name()
        if not name:
            return ""
        env = self._session_mgr.get_envelope(name)
        return env.workspace if env else ""

    # ------------------------------------------------------------------
    # Connector 专属 hook（子类可覆写）
    # ------------------------------------------------------------------

    async def _prompt_rollback_reason(self, record: dict) -> str:
        """
        !rollback 执行前询问回退原因。
        CLI 子类覆写为交互式 stdin 输入；其他接口直接返回空字符串。
        record 包含 commit_hash / created_at / project_root 等字段。
        """
        return ""

    # ------------------------------------------------------------------
    # Agent 调用
    # ------------------------------------------------------------------

    async def invoke_agent(
        self,
        user_input: str,
        extra_state: dict | None = None,
    ) -> str:
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
        pass

    def _on_stream_reset(self) -> None:
        pass

    @staticmethod
    def _extract_response(result_state: dict) -> str:
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
        处理通用命令。返回回复字符串；None 表示未识别，交子类处理。

        所有在 command_registry.ALL 中注册的命令均在此实现。
        connector 专属逻辑（如 Discord per-channel、CLI 交互式输入）
        通过覆写 _resolve_*() 和 _prompt_rollback_reason() 注入。
        """
        controller  = self._controller
        session_mgr = self._session_mgr

        # ── 帮助 ─────────────────────────────────────────────────────────
        if cmd == "!help":
            from framework.command_registry import REGISTRY
            connector = self._connector
            if connector is not None:
                cmds = [c for c in REGISTRY.values() if connector in c.connectors]
            else:
                cmds = list(REGISTRY.values())
            lines = ["可用命令："]
            for c in cmds:
                left = f"{c.name} {c.usage}".strip()
                lines.append(f"  {left:<30} {c.description}")
            return "\n".join(lines)

        # ── Agent 图拓扑（从 agent.json 递归展开 AGENT_REF 子图）─────────
        if cmd == "!topology":
            return self._loader.build_topology_mermaid()

        # ── Debug 状态 ────────────────────────────────────────────────────
        if cmd == "!debug":
            from framework.debug import is_debug
            return f"Debug mode: {'ON' if is_debug() else 'OFF'}"

        # ── 流式输出切换 ──────────────────────────────────────────────────
        if cmd == "!stream":
            self._streaming = not self._streaming
            return f"Streaming: {'ON' if self._streaming else 'OFF'}"

        # ── Token 统计 ────────────────────────────────────────────────────
        if cmd == "!tokens":
            from framework.token_tracker import get_token_stats, reset_token_stats
            if arg == "reset":
                reset_token_stats()
                return "Token 计数已重置。"
            s = get_token_stats()
            inp  = s["input_tokens"]
            out  = s["output_tokens"]
            cr   = s["cache_read_input_tokens"]
            cc   = s["cache_creation_input_tokens"]
            calls = s["calls"]
            cost_usd  = (inp * 3 + out * 15 + cr * 0.3 + cc * 3.75) / 1_000_000
            saved_usd = cr * (3 - 0.3) / 1_000_000
            return (
                f"调用次数      : {calls}\n"
                f"Input tokens  : {inp:,}\n"
                f"Output tokens : {out:,}\n"
                f"Cache read    : {cr:,}  (省了 ${saved_usd:.4f})\n"
                f"Cache create  : {cc:,}\n"
                f"估算费用      : ~${cost_usd:.4f} USD"
            )

        # ── 资源锁状态 ────────────────────────────────────────────────────
        if cmd == "!resources":
            from framework.resource_lock import format_resource_status
            return format_resource_status()

        # ── Session 管理 ──────────────────────────────────────────────────
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
            cur_tid = self._resolve_thread_id()
            lines = []
            for sname, env in all_sessions.items():
                marker = " ◀" if env.thread_id == cur_tid else ""
                lines.append(f"  {sname} → {env.thread_id}{marker}")
            return "\n".join(lines)

        if cmd == "!session":
            cur_tid  = self._resolve_thread_id()
            cur_name = self._resolve_session_name() or "（默认）"
            return f"当前 session: {cur_name} | thread_id: {cur_tid}"

        if cmd == "!clear":
            cur_name = self._resolve_session_name() or "default"
            old_env  = session_mgr.get_envelope(cur_name)
            workspace = old_env.workspace if old_env else ""
            session_mgr.delete(cur_name)
            new_env = session_mgr.create_session(cur_name, workspace=workspace)
            controller._active_thread_id = new_env.thread_id
            return f"Session '{cur_name}' 已重置。(new thread: {new_env.thread_id[:8]})"

        # ── Checkpoint 管理 ───────────────────────────────────────────────
        if cmd == "!memory":
            thread_id = self._resolve_thread_id()
            stats     = session_mgr.session_stats(thread_id)
            name      = self._resolve_session_name() or "default"
            return (
                f"Session 状态（{name}）\n"
                f"  thread     : {stats['thread_id']}\n"
                f"  checkpoints: {stats['message_count']} 条\n"
                f"  DB 大小    : {stats['db_size_kb']} KB"
            )

        if cmd == "!compact":
            try:
                keep = int(arg) if arg else 20
            except ValueError:
                keep = 20
            thread_id = self._resolve_thread_id()
            deleted   = session_mgr.compact(thread_id, keep_last=keep)
            return f"Compact 完成：删除了 {deleted} 条旧记录，保留最近 {keep} 条。"

        if cmd == "!reset":
            if arg != "confirm":
                return "此操作将清空当前 session 全部记忆（无法恢复）。确认请输入：!reset confirm"
            thread_id = self._resolve_thread_id()
            deleted   = session_mgr.reset(thread_id)
            return f"Session 已重置，清空了 {deleted} 条记录。"

        # ── 工作目录 ──────────────────────────────────────────────────────
        if cmd == "!setproject":
            if not arg:
                workspace = self._resolve_workspace() or "（未设置）"
                return f"当前项目目录：{workspace}\n用法：!setproject <路径>"
            path = os.path.expanduser(arg.strip())
            if not os.path.isdir(path):
                return f"路径不存在：{path}"
            name = self._resolve_session_name()
            if not name:
                return "❌ 当前没有活跃的命名 session"
            env = session_mgr.get_envelope(name)
            if not env:
                return "❌ Session envelope 不存在"
            env.workspace = path
            env.updated_at = datetime.now(timezone.utc).isoformat()
            session_mgr._save()
            return f"项目目录已设置为：{path}"

        if cmd == "!project":
            workspace = self._resolve_workspace() or "（未设置，全局模式）"
            return f"当前项目目录：{workspace}"

        # ── Git 快照 ──────────────────────────────────────────────────────
        if cmd == "!snapshots":
            thread_id = self._resolve_thread_id()
            history   = controller.rollback_log.get_history(thread_id, limit=10)
            if not history:
                return (
                    "当前 session 还没有任何 git 快照记录。\n"
                    "（需要 project_root 指向一个 git repo，每轮对话会自动快照）"
                )
            lines = [f"最近 {len(history)} 条快照（最新在前）："]
            for i, rec in enumerate(history, 1):
                ts   = rec["created_at"][:19].replace("T", " ")
                root = rec["project_root"] or "(无 project_root)"
                lines.append(f"  [{i}] {rec['commit_hash'][:8]}  {ts}  {root}")
            lines.append("用法：!rollback <序号>  （1=最近一次）")
            return "\n".join(lines)

        if cmd == "!rollback":
            if not arg:
                return "用法：!rollback <序号> [原因]  （序号见 !snapshots）"
            parts = arg.split(maxsplit=1)
            try:
                n = int(parts[0])
                if n < 1:
                    raise ValueError
            except ValueError:
                return "❌ 序号必须是正整数，例如：!rollback 3"

            thread_id = self._resolve_thread_id()
            record    = controller.rollback_log.get_nth_ago(thread_id, n)
            if not record:
                return f"❌ 没有找到第 {n} 条快照（用 !snapshots 查看当前记录数）"

            # 原因：优先使用 arg 中提供的，否则交子类 hook 询问
            reason = parts[1] if len(parts) > 1 else await self._prompt_rollback_reason(record)

            result = await controller.rollback_to_turn(n, reason=reason)
            if result["ok"]:
                suffix = "\n   已写入 .DO_NOT_REPEAT.md" if reason else ""
                return f"✅ {result['msg']}{suffix}"
            else:
                return f"❌ {result['msg']}"

        return None  # 未识别 → 子类处理

    # ------------------------------------------------------------------
    # 静态工具
    # ------------------------------------------------------------------

    @staticmethod
    def split_fence_aware(text: str, max_chars: int = 1900) -> list[str]:
        """Markdown fence-aware 文本分块，用于 Discord / GChat 消息长度限制。"""
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        remaining = text
        in_fence  = False
        fence_lang = ""

        while len(remaining) > max_chars:
            candidate   = remaining[:max_chars]
            cur_in_fence = in_fence
            cur_lang     = fence_lang

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
                chunk     = candidate + "\n```"
                remaining = f"```{cur_lang}\n" + remaining[max_chars:]
            else:
                chunk     = candidate
                remaining = remaining[max_chars:]

            in_fence   = cur_in_fence
            fence_lang = cur_lang
            chunks.append(chunk)

        if remaining:
            chunks.append(remaining)

        return chunks

    @staticmethod
    def extract_attachments(text: str) -> tuple[str, list[str]]:
        """从 agent 输出中提取所有 [SEND_FILE: /path/to/file] 标记。"""
        paths      = [m.group(1).strip() for m in _SEND_FILE_RE.finditer(text)]
        clean_text = _SEND_FILE_RE.sub("", text).strip()
        return clean_text, paths
