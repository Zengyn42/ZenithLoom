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

import asyncio
import json
import logging
import os
import queue
import re as _re
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, AIMessage

_SEND_FILE_RE = _re.compile(r"\[SEND_FILE:\s*([^\]]+)\]")

_logger = logging.getLogger(__name__)

# 严重告警阈值：连续失败 N 次升级为 critical
_CRITICAL_THRESHOLD = 3

# PENDING 消息标记前缀
_PENDING_PREFIX = "[PENDING]"


class BaseInterface:
    # 子类设置，用于 !help 过滤；None = 显示全部
    _connector = None  # type: Connector | None

    def __init__(self, loader) -> None:
        self._loader = loader
        self._controller = None
        self._session_mgr = None
        self._streaming: bool = True
        self._last_stream_chunk_count: int = 0
        # 后台任务完成通知队列（线程安全）
        self._completed_tasks_queue: queue.Queue[str] = queue.Queue()

    async def setup(self) -> None:
        """初始化 controller、session_mgr 和 config（所有子类应在 run() 开头调用）。"""
        self._controller = await self._loader.get_controller()
        self._session_mgr = self._controller.session_mgr
        self._config = self._loader.load_config()

    # ------------------------------------------------------------------
    # Heartbeat 告警回调（SSE 推送驱动，非轮询）
    # ------------------------------------------------------------------

    def _register_alert_callback(self) -> None:
        """
        将 _handle_alert 注册到 HeartbeatMCPProxy 的 logging_callback。
        MCP Server 失败时通过 SSE 主动推送 LoggingMessageNotification，
        Agent 端 ClientSession 收到后直接触发此回调。零轮询。
        """
        proxy = self._loader.heartbeat_proxy
        if proxy is None:
            return
        proxy.set_alert_callback(self._handle_alert)
        _logger.info("[base_interface] heartbeat alert callback registered (SSE push)")

    async def _handle_alert(self, alert: dict) -> None:
        """
        SSE 推送触发的告警处理入口。

        分级响应：
          - TASK_MONITOR 完成事件 → _on_task_completed()
          - consecutive_failures < 3 → _on_heartbeat_alert()（直接通知用户）
          - consecutive_failures >= 3 → _on_heartbeat_critical()（唤醒 Agent）
        """
        # TASK_MONITOR 完成事件
        alert_type = alert.get("type", "")
        if alert_type == "TASK_MONITOR":
            task_id = alert.get("task_id", "")
            status = alert.get("status", "")
            if task_id:
                if status in ("completed", "timeout", "failed"):
                    self._on_task_completed(task_id)
            return

        consecutive = alert.get("consecutive_failures", 1)
        alert_text = (
            f"[{alert.get('task_id', '?')}] {alert.get('type', '?')} FAILED "
            f"(×{consecutive}) at {alert.get('time', '?')}: {alert.get('error', '?')}"
        )

        if consecutive >= _CRITICAL_THRESHOLD:
            await self._on_heartbeat_critical(alert_text)
        else:
            await self._on_heartbeat_alert(alert_text)

    async def _on_heartbeat_alert(self, alert_text: str) -> None:
        """
        普通告警：直接通知用户（不经过 LLM）。
        子类覆写此方法实现具体输出。默认只 log。
        """
        _logger.warning(f"[heartbeat_alert] {alert_text}")

    async def _on_heartbeat_critical(self, alert_text: str) -> None:
        """
        严重告警：唤醒 Agent，让 LLM 分析失败并给用户建议。
        注入告警信息作为用户消息触发 Agent 响应。
        子类覆写 _deliver_agent_alert() 控制输出渠道。
        """
        _logger.warning(f"[heartbeat_critical] escalating to Agent: {alert_text}")
        prompt = (
            f"[SYSTEM ALERT — Heartbeat 失败告警]\n\n{alert_text}\n\n"
            "请分析上述 heartbeat 探针失败的情况，告知用户可能的原因和建议的修复措施。"
        )
        try:
            response = await self.invoke_agent(prompt)
            await self._deliver_agent_alert(alert_text, response)
        except Exception as e:
            _logger.error(f"[heartbeat_critical] agent invocation failed: {e}")
            # 降级为普通通知
            await self._on_heartbeat_alert(alert_text)

    async def _deliver_agent_alert(self, alert_text: str, agent_response: str) -> None:
        """
        投递 Agent 生成的告警响应。子类覆写。
        默认只 log。
        """
        _logger.warning(f"[heartbeat_agent_alert] {agent_response}")

    # ------------------------------------------------------------------
    # 后台任务完成处理（TASK_MONITOR 回调驱动）
    # ------------------------------------------------------------------

    def _on_task_completed(self, task_id: str) -> None:
        """后台任务完成回调：入队 + 响铃通知用户。

        由 Heartbeat SSE 事件或 _handle_alert 触发（线程安全）。
        """
        self._completed_tasks_queue.put(task_id)
        # 响铃提醒用户（\\a = BEL 字符）
        print(f"\a[TASK COMPLETED] {task_id} — 下次输入时自动注入结果。", flush=True)
        _logger.info(f"[base_interface] task completed: {task_id}")

    def _consume_pending_tasks(self, state: dict) -> dict:
        """invoke 前调用：从 Task Vault 取结果，覆写 state 中的 PENDING 消息。

        扫描 state["messages"] 中的 [PENDING] 标记消息，
        如果对应 task 已完成，用实际结果替换 PENDING 内容。

        Returns:
            修改后的 state（原地修改 messages 列表）。
        """
        # 先排空完成队列
        completed_ids: set[str] = set()
        while not self._completed_tasks_queue.empty():
            try:
                completed_ids.add(self._completed_tasks_queue.get_nowait())
            except queue.Empty:
                break

        if not completed_ids:
            return state

        from mcp_servers.heartbeat.task_vault import TaskVault

        mgr = TaskVault.get_instance()
        messages = state.get("messages", [])

        for i, msg in enumerate(messages):
            content = getattr(msg, "content", "")
            if not isinstance(content, str) or not content.startswith(_PENDING_PREFIX):
                continue

            # 提取 task_id
            task_id = None
            extra = getattr(msg, "additional_kwargs", {})
            if extra and "task_id" in extra:
                task_id = extra["task_id"]
            else:
                # fallback: 从 content 中解析 task_id
                for line in content.split("\n"):
                    if line.startswith("task_id:"):
                        task_id = line.split(":", 1)[1].strip()
                        break

            if task_id is None or task_id not in completed_ids:
                continue

            # 获取结果
            result = mgr.get_result(task_id)
            if result is None:
                result = f"[TASK {task_id}] 结果文件已丢失或为空。"

            # 覆写 PENDING 消息
            _logger.info(f"[base_interface] replacing PENDING message for {task_id}")
            messages[i] = AIMessage(content=result)

        return state

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
        """当前工作目录。session workspace 优先，fallback 到 entity workspace。"""
        name = self._resolve_session_name()
        if name:
            env = self._session_mgr.get_envelope(name)
            if env and env.workspace:
                return env.workspace
        # fallback: entity 级默认 workspace
        return self._config.workspace

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
        from framework.nodes.llm.llm_node import set_stream_callback

        thread_id = self._resolve_thread_id()
        workspace = self._resolve_workspace()
        config = {"configurable": {"thread_id": thread_id}}

        init_state: dict = {"messages": [HumanMessage(content=user_input)]}
        if self._connector is not None:
            init_state["connector"] = self._connector.value
        if workspace:
            init_state["workspace"] = workspace
        if extra_state:
            init_state.update(extra_state)

        # 后台任务结果注入：覆写 PENDING 消息
        init_state = self._consume_pending_tasks(init_state)

        self._last_stream_chunk_count = 0
        self._on_stream_reset()
        if self._streaming:
            set_stream_callback(self._on_stream_chunk)
        try:
            result_state = await self._controller.graph.ainvoke(init_state, config=config)
        finally:
            if self._streaming:
                set_stream_callback(None)

        # P3 fix: sync node_sessions to sessions.json (controller.run() does this; ainvoke() does not)
        self._controller.sync_node_sessions(result_state, thread_id)

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

        # ── Agent 图拓扑（从 entity.json 递归展开 SUBGRAPH_REF 子图）─────────
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
            name      = self._resolve_session_name() or "default"
            stats     = session_mgr.session_stats(thread_id)
            # 通过 checkpointer 连接获取准确的 checkpoint 数量
            stats["message_count"] = await controller.checkpoint_stats(thread_id)
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
            deleted   = await controller.compact_checkpoint(thread_id, keep_last=keep)
            return f"Compact 完成：删除了 {deleted} 条旧记录，保留最近 {keep} 条。"

        if cmd == "!reset":
            if arg != "confirm":
                return "此操作将清空当前 session 全部记忆（无法恢复）。确认请输入：!reset confirm"
            thread_id = self._resolve_thread_id()
            deleted = await controller.reset_checkpoint(thread_id)
            return f"Session 已重置，清空了 {deleted} 条记录。"

        # ── 工作目录 ──────────────────────────────────────────────────────
        if cmd == "!setproject":
            if not arg:
                workspace = self._resolve_workspace() or "（未设置）"
                return f"当前项目目录：{workspace}\n用法：!setproject <路径>  （传 clear 可清空，回退到 entity 默认值）"
            name = self._resolve_session_name()
            if not name:
                return "❌ 当前没有活跃的命名 session"
            env = session_mgr.get_envelope(name)
            if not env:
                return "❌ Session envelope 不存在"
            stripped = arg.strip()
            if stripped.lower() == "clear":
                env.workspace = ""
                env.updated_at = datetime.now(timezone.utc).isoformat()
                session_mgr._save()
                fallback = self._config.workspace if self._config else ""
                if fallback:
                    return f"Session 工作目录已清空，回退到 entity 默认值：{fallback}"
                return "Session 工作目录已清空（entity 也未设置默认值）"
            path = os.path.expanduser(stripped)
            if not os.path.isdir(path):
                return f"路径不存在：{path}"
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

        # ── 工具发现 ─────────────────────────────────────────────────────
        if cmd == "!discover":
            if not arg:
                return "用法：!discover <需求描述>  例如：!discover 找一个能写 Google Slides 的 AI 工具"
            # 转化为普通消息，由 Claude 自动路由到 tool_discovery 子图
            return await self.invoke_agent(
                f"请帮我搜索和评估开源工具：{arg}",
                extra_state={"routing_target": "tool_discovery", "routing_context": arg},
            )

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
