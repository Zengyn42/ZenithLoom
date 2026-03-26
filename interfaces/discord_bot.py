"""
无垠智穹 — Discord 远程接口（Agent 无关）

由 main.py 注入 AgentLoader 后调用 run_discord(loader)。
所有 agent 专属名称从 loader.name 动态获取。

流式输出 429 Rate Limit 防护：
  Discord 对消息编辑（PATCH /channels/{id}/messages/{id}）有严格的频率限制。
  当 LLM 流式输出大量 token 时，bot 频繁 edit 同一条消息会触发 429 Too Many Requests。
  discord.py 库虽有内置重试，但重试期间会阻塞后续编辑，导致日志刷屏且用户体验卡顿。

  防护机制（_editor 内的 _safe_edit）：
    1. 基线节流：TEXT_THROTTLE=1.0s、THINKING_THROTTLE=1.5s，控制编辑频率上限
    2. 动态退避：捕获 429 后升级退避级别（+2s/+5s/+10s），退避期内跳过编辑
       - 跳过意味着中间状态被截断，用户看到流式输出暂停几秒后直接跳到最新内容
       - 优于阻塞重试：不堆积请求，不触发连锁 429
    3. 自动恢复：15 秒无 429 后逐级降回正常频率
    4. retry_after 优先：使用 Discord 返回的精确等待时间，兜底用退避步长表
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
import urllib.request
from datetime import datetime, timezone

import discord
from discord.ext import commands
from dotenv import load_dotenv
from framework.base_interface import BaseInterface
from framework.command_registry import Connector
from framework.debug import is_debug
from framework.token_tracker import get_token_stats, reset_token_stats

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG if is_debug() else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("discord_bot")

DISCORD_MAX_CHARS = 1900

# 由 run_discord() 注入
_loader = None
_controller = None
_session_mgr = None

# 流式输出开关（!stream 切换）
_discord_streaming: bool = True

# ==========================================
# Mermaid → PNG（via mermaid.ink，无额外依赖）
# ==========================================

async def _fetch_mermaid_png(mermaid_text: str) -> bytes | None:
    """
    调用 mermaid.ink 将 Mermaid 文本渲染成 PNG bytes。
    URL 格式与 LangGraph 官方实现保持一致：
      base64.b64encode + ?type=png&bgColor=white
    mermaid.ink 会拒绝 Python 默认 User-Agent，需显式设置。
    """
    try:
        import urllib.parse
        encoded  = base64.b64encode(mermaid_text.encode("utf-8")).decode("ascii")
        bg_color = urllib.parse.quote("white", safe="")
        url      = f"https://mermaid.ink/img/{encoded}?type=png&bgColor={bg_color}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BootstrapBuilder/1.0)"},
        )

        def _blocking_fetch() -> bytes:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _blocking_fetch)
    except Exception as e:
        logger.warning(f"[discord] mermaid.ink PNG 获取失败: {e}")
        return None


# Per-channel：消息队列（入队即返回，消费者顺序处理，不丢弃消息）
_channel_queues: dict[int, asyncio.Queue] = {}

# Per-channel：消费者协程 task（每个频道一个，队列空闲 60s 后自动退出）
_channel_consumers: dict[int, asyncio.Task] = {}

# Per-channel：活跃 session 名称（channel_id → session name in sessions.json）
_channel_active_session: dict[int, str] = {}

# Per-channel：正在运行的 agent task（用于 !stop 取消当前请求）
_channel_tasks: dict[int, asyncio.Task] = {}

# 最后活跃的频道 ID（用于 heartbeat 告警推送）
_last_active_channel_id: int | None = None

# PENDING 后台任务 → 频道映射（task_id → channel_id）
_pending_task_channels: dict[str, int] = {}

# 后台 poller task
_pending_poller_task: asyncio.Task | None = None

_PENDING_POLL_INTERVAL = 30  # seconds


# ==========================================
# Per-channel session 辅助函数
# ==========================================
def _channel_prefix(channel_id: int) -> str:
    """Discord 频道的 session name 前缀。"""
    return f"discord-{channel_id}"


def _channel_default_session(channel_id: int) -> str:
    """频道的默认 session 名称。"""
    return f"discord-{channel_id}"


def _ensure_channel_session(channel_id: int) -> str:
    """确保频道有 session，不存在则自动创建。返回 session name。"""
    name = _channel_active_session.get(channel_id)
    if name and _session_mgr and _session_mgr.get_envelope(name):
        return name
    default = _channel_default_session(channel_id)
    if _session_mgr and not _session_mgr.get_envelope(default):
        _session_mgr.create_session(default)
    _channel_active_session[channel_id] = default
    return default


def _get_channel_config(channel_id: int) -> dict:
    """获取频道活跃 session 对应的 LangGraph config。"""
    name = _channel_active_session.get(channel_id, _channel_default_session(channel_id))
    env = _session_mgr.get_envelope(name) if _session_mgr else None
    tid = env.thread_id if env else _channel_default_session(channel_id)
    return {"configurable": {"thread_id": tid}}


def _get_channel_workspace(channel_id: int) -> str:
    """获取频道活跃 session 的工作目录。"""
    name = _channel_active_session.get(channel_id, _channel_default_session(channel_id))
    env = _session_mgr.get_envelope(name) if _session_mgr else None
    return env.workspace if env else ""


async def _cleanup_channel(channel_id: int) -> int:
    """删除频道的所有 sessions + checkpoints，清理内存数据。返回删除的 session 数。"""
    if not _session_mgr:
        return 0
    prefix = _channel_prefix(channel_id)
    deleted = _session_mgr.delete_by_prefix(prefix)
    _channel_active_session.pop(channel_id, None)
    _channel_tasks.pop(channel_id, None)
    consumer = _channel_consumers.pop(channel_id, None)
    if consumer and not consumer.done():
        consumer.cancel()
    _channel_queues.pop(channel_id, None)
    return deleted


# ==========================================
# 身份认证白名单（从 loader.config 读取）
# ==========================================
def _get_allowed_users() -> set[int]:
    if _loader is None:
        return set()
    users = _loader.load_config().discord_allowed_users
    return {int(u) for u in users if u.isdigit()}


def _is_authorized(user: discord.User | discord.Member) -> bool:
    allowed = _get_allowed_users()
    if not allowed:
        return True
    return user.id in allowed


# ==========================================
# Fence-Aware Chunking — delegated to BaseInterface
# ==========================================
def split_fence_aware(text: str, max_chars: int = DISCORD_MAX_CHARS) -> list[str]:
    return BaseInterface.split_fence_aware(text, max_chars)


# ==========================================
# 文件发送解析（图片 / 视频）— delegated to BaseInterface
# ==========================================
def _extract_attachments(text: str) -> tuple[str, list[str]]:
    return BaseInterface.extract_attachments(text)


# ==========================================
# 频道历史 — 写入文件（Agent 按需用 Read 工具读取）
# ==========================================
_HISTORY_FILENAME = ".discord_channel_history.txt"


async def _refresh_history_file(channel, limit: int, exclude_msg_id: int) -> str | None:
    """
    拉取最近 limit 条频道消息，写入 workspace/.discord_channel_history.txt。
    返回文件绝对路径，写入失败或无 workspace 时返回 None。
    Agent 仅在用户主动要求时通过 Read 工具读取该文件，不自动注入。
    """
    workspace = (_loader.json.get("workspace", "") if _loader else "") or _get_channel_workspace(channel.id)
    if not workspace:
        return None

    lines = []
    async for msg in channel.history(limit=limit + 1):
        if msg.id == exclude_msg_id:
            continue
        if msg.content.startswith("!"):
            continue
        ts = msg.created_at.strftime("%H:%M")
        lines.append(f"[{ts}] {msg.author.display_name}: {msg.content.strip()}")
        if len(lines) >= limit:
            break
    lines.reverse()  # 由旧到新

    path = os.path.join(workspace, _HISTORY_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Discord 频道历史（最近 {limit} 条，截止本消息前）\n\n")
            f.write("\n".join(lines))
        return path
    except OSError as e:
        logger.warning(f"[discord] 历史文件写入失败: {e}")
        return None


# ==========================================
# Per-channel 消息消费者
# ==========================================
async def _channel_consumer(channel_id: int) -> None:
    """
    每个频道一个消费者协程：从队列中顺序取出消息并处理。
    队列空闲 60s 后自动退出，下次有消息时重新创建。
    !stop 取消当前 agent task → CancelledError 传到此处 → 清空剩余队列并退出。
    """
    queue = _channel_queues[channel_id]
    agent_name = _loader.name if _loader else "Agent"
    while True:
        try:
            user_input, message = await asyncio.wait_for(queue.get(), timeout=60.0)
        except asyncio.TimeoutError:
            _channel_consumers.pop(channel_id, None)
            return

        # ── 积压合并：如果队列里还有消息，全部取出合并为一条 ──────────
        merged_inputs = [user_input]
        extra_drained = 0
        while not queue.empty():
            try:
                extra_input, extra_msg = queue.get_nowait()
                merged_inputs.append(extra_input)
                message = extra_msg  # 用最新的 message 对象做回复锚点
                extra_drained += 1
            except asyncio.QueueEmpty:
                break
        if extra_drained:
            user_input = "\n\n".join(merged_inputs)
            logger.info(f"[discord] 合并了 {extra_drained + 1} 条积压消息 (channel={channel_id})")

        iface = _DiscordInterface(_loader, channel_id=channel_id)
        task = asyncio.create_task(iface.invoke(user_input, message))
        _channel_tasks[channel_id] = task
        try:
            await task
        except asyncio.CancelledError:
            drained = 0
            while not queue.empty():
                queue.get_nowait()
                drained += 1
            if drained:
                await message.channel.send(f"已停止，清除了 {drained} 条待处理消息。")
            _channel_consumers.pop(channel_id, None)
            return
        except Exception as e:
            logger.error(f"[agent] 出错: {e}", exc_info=is_debug())
            await message.channel.send(f"{agent_name} 出错了: {e}")
        finally:
            _channel_tasks.pop(channel_id, None)

        # 检测刚发送的回复中是否有 PENDING 标记，注册后台 poller
        _register_pending_tasks_for_channel(channel_id)
        queue.task_done()
        # 补偿额外 drain 的 task_done 计数
        for _ in range(extra_drained):
            queue.task_done()


# ==========================================
# DiscordInterface — BaseInterface 的 Discord 子类
# ==========================================

class _DiscordInterface(BaseInterface):
    """
    Discord connector 的 BaseInterface 实现。

    - _connector = Connector.DISCORD：!help 自动过滤 Discord 命令
    - channel_id：per-channel session 解析，覆写 _resolve_thread_id /
      _resolve_session_name，确保所有通用命令操作正确的频道 session
    - invoke(user_input, message)：统一入口，处理历史刷新 + 流式/非流式 Discord UI
    """

    _connector = Connector.DISCORD

    def __init__(self, loader, channel_id: int | None = None) -> None:
        super().__init__(loader)
        self._controller  = _controller
        self._session_mgr = _session_mgr
        self._config      = loader.load_config()
        self._channel_id  = channel_id
        self._streaming   = _discord_streaming
        self._event_queue: asyncio.Queue = asyncio.Queue()

    # ── per-channel session 解析 ────────────────────────────────────

    def _resolve_thread_id(self) -> str:
        if self._channel_id is not None and _session_mgr is not None:
            name = _ensure_channel_session(self._channel_id)
            env  = _session_mgr.get_envelope(name)
            if env:
                return env.thread_id
        return super()._resolve_thread_id()

    def _resolve_session_name(self) -> str | None:
        if self._channel_id is not None:
            return _channel_active_session.get(
                self._channel_id,
                _channel_default_session(self._channel_id),
            )
        return super()._resolve_session_name()

    # ── 流式回调（invoke_agent 经由 BaseInterface 调用）─────────────

    def _on_stream_chunk(self, text: str, is_thinking: bool = False) -> None:
        if not text:
            return
        self._event_queue.put_nowait(("thinking" if is_thinking else "text", text))

    def _on_stream_reset(self) -> None:
        self._event_queue = asyncio.Queue()

    # ── 统一调用入口 ──────────────────────────────────────────────────

    async def invoke(self, user_input: str, message) -> None:
        """统一入口：处理历史刷新 + 流式/非流式 Discord UI，再调用 invoke_agent()。"""
        history_limit = (_loader.json.get("channel_history_limit", 0) if _loader else 0)
        if history_limit and self._channel_id is not None:
            await _refresh_history_file(
                message.channel, history_limit, exclude_msg_id=message.id
            )
        if self._streaming:
            await self._invoke_streaming(user_input, message)
        else:
            await self._invoke_non_streaming(user_input, message)

    async def _invoke_non_streaming(self, user_input: str, message) -> None:
        agent_name = _loader.name if _loader else "Agent"
        async with message.channel.typing():
            result = await self.invoke_agent(user_input)
        if result:
            clean_text, file_paths = _extract_attachments(result)
            for chunk in (split_fence_aware(clean_text) if clean_text else []):
                await message.channel.send(chunk)
            for path in file_paths:
                if not os.path.isfile(path):
                    await message.channel.send(f"⚠️ 文件不存在：`{path}`")
                    continue
                try:
                    await message.channel.send(file=discord.File(path))
                except discord.HTTPException as e:
                    await message.channel.send(f"⚠️ 文件发送失败：{e}")
        else:
            await message.channel.send(f"（{agent_name} 没有输出，请重试）")

    async def _invoke_streaming(self, user_input: str, message) -> None:
        """
        流式模式：
        - 立即发送 typing 指示器，每 8s 续命（Discord 有效期 ~10s）
        - 由调用方（_channel_consumer）保证串行，此函数不加锁
        - 思考块 → 独立斜体消息，实时更新，保留在频道（不追加最终编辑）
        - 文字 token → 独立草稿，从头追加（不截尾），完成后删除
        - 最终回复以全新消息发出（无 "(edited)"）
        """
        agent_name = _loader.name if _loader else "Agent"

        # ── Typing 指示器 ─────────────────────────────────────────────
        typing_ctx = message.channel.typing()
        await typing_ctx.__aenter__()

        # ── 编辑器任务状态 ────────────────────────────────────────────
        thinking_msg = [None]
        thinking_buf: list[str] = []
        text_draft   = [None]
        text_buf: list[str]     = []

        async def _editor():
            loop = asyncio.get_event_loop()
            thinking_last     = 0.0
            text_last         = 0.0
            THINKING_THROTTLE = 1.5
            TEXT_THROTTLE     = 1.0
            # 429 动态退避：遇到 rate limit 后逐步加大编辑间隔，恢复后递减
            _backoff_until    = 0.0      # 退避截止时间（loop.time）
            _backoff_level    = 0        # 当前退避级别（0=正常）
            _BACKOFF_STEPS    = [0, 2.0, 5.0, 10.0]  # 级别 → 额外等待秒数
            _BACKOFF_DECAY    = 15.0     # 无 429 持续 N 秒后降一级

            _last_429_time    = 0.0

            async def _safe_edit(msg_obj, content: str) -> bool:
                """编辑消息，捕获 429 并触发退避。返回 True=成功/跳过，False=429。"""
                nonlocal _backoff_until, _backoff_level, _last_429_time
                now = loop.time()
                # 退避期内直接跳过编辑
                if now < _backoff_until:
                    return True
                try:
                    await msg_obj.edit(content=content)
                    # 成功：如果距上次 429 够久，降级
                    if _backoff_level > 0 and now - _last_429_time > _BACKOFF_DECAY:
                        _backoff_level = max(0, _backoff_level - 1)
                        if _backoff_level == 0:
                            logger.debug("[discord-editor] 退避已恢复到正常频率")
                    return True
                except discord.HTTPException as e:
                    if e.status == 429:
                        # 提取 Discord 返回的 retry_after，兜底用退避步长
                        retry_after = getattr(e, "retry_after", None) or _BACKOFF_STEPS[min(_backoff_level + 1, len(_BACKOFF_STEPS) - 1)]
                        _backoff_level = min(_backoff_level + 1, len(_BACKOFF_STEPS) - 1)
                        _backoff_until = now + retry_after
                        _last_429_time = now
                        logger.warning(
                            f"[discord-editor] 429 rate limit, 退避 {retry_after:.1f}s "
                            f"(level={_backoff_level})"
                        )
                        return False
                    return True  # 非 429 错误静默跳过
                except Exception:
                    return True

            while True:
                sentinel = False
                while True:
                    try:
                        kind, chunk = self._event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if kind is None:
                        sentinel = True
                        break
                    if kind == "thinking":
                        thinking_buf.append(chunk)
                    else:
                        text_buf.append(chunk)

                now = loop.time()
                # 退避期额外节流
                extra_wait = _BACKOFF_STEPS[_backoff_level] if _backoff_level > 0 else 0

                # 思考消息（💭 emoji + blockquote，流式更新）
                if thinking_buf and (sentinel or now - thinking_last >= THINKING_THROTTLE + extra_wait):
                    thinking_last = now
                    preview   = "".join(thinking_buf)
                    max_body  = 1860
                    body_text = f"…{preview[-max_body:]}" if len(preview) > max_body else preview
                    quoted    = "\n".join(f"> {line}" if line else ">" for line in body_text.splitlines())
                    display   = f"*💭*\n{quoted}"
                    if thinking_msg[0] is None:
                        thinking_msg[0] = await message.channel.send(display)
                    else:
                        await _safe_edit(thinking_msg[0], display)

                # 文字草稿（从头追加，不截尾；完成后删除）
                if text_buf and (sentinel or now - text_last >= TEXT_THROTTLE + extra_wait):
                    text_last = now
                    preview = "".join(text_buf)
                    cursor  = "" if sentinel else " ▌"
                    if len(preview) + len(cursor) > 1900:
                        display = "…" + preview[-(1897 - len(cursor)):] + cursor
                    else:
                        display = preview + cursor
                    if text_draft[0] is None:
                        text_draft[0] = await message.channel.send("▌")
                    await _safe_edit(text_draft[0], display)

                if sentinel:
                    break
                await asyncio.sleep(0.1)

        editor_task = asyncio.create_task(_editor())
        try:
            result = await self.invoke_agent(user_input)
        except asyncio.CancelledError:
            self._event_queue.put_nowait((None, None))
            await editor_task
            await typing_ctx.__aexit__(None, None, None)
            if text_draft[0]:
                try:
                    await text_draft[0].delete()
                except Exception:
                    pass
            raise
        finally:
            self._event_queue.put_nowait((None, None))
            await editor_task
            await typing_ctx.__aexit__(None, None, None)

        if text_draft[0]:
            try:
                await text_draft[0].delete()
            except Exception:
                pass

        # 用 streaming 累积的完整文本作为最终输出（覆盖 result），
        # 避免 result 只含最后一轮 LLM 输出时丢失工具调用中间内容。
        streamed_full = "".join(text_buf) if text_buf else ""
        final_text = streamed_full or result or ""

        if not final_text:
            await message.channel.send(f"（{agent_name} 没有输出，请重试）")
            return

        clean_text, file_paths = _extract_attachments(final_text)
        for chunk in (split_fence_aware(clean_text) if clean_text else []):
            await message.channel.send(chunk)
        for path in file_paths:
            if not os.path.isfile(path):
                await message.channel.send(f"⚠️ 文件不存在：`{path}`")
                continue
            try:
                await message.channel.send(file=discord.File(path))
            except discord.HTTPException as e:
                await message.channel.send(f"⚠️ 文件发送失败：{e}")


# ==========================================
# PENDING task → channel 后台 poller
# ==========================================


def _register_pending_tasks_for_channel(channel_id: int) -> None:
    """扫描频道的 checkpoint 消息历史，找到 PENDING 标记并注册 poller。"""
    global _pending_poller_task
    try:
        from mcp_servers.heartbeat.task_vault import TaskVault
        vault = TaskVault.get_instance()
        # 扫描该频道对应 session 的 thread_id checkpoint
        session_name = _channel_active_session.get(channel_id)
        if not session_name or not _controller:
            return

        env = _controller.session_mgr.get_envelope(session_name)
        if not env:
            return

        # 获取 checkpoint 中的 messages
        import asyncio
        loop = asyncio.get_event_loop()
        # 使用 controller 的 checkpointer 获取最新 state
        checkpoint = _controller._checkpointer
        if not checkpoint:
            return

        # 直接查 TaskVault 的活跃任务（更简单可靠）
        for task_id, record in vault._tasks.items():
            from mcp_servers.heartbeat.task_vault import TaskStatus
            if record.status == TaskStatus.RUNNING:
                _pending_task_channels[task_id] = channel_id
                logger.info(f"[discord] registered pending task {task_id} → channel {channel_id}")

    except Exception as e:
        logger.warning(f"[discord] _register_pending_tasks_for_channel error: {e}")

    # 确保 poller 在运行
    if _pending_task_channels and (_pending_poller_task is None or _pending_poller_task.done()):
        _pending_poller_task = asyncio.get_event_loop().create_task(
            _pending_tasks_poller(), name="pending_tasks_poller"
        )


async def _pending_tasks_poller() -> None:
    """后台 poller：每 30s 检查 TaskVault，任务完成后主动推送到对应频道。"""
    while _pending_task_channels:
        await asyncio.sleep(_PENDING_POLL_INTERVAL)

        if not _pending_task_channels:
            break

        try:
            from mcp_servers.heartbeat.task_vault import TaskVault, TaskStatus
            vault = TaskVault.get_instance()

            completed: list[tuple[str, int]] = []  # (task_id, channel_id)

            for task_id, channel_id in list(_pending_task_channels.items()):
                status = vault.query_task(task_id)
                if status is None or status == TaskStatus.RUNNING:
                    continue
                completed.append((task_id, channel_id))

            for task_id, channel_id in completed:
                _pending_task_channels.pop(task_id, None)
                status = vault.query_task(task_id)
                result = vault.get_result(task_id)

                if result is None:
                    result = f"结果文件已丢失或为空。"

                if status == TaskStatus.TIMEOUT:
                    header = "⏰ **后台任务超时**"
                elif status == TaskStatus.FAILED:
                    header = "❌ **后台任务失败**"
                else:
                    header = "✅ **后台任务完成**"

                msg_text = f"{header} `{task_id}`\n```\n{result[:1800]}\n```"

                channel = bot.get_channel(channel_id)
                if channel:
                    try:
                        await channel.send(msg_text)
                        logger.info(f"[discord] sent pending task result: {task_id} → channel {channel_id}")
                    except Exception as e:
                        logger.warning(f"[discord] failed to send task result: {e}")
                else:
                    logger.warning(f"[discord] channel {channel_id} not found for task {task_id}")

        except Exception as e:
            logger.warning(f"[discord] pending_tasks_poller error: {e}")

    logger.info("[discord] pending_tasks_poller exiting (no pending tasks)")


# ==========================================
# Heartbeat Alert Callback (Discord, SSE push)
# ==========================================

_CRITICAL_THRESHOLD = 3


def _find_alert_channel():
    """返回最后活跃的频道；如果没有活跃记录，回退到第一个可发送的频道。"""
    if _last_active_channel_id is not None:
        ch = bot.get_channel(_last_active_channel_id)
        if ch is not None:
            return ch
    for guild in bot.guilds:
        for ch in guild.text_channels:
            if ch.permissions_for(guild.me).send_messages:
                return ch
    return None


def _register_discord_alert_callback():
    """将 Discord 告警处理注册到 HeartbeatMCPProxy 的 SSE 回调。"""
    if _loader is None or _loader.heartbeat_proxy is None:
        return
    _loader.heartbeat_proxy.set_alert_callback(_discord_handle_alert)
    logger.info("[Discord] heartbeat alert callback registered (SSE push)")


async def _discord_handle_alert(alert: dict):
    """
    SSE 推送触发的 Discord 告警处理。
    level=info/warning → 任务完成报告；level=error → 失败告警。
    """
    channel = _find_alert_channel()
    if channel is None:
        logger.warning(f"[discord_alert] no channel to send alert: {alert}")
        return

    level   = alert.get("level", "error")
    task_id = alert.get("task_id", "?")
    time_   = alert.get("time", "?")
    next_run = alert.get("next_run", "")

    # ── 完成报告（info / warning）────────────────────────────────────────
    if level in ("info", "warning"):
        icon = "✅" if level == "info" else "⚠️"
        content = alert.get("content", "")
        next_line = f" | 下次: `{next_run}`" if next_run else ""
        msg = f"{icon} `{task_id}` | 时间: `{time_}`{next_line}"
        if content:
            msg += f"\n> {content[:200]}"
        await channel.send(msg)
        return

    # ── 失败告警（error）─────────────────────────────────────────────────
    consecutive = alert.get("consecutive_failures", 1)
    error       = alert.get("error", "?")
    next_line   = f" → 下次: {next_run}" if next_run else ""
    alert_text  = f"[{task_id}] FAILED (×{consecutive}) at {time_}: {error}{next_line}"

    if consecutive >= _CRITICAL_THRESHOLD and _controller:
        agent_name = _loader.name if _loader else "Agent"
        prompt = (
            f"[SYSTEM ALERT — Heartbeat 失败告警]\n\n{alert_text}\n\n"
            "请分析上述 heartbeat 探针失败的情况，告知用户可能的原因和建议的修复措施。"
        )
        try:
            iface = _DiscordInterface(_loader)
            response = await iface.invoke_agent(prompt)
            await channel.send(
                f"🚨 **Heartbeat Critical — {agent_name} 分析**\n{response}"
            )
        except Exception as e:
            logger.error(f"[discord_alert] agent invocation failed: {e}")
            await channel.send(f"⚠ **Heartbeat Alert**\n```\n{alert_text}\n```")
    else:
        await channel.send(f"⚠ **Heartbeat Alert**\n```\n{alert_text}\n```")


# ==========================================
# Discord Bot
# ==========================================
intents = discord.Intents.default()
intents.message_content = True  # Content Intent（开发者后台已启用）

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


@bot.event
async def on_ready():
    global _controller, _session_mgr
    # 在 discord.py 的事件循环内初始化 controller（aiosqlite 连接绑定到正确的 loop）
    if _loader and _controller is None:
        _controller = await _loader.get_controller()
        _session_mgr = getattr(_controller, "session_mgr", _session_mgr)
        logger.info(f"[Discord] controller 已初始化（graph 已编译）")

    agent_name = _loader.name if _loader else "Agent"
    logger.info(f"[Discord] {agent_name} 已上线: {bot.user}")
    allowed = _get_allowed_users()
    if allowed:
        logger.info(f"[Auth] 白名单已启用: {allowed}")
    else:
        logger.warning("[Auth] 白名单为空，任何人均可使用")

    # ── 从 sessions.json 恢复 per-channel 活跃 session ──────────────
    if _session_mgr:
        best: dict[int, tuple[str, str]] = {}  # channel_id → (name, updated_at)
        for sname, env in _session_mgr.list_by_prefix("discord-").items():
            parts = sname.split("-", 2)  # "discord-{channel_id}" or "discord-{channel_id}-{name}"
            if len(parts) < 2:
                continue
            try:
                cid = int(parts[1])
            except ValueError:
                continue
            if cid not in best or env.updated_at > best[cid][1]:
                best[cid] = (sname, env.updated_at)
        for cid, (sname, _) in best.items():
            _channel_active_session[cid] = sname
        if best:
            logger.info(f"[Discord] 恢复了 {len(best)} 个频道的活跃 session")

    if _loader:
        mgr = await _loader.start_heartbeat()
        if mgr is not None:
            logger.info(f"[Discord] heartbeat 已启动（agent={agent_name}）")
            # 注册 SSE 推送告警回调
            _register_discord_alert_callback()


@bot.event
async def on_guild_channel_delete(channel):
    """频道被删除时自动清理所有相关 sessions + checkpoints。"""
    task = _channel_tasks.get(channel.id)
    if task and not task.done():
        task.cancel()
    deleted = await _cleanup_channel(channel.id)
    if deleted:
        logger.info(f"[Discord] 已清理已删频道 {channel.id} 的 {deleted} 个 sessions")


@bot.event
async def on_message(message: discord.Message):
    global _last_active_channel_id
    if message.author.bot:
        return
    if not _is_authorized(message.author):
        return
    _last_active_channel_id = message.channel.id
    if message.content.startswith("!"):
        ctx = await bot.get_context(message)
        if ctx.command is not None:
            await bot.invoke(ctx)
        else:
            # 未被 @bot.command 命中 → BaseInterface 通用命令（含 channel_id）
            parts = message.content.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if _controller is not None:
                iface = _DiscordInterface(_loader, channel_id=message.channel.id)
                reply = await iface.handle_command(cmd, arg)
                if reply is not None:
                    if cmd == "!topology":
                        # Mermaid 文本 + PNG 图片
                        for chunk in BaseInterface.split_fence_aware(
                            f"```mermaid\n{reply}\n```", DISCORD_MAX_CHARS
                        ):
                            await message.channel.send(chunk)
                        png = await _fetch_mermaid_png(reply)
                        if png:
                            await message.channel.send(
                                file=discord.File(io.BytesIO(png), filename="graph.png")
                            )
                        else:
                            await message.channel.send("⚠️ PNG 生成失败，请粘贴上方 Mermaid 文本到 https://mermaid.live 查看")
                    elif "\n" in reply:
                        # 多行回复用代码块包裹
                        for chunk in BaseInterface.split_fence_aware(
                            f"```\n{reply}\n```", DISCORD_MAX_CHARS
                        ):
                            await message.channel.send(chunk)
                    else:
                        await message.channel.send(reply)
        return

    user_input = message.content.strip()
    channel_id = message.channel.id

    # ── 确保频道有 session ──────────────────────────────────────────
    _ensure_channel_session(channel_id)

    # ── 附件处理：下载到临时目录，路径附加到 user_input ──────────────
    if message.attachments:
        workspace = (_loader.json.get("workspace", "") if _loader else "") or _get_channel_workspace(channel_id)
        attach_dir = os.path.join(workspace, ".discord_attachments") if workspace else tempfile.mkdtemp(prefix="discord_attach_")
        os.makedirs(attach_dir, exist_ok=True)
        attached_paths: list[str] = []
        for att in message.attachments:
            safe_name = f"{message.id}_{att.filename}"
            dest = os.path.join(attach_dir, safe_name)
            try:
                await att.save(dest)
                attached_paths.append(dest)
                logger.info(f"[discord] 附件已保存: {dest} ({att.size} bytes)")
            except Exception as e:
                logger.warning(f"[discord] 附件下载失败 {att.filename}: {e}")
        if attached_paths:
            hint = "\n\n[用户上传了以下文件，请用 Read 工具查看]\n" + "\n".join(attached_paths)
            user_input = (user_input + hint) if user_input else hint.strip()

    if not user_input:
        return

    # ── 消息入队，消费者串行处理（不丢弃消息）──────────────────────────
    if channel_id not in _channel_queues:
        _channel_queues[channel_id] = asyncio.Queue()
    await _channel_queues[channel_id].put((user_input, message))

    # 启动消费者（如果尚未运行或已退出）
    consumer = _channel_consumers.get(channel_id)
    if consumer is None or consumer.done():
        _channel_consumers[channel_id] = asyncio.create_task(_channel_consumer(channel_id))


# ==========================================
# 命令前置认证
# ==========================================
async def _check_auth(ctx) -> bool:
    return _is_authorized(ctx.author)


# ==========================================
# 命令
# ==========================================
@bot.command(name="new")
async def new_session_cmd(ctx, *, name: str = ""):
    if not await _check_auth(ctx):
        return
    if not name:
        await ctx.send("用法：`!new <名称>`")
        return
    channel_id = ctx.channel.id
    full_name = f"{_channel_prefix(channel_id)}-{name}"
    try:
        env = _session_mgr.create_session(full_name)
        _channel_active_session[channel_id] = full_name
        await ctx.send(f"已创建并切换到 session `{name}`\nthread: `{env.thread_id}`")
    except ValueError as e:
        await ctx.send(f"❌ {e}")


@bot.command(name="switch")
async def switch_session_cmd(ctx, *, name: str = ""):
    if not await _check_auth(ctx):
        return
    if not name:
        await ctx.send("用法：`!switch <名称>` 或 `!switch default`")
        return
    channel_id = ctx.channel.id
    if name == "default":
        full_name = _channel_default_session(channel_id)
    else:
        full_name = f"{_channel_prefix(channel_id)}-{name}"
    env = _session_mgr.get_envelope(full_name)
    if not env:
        await ctx.send(f"❌ Session `{name}` 不存在。用 `!sessions` 查看可用 sessions。")
        return
    _channel_active_session[channel_id] = full_name
    await ctx.send(f"已切换到 session `{name}`\nthread: `{env.thread_id}`")


@bot.command(name="session")
async def show_session(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    name = _ensure_channel_session(channel_id)
    env = _session_mgr.get_envelope(name)
    display = name.replace(_channel_prefix(channel_id), "").lstrip("-") or "default"
    await ctx.send(f"当前 session: `{display}`\nthread: `{env.thread_id}`")


@bot.command(name="sessions")
async def list_sessions_cmd(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    prefix = _channel_prefix(channel_id)
    sessions = _session_mgr.list_by_prefix(prefix)
    if not sessions:
        await ctx.send("当前频道没有 session。")
        return
    active = _channel_active_session.get(channel_id, _channel_default_session(channel_id))
    lines = []
    for name, env in sessions.items():
        display = name.replace(prefix, "").lstrip("-") or "default"
        marker = " ◀" if name == active else ""
        lines.append(f"  {display:<20} thread={env.thread_id}{marker}")
    await ctx.send(f"**频道 Sessions**\n```\n" + "\n".join(lines) + "\n```")


@bot.command(name="stop")
async def stop_task(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    task = _channel_tasks.get(channel_id)
    if task and not task.done():
        # 取消当前 agent task；consumer 会捕获 CancelledError 并清空队列
        task.cancel()
        await ctx.send("已停止。")
        return
    # 无正在运行的 task，但 consumer 可能在等待队列
    consumer = _channel_consumers.get(channel_id)
    queue = _channel_queues.get(channel_id)
    pending = queue.qsize() if queue else 0
    if consumer and not consumer.done():
        consumer.cancel()
        _channel_consumers.pop(channel_id, None)
        if queue:
            while not queue.empty():
                queue.get_nowait()
        msg = f"已停止。" + (f" 清除了 {pending} 条待处理消息。" if pending else "")
        await ctx.send(msg)
        return
    await ctx.send("没有正在运行的任务。")


@bot.command(name="channels")
async def list_channels_cmd(ctx):
    if not await _check_auth(ctx):
        return
    all_discord = _session_mgr.list_by_prefix("discord-")
    if not all_discord:
        await ctx.send("没有任何 Discord 频道 session。")
        return
    # 按 channel_id 分组
    grouped: dict[int, list[str]] = {}
    for sname in all_discord:
        parts = sname.split("-", 2)
        if len(parts) < 2:
            continue
        try:
            cid = int(parts[1])
        except ValueError:
            continue
        grouped.setdefault(cid, []).append(sname)
    # 输出
    lines = []
    guild = ctx.guild
    for cid, names in sorted(grouped.items()):
        ch = guild.get_channel(cid) if guild else None
        ch_label = f"#{ch.name}" if ch else f"⚠ orphan({cid})"
        active = _channel_active_session.get(cid, "")
        lines.append(f"{ch_label}  ({len(names)} sessions, active={active.split('-', 2)[-1] if active else '?'})")
        for sname in names:
            display = sname.replace(f"discord-{cid}", "").lstrip("-") or "default"
            marker = " ◀" if sname == active else ""
            lines.append(f"    {display}{marker}")
    await ctx.send(f"**所有频道 Sessions**\n```\n" + "\n".join(lines) + "\n```")


@bot.command(name="stream")
async def toggle_stream(ctx):
    if not await _check_auth(ctx):
        return
    global _discord_streaming
    _discord_streaming = not _discord_streaming
    state = "ON" if _discord_streaming else "OFF"
    await ctx.send(f"Streaming: {state}")


@bot.command(name="whoami")
async def whoami(ctx):
    authorized = "已授权" if _is_authorized(ctx.author) else "未授权"
    await ctx.send(f"你的 Discord ID: `{ctx.author.id}` ({authorized})")


@bot.command(name="clear")
async def clear_session(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    name = _ensure_channel_session(channel_id)
    env = _session_mgr.get_envelope(name)
    workspace = env.workspace if env else ""
    _session_mgr.delete(name)
    new_env = _session_mgr.create_session(name, workspace=workspace)
    _channel_active_session[channel_id] = name
    await ctx.send(f"Session 已重置。(new thread: `{new_env.thread_id[:8]}`)")


# ==========================================
# 入口
# ==========================================
def run_discord(loader=None):
    global _loader, _session_mgr
    _loader = loader
    _session_mgr = loader.session_mgr if loader else None
    # _controller 延迟到 on_ready() 内初始化，确保 aiosqlite 绑定到 discord.py 的事件循环

    token = loader.load_config().discord_token if loader else ""
    if not token:
        print("请在 entity.json 或 .env（DISCORD_BOT_TOKEN）中设置 Discord Bot Token")
        return
    logger.info(f"[Discord] 启动中... agent={loader.name if loader else '?'} DEBUG={'ON' if is_debug() else 'OFF'}")
    bot.run(token, log_handler=None)

    # bot.run() 返回说明 bot 已关闭 — 清理 heartbeat
    if _loader:
        import asyncio
        try:
            asyncio.run(_loader.stop_heartbeat())
        except RuntimeError:
            # 如果 event loop 已关闭，尝试创建新 loop
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_loader.stop_heartbeat())
            loop.close()
