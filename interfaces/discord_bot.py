"""
无垠智穹 — Discord 远程接口（Agent 无关）

由 main.py 注入 AgentLoader 后调用 run_discord(loader)。
所有 agent 专属名称从 loader.name 动态获取。
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone

import discord
from discord.ext import commands
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from framework.base_interface import BaseInterface
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

# 每个频道的并发锁（防止多条消息同时被处理）
_channel_locks: dict = {}

# Per-channel：活跃 session 名称（channel_id → session name in sessions.json）
_channel_active_session: dict[int, str] = {}

# Per-channel：正在运行的 agent task（用于 !stop）
_channel_tasks: dict[int, asyncio.Task] = {}


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
    _channel_locks.pop(channel_id, None)
    _channel_tasks.pop(channel_id, None)
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
# Agent 调用
# ==========================================
async def _invoke_agent_async(user_input: str, message=None) -> str:
    engine = _controller._graph
    channel_id = message.channel.id if message else 0
    config = _get_channel_config(channel_id)
    workspace = _get_channel_workspace(channel_id)

    # 后台刷新历史文件（Agent 不会自动读取，只在被要求时用 Read 工具读取）
    history_limit = (_loader.json.get("channel_history_limit", 0) if _loader else 0)
    if history_limit and message is not None:
        history_path = await _refresh_history_file(
            message.channel, history_limit, exclude_msg_id=message.id
        )
        if history_path and is_debug():
            logger.debug(f"[discord] 历史文件已刷新: {history_path}")

    if is_debug():
        logger.debug(f"[agent] input_len={len(user_input)} workspace={workspace!r}")

    init_state: dict = {"messages": [HumanMessage(content=user_input)]}
    if workspace:
        init_state["project_root"] = workspace

    full_response = []
    async for chunk, _ in engine.astream(
        init_state,
        config=config,
        stream_mode="messages",
    ):
        if hasattr(chunk, "content") and isinstance(chunk.content, str):
            full_response.append(chunk.content)

    result = "".join(full_response)
    if is_debug():
        logger.debug(f"[agent] output={result[:500]!r}")
    return result


async def _invoke_agent_streaming(user_input: str, message) -> None:
    """
    流式模式：
    - 立即发送 typing 指示器，每 8s 续命（Discord 有效期 ~10s）
    - 每个频道加锁，避免并发打断
    - 思考块 → 独立斜体消息，实时更新，保留在频道（不追加最终编辑）
    - 文字 token → 独立草稿，从头追加（不截尾），完成后删除
    - 最终回复以全新消息发出（无 "(edited)"）
    """
    from framework.claude.node import set_stream_callback
    from framework.base_interface import BaseInterface

    channel_id = message.channel.id
    if channel_id not in _channel_locks:
        _channel_locks[channel_id] = asyncio.Lock()

    lock = _channel_locks[channel_id]
    if lock.locked():
        await message.channel.send("⏳ 正在处理上一条消息，请稍候…")
        return

    async with lock:
        engine = _controller._graph
        config = _get_channel_config(channel_id)
        workspace = _get_channel_workspace(channel_id)
        agent_name = _loader.name if _loader else "Agent"

        history_limit = (_loader.json.get("channel_history_limit", 0) if _loader else 0)
        if history_limit:
            await _refresh_history_file(message.channel, history_limit, exclude_msg_id=message.id)

        init_state: dict = {"messages": [HumanMessage(content=user_input)]}
        if workspace:
            init_state["project_root"] = workspace

        # ── Typing 指示器：context manager 自动续命，兼容 DM/群组频道 ────────
        typing_ctx = message.channel.typing()
        await typing_ctx.__aenter__()

        # ── 事件队列：区分思考 / 文字 token ────────────────────────────────
        event_queue: asyncio.Queue = asyncio.Queue()

        def _discord_cb(text: str, is_thinking: bool = False) -> None:
            if not text:
                return
            event_queue.put_nowait(("thinking" if is_thinking else "text", text))

        # ── 编辑器任务 ──────────────────────────────────────────────────────
        thinking_msg = [None]
        thinking_buf: list[str] = []
        text_draft = [None]
        text_buf: list[str] = []

        async def _editor():
            loop = asyncio.get_event_loop()
            thinking_last = 0.0
            text_last = 0.0
            THINKING_THROTTLE = 1.5
            TEXT_THROTTLE = 0.5

            while True:
                sentinel = False
                while True:
                    try:
                        kind, chunk = event_queue.get_nowait()
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

                # 思考消息（💭 emoji + blockquote，流式更新；sentinel 时加 [/thinking] 结尾）
                if thinking_buf and (sentinel or now - thinking_last >= THINKING_THROTTLE):
                    thinking_last = now
                    preview = "".join(thinking_buf)
                    max_body = 1860
                    body_text = f"…{preview[-max_body:]}" if len(preview) > max_body else preview
                    quoted = "\n".join(f"> {line}" if line else ">" for line in body_text.splitlines())
                    display = f"*💭*\n{quoted}"
                    if thinking_msg[0] is None:
                        thinking_msg[0] = await message.channel.send(display)
                    else:
                        try:
                            await thinking_msg[0].edit(content=display)
                        except Exception:
                            pass

                # 文字草稿（从头追加，不截尾；完成后删除）
                if text_buf and (sentinel or now - text_last >= TEXT_THROTTLE):
                    text_last = now
                    preview = "".join(text_buf)
                    cursor = "" if sentinel else " ▌"
                    # 从头显示，只在超长时截掉开头（保持末尾可见）
                    if len(preview) + len(cursor) > 1900:
                        display = "…" + preview[-(1897 - len(cursor)):] + cursor
                    else:
                        display = preview + cursor
                    if text_draft[0] is None:
                        text_draft[0] = await message.channel.send("▌")
                    try:
                        await text_draft[0].edit(content=display)
                    except Exception:
                        pass

                if sentinel:
                    break
                await asyncio.sleep(0.1)

        editor_task = asyncio.create_task(_editor())
        set_stream_callback(_discord_cb)
        try:
            result_state = await engine.ainvoke(init_state, config=config)
        except asyncio.CancelledError:
            set_stream_callback(None)
            event_queue.put_nowait((None, None))
            await editor_task
            await typing_ctx.__aexit__(None, None, None)
            if text_draft[0]:
                try:
                    await text_draft[0].delete()
                except Exception:
                    pass
            raise
        finally:
            set_stream_callback(None)
            event_queue.put_nowait((None, None))
            await editor_task
            await typing_ctx.__aexit__(None, None, None)

        final_text = BaseInterface._extract_response(result_state)

        # 删除文字草稿（(edited) 随草稿消失）
        if text_draft[0]:
            try:
                await text_draft[0].delete()
            except Exception:
                pass

        # 思考消息保留原样（不追加最终编辑，避免 "(edited)" 出现在最后）

        if not final_text:
            await message.channel.send(f"（{agent_name} 没有输出，请重试）")
            return

        clean_text, file_paths = _extract_attachments(final_text)
        chunks = split_fence_aware(clean_text) if clean_text else []

        for chunk in chunks:
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

    hb_cfg = _loader and _loader.json.get("heartbeat")
    if hb_cfg:
        from framework.heartbeat import heartbeat_loop, run_heartbeat_once
        probes = (hb_cfg if isinstance(hb_cfg, dict) else {}).get("probes", [])
        logger.info(f"[Discord] heartbeat 已启动（agent={agent_name} probes={probes}）")
        await run_heartbeat_once(probes)
        asyncio.create_task(heartbeat_loop(probes))


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
    if message.author.bot:
        return
    if not _is_authorized(message.author):
        return
    if message.content.startswith("!"):
        await bot.process_commands(message)
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

    agent_name = _loader.name if _loader else "Agent"
    full_text = None

    if _discord_streaming:
        # 流式模式：wrapped in tracked task for !stop
        task = asyncio.create_task(_invoke_agent_streaming(user_input, message))
        _channel_tasks[channel_id] = task
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"[discord] task cancelled for channel {channel_id}")
        except Exception as e:
            logger.error(f"[agent] 流式调用出错: {e}", exc_info=is_debug())
            await message.channel.send(f"{agent_name} 出错了: {e}")
        finally:
            _channel_tasks.pop(channel_id, None)
        return
    else:
        # 非流式模式：wrapped in tracked task for !stop
        async with message.channel.typing():
            async def _run_async():
                return await _invoke_agent_async(user_input, message=message)
            task = asyncio.create_task(_run_async())
            _channel_tasks[channel_id] = task
            try:
                full_text = await task
            except asyncio.CancelledError:
                logger.info(f"[discord] task cancelled for channel {channel_id}")
                return
            except Exception as e:
                logger.error(f"[agent] 调用出错: {e}", exc_info=is_debug())
                await message.channel.send(f"{agent_name} 出错了: {e}")
                return
            finally:
                _channel_tasks.pop(channel_id, None)

    if not full_text:
        await message.channel.send(f"（{agent_name} 没有输出，请重试）")
        return

    # 提取文件附件标记，分块发送
    clean_text, file_paths = _extract_attachments(full_text)

    if clean_text:
        for chunk in split_fence_aware(clean_text):
            await message.channel.send(chunk)

    for path in file_paths:
        if not os.path.isfile(path):
            await message.channel.send(f"⚠️ 文件不存在，无法发送：`{path}`")
            logger.warning(f"[discord] SEND_FILE 文件不存在: {path}")
            continue
        try:
            await message.channel.send(file=discord.File(path))
            logger.info(f"[discord] 已发送文件: {path}")
        except discord.HTTPException as e:
            await message.channel.send(f"⚠️ 文件发送失败：{e}")
            logger.error(f"[discord] 文件发送失败 {path}: {e}")


# ==========================================
# 命令前置认证
# ==========================================
async def _check_auth(ctx) -> bool:
    return _is_authorized(ctx.author)


# ==========================================
# 命令
# ==========================================
@bot.command(name="help")
async def show_help(ctx):
    if not await _check_auth(ctx):
        return
    await ctx.send(
        "**Agent 命令手册**\n"
        "```\n"
        "── 频道 Session ─────────────────────────────────\n"
        "!new <名称>        在当前频道创建新 session\n"
        "!switch <名称>     切换当前频道的 session\n"
        "!session           显示当前频道的 session 信息\n"
        "!sessions          列出当前频道的所有 sessions\n"
        "!stop              停止当前频道正在运行的任务\n"
        "!channels          列出所有频道的 session 数据\n"
        "\n"
        "── 记忆管理 ─────────────────────────────────────\n"
        "!memory            查看当前 session 的 checkpoint 统计\n"
        "!compact [N]       压缩当前 session，保留最近 N 条\n"
        "!reset confirm     清空当前 session 全部记忆\n"
        "!clear             重置当前 session（无需确认）\n"
        "\n"
        "── 工具 ─────────────────────────────────────────\n"
        "!tokens            查看 token 消耗统计\n"
        "!tokens reset      重置 token 计数\n"
        "!setproject <路径>  设置当前频道的工作目录\n"
        "!project           查看当前频道的项目目录\n"
        "!stream            切换流式输出 ON/OFF\n"
        "!debug             查看 debug 模式状态\n"
        "!whoami            显示你的 Discord ID\n"
        "!help              显示此帮助信息\n"
        "```"
    )


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
    task = _channel_tasks.get(ctx.channel.id)
    if task is None or task.done():
        await ctx.send("没有正在运行的任务。")
        return
    task.cancel()
    await ctx.send("已停止。")


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


@bot.command(name="debug")
async def toggle_debug(ctx):
    if not await _check_auth(ctx):
        return
    status = "ON" if is_debug() else "OFF（设置 DEBUG=1 启动以开启）"
    await ctx.send(f"Debug mode: {status}")


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


@bot.command(name="memory")
async def show_memory(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    name = _ensure_channel_session(channel_id)
    env = _session_mgr.get_envelope(name)
    stats = _session_mgr.session_stats(env.thread_id)
    display = name.replace(_channel_prefix(channel_id), "").lstrip("-") or "default"
    await ctx.send(
        f"**Session 状态**（`{display}`）\n"
        f"- thread: `{stats['thread_id']}`\n"
        f"- checkpoints: `{stats['message_count']}` 条\n"
        f"- DB 大小: `{stats['db_size_kb']} KB`"
    )


@bot.command(name="compact")
async def compact_session(ctx, keep: int = 20):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    name = _ensure_channel_session(channel_id)
    env = _session_mgr.get_envelope(name)
    deleted = _session_mgr.compact(env.thread_id, keep_last=keep)
    await ctx.send(f"Compact 完成：删除了 `{deleted}` 条旧记录，保留最近 `{keep}` 条。")


@bot.command(name="reset")
async def reset_session(ctx, confirm: str = ""):
    if not await _check_auth(ctx):
        return
    agent_name = _loader.name if _loader else "Agent"
    if confirm != "confirm":
        await ctx.send(
            f"此操作将**清空当前频道 session 的全部记忆**，无法恢复。\n"
            "确认请输入：`!reset confirm`"
        )
        return
    channel_id = ctx.channel.id
    name = _ensure_channel_session(channel_id)
    env = _session_mgr.get_envelope(name)
    deleted = _session_mgr.reset(env.thread_id)
    await ctx.send(f"Session 已重置，清空了 `{deleted}` 条记录。{agent_name} 从零开始。")


@bot.command(name="tokens")
async def show_tokens(ctx, reset: str = ""):
    if not await _check_auth(ctx):
        return
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!tokens", reset)
    await ctx.send(f"**Token 统计（本次启动后）**\n```\n{reply}\n```")


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


@bot.command(name="resources")
async def show_resources(ctx):
    if not await _check_auth(ctx):
        return
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!resources", "")
    await ctx.send(f"**资源锁状态**\n```\n{reply}\n```")


@bot.command(name="setproject")
async def set_project(ctx, *, path: str = ""):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    name = _ensure_channel_session(channel_id)
    env = _session_mgr.get_envelope(name)
    if not path:
        current = (env.workspace if env else "") or "（未设置）"
        await ctx.send(f"当前项目目录：`{current}`")
        return
    path = os.path.expanduser(path.strip())
    if not os.path.isdir(path):
        await ctx.send(f"路径不存在：`{path}`")
        return
    env.workspace = path
    env.updated_at = datetime.now(timezone.utc).isoformat()
    _session_mgr._save()
    await ctx.send(f"项目目录已设置为：`{path}`\nGit 时间机器已就位。")


@bot.command(name="project")
async def show_project(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    name = _ensure_channel_session(channel_id)
    env = _session_mgr.get_envelope(name)
    current = (env.workspace if env else "") or "（未设置，全局模式）"
    await ctx.send(f"当前项目目录：`{current}`")


# ==========================================
# DiscordInterface — wraps BaseInterface for command delegation
# ==========================================
class _DiscordInterface(BaseInterface):
    """
    Thin wrapper so discord command handlers can delegate to BaseInterface.handle_command()
    without managing controller lifecycle themselves.  The module-level _controller /
    _session_mgr are already initialised by run_discord(); we just reference them.
    """

    def __init__(self, loader) -> None:
        super().__init__(loader)
        # Reuse the already-initialised globals
        self._controller = _controller
        self._session_mgr = _session_mgr


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
        print("请在 agent.json 或 .env（DISCORD_BOT_TOKEN）中设置 Discord Bot Token")
        return
    logger.info(f"[Discord] 启动中... agent={loader.name if loader else '?'} DEBUG={'ON' if is_debug() else 'OFF'}")
    bot.run(token, log_handler=None)
