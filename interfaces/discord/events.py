"""
Discord connector — @bot.event handlers and run_discord entry point.
"""

import asyncio
import io
import logging
import os
import tempfile

import discord
from framework.debug import is_debug
from interfaces.discord import state as _state
from interfaces.discord.alerts import _register_discord_alert_callback
from interfaces.discord.formatting import _fetch_mermaid_png
from interfaces.discord.interface import _DiscordInterface, _channel_consumer
from interfaces.discord.messaging import send_to_channel
from interfaces.discord.state import bot

# Import commands module to trigger @bot.command registration
from interfaces.discord import commands  # noqa: F401

logger = logging.getLogger("discord_bot")


@bot.event
async def on_socket_response(msg):
    if msg.get("t") == "MESSAGE_CREATE":
        print(f"DEBUG: RAW GATEWAY MESSAGE_CREATE: {msg['d'].get('content')}", flush=True)


@bot.event
async def on_ready():
    if _state._loader and _state._controller is None:
        try:
            _state._controller = await _state._loader.get_controller()
            _state._session_mgr = getattr(_state._controller, "session_mgr", _state._session_mgr)
            logger.info(f"[Discord] controller 已初始化（graph 已编译）")
        except Exception as e:
            logger.error(f"[Discord] controller 初始化失败: {e}", exc_info=True)

    agent_name = _state._loader.name if _state._loader else "Agent"
    logger.info(f"[Discord] {agent_name} 已上线: {bot.user}")

    print("DEBUG: Bot is in the following guilds:", flush=True)
    for guild in bot.guilds:
        print(f"  - {guild.name} (ID: {guild.id}, Members: {guild.member_count})", flush=True)
        for channel in guild.text_channels:
            perms = channel.permissions_for(guild.me)
            if perms.read_messages:
                print(f"    - 可见频道: {channel.name} (ID: {channel.id})", flush=True)

    allowed = _state._get_allowed_users()
    if allowed:
        logger.info(f"[Auth] 白名单已启用: {allowed}")
    else:
        logger.warning("[Auth] 白名单为空，任何人均可使用")

    # ── 从 sessions.json 恢复 per-channel 活跃 session ──────────────
    if _state._session_mgr:
        best: dict[int, tuple[str, str]] = {}  # channel_id → (name, updated_at)
        for sname, env in _state._session_mgr.list_by_prefix("discord-").items():
            parts = sname.split("-", 2)
            if len(parts) < 2:
                continue
            try:
                cid = int(parts[1])
            except ValueError:
                continue
            if cid not in best or env.updated_at > best[cid][1]:
                best[cid] = (sname, env.updated_at)
        for cid, (sname, _) in best.items():
            _state._channel_active_session[cid] = sname
        if best:
            logger.info(f"[Discord] 恢复了 {len(best)} 个频道的活跃 session")

    if _state._loader:
        mgr = await _state._loader.start_heartbeat()
        if mgr is not None:
            logger.info(f"[Discord] heartbeat 已启动（agent={agent_name}）")
            _register_discord_alert_callback()

        await _state._loader.start_mcp_servers()
        logger.info(f"[Discord] mcp servers 已启动（agent={agent_name}）")

        from interfaces.discord.tool_server import start_tool_server
        port = await start_tool_server()
        logger.info(f"[Discord] discord tool server 已启动（port={port}）")


@bot.event
async def on_guild_channel_delete(channel):
    """频道被删除时自动清理所有相关 sessions + checkpoints。"""
    task = _state._channel_tasks.get(channel.id)
    if task and not task.done():
        task.cancel()
    deleted = await _state._cleanup_channel(channel.id)
    if deleted:
        logger.info(f"[Discord] 已清理已删频道 {channel.id} 的 {deleted} 个 sessions")


@bot.event
async def on_message(message: discord.Message):
    from interfaces.discord.commands import stop_task

    print(f"DEBUG: on_message received from {message.author.name} (ID: {message.author.id}): {message.content[:50]}", flush=True)
    if message.author.bot:
        return
    if not _state._is_authorized(message.author):
        print(f"DEBUG: Unauthorized user: {message.author.id}", flush=True)
        return
    _state._last_active_channel_id = message.channel.id
    if message.content.startswith("!"):
        ctx = await bot.get_context(message)
        if ctx.command is not None:
            await bot.invoke(ctx)
        else:
            parts = message.content.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if _state._controller is not None:
                _long_running_cmds = {"!discover"}
                _long_running_hints = {
                    "!discover": "🔍 正在搜索工具，请稍候…（搜索 → 设计评估维度，约 1-2 分钟）",
                }
                _typing_ctx = None
                if cmd in _long_running_cmds:
                    await message.channel.send(_long_running_hints.get(cmd, "⏳ 处理中…"))
                    _typing_ctx = message.channel.typing()
                    await _typing_ctx.__aenter__()

                iface = _DiscordInterface(_state._loader, channel_id=message.channel.id)
                try:
                    reply = await iface.handle_command(cmd, arg)
                finally:
                    if _typing_ctx:
                        await _typing_ctx.__aexit__(None, None, None)
                if reply is not None:
                    if cmd == "!topology":
                        await send_to_channel(message.channel, f"```mermaid\n{reply}\n```")
                        png = await _fetch_mermaid_png(reply)
                        if png:
                            await message.channel.send(
                                file=discord.File(io.BytesIO(png), filename="topology.png")
                            )
                        else:
                            await message.channel.send("⚠️ PNG 生成失败，请粘贴上方 Mermaid 文本到 https://mermaid.live 查看")
                    elif "\n" in reply:
                        await send_to_channel(message.channel, f"```\n{reply}\n```")
                    else:
                        await message.channel.send(reply)
        return

    user_input = message.content.strip()
    author_name = message.author.name  # 使用用户名而非昵称
    if user_input:
        user_input = f"{author_name} (DISCORD): {user_input}"
    channel_id = message.channel.id

    # ── 软停止关键词：stop / wait / 停 / 停止（整行，大小写不限）──────
    _SOFT_STOP_WORDS = {"stop", "wait", "停", "停止"}
    if user_input.lower() in _SOFT_STOP_WORDS:
        ctx = await bot.get_context(message)
        await stop_task(ctx)
        return

    # ── 确保频道有 session ──────────────────────────────────────────
    _state._ensure_channel_session(channel_id)

    # ── 附件处理：下载到临时目录，路径附加到 user_input ──────────────
    if message.attachments:
        workspace = (
            (_state._loader.json.get("workspace", "") if _state._loader else "")
            or _state._get_channel_workspace(channel_id)
        )
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
    if channel_id not in _state._channel_queues:
        _state._channel_queues[channel_id] = asyncio.Queue()
    await _state._channel_queues[channel_id].put((user_input, message))

    consumer = _state._channel_consumers.get(channel_id)
    if consumer is None or consumer.done():
        _state._channel_consumers[channel_id] = asyncio.create_task(_channel_consumer(channel_id))


def run_discord(loader=None):
    _state._loader = loader
    _state._session_mgr = loader.session_mgr if loader else None
    # _controller 延迟到 on_ready() 内初始化，确保 aiosqlite 绑定到 discord.py 的事件循环

    token = loader.load_config().discord_token if loader else ""
    if not token:
        print("请在 entity.json 或 .env（DISCORD_BOT_TOKEN）中设置 Discord Bot Token")
        return
    logger.info(f"[Discord] 启动中... agent={loader.name if loader else '?'} DEBUG={'ON' if is_debug() else 'OFF'}")
    bot.run(token, log_handler=None)

    if _state._loader:
        async def _cleanup():
            from interfaces.discord.tool_server import stop_tool_server
            await stop_tool_server()
            await _state._loader.stop_heartbeat()
            await _state._loader.stop_mcp_servers()

        try:
            asyncio.run(_cleanup())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_cleanup())
            loop.close()
