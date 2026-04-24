"""
Discord connector — @bot.command handlers.
Session/stream commands delegate to _DiscordInterface.handle_command();
Discord-specific commands (!stop, !channels, !whoami) are implemented here.
"""

import logging

from interfaces.discord import state as _state
from interfaces.discord.interface import _DiscordInterface
from interfaces.discord.state import bot

logger = logging.getLogger("discord_bot")


async def _check_auth(ctx) -> bool:
    return _state._is_authorized(ctx.author)


async def _delegate(ctx, cmd: str, arg: str = "") -> None:
    iface = _DiscordInterface(_state._loader, channel_id=ctx.channel.id)
    reply = await iface.handle_command(cmd, arg)
    if reply:
        await ctx.send(reply)


@bot.command(name="new")
async def new_session_cmd(ctx, *, name: str = ""):
    if not await _check_auth(ctx):
        return
    await _delegate(ctx, "!new", name)


@bot.command(name="switch")
async def switch_session_cmd(ctx, *, name: str = ""):
    if not await _check_auth(ctx):
        return
    await _delegate(ctx, "!switch", name)


@bot.command(name="session")
async def show_session(ctx):
    if not await _check_auth(ctx):
        return
    await _delegate(ctx, "!session")


@bot.command(name="sessions")
async def list_sessions_cmd(ctx):
    if not await _check_auth(ctx):
        return
    await _delegate(ctx, "!sessions")


@bot.command(name="stream")
async def toggle_stream(ctx):
    if not await _check_auth(ctx):
        return
    await _delegate(ctx, "!stream")


@bot.command(name="stop")
async def stop_task(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    task = _state._channel_tasks.get(channel_id)
    if task and not task.done():
        task.cancel()
        await ctx.send("已停止。")
        return
    consumer = _state._channel_consumers.get(channel_id)
    queue = _state._channel_queues.get(channel_id)
    pending = queue.qsize() if queue else 0
    if consumer and not consumer.done():
        consumer.cancel()
        _state._channel_consumers.pop(channel_id, None)
        if queue:
            while not queue.empty():
                queue.get_nowait()
        msg = "已停止。" + (f" 清除了 {pending} 条待处理消息。" if pending else "")
        await ctx.send(msg)
        return
    await ctx.send("没有正在运行的任务。")


@bot.command(name="channels")
async def list_channels_cmd(ctx):
    if not await _check_auth(ctx):
        return
    all_discord = _state._session_mgr.list_by_prefix("discord-")
    if not all_discord:
        await ctx.send("没有任何 Discord 频道 session。")
        return
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
    lines = []
    guild = ctx.guild
    for cid, names in sorted(grouped.items()):
        ch = guild.get_channel(cid) if guild else None
        ch_label = f"#{ch.name}" if ch else f"⚠ orphan({cid})"
        active = _state._channel_active_session.get(cid, "")
        lines.append(f"{ch_label}  ({len(names)} sessions, active={active.split('-', 2)[-1] if active else '?'})")
        for sname in names:
            display = sname.replace(f"discord-{cid}", "").lstrip("-") or "default"
            marker = " ◀" if sname == active else ""
            lines.append(f"    {display}{marker}")
    await ctx.send(f"**所有频道 Sessions**\n```\n" + "\n".join(lines) + "\n```")


@bot.command(name="whoami")
async def whoami(ctx):
    authorized = "已授权" if _state._is_authorized(ctx.author) else "未授权"
    await ctx.send(f"你的 Discord ID: `{ctx.author.id}` ({authorized})")
