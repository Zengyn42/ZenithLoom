"""
Discord connector — @bot.command handlers.
"""

import logging

from interfaces.discord import state as _state
from interfaces.discord.state import bot

logger = logging.getLogger("discord_bot")


async def _check_auth(ctx) -> bool:
    return _state._is_authorized(ctx.author)


@bot.command(name="new")
async def new_session_cmd(ctx, *, name: str = ""):
    if not await _check_auth(ctx):
        return
    if not name:
        await ctx.send("用法：`!new <名称>`")
        return
    channel_id = ctx.channel.id
    full_name = f"{_state._channel_prefix(channel_id)}-{name}"
    try:
        env = _state._session_mgr.create_session(full_name)
        _state._channel_active_session[channel_id] = full_name
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
        full_name = _state._channel_default_session(channel_id)
    else:
        full_name = f"{_state._channel_prefix(channel_id)}-{name}"
    env = _state._session_mgr.get_envelope(full_name)
    if not env:
        await ctx.send(f"❌ Session `{name}` 不存在。用 `!sessions` 查看可用 sessions。")
        return
    _state._channel_active_session[channel_id] = full_name
    await ctx.send(f"已切换到 session `{name}`\nthread: `{env.thread_id}`")


@bot.command(name="session")
async def show_session(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    name = _state._ensure_channel_session(channel_id)
    env = _state._session_mgr.get_envelope(name)
    ns = env.node_sessions if env else {}
    ns_display = ", ".join(f"{k}={v[:8]}" for k, v in ns.items() if v) if ns else "（无）"
    await ctx.send(
        f"session: `{name}`\n"
        f"thread: `{env.thread_id}`\n"
        f"node_sessions: `{ns_display}`"
    )


@bot.command(name="sessions")
async def list_sessions_cmd(ctx):
    if not await _check_auth(ctx):
        return
    channel_id = ctx.channel.id
    prefix = _state._channel_prefix(channel_id)
    sessions = _state._session_mgr.list_by_prefix(prefix)
    if not sessions:
        await ctx.send("当前频道没有 session。")
        return
    active = _state._channel_active_session.get(channel_id, _state._channel_default_session(channel_id))
    lines = []
    for name, env in sessions.items():
        marker = " ◀" if name == active else ""
        lines.append(f"  {name:<40} thread={env.thread_id}{marker}")
    await ctx.send(f"**频道 Sessions**\n```\n" + "\n".join(lines) + "\n```")


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
        msg = f"已停止。" + (f" 清除了 {pending} 条待处理消息。" if pending else "")
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


@bot.command(name="stream")
async def toggle_stream(ctx):
    if not await _check_auth(ctx):
        return
    _state._discord_streaming = not _state._discord_streaming
    state_str = "ON" if _state._discord_streaming else "OFF"
    await ctx.send(f"Streaming: {state_str}")


@bot.command(name="whoami")
async def whoami(ctx):
    authorized = "已授权" if _state._is_authorized(ctx.author) else "未授权"
    await ctx.send(f"你的 Discord ID: `{ctx.author.id}` ({authorized})")
