"""
Discord Tool Server — interfaces/discord/tool_server.py

轻量 aiohttp HTTP server，暴露 Discord in-process 数据给 EXTERNAL_TOOL 子进程。
启动时随机选端口，存入 os.environ["DISCORD_TOOL_PORT"] 供子进程读取。

端点：
  GET /history?limit=N          — 当前频道最近 N 条消息（默认 20）
  GET /search?q=keyword&limit=N — 在最近 limit 条里搜关键词（默认搜 100 条，返回匹配）
  GET /channels                 — 列出 server 所有文字频道
  GET /user?id=USER_ID          — 查询用户基本信息（display_name、roles 等）
"""

import asyncio
import json
import logging
import os
import socket

from aiohttp import web

logger = logging.getLogger("discord_bot")

# 当前处理消息的 channel 对象（由 interface.py invoke() 注入）
_current_channel = None


def set_current_channel(channel) -> None:
    global _current_channel
    _current_channel = channel


# ---------------------------------------------------------------------------
# 端点处理器
# ---------------------------------------------------------------------------

async def _handle_history(request: web.Request) -> web.Response:
    channel = _current_channel
    if channel is None:
        return web.Response(status=503, text="no active channel")

    limit = min(int(request.rel_url.query.get("limit", 20)), 200)
    lines = []
    try:
        async for msg in channel.history(limit=limit):
            if msg.content.startswith("!"):
                continue
            ts = msg.created_at.strftime("%H:%M")
            lines.append({
                "ts": ts,
                "author": msg.author.display_name,
                "content": msg.content.strip(),
            })
        lines.reverse()  # 旧 → 新
    except Exception as exc:
        logger.warning(f"[tool_server] history fetch error: {exc}")
        return web.Response(status=500, text=str(exc))

    return web.Response(
        content_type="application/json",
        text=json.dumps(lines, ensure_ascii=False),
    )


async def _handle_search(request: web.Request) -> web.Response:
    channel = _current_channel
    if channel is None:
        return web.Response(status=503, text="no active channel")

    query = request.rel_url.query.get("q", "").lower()
    limit = min(int(request.rel_url.query.get("limit", 100)), 500)
    if not query:
        return web.Response(status=400, text="missing q param")

    matches = []
    try:
        async for msg in channel.history(limit=limit):
            if query in msg.content.lower():
                ts = msg.created_at.strftime("%Y-%m-%d %H:%M")
                matches.append({
                    "ts": ts,
                    "author": msg.author.display_name,
                    "content": msg.content.strip(),
                })
        matches.reverse()
    except Exception as exc:
        logger.warning(f"[tool_server] search error: {exc}")
        return web.Response(status=500, text=str(exc))

    return web.Response(
        content_type="application/json",
        text=json.dumps(matches, ensure_ascii=False),
    )


async def _handle_channels(request: web.Request) -> web.Response:
    channel = _current_channel
    if channel is None:
        return web.Response(status=503, text="no active channel")

    guild = channel.guild
    if guild is None:
        return web.Response(status=404, text="no guild")

    result = []
    for ch in guild.text_channels:
        result.append({
            "id": str(ch.id),
            "name": ch.name,
            "category": ch.category.name if ch.category else None,
            "current": ch.id == channel.id,
        })

    return web.Response(
        content_type="application/json",
        text=json.dumps(result, ensure_ascii=False),
    )


async def _handle_user(request: web.Request) -> web.Response:
    channel = _current_channel
    if channel is None:
        return web.Response(status=503, text="no active channel")

    user_id_str = request.rel_url.query.get("id", "")
    if not user_id_str:
        return web.Response(status=400, text="missing id param")

    guild = channel.guild
    if guild is None:
        return web.Response(status=404, text="no guild")

    try:
        member = guild.get_member(int(user_id_str))
        if member is None:
            member = await guild.fetch_member(int(user_id_str))
    except Exception as exc:
        return web.Response(status=404, text=f"user not found: {exc}")

    result = {
        "id": str(member.id),
        "name": member.name,
        "display_name": member.display_name,
        "roles": [r.name for r in member.roles if r.name != "@everyone"],
        "joined_at": member.joined_at.isoformat() if member.joined_at else None,
    }
    return web.Response(
        content_type="application/json",
        text=json.dumps(result, ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# 启动 / 停止
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_runner: web.AppRunner | None = None


async def start_tool_server() -> int:
    """启动 tool server，返回监听端口。"""
    global _runner

    app = web.Application()
    app.router.add_get("/history", _handle_history)
    app.router.add_get("/search", _handle_search)
    app.router.add_get("/channels", _handle_channels)
    app.router.add_get("/user", _handle_user)

    port = _find_free_port()
    _runner = web.AppRunner(app)
    await _runner.setup()
    site = web.TCPSite(_runner, "127.0.0.1", port)
    await site.start()

    os.environ["DISCORD_TOOL_PORT"] = str(port)
    logger.info(f"[tool_server] Discord tool server started on port {port}")
    return port


async def stop_tool_server() -> None:
    global _runner
    if _runner is not None:
        await _runner.cleanup()
        _runner = None
        os.environ.pop("DISCORD_TOOL_PORT", None)
        logger.info("[tool_server] Discord tool server stopped")
