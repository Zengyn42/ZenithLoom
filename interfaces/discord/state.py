"""
Discord connector — global state, bot instance, per-channel helpers, auth.
"""

import asyncio
import logging
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv
from framework.debug import is_debug

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

# 当前正在处理消息的 channel 对象（供 Discord tool server 使用）
_current_channel = None

# PENDING 后台任务 → 频道映射（task_id → channel_id）
_pending_task_channels: dict[str, int] = {}

# 后台 poller task
_pending_poller_task: asyncio.Task | None = None

_PENDING_POLL_INTERVAL = 30  # seconds

# Discord Bot instance
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


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
        return True  # 如果未设置白名单，默认允许所有人
    return user.id in allowed
