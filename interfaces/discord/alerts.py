"""
Discord connector — heartbeat alert callback and pending task poller.
"""

import asyncio
import logging

from interfaces.discord import state as _state
from interfaces.discord.messaging import send_to_channel

logger = logging.getLogger("discord_bot")

_CRITICAL_THRESHOLD = 3


def _register_pending_tasks_for_channel(channel_id: int) -> None:
    """查找该频道刚产生的新 PENDING 任务，绑定 task_id → channel_id。

    只绑定尚未在 _pending_task_channels 中的 RUNNING 任务，
    避免多频道时错误覆盖。
    """
    try:
        from mcp_servers.heartbeat.task_vault import TaskVault, TaskStatus
        vault = TaskVault.get_instance()

        for task_id, record in vault._tasks.items():
            if record.status == TaskStatus.RUNNING and task_id not in _state._pending_task_channels:
                _state._pending_task_channels[task_id] = channel_id
                logger.info(f"[discord] bound pending task {task_id} → channel {channel_id}")

    except Exception as e:
        logger.warning(f"[discord] _register_pending_tasks_for_channel error: {e}")

    # 确保 poller 在运行（作为 SSE 推送的兜底）
    if _state._pending_task_channels and (
        _state._pending_poller_task is None or _state._pending_poller_task.done()
    ):
        _state._pending_poller_task = asyncio.get_event_loop().create_task(
            _pending_tasks_poller(), name="pending_tasks_poller"
        )


async def _pending_tasks_poller() -> None:
    """后台 poller：每 30s 检查 TaskVault，任务完成后主动推送到对应频道。"""
    while _state._pending_task_channels:
        await asyncio.sleep(_state._PENDING_POLL_INTERVAL)

        if not _state._pending_task_channels:
            break

        try:
            from mcp_servers.heartbeat.task_vault import TaskVault, TaskStatus
            vault = TaskVault.get_instance()

            completed: list[tuple[str, int]] = []  # (task_id, channel_id)

            for task_id, channel_id in list(_state._pending_task_channels.items()):
                status = vault.query_task(task_id)
                if status is None or status == TaskStatus.RUNNING:
                    continue
                completed.append((task_id, channel_id))

            for task_id, channel_id in completed:
                _state._pending_task_channels.pop(task_id, None)
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

                msg_text = f"{header} `{task_id}`\n```\n{result}\n```"

                channel = _state.bot.get_channel(channel_id)
                if channel:
                    try:
                        await send_to_channel(channel, msg_text)
                        logger.info(f"[discord] sent pending task result: {task_id} → channel {channel_id}")
                    except Exception as e:
                        logger.warning(f"[discord] failed to send task result: {e}")
                else:
                    logger.warning(f"[discord] channel {channel_id} not found for task {task_id}")

        except Exception as e:
            logger.warning(f"[discord] pending_tasks_poller error: {e}")

    logger.info("[discord] pending_tasks_poller exiting (no pending tasks)")


def _find_alert_channel():
    """返回最后活跃的频道；如果没有活跃记录，回退到第一个可发送的频道。"""
    if _state._last_active_channel_id is not None:
        ch = _state.bot.get_channel(_state._last_active_channel_id)
        if ch is not None:
            return ch
    for guild in _state.bot.guilds:
        for ch in guild.text_channels:
            if ch.permissions_for(guild.me).send_messages:
                return ch
    return None


def _register_discord_alert_callback():
    """将 Discord 告警处理注册到 HeartbeatMCPProxy 的 SSE 回调。

    使用 on_proxy_ready hook：如果 proxy 已存在立即注册，
    否则等 ExternalToolNode 按需创建 proxy 时自动触发。
    """
    from framework.nodes.llm.heartbeat_tools import on_proxy_ready

    def _bind_callback(proxy):
        proxy.set_alert_callback(_discord_handle_alert)
        logger.info("[Discord] heartbeat alert callback registered (SSE push)")

    on_proxy_ready(_bind_callback)


async def _discord_handle_alert(alert: dict):
    """
    SSE 推送触发的 Discord 告警处理。
    TASK_MONITOR 事件 → 通过 task_id 查找绑定的频道（精确投递）
    其他告警 → 发到最后活跃频道（兜底）
    """
    alert_type = alert.get("type", "")
    task_id = alert.get("task_id", "")
    if alert_type == "TASK_MONITOR" and task_id in _state._pending_task_channels:
        channel_id = _state._pending_task_channels[task_id]
        channel = _state.bot.get_channel(channel_id)
        status = alert.get("status", "")

        if channel and status in ("completed", "timeout", "failed"):
            _state._pending_task_channels.pop(task_id, None)
            content = alert.get("content", "")
            if status == "completed":
                header = "✅ **后台任务完成**"
            elif status == "timeout":
                header = "⏰ **后台任务超时**"
            else:
                header = "❌ **后台任务失败**"
            msg = f"{header} `{task_id}`"
            if content:
                msg += f"\n```\n{content}\n```"
            await send_to_channel(channel, msg)
            logger.info(f"[discord_alert] TASK_MONITOR {status}: {task_id} → channel {channel_id}")
            return
        elif channel and status == "monitoring":
            return

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
        status = alert.get("status", "")
        if status == "DEAD":
            icon = "❌"
        else:
            icon = "✅" if level == "info" else "⚠️"
        content = alert.get("content", "")
        next_line = f" | 下次: `{next_run}`" if next_run else ""
        msg = f"{icon} `{task_id}` | 时间: `{time_}`{next_line}"
        if content:
            msg += f"\n> {content[:200]}"
        await send_to_channel(channel, msg)
        return

    # ── 失败告警（error）─────────────────────────────────────────────────
    consecutive = alert.get("consecutive_failures", 1)
    error       = alert.get("error", "?")
    next_line   = f" → 下次: {next_run}" if next_run else ""
    alert_text  = f"[{task_id}] FAILED (×{consecutive}) at {time_}: {error}{next_line}"

    if consecutive >= _CRITICAL_THRESHOLD and _state._controller:
        agent_name = _state._loader.name if _state._loader else "Agent"
        prompt = (
            f"[SYSTEM ALERT — Heartbeat 失败告警]\n\n{alert_text}\n\n"
            "请分析上述 heartbeat 探针失败的情况，告知用户可能的原因和建议的修复措施。"
        )
        try:
            from interfaces.discord.interface import _DiscordInterface
            iface = _DiscordInterface(_state._loader)
            response = await iface.invoke_agent(prompt)
            await send_to_channel(
                channel,
                f"🚨 **Heartbeat Critical — {agent_name} 分析**\n{response}",
            )
        except Exception as e:
            logger.error(f"[discord_alert] agent invocation failed: {e}")
            await send_to_channel(channel, f"⚠ **Heartbeat Alert**\n```\n{alert_text}\n```")
    else:
        await send_to_channel(channel, f"⚠ **Heartbeat Alert**\n```\n{alert_text}\n```")
