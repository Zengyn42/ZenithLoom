"""
Discord connector — message sending and channel history utilities.
"""

import logging
import os

from interfaces.base_interface import BaseInterface
from interfaces.discord import state as _state

logger = logging.getLogger("discord_bot")

_HISTORY_FILENAME = ".discord_channel_history.txt"


async def send_to_channel(
    channel,
    text: str,
    *,
    max_chars: int = _state.DISCORD_MAX_CHARS,
    files: list | None = None,
) -> None:
    """
    统一消息发送工具：将长文本自动分段发送，不截断任何内容。

    行为：
      1. 用 split_fence_aware 把文本按 max_chars 分成多段
         （尊重代码块边界，避免在 ``` 内部切断）
      2. 逐段 await channel.send(chunk)
      3. 若提供 files，在最后一段文本之后发送文件附件
    """
    chunks = BaseInterface.split_fence_aware(text, max_chars) if text else []
    for i, chunk in enumerate(chunks):
        if i == len(chunks) - 1 and files:
            await channel.send(chunk, files=files)
        else:
            await channel.send(chunk)
    if not chunks and files:
        await channel.send(files=files)


async def _refresh_history_file(channel, limit: int, exclude_msg_id: int) -> str | None:
    """
    拉取最近 limit 条频道消息，写入 workspace/.discord_channel_history.txt。
    返回文件绝对路径，写入失败或无 workspace 时返回 None。
    """
    workspace = (
        (_state._loader.json.get("workspace", "") if _state._loader else "")
        or _state._get_channel_workspace(channel.id)
    )
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
