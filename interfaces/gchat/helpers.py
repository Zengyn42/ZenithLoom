"""
GChat connector — subprocess helpers for gws CLI.
"""

import asyncio
import logging
import subprocess

logger = logging.getLogger("gchat_bot")


def _run_gws_send(space: str, text: str) -> None:
    """Synchronous helper: send a GChat message via gws (called in thread pool)."""
    subprocess.run(
        ["gws", "chat", "+send", "--space", space, "--text", text],
        capture_output=True,
    )


def _stream_gws_events(cmd: list[str], queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """
    Blocking reader: run gws events +subscribe and push each NDJSON line into queue.
    Runs in a thread pool so it doesn't block the event loop.
    Pushes None as sentinel when the process exits.
    """
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            line = line.strip()
            if line:
                loop.call_soon_threadsafe(queue.put_nowait, line)
    except Exception as e:
        logger.error("[GChat] 事件流异常: %s", e)
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel — signals EOF
