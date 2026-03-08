"""
框架级心跳后台任务

在 main.py 中用 asyncio.create_task(heartbeat_loop()) 启动。
与任何 agent 完全解耦：只检测 Claude CLI 和 Gemini CLI 是否存活。
"""

import asyncio
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 23 * 3600  # 23小时


async def heartbeat_loop() -> None:
    """后台无限循环心跳。"""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        await _run_heartbeat()


async def run_heartbeat_once() -> bool:
    """启动时立即跑一次。"""
    return await _run_heartbeat()


async def _run_heartbeat() -> bool:
    logger.info("[heartbeat] 开始检测...")
    loop = asyncio.get_event_loop()
    claude_ok, gemini_ok = await asyncio.gather(
        loop.run_in_executor(None, _probe_claude),
        loop.run_in_executor(None, _probe_gemini),
    )
    if claude_ok and gemini_ok:
        logger.info("[heartbeat] 均存活")
    else:
        logger.warning(
            f"[heartbeat] Claude={'OK' if claude_ok else 'DEAD'}, "
            f"Gemini={'OK' if gemini_ok else 'DEAD'}"
        )
    return claude_ok and gemini_ok


def _probe_claude() -> bool:
    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_SESSION", None)
        r = subprocess.run(
            ["claude", "-p", "Reply with just OK.", "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        return "ok" in r.stdout.lower()
    except Exception:
        return False


def _probe_gemini() -> bool:
    try:
        r = subprocess.run(
            ["gemini", "-p", "Reply with just OK."],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        return "ok" in r.stdout.lower()
    except Exception:
        return False
