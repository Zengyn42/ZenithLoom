"""
框架级心跳后台任务

任务列表由 agent.json 的 heartbeat.tasks 声明，例如：

  "heartbeat": {
    "tasks": [
      {"type": "probe", "name": "claude"},
      {"type": "probe", "name": "gemini"}
    ]
  }

当前支持的 task type：
  "probe"  — 服务存活检测，name 可选值：
               "claude"  调用 Claude CLI（keep-alive + auth 检测）
               "gemini"  调用 Gemini CLI（keep-alive + auth 检测）
               "ollama"  GET /api/tags（无推理，仅检测服务在线）

扩展新 task type：在 _run_task() 中增加 elif 分支即可，无需改动接口层。

接口层调用方式：
  hb_cfg = loader.json.get("heartbeat") or {}
  await run_heartbeat_once(hb_cfg)
  asyncio.create_task(heartbeat_loop(hb_cfg))
"""

import asyncio
import logging
import os
import subprocess

import httpx

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 23 * 3600  # 23小时


async def heartbeat_loop(cfg: dict) -> None:
    """后台无限循环心跳。"""
    interval = cfg.get("interval_hours", 23) * 3600
    while True:
        await asyncio.sleep(interval)
        await _run_heartbeat(cfg)


async def run_heartbeat_once(cfg: dict) -> bool:
    """启动时立即跑一次。"""
    return await _run_heartbeat(cfg)


async def _run_heartbeat(cfg: dict) -> bool:
    tasks = cfg.get("tasks", [])
    if not tasks:
        return True

    logger.info("[heartbeat] 开始检测...")
    results = await asyncio.gather(*(_run_task(t) for t in tasks))
    statuses = ", ".join(
        f"{label.capitalize()}={'OK' if ok else 'DEAD'}" for label, ok in results
    )
    all_ok = all(ok for _, ok in results)
    (logger.info if all_ok else logger.warning)(f"[heartbeat] {statuses}")
    return all_ok


async def _run_task(task: dict) -> tuple[str, bool]:
    """执行单个 task，返回 (label, ok)。"""
    ttype = task.get("type")

    if ttype == "probe":
        return await _run_probe(task)

    # 未来扩展：elif ttype == "shell": ...
    logger.warning(f"[heartbeat] 未知 task type: {ttype!r}")
    return str(ttype), False


# ── probe type ──────────────────────────────────────────────────────────────

_SYNC_PROBES = {
    "claude": None,   # set below (forward ref)
    "gemini": None,
}

async def _run_probe(task: dict) -> tuple[str, bool]:
    name = task.get("name", "")
    loop = asyncio.get_event_loop()

    if name == "claude":
        ok = await loop.run_in_executor(None, _probe_claude)
    elif name == "gemini":
        ok = await loop.run_in_executor(None, _probe_gemini)
    elif name == "ollama":
        endpoint = task.get("endpoint", "http://localhost:11434")
        ok = await _probe_ollama(endpoint)
    else:
        logger.warning(f"[heartbeat] 未知 probe name: {name!r}")
        ok = False

    return name, ok


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
            ["gemini", "-m", "gemini-2.5-flash", "-p", "Reply with just OK."],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        return "ok" in r.stdout.lower()
    except Exception:
        return False


async def _probe_ollama(endpoint: str = "http://localhost:11434") -> bool:
    """Ping Ollama /api/tags — fast, no LLM inference needed."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{endpoint}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
