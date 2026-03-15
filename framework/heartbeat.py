"""
框架级心跳后台任务

探针由 agent.json 的 heartbeat.probes 列表声明，例如：

  "heartbeat": {
    "probes": ["claude", "gemini"]   # Hani
  }

  "heartbeat": {
    "probes": ["ollama"]             # Asa
  }

可用探针：
  "claude"  — 调用 Claude CLI，检测 auth 和 session 存活
  "gemini"  — 调用 Gemini CLI，同上
  "ollama"  — GET /api/tags，检测 Ollama 服务是否在线（无 LLM 推理）

接口层调用方式：
  hb_cfg = loader.json.get("heartbeat") or {}
  probes = hb_cfg.get("probes", [])
  await run_heartbeat_once(probes)
  asyncio.create_task(heartbeat_loop(probes))
"""

import asyncio
import logging
import os
import subprocess

import httpx

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL = 23 * 3600  # 23小时


async def heartbeat_loop(probes: list[str]) -> None:
    """后台无限循环心跳。"""
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        await _run_heartbeat(probes)


async def run_heartbeat_once(probes: list[str]) -> bool:
    """启动时立即跑一次。"""
    return await _run_heartbeat(probes)


async def _run_heartbeat(probes: list[str]) -> bool:
    if not probes:
        return True

    logger.info(f"[heartbeat] 开始检测 probes={probes}...")
    loop = asyncio.get_event_loop()

    tasks = []
    for name in probes:
        if name == "claude":
            tasks.append(("claude", loop.run_in_executor(None, _probe_claude)))
        elif name == "gemini":
            tasks.append(("gemini", loop.run_in_executor(None, _probe_gemini)))
        elif name == "ollama":
            tasks.append(("ollama", _probe_ollama()))

    results = await asyncio.gather(*(t for _, t in tasks))
    name_results = list(zip([n for n, _ in tasks], results))

    statuses = ", ".join(f"{n.capitalize()}={'OK' if ok else 'DEAD'}" for n, ok in name_results)
    all_ok = all(ok for _, ok in name_results)
    if all_ok:
        logger.info(f"[heartbeat] {statuses}")
    else:
        logger.warning(f"[heartbeat] {statuses}")
    return all_ok


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
