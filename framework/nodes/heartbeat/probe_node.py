"""
PROBE 节点 — 服务存活检测节点，不继承 AgentNode。

将探测结果以 AIMessage 写入 state["messages"]，供后续节点参考。

支持的 probe name：
  "claude"  — Claude CLI subprocess（keep-alive + auth 检测）
  "gemini"  — Gemini CLI subprocess（keep-alive + auth 检测）
  "ollama"  — GET /api/tags（无推理，纯在线检测）

node_config 字段：
  name      str  探针名称（必填）
  endpoint  str  Ollama endpoint（可选，默认 http://localhost:11434）
  timeout   int  探测超时秒数（可选，默认 30；ollama 默认 5）
"""

import asyncio
import logging
import os
import subprocess

import httpx
from langchain_core.messages import AIMessage

from framework.debug import is_debug

logger = logging.getLogger(__name__)


class ProbeNode:
    """服务存活探针节点，实现标准 LangGraph node 接口 __call__(state) -> dict。"""

    def __init__(self, node_config: dict):
        self._name = node_config.get("name", "")
        self._endpoint = node_config.get("endpoint", "http://localhost:11434")
        self._timeout = node_config.get("timeout", 30)  # entity 级可覆写

    async def __call__(self, state: dict) -> dict:
        name = self._name
        timeout = self._timeout
        loop = asyncio.get_event_loop()

        if is_debug():
            logger.debug(f"[probe] 开始探测 {name!r} endpoint={self._endpoint!r} timeout={timeout}s")

        if name == "claude":
            ok = await loop.run_in_executor(None, lambda: _probe_claude(timeout))
        elif name == "gemini":
            ok = await loop.run_in_executor(None, lambda: _probe_gemini(timeout))
        elif name == "ollama":
            ok = await _probe_ollama(self._endpoint, timeout)
        else:
            logger.warning(f"[probe] 未知 probe name: {name!r}")
            ok = False

        label = f"{name}:{'OK' if ok else 'DEAD'}"
        (logger.info if ok else logger.warning)(f"[probe] {label}")
        return {"messages": [AIMessage(content=label)]}


# ── 探针实现 ─────────────────────────────────────────────────────────────────

def _probe_claude(timeout: int = 30) -> bool:
    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_SESSION", None)
        r = subprocess.run(
            ["claude", "-p", "Reply with just OK.", "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        return "ok" in r.stdout.lower()
    except Exception:
        return False


def _probe_gemini(timeout: int = 30) -> bool:
    try:
        r = subprocess.run(
            ["gemini", "-m", "gemini-2.5-flash", "-p", "Reply with just OK."],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return "ok" in r.stdout.lower()
    except Exception:
        return False


async def _probe_ollama(endpoint: str, timeout: int = 5) -> bool:
    """Ping Ollama /api/tags — fast, no LLM inference needed."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{endpoint}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
