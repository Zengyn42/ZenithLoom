"""
framework/mcp_manager.py — 全局 MCP Server 进程生命周期管理器

职责：
  - 按需启动 MCP Server 进程（subprocess.Popen，start_new_session=True）
  - 引用计数：多个 agent 可共享同一个 server，引用归零时停止进程
  - 幂等性：server 已在运行（外部启动）时直接复用，不重复启动
  - 为 ClaudeAgentOptions.mcp_servers 提供 SSE 配置字典

entity.json "mcp" 字段格式（顶层，与 "graph" 平级）：
  "mcp": [
    {
      "name": "obsidian-vault",
      "module": "mcp_servers.obsidian.server",
      "module_args": ["--transport", "sse", "--port", "8101",
                      "--vault", "/home/kingy/Foundation/Vault"],
      "url": "http://localhost:8101/sse",
      "shared": true
    }
  ]

字段说明：
  name         — 服务唯一标识（ClaudeAgentOptions.mcp_servers 的 key）
  module       — 用 python -m <module> 启动
  module_args  — 额外命令行参数（默认含 --transport sse）
  url          — SSE 端点 URL，供 ClaudeAgentOptions 使用，也用于存活探测
  shared       — true（默认）表示跨 agent 共享；false 表示每 agent 独占

用法示例：
  mgr = MCPManager.get_instance()
  await mgr.acquire(spec, agent_name="hani")
  configs = mgr.get_all_configs()   # → {"obsidian-vault": {"type":"sse","url":...}}
  await mgr.release("obsidian-vault", agent_name="hani")
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 项目根目录（mcp server 模块的 cwd）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------

@dataclass
class _ServerEntry:
    name: str
    url: str
    agents: set = field(default_factory=set)
    proc: Optional[subprocess.Popen] = None  # None = 外部已启动，不由我们管理


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------

class MCPManager:
    """
    全局单例：MCP Server 进程生命周期管理。

    acquire() → 启动（如未运行）并注册 agent
    release() → 取消注册，引用归零时停止进程
    get_all_configs() → 返回 ClaudeAgentOptions 可用的 mcp_servers 字典
    """

    _instance: Optional["MCPManager"] = None

    @classmethod
    def get_instance(cls) -> "MCPManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._servers: dict[str, _ServerEntry] = {}
        self._lock: Optional[asyncio.Lock] = None  # 懒初始化，避免跨事件循环问题

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ------------------------------------------------------------------
    # 存活探测
    # ------------------------------------------------------------------

    def _is_reachable(self, url: str) -> bool:
        """
        GET url 的父路径（或 url 本身）检测 server 是否可达。
        SSE URL 如 http://localhost:8101/sse → 探测 http://localhost:8101/
        """
        # 去掉 /sse 后缀后探测根路径
        check = url
        if check.endswith("/sse"):
            check = check[:-4]
        if not check.endswith("/"):
            check += "/"
        try:
            req = urllib.request.Request(check, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def _wait_ready(self, url: str, timeout: int = 20) -> bool:
        """轮询直到 server 可达或超时。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_reachable(url):
                return True
            time.sleep(0.4)
        return False

    # ------------------------------------------------------------------
    # 进程启动
    # ------------------------------------------------------------------

    def _start_process(self, spec: dict) -> subprocess.Popen:
        """
        以 python -m <module> [module_args] 启动 MCP Server。
        start_new_session=True → 与父进程解耦，不随父进程 SIGINT 退出。
        """
        module = spec["module"]
        args = spec.get("module_args", [])
        cmd = [sys.executable, "-m", module, *args]
        proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info(f"[mcp_manager] spawned {spec['name']} pid={proc.pid} cmd={' '.join(cmd[:4])}")
        return proc

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    async def acquire(self, spec: dict, agent_name: str) -> bool:
        """
        确保指定 MCP Server 正在运行，并将 agent_name 注册为消费者。

        如果 server 已在 _servers 中（其他 agent 已启动），直接复用。
        如果 server 已在对应端口运行（外部启动），标记为外部所有（proc=None）。
        否则启动新进程并等待就绪。

        返回 True 表示 server 可用，False 表示启动失败。
        """
        name = spec["name"]
        url = spec["url"]

        async with self._ensure_lock():
            if name in self._servers:
                # 已有记录：仅增加引用
                self._servers[name].agents.add(agent_name)
                logger.debug(
                    f"[mcp_manager] {name}: {agent_name!r} joined "
                    f"(refs={len(self._servers[name].agents)})"
                )
                return True

            # 检测是否已在运行（外部启动或跨重启复用）
            if self._is_reachable(url):
                entry = _ServerEntry(name=name, url=url, proc=None)
                entry.agents.add(agent_name)
                self._servers[name] = entry
                logger.info(f"[mcp_manager] {name}: already running at {url} (external)")
                return True

            # 需要启动
            try:
                proc = self._start_process(spec)
            except Exception as exc:
                logger.error(f"[mcp_manager] {name}: spawn failed: {exc}")
                return False

            ready = self._wait_ready(url)
            if not ready:
                logger.error(f"[mcp_manager] {name}: did not become ready (timeout)")
                try:
                    proc.kill()
                except Exception:
                    pass
                return False

            entry = _ServerEntry(name=name, url=url, proc=proc)
            entry.agents.add(agent_name)
            self._servers[name] = entry
            logger.info(f"[mcp_manager] {name}: ready at {url}")
            return True

    async def release(self, server_name: str, agent_name: str) -> None:
        """
        注销 agent_name 对指定 server 的引用。
        引用归零且进程由我们管理时，终止进程。
        """
        async with self._ensure_lock():
            entry = self._servers.get(server_name)
            if entry is None:
                return

            entry.agents.discard(agent_name)
            logger.debug(
                f"[mcp_manager] {server_name}: {agent_name!r} released "
                f"(refs={len(entry.agents)})"
            )

            if entry.agents:
                return  # 还有其他 agent 在用

            del self._servers[server_name]

            if entry.proc is not None:
                # 我们启动的，我们负责停止
                try:
                    entry.proc.terminate()
                    entry.proc.wait(timeout=5)
                    logger.info(f"[mcp_manager] {server_name}: stopped (no more agents)")
                except subprocess.TimeoutExpired:
                    entry.proc.kill()
                    logger.warning(f"[mcp_manager] {server_name}: SIGKILL after terminate timeout")
                except Exception as exc:
                    logger.warning(f"[mcp_manager] {server_name}: stop error: {exc}")
            else:
                logger.info(f"[mcp_manager] {server_name}: released (external process, not stopped)")

    async def release_all(self, agent_name: str) -> None:
        """释放指定 agent 持有的所有 server 引用。"""
        names = list(self._servers.keys())
        for name in names:
            await self.release(name, agent_name)

    def get_sse_configs(self, server_names: list[str]) -> dict:
        """
        返回指定 server 名称的 SSE 配置字典（仅包含当前已运行的 server）。
        格式兼容 ClaudeAgentOptions.mcp_servers：
          {name: {"type": "sse", "url": "..."}}
        """
        result = {}
        for name in server_names:
            entry = self._servers.get(name)
            if entry:
                result[name] = {"type": "sse", "url": entry.url}
        return result

    def get_all_configs(self) -> dict:
        """
        返回所有当前运行 server 的 SSE 配置字典。
        格式兼容 ClaudeAgentOptions.mcp_servers。
        """
        return {
            name: {"type": "sse", "url": entry.url}
            for name, entry in self._servers.items()
        }

    def running_servers(self) -> list[str]:
        """返回当前运行中的 server 名称列表（调试用）。"""
        return list(self._servers.keys())
