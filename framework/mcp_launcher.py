"""
通用 MCP 自启动器 — framework/mcp_launcher.py

MCPLauncher 处理所有 MCP 的"检查 → 自启动 → 连接"逻辑，
解耦具体 MCP 实现，由 entity.json 的 mcps 字段声明依赖。

用法（由 agent_loader.start_mcps() 调用）：
  proxy = await MCPLauncher.ensure_and_connect(mcp_conf, proxy_class)

mcp_conf 格式：
  {
    "name": "agent_mail",
    "module": "mcp_servers.agent_mail",
    "transport": "sse",
    "host": "127.0.0.1",
    "port": 8200,
    "pid_file": "data/agent_mail/mail.pid"
  }
"""

import asyncio
import fcntl
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 项目根：framework/ 的父目录，即 BootstrapBuilder/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MCPLauncher:
    """通用 MCP 进程检查、启动、连接工具类。所有方法均为静态方法。"""

    # ------------------------------------------------------------------
    # 1. 进程存活检测
    # ------------------------------------------------------------------

    @staticmethod
    def is_running(pid_file: Path) -> bool:
        """检查 PID 文件对应的进程是否存活。

        若 PID 文件不存在或进程已死，返回 False 并清理残留 PID 文件。
        """
        # 支持相对路径（相对 _PROJECT_ROOT）
        if not pid_file.is_absolute():
            pid_file = _PROJECT_ROOT / pid_file

        if not pid_file.exists():
            return False
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # signal 0 = 存活检测，不发送实际信号
            return True
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            # PID 文件残留但进程已死 → 清理
            pid_file.unlink(missing_ok=True)
            return False

    # ------------------------------------------------------------------
    # 2. 启动 MCP Server（内部实现）
    # ------------------------------------------------------------------

    @staticmethod
    def _do_launch(module: str, host: str, port: int, pid_file: Path) -> None:
        """detach 启动 MCP server 进程，写 PID 文件。"""
        pid_file.parent.mkdir(parents=True, exist_ok=True)

        proc = subprocess.Popen(
            [
                sys.executable, "-m", module,
                "--transport", "sse",
                "--host", host,
                "--port", str(port),
            ],
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach from parent process group
        )
        pid_file.write_text(str(proc.pid))
        logger.info(
            f"[mcp_launcher] launched {module!r} pid={proc.pid} "
            f"at http://{host}:{port}/sse"
        )

    # ------------------------------------------------------------------
    # 3. 文件锁保护的启动入口
    # ------------------------------------------------------------------

    @staticmethod
    def launch(module: str, host: str, port: int, pid_file: Path) -> None:
        """用文件锁保护，detach 启动 MCP server 进程，写 PID 文件。

        文件锁路径：pid_file.with_suffix('.launch.lock')
        double-check：加锁后再次检查 is_running，避免重复启动。
        """
        if not pid_file.is_absolute():
            pid_file = _PROJECT_ROOT / pid_file

        pid_file.parent.mkdir(parents=True, exist_ok=True)
        lock_path = pid_file.with_suffix(".launch.lock")

        _locked = False
        with open(lock_path, "w") as lf:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _locked = True
                # double-check：加锁后再次验证（另一进程可能已完成启动）
                if not MCPLauncher.is_running(pid_file):
                    MCPLauncher._do_launch(module, host, port, pid_file)
                else:
                    logger.debug(
                        f"[mcp_launcher] {module!r} already running (double-check), skip launch"
                    )
            except BlockingIOError:
                # 另一进程正在持有锁（正在启动），等 wait_ready 即可
                logger.debug(
                    f"[mcp_launcher] {module!r} launch lock held by another process, skipping"
                )
            finally:
                if _locked:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # 4. 等待 SSE 端点就绪
    # ------------------------------------------------------------------

    @staticmethod
    async def wait_ready(url: str, timeout: float = 10.0) -> bool:
        """轮询等待 SSE 端点就绪（每 0.5s 重试，最多 timeout 秒）。

        返回 True 表示端点可达，False 表示超时未就绪。
        """
        import aiohttp

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        attempt = 0
        while loop.time() < deadline:
            await asyncio.sleep(0.5)
            attempt += 1
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                        if resp.status < 500:
                            logger.info(
                                f"[mcp_launcher] {url} ready (attempt {attempt}, "
                                f"status={resp.status})"
                            )
                            return True
            except Exception:
                pass  # server 尚未就绪，继续等待

        logger.warning(f"[mcp_launcher] {url} not ready after {timeout}s")
        return False

    # ------------------------------------------------------------------
    # 5. 完整流程：检查 → 自启动 → 等待就绪 → 连接
    # ------------------------------------------------------------------

    @staticmethod
    async def ensure_and_connect(mcp_conf: dict, proxy_class) -> object | None:
        """完整流程：检查 → 自启动 → 等待就绪 → 连接。

        mcp_conf 格式：
          {
            "name": "agent_mail",
            "module": "mcp_servers.agent_mail",
            "transport": "sse",
            "host": "127.0.0.1",
            "port": 8200,
            "pid_file": "data/agent_mail/mail.pid"
          }

        pid_file 路径支持相对路径（相对 _PROJECT_ROOT）和绝对路径。
        proxy_class 由调用方传入（如 HeartbeatMCPProxy 或 AgentMailProxy）。

        返回已连接的 proxy 实例，失败时返回 None。
        """
        name = mcp_conf.get("name", "unknown")
        module = mcp_conf.get("module", "")
        host = mcp_conf.get("host", "127.0.0.1")
        port = mcp_conf.get("port", 8000)
        transport = mcp_conf.get("transport", "sse")
        pid_file_raw = mcp_conf.get("pid_file", f"data/{name}/{name}.pid")

        # 解析 pid_file 路径
        pid_file = Path(pid_file_raw)
        if not pid_file.is_absolute():
            pid_file = _PROJECT_ROOT / pid_file

        if transport != "sse":
            logger.error(
                f"[mcp_launcher] {name!r}: transport={transport!r} is not supported; "
                "only 'sse' transport is handled by MCPLauncher"
            )
            return None

        server_url = f"http://{host}:{port}/sse"

        # 1. 检查并按需自启动
        if not MCPLauncher.is_running(pid_file):
            logger.info(f"[mcp_launcher] {name!r}: not running, launching {module!r}")
            MCPLauncher.launch(module, host, port, pid_file)
        else:
            logger.debug(f"[mcp_launcher] {name!r}: already running, skip launch")

        # 2. 等待 SSE 端点就绪
        ready = await MCPLauncher.wait_ready(server_url, timeout=10.0)
        if not ready:
            logger.error(
                f"[mcp_launcher] {name!r}: server at {server_url} not ready after 10s"
            )
            return None

        # 3. 连接（proxy_class 负责具体连接逻辑）
        proxy = proxy_class(server_url)
        try:
            await proxy.connect()
            logger.info(f"[mcp_launcher] {name!r}: connected to {server_url}")
            return proxy
        except Exception as e:
            logger.error(f"[mcp_launcher] {name!r}: connect failed: {e}")
            return None
