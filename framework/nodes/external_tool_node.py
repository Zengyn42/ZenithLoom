"""ExternalToolNode — EXTERNAL_TOOL node type.

Executes any external CLI as a LangGraph node.  Supports:
- CLI-Anything harnesses  (cli-anything-blender --json ...)
- Native structured CLIs  (gws gmail +triage --json)
- Official CLIs           (obsidian search --query ...)

node_config fields:
  command     list[str]  required  Base command + args; supports {field} templates
  timeout     float      30.0      Max subprocess runtime (seconds); also used as hard_timeout for background monitoring
  inject_as   str        "message" "message" -> AIMessage; anything else -> state field name
  description str        ""        Human-readable label shown by !topology

固有行为（不可配置）：
  120s 内完成 → 正常返回
  超过 120s  → 自动启动 heartbeat MCP 后台监控，返回 PENDING 消息
"""

import asyncio
import io
import json
import logging
import re
import shlex
import subprocess
import threading
import time
import uuid

from langchain_core.messages import AIMessage

from framework.config import AgentConfig
from framework.debug import is_debug

logger = logging.getLogger(__name__)

# 所有 EXTERNAL_TOOL 的固有超时阈值：超过此时间自动转入 heartbeat 后台监控
_ASYNC_THRESHOLD = 120.0  # seconds


def _substitute(template: str, state: dict) -> str:
    """Replace {word} placeholders from state; leave all other {…} (e.g. JSON) untouched."""
    return re.sub(r"\{(\w+)\}", lambda m: str(state[m.group(1)]) if m.group(1) in state else m.group(0), template)


def _run_code_exec(cmd: list[str], timeout: float, cwd: str | None) -> subprocess.CompletedProcess:
    """Run cmd (arg-list, no shell) with timeout and optional cwd."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd or None,
    )


class ExternalToolNode:
    """
    EXTERNAL_TOOL node: calls any external CLI as a subprocess (arg-list, no shell),
    parses stdout as JSON when possible, and injects the result into LangGraph state.

    Routing pattern:
        Claude outputs {"route": "<node_id>", "context": "..."} -> ExternalToolNode runs
        -> result injected as AIMessage -> Claude reads and continues.

    Async timeout:
        120s 内完成 → 正常返回结果
        超过 120s  → 注册 heartbeat 后台监控 → 返回 PENDING 消息 → heartbeat 每 60s 报告
        超过 hard_timeout → kill 进程 → 标记 TIMEOUT
    """

    def __init__(self, config: AgentConfig, node_config: dict) -> None:
        # node_config may be the full node def {"id":..., "type":..., "node_config":{...}}
        # or just the inner node_config dict directly — handle both
        inner = node_config.get("node_config", node_config)
        self._backend: str = inner.get("backend", "cli")
        raw = inner.get("command")
        if self._backend != "code_execution" and not raw:
            raise ValueError("ExternalToolNode: 'command' must be a non-empty list")
        self._command: list[str] = shlex.split(raw) if isinstance(raw, str) else list(raw or [])
        self._timeout = float(inner.get("timeout", 30.0))
        self._inject_as: str = inner.get("inject_as", "message")
        self._description: str = inner.get("description", "")

        # hard_timeout: 后台监控的最大运行时间，取 max(timeout, 120s)
        self._hard_timeout: float = max(self._timeout, _ASYNC_THRESHOLD)

    async def _run_code_execution(self, state: dict) -> dict:
        """code_execution 后端：从 state 读取命令执行。"""
        cmd_str: str = state.get("execution_command", "")
        if not cmd_str.strip():
            if is_debug():
                logger.debug("[external_tool] code_execution: execution_command is empty, skipping")
            return {
                "execution_stdout": "",
                "execution_stderr": "",
                "execution_returncode": 0,
            }
        working_dir: str = state.get("working_directory") or ""
        cmd = shlex.split(cmd_str)
        cwd = working_dir if working_dir else None
        try:
            result = await asyncio.to_thread(_run_code_exec, cmd, self._timeout, cwd)
        except subprocess.TimeoutExpired:
            return {
                "execution_stdout": "",
                "execution_stderr": f"timeout after {self._timeout}s",
                "execution_returncode": -1,
            }
        return {
            "execution_stdout": result.stdout,
            "execution_stderr": result.stderr,
            "execution_returncode": result.returncode,
        }

    def _tee_subprocess(self, cmd: list[str]) -> tuple[subprocess.Popen, io.BytesIO, "BoundedFileWriter"]:
        """启动子进程 + tee 线程：stdout/stderr 同时写到 BoundedFile 和内存 buffer。

        Returns:
            (proc, memory_buffer, bounded_writer)
        """
        from framework.bounded_file_writer import BoundedFileWriter
        from mcp_servers.heartbeat.task_vault import _MONITORS_DIR

        task_id = f"tool_{uuid.uuid4().hex[:12]}"
        output_path = _MONITORS_DIR / f"{task_id}.output"

        writer = BoundedFileWriter(output_path)
        buffer = io.BytesIO()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )

        def _tee_reader():
            """后台线程：从 proc.stdout 读取，同时写入 buffer 和 file。"""
            try:
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    writer.write(chunk)
            except Exception as e:
                logger.warning(f"[external_tool] tee reader error: {e}")
            finally:
                writer.close()

        tee_thread = threading.Thread(target=_tee_reader, daemon=True, name=f"tee:{task_id}")
        tee_thread.start()

        # 将 task_id 和 tee_thread 挂在 proc 上，方便后续引用
        proc._task_id = task_id  # type: ignore[attr-defined]
        proc._tee_thread = tee_thread  # type: ignore[attr-defined]
        proc._output_path = str(output_path)  # type: ignore[attr-defined]

        return proc, buffer, writer

    def _ensure_heartbeat_server(self) -> None:
        """确保 Heartbeat MCP Server 正在运行，未运行则自动启动。

        最多等待 10 秒，每 0.5 秒检查一次。
        """
        from framework.nodes.llm.heartbeat_tools import _is_server_running, _launch_server

        if _is_server_running():
            return

        logger.info("[external_tool] heartbeat MCP server not running, launching...")
        _launch_server("http://127.0.0.1:8100/sse")

        for _ in range(20):
            time.sleep(0.5)
            if _is_server_running():
                logger.info("[external_tool] heartbeat MCP server started successfully")
                return

        logger.warning("[external_tool] heartbeat MCP server failed to start within 10s")

    async def _on_timeout(self, proc: subprocess.Popen, task_id: str, output_path: str) -> dict:
        """120s 超时后的处理：启动 heartbeat 监控，返回 PENDING 消息。

        1. 确保 Heartbeat MCP Server 正在运行（自动启动）
        2. 通过 TaskVault 注册 PID（本地持久化）
        3. 通过 MCP 调用 heartbeat_register_monitor 启动监控循环
        4. 返回 PENDING 消息

        Returns:
            LangGraph state dict。
        """
        self._ensure_heartbeat_server()

        from mcp_servers.heartbeat.task_vault import TaskVault

        TaskVault.get_instance().register_task(
            task_id=task_id,
            pid=proc.pid,
            output_path=output_path,
            hard_timeout=self._hard_timeout,
        )

        # 通过 MCP 调用 heartbeat_register_monitor 启动实际监控循环
        await self._register_monitor_via_mcp(task_id, proc.pid, output_path)

        pending_content = (
            f"[PENDING] 命令执行超过 {_ASYNC_THRESHOLD:.0f}s，已转入后台继续运行。\n"
            f"task_id: {task_id}\n"
            f"完成后系统会通知你。你可以继续其他工作。"
        )
        logger.info(f"[external_tool] timeout: {task_id} pid={proc.pid} backgrounded")

        if self._inject_as == "message":
            return {
                "messages": [AIMessage(content=pending_content, additional_kwargs={"task_id": task_id})],
                "routing_target": "",
                "routing_context": "",
            }
        else:
            return {self._inject_as: pending_content}

    async def _register_monitor_via_mcp(self, task_id: str, pid: int, output_path: str) -> None:
        """通过 MCP 协议调用 heartbeat_register_monitor 启动监控循环。"""
        try:
            from mcp.client.sse import sse_client
            from mcp import ClientSession

            server_url = "http://127.0.0.1:8100/sse"
            async with sse_client(server_url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "heartbeat_register_monitor",
                        {
                            "task_id": task_id,
                            "pid": pid,
                            "output_path": output_path,
                            "hard_timeout": self._hard_timeout,
                            "agent_id": "",
                        },
                    )
                    logger.info(f"[external_tool] MCP register_monitor result: {result}")
        except Exception as e:
            logger.warning(f"[external_tool] failed to register monitor via MCP: {e}")
            # TaskVault 已注册，最坏情况是没有主动监控，但 PID 文件仍在
            # 下次 TaskVault 实例化时 reconciliation 会处理

    async def __call__(self, state: dict) -> dict:
        if self._backend == "code_execution":
            if is_debug():
                cmd_str = state.get("execution_command", "")
                logger.debug(f"[external_tool] code_execution: cmd={cmd_str!r}")
            return await self._run_code_execution(state)

        cmd = [_substitute(s, state) for s in self._command]

        if is_debug():
            logger.debug(f"[external_tool] cmd={cmd} timeout={self._timeout}s inject_as={self._inject_as!r}")
        else:
            logger.debug("[external_tool] run: %s", cmd)

        return await self._run_tee(cmd)

    async def _run_tee(self, cmd: list[str]) -> dict:
        """tee + 120s 异步超时执行。"""
        try:
            proc, buffer, writer = self._tee_subprocess(cmd)
        except FileNotFoundError:
            raise RuntimeError(
                f"ExternalToolNode: '{cmd[0]}' not found in PATH. Please install it first."
            )

        task_id: str = proc._task_id  # type: ignore[attr-defined]
        output_path: str = proc._output_path  # type: ignore[attr-defined]
        tee_thread: threading.Thread = proc._tee_thread  # type: ignore[attr-defined]

        if is_debug():
            logger.debug(
                f"[external_tool] tee started: task_id={task_id} "
                f"async_threshold={_ASYNC_THRESHOLD}s hard_timeout={self._hard_timeout}s"
            )

        try:
            returncode = await asyncio.wait_for(
                asyncio.to_thread(proc.wait),
                timeout=_ASYNC_THRESHOLD,
            )
        except asyncio.TimeoutError:
            logger.info(f"[external_tool] 120s threshold reached for {task_id}")
            return await self._on_timeout(proc, task_id, output_path)

        # 子进程在 120s 内完成 — 正常路径
        tee_thread.join(timeout=5.0)

        output = buffer.getvalue().decode("utf-8", errors="replace").strip()

        if returncode != 0:
            output = f"[exit={returncode}] {output}"
            logger.warning("[external_tool] non-zero exit %s: %s", returncode, cmd)
        else:
            try:
                output = json.dumps(json.loads(output), ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, ValueError):
                pass

        if is_debug():
            logger.debug(
                f"[external_tool] exit={returncode} "
                f"output_len={len(output)} preview={output[:200]!r}"
            )

        writer.close()

        if self._inject_as == "message":
            return {
                "messages": [AIMessage(content=output)],
                "routing_target": "",
                "routing_context": "",
            }
        else:
            return {self._inject_as: output}
