"""ExternalToolNode — EXTERNAL_TOOL node type.

Executes any external CLI as a LangGraph node.  Supports:
- CLI-Anything harnesses  (cli-anything-blender --json ...)
- Native structured CLIs  (gws gmail +triage --json)
- Official CLIs           (obsidian search --query ...)

node_config fields:
  command     list[str]  required  Base command + args; supports {field} templates
  timeout     float      30.0      Subprocess timeout in seconds
  inject_as   str        "message" "message" -> AIMessage; anything else -> state field name
  description str        ""        Human-readable label shown by !topology
  soft_timeout float     None      Seconds before backgrounding (None = disabled, use legacy timeout)
  hard_timeout float     300.0     Max background seconds before kill
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

_DEFAULT_SOFT_TIMEOUT = 120.0  # Agent 自己等 120s，超过才启动 heartbeat
_DEFAULT_HARD_TIMEOUT = 300.0


def _substitute(template: str, state: dict) -> str:
    """Replace {word} placeholders from state; leave all other {…} (e.g. JSON) untouched."""
    return re.sub(r"\{(\w+)\}", lambda m: str(state[m.group(1)]) if m.group(1) in state else m.group(0), template)


def _run_subprocess(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run cmd as an arg-list (no shell) with timeout. Raises on FileNotFoundError."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


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

    Async timeout pattern (soft_timeout enabled):
        子进程在 soft_timeout 内完成 → 正常返回
        soft_timeout 到达 → on_soft_timeout() → 注册后台监控 → 返回 PENDING 消息
        hard_timeout 到达 → kill 进程 → 标记 TIMEOUT
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

        # Async timeout 配置（None = 禁用，走原有 timeout 逻辑）
        raw_soft = inner.get("soft_timeout")
        self._soft_timeout: float | None = float(raw_soft) if raw_soft is not None else None
        self._hard_timeout: float = float(inner.get("hard_timeout", _DEFAULT_HARD_TIMEOUT))

    async def _run_code_execution(self, state: dict) -> dict:
        cmd_str: str = state.get("execution_command", "")
        if not cmd_str.strip():
            # execution_command 为空 → 跳过（code_gen 可能直接用工具写文件，无需额外命令）
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
        from mcp_servers.heartbeat.task_vault import _MONITORS_DIR as _TASKS_DIR

        task_id = f"tool_{uuid.uuid4().hex[:12]}"
        output_path = _TASKS_DIR / f"{task_id}.output"

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

        # 等待最多 10 秒让 server 启动
        for _ in range(20):
            time.sleep(0.5)
            if _is_server_running():
                logger.info("[external_tool] heartbeat MCP server started successfully")
                return

        logger.warning("[external_tool] heartbeat MCP server failed to start within 10s")

    def on_soft_timeout(
        self,
        proc: subprocess.Popen,
        task_id: str,
        output_path: str,
    ) -> dict:
        """soft_timeout 到达时的钩子。

        默认实现：
        1. 确保 Heartbeat MCP Server 正在运行（自动启动）
        2. 通过 Heartbeat TaskVault 注册后台监控
        3. 返回 PENDING ToolMessage

        子类可覆写（如 ClaudeSDKNode 仅打印进度提示）。

        Returns:
            LangGraph state dict，包含 PENDING 消息。
        """
        # 需求 6：确保 heartbeat MCP server 正在运行
        self._ensure_heartbeat_server()

        from mcp_servers.heartbeat.task_vault import TaskVault

        mgr = TaskVault.get_instance()
        mgr.register_task(
            task_id=task_id,
            pid=proc.pid,
            output_path=output_path,
            hard_timeout=self._hard_timeout,
        )

        pending_content = (
            f"[PENDING] 命令执行超过 {self._soft_timeout}s，已转入后台继续运行。\n"
            f"task_id: {task_id}\n"
            f"完成后系统会通知你。你可以继续其他工作。"
        )
        logger.info(
            f"[external_tool] soft_timeout: {task_id} pid={proc.pid} backgrounded"
        )

        if self._inject_as == "message":
            return {
                "messages": [AIMessage(content=pending_content, additional_kwargs={"task_id": task_id})],
                "routing_target": "",
                "routing_context": "",
            }
        else:
            return {self._inject_as: pending_content}

    def _build_error_result(self, error_msg: str) -> dict:
        """构建错误结果 dict。"""
        if self._inject_as == "message":
            return {
                "messages": [AIMessage(content=error_msg)],
                "routing_target": "",
                "routing_context": "",
            }
        else:
            return {self._inject_as: error_msg}

    async def __call__(self, state: dict) -> dict:
        if self._backend == "code_execution":
            if is_debug():
                cmd_str = state.get("execution_command", "")
                logger.debug(f"[external_tool] code_execution: cmd={cmd_str!r}")
            return await self._run_code_execution(state)
        # 1. Resolve {field} templates from state (only \w+ keys; JSON braces are left untouched)
        cmd = [_substitute(s, state) for s in self._command]

        if is_debug():
            logger.debug(f"[external_tool] cmd={cmd} timeout={self._timeout}s inject_as={self._inject_as!r}")
        else:
            logger.debug("[external_tool] run: %s", cmd)

        # ── soft_timeout 分支：tee + 异步超时 ──────────────────────────────
        if self._soft_timeout is not None:
            return await self._run_with_soft_timeout(cmd)

        # ── 原有同步路径（soft_timeout 未配置时保持不变）──────────────────
        # 2. Run subprocess in thread pool (arg-list, no shell)
        try:
            result = await asyncio.to_thread(_run_subprocess, cmd, self._timeout)
        except FileNotFoundError:
            raise RuntimeError(
                f"ExternalToolNode: '{cmd[0]}' not found in PATH. Please install it first."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ExternalToolNode: command timed out ({self._timeout}s): {cmd}"
            )

        # 3. Parse output
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            output = f"[exit={result.returncode}] {err or output}"
            logger.warning("[external_tool] non-zero exit %s: %s", result.returncode, cmd)
        else:
            # Pretty-print JSON when the tool honours a --json flag
            try:
                output = json.dumps(json.loads(output), ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, ValueError):
                pass  # keep as plain text

        if is_debug():
            logger.debug(
                f"[external_tool] exit={result.returncode} "
                f"output_len={len(output)} preview={output[:200]!r}"
            )

        # 4. Inject into state
        if self._inject_as == "message":
            return {
                "messages": [AIMessage(content=output)],
                "routing_target": "",
                "routing_context": "",
            }
        else:
            return {self._inject_as: output}

    async def _run_with_soft_timeout(self, cmd: list[str]) -> dict:
        """tee + soft_timeout 异步执行路径。"""
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
                f"soft_timeout={self._soft_timeout}s hard_timeout={self._hard_timeout}s"
            )

        # 等待子进程完成或 soft_timeout 到达
        try:
            returncode = await asyncio.wait_for(
                asyncio.to_thread(proc.wait),
                timeout=self._soft_timeout,
            )
        except asyncio.TimeoutError:
            # soft_timeout 到达 — 不 kill，交给 on_soft_timeout 处理
            logger.info(
                f"[external_tool] soft_timeout reached ({self._soft_timeout}s) for {task_id}"
            )
            return self.on_soft_timeout(proc, task_id, output_path)

        # 子进程在 soft_timeout 内完成 — 正常路径
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

        # 清理 writer（tee_reader 线程已关闭，但以防万一）
        writer.close()

        if self._inject_as == "message":
            return {
                "messages": [AIMessage(content=output)],
                "routing_target": "",
                "routing_context": "",
            }
        else:
            return {self._inject_as: output}
