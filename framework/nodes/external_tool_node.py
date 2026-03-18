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
"""

import asyncio
import json
import logging
import re
import shlex
import subprocess

from langchain_core.messages import AIMessage

from framework.config import AgentConfig

logger = logging.getLogger(__name__)


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

    async def _run_code_execution(self, state: dict) -> dict:
        cmd_str: str = state.get("execution_command", "")
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

    async def __call__(self, state: dict) -> dict:
        if self._backend == "code_execution":
            return await self._run_code_execution(state)
        # 1. Resolve {field} templates from state (only \w+ keys; JSON braces are left untouched)
        cmd = [_substitute(s, state) for s in self._command]

        logger.debug("[external_tool] run: %s", cmd)

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

        # 4. Inject into state
        if self._inject_as == "message":
            return {
                "messages": [AIMessage(content=output)],
                "routing_target": "",
                "routing_context": "",
            }
        else:
            return {self._inject_as: output}
