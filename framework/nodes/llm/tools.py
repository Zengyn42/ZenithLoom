"""Tool registry for OllamaNode tool-calling loop.

Tools:
  read_file          — read a file from the filesystem
  write_file         — write content to a file
  bash_exec          — run a command (arg-list form via shlex.split, no shell)
  list_dir           — list directory entries
  submit_validation  — structured validation output (terminates the tool loop)

TOOL_REGISTRY   dict[name → async callable]
TOOL_SCHEMAS    dict[name → OpenAI-style function schema]
build_tool_schemas(names) → list of schemas for use in /v1/chat/completions payload
"""

import asyncio
import json
import shlex
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def read_file(path: str) -> dict:
    try:
        content = Path(path).read_text(encoding="utf-8")
        lines = content.splitlines()
        numbered = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))
        return {"content": numbered, "total_lines": len(lines)}
    except Exception as exc:
        return {"error": str(exc)}


async def replace_lines(
    path: str, start_line: int, end_line: int, new_content: str
) -> dict:
    """Replace lines start_line..end_line (1-indexed, inclusive) with new_content."""
    try:
        p = Path(path)
        lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
        total = len(lines)
        if start_line < 1 or end_line > total or start_line > end_line:
            return {
                "error": f"Invalid range {start_line}-{end_line}. "
                f"File has {total} lines (1-indexed)."
            }
        new_lines = new_content.splitlines(keepends=True)
        # Ensure trailing newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        lines[start_line - 1 : end_line] = new_lines
        p.write_text("".join(lines), encoding="utf-8")
        return {
            "replaced": True,
            "old_line_count": end_line - start_line + 1,
            "new_line_count": len(new_lines),
            "total_lines": len(lines),
        }
    except Exception as exc:
        return {"error": str(exc)}


async def write_file(path: str, content: str) -> dict:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return {"written": True}
    except Exception as exc:
        return {"error": str(exc)}


def _exec_cmd(args: list[str], timeout: int, cwd: str | None = None) -> dict:
    """Synchronous helper: run args (no shell) and capture output."""
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or None,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"timeout after {timeout}s", "returncode": -1}
    except FileNotFoundError as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": 127}


async def bash_exec(command: str, timeout: int = 30, cwd: str = "") -> dict:
    """Execute a shell command. Use cwd to set the working directory."""
    args = shlex.split(command)
    return await asyncio.to_thread(_exec_cmd, args, timeout, cwd or None)


async def list_dir(path: str) -> dict:
    try:
        entries = sorted(str(p) for p in Path(path).iterdir())
        return {"entries": entries}
    except Exception as exc:
        return {"error": str(exc)}


async def submit_validation(
    status: str,
    category: str,
    severity: str,
    rationale: str,
    affected_scope: str = "",
    is_regression: bool = False,
    raw_stderr: str = "",
) -> dict:
    """Structured validation output — signals the tool loop to terminate."""
    return {
        "status": status,
        "category": category,
        "severity": severity,
        "rationale": rationale,
        "affected_scope": affected_scope,
        "is_regression": is_regression,
        "raw_stderr": raw_stderr,
        "_terminal": True,
    }


# ---------------------------------------------------------------------------
# Registry + schemas
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict = {
    "read_file": read_file,
    "write_file": write_file,
    "replace_lines": replace_lines,
    "bash_exec": bash_exec,
    "list_dir": list_dir,
    "submit_validation": submit_validation,
}

TOOL_SCHEMAS: dict = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute or relative file path"}},
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates parent dirs if needed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "replace_lines": {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": (
                "Replace a range of lines in a file. Lines are 1-indexed, inclusive. "
                "IMPORTANT: Always read_file first to get current line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "start_line": {"type": "integer", "description": "First line to replace (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to replace (inclusive)"},
                    "new_content": {"type": "string", "description": "Replacement content"},
                },
                "required": ["path", "start_line", "end_line", "new_content"],
            },
        },
    },
    "bash_exec": {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Execute a shell command and return stdout/stderr/returncode. Use cwd to set the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 30},
                    "cwd": {"type": "string", "description": "Working directory for the command"},
                },
                "required": ["command"],
            },
        },
    },
    "list_dir": {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List all entries in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "submit_validation": {
        "type": "function",
        "function": {
            "name": "submit_validation",
            "description": "Submit a structured validation result (terminates the tool loop)",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["pass", "fail", "abort"]},
                    "category": {"type": "string", "description": "Error category (e.g. syntax_error, cross_task)"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "rationale": {"type": "string", "description": "Human-readable explanation"},
                    "affected_scope": {"type": "string", "description": "Comma-separated task IDs affected"},
                    "is_regression": {"type": "boolean"},
                    "raw_stderr": {"type": "string"},
                },
                "required": ["status", "category", "severity", "rationale"],
            },
        },
    },
}


def build_tool_schemas(tool_names: list) -> list:
    """Return a list of tool schemas for the given tool names (order preserved)."""
    return [TOOL_SCHEMAS[n] for n in tool_names if n in TOOL_SCHEMAS]
