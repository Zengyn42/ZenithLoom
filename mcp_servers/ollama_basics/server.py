"""
Ollama Basics MCP Server — mcp_servers/ollama_basics/server.py

Provides basic operating capabilities for Ollama nodes: file operations,
bash execution, file search, content grep, and web fetching.
These are abilities that Claude and Gemini have natively but Ollama models
need via tool-calling.

Tools:
  read_file       — read a file's contents
  write_file      — write content to a file
  list_directory   — list files and directories
  run_command      — execute a bash command
  search_files     — search for files by glob pattern
  grep_content     — search file contents using regex
  web_fetch        — fetch a URL and return text content

Startup:
  python -m mcp_servers.ollama_basics                              # stdio
  python -m mcp_servers.ollama_basics --transport sse              # SSE
  python -m mcp_servers.ollama_basics --transport sse --port 8102  # SSE on port

Security:
  - run_command: enforced timeout (max 120s), captured output
  - read_file/write_file: no path restrictions beyond OS-level
  - web_fetch: http/https only, 15s timeout, 50K char limit
"""

import argparse
import asyncio
import logging
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("ollama_basics_mcp")

MAX_FILE_CHARS = 50_000
MAX_WEB_CHARS = 50_000
MAX_TIMEOUT = 120


@asynccontextmanager
async def lifespan(server):
    """MCP Server lifecycle management."""
    logger.info("Ollama Basics MCP starting")
    logger.info(f"Registered tools: {len(server._tool_manager._tools)}")
    yield
    logger.info("Ollama Basics MCP shutting down")


mcp = FastMCP(
    name="ollama-basics",
    instructions=(
        "Basic operating tools for Ollama LLM nodes. "
        "Provides file read/write, directory listing, bash execution, "
        "file search, content grep, and web fetching capabilities."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def read_file(path: str) -> dict:
    """Read a file's contents from the filesystem.

    Args:
        path: Absolute or relative file path to read.

    Returns:
        dict with ok, content, line_count (or ok=False with error).
    """
    try:
        p = Path(path).resolve()
        if not p.is_file():
            return {"ok": False, "error": f"Not a file or does not exist: {path}"}
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS]
            truncated = True
        else:
            truncated = False
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return {
            "ok": True,
            "content": content,
            "line_count": line_count,
            "truncated": truncated,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
async def write_file(path: str, content: str) -> dict:
    """Write content to a file, creating parent directories if needed.

    Args:
        path: Absolute or relative file path to write.
        content: The text content to write.

    Returns:
        dict with ok, bytes_written (or ok=False with error).
    """
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True, "bytes_written": len(content.encode("utf-8"))}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
async def list_directory(path: str = ".") -> dict:
    """List files and directories at the given path.

    Args:
        path: Directory path to list (default: current directory).

    Returns:
        dict with ok, entries (list of {name, type, size}).
    """
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return {"ok": False, "error": f"Not a directory: {path}"}
        entries = []
        for item in sorted(p.iterdir(), key=lambda x: x.name):
            entry = {"name": item.name}
            if item.is_dir():
                entry["type"] = "dir"
                entry["size"] = None
            else:
                entry["type"] = "file"
                try:
                    entry["size"] = item.stat().st_size
                except OSError:
                    entry["size"] = None
            entries.append(entry)
        return {"ok": True, "entries": entries, "count": len(entries)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _run_cmd(command: str, timeout: int) -> dict:
    """Synchronous helper: run command via shell and capture output."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": True,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "exit_code": -1,
        }
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc), "exit_code": -1}


@mcp.tool()
async def run_command(command: str, timeout: int = 30) -> dict:
    """Execute a bash command and return its output.

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds (default 30, max 120).

    Returns:
        dict with ok, stdout, stderr, exit_code.
    """
    timeout = max(1, min(timeout, MAX_TIMEOUT))
    return await asyncio.to_thread(_run_cmd, command, timeout)


@mcp.tool()
async def search_files(pattern: str, path: str = ".", max_results: int = 20) -> dict:
    """Search for files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g. "**/*.py", "*.json").
        path: Root directory to search from (default: current directory).
        max_results: Maximum number of results to return (default 20).

    Returns:
        dict with ok, matches (list of file paths), count.
    """
    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return {"ok": False, "error": f"Not a directory: {path}"}
        matches = []
        for match in p.glob(pattern):
            matches.append(str(match))
            if len(matches) >= max_results:
                break
        return {"ok": True, "matches": matches, "count": len(matches)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
async def grep_content(
    pattern: str,
    path: str = ".",
    file_glob: str = "*",
    max_results: int = 20,
) -> dict:
    """Search file contents using regex pattern.

    Args:
        pattern: Regular expression pattern to search for.
        path: Root directory to search in (default: current directory).
        file_glob: Glob pattern to filter which files to search (default: "*").
        max_results: Maximum number of matching lines to return (default 20).

    Returns:
        dict with ok, results (list of {file, line_number, line}).
    """
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return {"ok": False, "error": f"Invalid regex: {exc}"}

    try:
        p = Path(path).resolve()
        if not p.is_dir():
            return {"ok": False, "error": f"Not a directory: {path}"}

        results = []
        for filepath in p.rglob(file_glob):
            if not filepath.is_file():
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append({
                        "file": str(filepath),
                        "line_number": i,
                        "line": line[:500],
                    })
                    if len(results) >= max_results:
                        return {"ok": True, "results": results, "count": len(results)}
        return {"ok": True, "results": results, "count": len(results)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for web pages."""
    # Remove script and style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_url(url: str) -> dict:
    """Synchronous helper: fetch URL content."""
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "error": "Only http:// and https:// URLs are supported"}
    try:
        req = Request(url, headers={"User-Agent": "OllamaBasicsMCP/1.0"})
        with urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8", errors="replace")

            if "html" in content_type.lower():
                text = _strip_html(raw)
            else:
                text = raw

            if len(text) > MAX_WEB_CHARS:
                text = text[:MAX_WEB_CHARS]
                truncated = True
            else:
                truncated = False

            return {
                "ok": True,
                "content": text,
                "content_type": content_type,
                "length": len(text),
                "truncated": truncated,
            }
    except URLError as exc:
        return {"ok": False, "error": f"URL error: {exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
async def web_fetch(url: str) -> dict:
    """Fetch a URL and return its text content.

    For HTML pages, tags are stripped to return plain text.
    Only http:// and https:// URLs are supported.

    Args:
        url: The URL to fetch.

    Returns:
        dict with ok, content, content_type, length, truncated.
    """
    return await asyncio.to_thread(_fetch_url, url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ollama Basics MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host")
    parser.add_argument("--port", type=int, default=8102, help="SSE port")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    logger.info(f"Starting Ollama Basics MCP Server (transport={args.transport})")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
