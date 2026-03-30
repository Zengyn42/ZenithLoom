"""
Google Workspace MCP Server — mcp_servers/google_workspace/server.py

Wraps the existing `gws` CLI tool to provide Gmail, Drive, Slides, and Docs
operations as MCP tools. All commands are validated against shell injection.

Tools:
  gws_gmail_read   - Read Gmail messages
  gws_drive_list   - List Google Drive files
  gws_slides_exec  - Execute Google Slides operations
  gws_docs_exec    - Execute Google Docs operations

Startup:
  python -m mcp_servers.google_workspace.server                   # stdio
  python -m mcp_servers.google_workspace.server --transport sse   # SSE
"""

import argparse
import logging
import re
import shlex
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("google_workspace_mcp")

# Shell metacharacters that are forbidden in commands
_DANGEROUS_CHARS = re.compile(r'[\$`|;&><]')


@asynccontextmanager
async def lifespan(server):
    """MCP Server lifecycle management."""
    logger.info("Google Workspace MCP starting")
    logger.info(f"Registered tools: {len(server._tool_manager._tools)}")
    yield
    logger.info("Google Workspace MCP shutting down")


mcp = FastMCP(
    name="google-workspace",
    instructions=(
        "Google Workspace tools. "
        "Provides Gmail read, Drive listing, Slides and Docs operations "
        "via the gws CLI. All commands are validated for safety."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_command(cmd: str, required_prefix: str) -> str | None:
    """Validate a gws command string.

    Returns an error message if invalid, None if OK.
    """
    stripped = cmd.strip()
    if not stripped.startswith(required_prefix):
        return f"Command must start with '{required_prefix}'. Got: {stripped[:60]}"
    if _DANGEROUS_CHARS.search(stripped):
        return "Shell metacharacters ($ ` | ; & > <) are not allowed in gws commands."
    return None


def _run_gws(args: list[str], timeout: int = 60) -> dict:
    """Run a gws command as a subprocess and return structured output."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout}s",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "gws CLI not found. Ensure it is installed and on PATH.",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def gws_gmail_read(query: str = "is:unread") -> dict:
    """Read Gmail messages matching a search query.

    Uses `gws gmail list` to find messages and returns their content.
    The query uses standard Gmail search syntax (e.g. "is:unread",
    "from:alice@example.com", "subject:meeting").
    """
    error = _validate_command(f"gws gmail list {query}", "gws gmail")
    if error:
        return {"success": False, "error": error}

    # List messages matching the query
    list_result = _run_gws(["gws", "gmail", "list", "--query", query])
    return list_result


@mcp.tool()
def gws_drive_list(folder: str = "") -> dict:
    """List files in Google Drive.

    Optionally specify a folder name or ID to scope the listing.
    If empty, lists files in the root/default location.
    """
    if folder and _DANGEROUS_CHARS.search(folder):
        return {
            "success": False,
            "error": "Shell metacharacters are not allowed in folder parameter.",
        }

    args = ["gws", "drive", "list"]
    if folder:
        args.extend(["--folder", folder])

    return _run_gws(args)


@mcp.tool()
def gws_slides_exec(command: str) -> dict:
    """Execute a Google Slides operation via the gws CLI.

    The command must be a full gws slides command string,
    e.g. "gws slides list" or "gws slides export --id <id> --format pdf".

    Shell metacharacters ($ ` | ; & > <) are blocked for security.
    """
    error = _validate_command(command, "gws slides")
    if error:
        return {"success": False, "error": error}

    args = shlex.split(command)
    return _run_gws(args)


@mcp.tool()
def gws_docs_exec(command: str) -> dict:
    """Execute a Google Docs operation via the gws CLI.

    The command must be a full gws docs command string,
    e.g. "gws docs list" or "gws docs get --id <doc_id>".

    Shell metacharacters ($ ` | ; & > <) are blocked for security.
    """
    error = _validate_command(command, "gws docs")
    if error:
        return {"success": False, "error": error}

    args = shlex.split(command)
    return _run_gws(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Google Workspace MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host")
    parser.add_argument("--port", type=int, default=8103, help="SSE port")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    logger.info(f"Starting Google Workspace MCP Server (transport={args.transport})")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
