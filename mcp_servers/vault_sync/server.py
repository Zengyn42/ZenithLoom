"""
Vault Sync MCP Server — mcp_servers/vault_sync/server.py

Wraps rsync operations for syncing the Obsidian Vault between Windows and WSL.
Provides pull (Windows -> WSL), push (WSL -> Windows), and dry-run status tools.

Tools:
  vault_sync_pull   - Sync vault from Windows to WSL
  vault_sync_push   - Sync vault from WSL to Windows
  vault_sync_status - Dry-run both directions to show pending changes

Startup:
  python -m mcp_servers.vault_sync.server                    # stdio
  python -m mcp_servers.vault_sync.server --transport sse    # SSE (multi-client)

Configuration (env vars):
  VAULT_WSL_PATH  - WSL-side vault path   (default: /home/kingy/Foundation/Vault/)
  VAULT_WIN_PATH  - Windows-side vault path (default: /mnt/d/KnowledgeBase/Vault/)
"""

import argparse
import logging
import os
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
logger = logging.getLogger("vault_sync_mcp")

# Configurable paths via environment variables
_VAULT_WSL_PATH = os.environ.get("VAULT_WSL_PATH", "/home/kingy/Foundation/Vault/")
_VAULT_WIN_PATH = os.environ.get("VAULT_WIN_PATH", "/mnt/d/KnowledgeBase/Vault/")

# Ensure trailing slashes (rsync semantics)
if not _VAULT_WSL_PATH.endswith("/"):
    _VAULT_WSL_PATH += "/"
if not _VAULT_WIN_PATH.endswith("/"):
    _VAULT_WIN_PATH += "/"


@asynccontextmanager
async def lifespan(server):
    """MCP Server lifecycle management."""
    logger.info(
        f"Vault Sync MCP started — WSL: {_VAULT_WSL_PATH}, Windows: {_VAULT_WIN_PATH}"
    )
    logger.info(f"Registered tools: {len(server._tool_manager._tools)}")
    yield
    logger.info("Vault Sync MCP shutting down")


mcp = FastMCP(
    name="vault-sync",
    instructions=(
        "Vault sync tools for rsync-based synchronisation between Windows and WSL. "
        "Use vault_sync_pull to copy from Windows to WSL, vault_sync_push to copy "
        "from WSL to Windows, and vault_sync_status to preview changes without syncing."
    ),
    lifespan=lifespan,
)


def _run_rsync(args: list[str], timeout: int = 60) -> dict:
    """Execute an rsync command and return structured results."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Parse rsync output for a summary
        stdout_lines = result.stdout.strip().splitlines() if result.stdout else []
        stderr_lines = result.stderr.strip().splitlines() if result.stderr else []

        # Filter out blank lines for the file list
        changed_files = [
            line for line in stdout_lines
            if line and not line.startswith("sending ")
            and not line.startswith("sent ")
            and not line.startswith("total size")
            and not line.startswith("receiving ")
            and not line.startswith("building file list")
            and not line == "./"
        ]

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "files_changed": len(changed_files),
            "changed": changed_files[:100],  # cap at 100 to avoid huge responses
            "summary": (
                f"{len(changed_files)} file(s) changed"
                if result.returncode == 0
                else f"rsync failed (exit {result.returncode})"
            ),
            "stderr": "\n".join(stderr_lines) if stderr_lines else None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "return_code": -1,
            "files_changed": 0,
            "changed": [],
            "summary": f"rsync timed out after {timeout}s",
            "stderr": "Process timed out",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "return_code": -1,
            "files_changed": 0,
            "changed": [],
            "summary": "rsync not found — is rsync installed?",
            "stderr": "rsync binary not found on PATH",
        }
    except Exception as exc:
        return {
            "success": False,
            "return_code": -1,
            "files_changed": 0,
            "changed": [],
            "summary": f"Unexpected error: {exc}",
            "stderr": str(exc),
        }


@mcp.tool()
def vault_sync_pull() -> dict:
    """Sync vault from Windows to WSL (rsync -a --delete Windows -> WSL).

    Copies the Obsidian Vault from the Windows mount to the WSL filesystem.
    Uses --delete to mirror exactly, removing files in WSL that no longer exist
    on the Windows side.
    """
    logger.info(f"vault_sync_pull: {_VAULT_WIN_PATH} -> {_VAULT_WSL_PATH}")
    result = _run_rsync([
        "rsync", "-a", "--delete",
        _VAULT_WIN_PATH,
        _VAULT_WSL_PATH,
    ])
    result["direction"] = "pull (Windows -> WSL)"
    result["source"] = _VAULT_WIN_PATH
    result["destination"] = _VAULT_WSL_PATH
    return result


@mcp.tool()
def vault_sync_push() -> dict:
    """Sync vault from WSL to Windows (rsync -a --delete WSL -> Windows).

    Copies the Obsidian Vault from the WSL filesystem to the Windows mount.
    Uses --delete to mirror exactly, removing files on Windows that no longer
    exist on the WSL side.
    """
    logger.info(f"vault_sync_push: {_VAULT_WSL_PATH} -> {_VAULT_WIN_PATH}")
    result = _run_rsync([
        "rsync", "-a", "--delete",
        _VAULT_WSL_PATH,
        _VAULT_WIN_PATH,
    ])
    result["direction"] = "push (WSL -> Windows)"
    result["source"] = _VAULT_WSL_PATH
    result["destination"] = _VAULT_WIN_PATH
    return result


@mcp.tool()
def vault_sync_status() -> dict:
    """Dry-run both sync directions to show what would change without syncing.

    Returns a preview of which files would be added, updated, or deleted
    in each direction, without making any actual changes.
    """
    logger.info("vault_sync_status: dry-run both directions")

    pull_result = _run_rsync([
        "rsync", "-a", "--delete", "--dry-run",
        _VAULT_WIN_PATH,
        _VAULT_WSL_PATH,
    ])
    pull_result["direction"] = "pull (Windows -> WSL)"

    push_result = _run_rsync([
        "rsync", "-a", "--delete", "--dry-run",
        _VAULT_WSL_PATH,
        _VAULT_WIN_PATH,
    ])
    push_result["direction"] = "push (WSL -> Windows)"

    return {
        "pull_preview": pull_result,
        "push_preview": push_result,
        "note": "Dry-run only — no files were changed.",
    }


def main():
    parser = argparse.ArgumentParser(description="Vault Sync MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host")
    parser.add_argument("--port", type=int, default=8105, help="SSE port")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    logger.info(f"Starting Vault Sync MCP Server (transport={args.transport})")
    logger.info(f"  WSL path: {_VAULT_WSL_PATH}")
    logger.info(f"  Windows path: {_VAULT_WIN_PATH}")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
