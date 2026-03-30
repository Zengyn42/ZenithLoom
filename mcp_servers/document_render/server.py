"""
Document Render MCP Server — mcp_servers/document_render/server.py

Wraps Presenton (PDF slides) and Pandoc (DOCX documents) rendering into MCP tools.
Replicates the logic from:
  - skills/presenton/scripts/render_slides.sh
  - skills/pandoc/scripts/render_docs.sh

Tools:
  render_slides  — POST content to Presenton API, return PDF path
  render_docs    — Write Markdown to temp file, run pandoc, return DOCX path

Startup:
  python -m mcp_servers.document_render.server                  # stdio
  python -m mcp_servers.document_render.server --transport sse  # SSE
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("document_render_mcp")

# --- Presenton configuration ---
PRESENTON_URL = os.environ.get("PRESENTON_API_URL", "http://localhost:5000")
PRESENTON_CONTAINER = os.environ.get("PRESENTON_CONTAINER_NAME", "presenton")
PRESENTON_API_KEY = os.environ.get("PRESENTON_API_KEY", "")

# --- Pandoc configuration ---
_TEMPLATE_DIR = _PROJECT_ROOT / "skills" / "pandoc" / "templates"
_REFERENCE_DOC = _TEMPLATE_DIR / "professional.docx"


@asynccontextmanager
async def lifespan(server):
    """MCP Server lifecycle management."""
    logger.info("Document Render MCP started")
    logger.info(f"Registered tools: {len(server._tool_manager._tools)}")
    yield
    logger.info("Document Render MCP stopped")


mcp = FastMCP(
    name="document-render",
    instructions=(
        "Document rendering tools. "
        "render_slides: convert content to PDF slides via Presenton API. "
        "render_docs: convert Markdown to DOCX (or other formats) via Pandoc. "
        "Output files are saved to /tmp/ and the path is returned."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Presenton helpers
# ---------------------------------------------------------------------------

def _presenton_ready() -> bool:
    """Check if the Presenton API is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{PRESENTON_URL}/api/v1/ppt/presentation/all",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ensure_presenton_running() -> None:
    """Start the Presenton Docker container if it is not already running."""
    if _presenton_ready():
        return

    logger.info("Presenton not running. Starting Docker container...")

    # Check if container exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    container_names = result.stdout.strip().splitlines()

    if PRESENTON_CONTAINER not in container_names:
        raise RuntimeError(
            f"Presenton container '{PRESENTON_CONTAINER}' not found. "
            f"Hint: docker run -d --name {PRESENTON_CONTAINER} -p 5000:5000 presenton/presenton"
        )

    subprocess.run(
        ["docker", "start", PRESENTON_CONTAINER],
        capture_output=True, text=True,
    )

    # Wait up to 60 seconds for readiness
    for _ in range(30):
        if _presenton_ready():
            logger.info("Presenton is ready")
            return
        time.sleep(2)

    raise RuntimeError("Presenton container started but did not become ready within 60 seconds")


def _stop_presenton() -> None:
    """Stop the Presenton Docker container to conserve resources."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        if PRESENTON_CONTAINER in result.stdout.strip().splitlines():
            subprocess.run(
                ["docker", "stop", PRESENTON_CONTAINER],
                capture_output=True, text=True,
            )
            logger.info("Presenton container stopped")
    except Exception:
        pass


def _ensure_presenton_configured() -> None:
    """Ensure DISABLE_IMAGE_GENERATION is true in Presenton config."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{PRESENTON_URL}/api/user-config", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            cfg = json.loads(resp.read().decode())

        if cfg.get("DISABLE_IMAGE_GENERATION"):
            return

        cfg["DISABLE_IMAGE_GENERATION"] = True
        cfg.setdefault("IMAGE_PROVIDER", "")

        data = json.dumps(cfg).encode()
        req = urllib.request.Request(
            f"{PRESENTON_URL}/api/user-config",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception as exc:
        logger.warning(f"Could not verify/set Presenton config: {exc}")


# ---------------------------------------------------------------------------
# Background task runner (fire-and-forget)
# ---------------------------------------------------------------------------

def _run_in_background(fn, *args, **kwargs) -> tuple[int, str]:
    """Run fn(*args, **kwargs) in a daemon thread. Returns (pid, log_path).

    The thread PID is the current process PID (MCP server process) since we
    use threads, not subprocesses. The log_path file captures stdout/stderr
    written by fn via the logger.

    For heartbeat monitoring we write a sentinel file:
      {log_path}.done  → contains exit status JSON when complete
    """
    ts = int(time.time() * 1000)
    log_path = f"/tmp/render_task_{ts}.log"

    def _wrapper():
        try:
            result = fn(*args, **kwargs)
            with open(f"{log_path}.done", "w") as f:
                json.dump({"status": "success", **result}, f)
        except Exception as exc:
            with open(f"{log_path}.done", "w") as f:
                json.dump({"status": "error", "error": str(exc)}, f)

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()
    return os.getpid(), log_path


# ---------------------------------------------------------------------------
# Core rendering functions (synchronous, called from background thread)
# ---------------------------------------------------------------------------

def _render_slides_sync(content: str, output_path: str) -> dict:
    """Synchronous Presenton render — runs inside background thread."""
    _ensure_presenton_running()
    _ensure_presenton_configured()

    api_url = f"{PRESENTON_URL}/api/v1/ppt/presentation/generate"
    request_body = json.dumps({
        "content": content,
        "n_slides": 10,
        "language": "Chinese",
        "template": "general",
        "export_as": "pdf",
    }).encode()

    try:
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if PRESENTON_API_KEY:
            headers["Authorization"] = f"Bearer {PRESENTON_API_KEY}"

        req = urllib.request.Request(api_url, data=request_body, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())

        download_url = body.get("path", "")
        if not download_url:
            return {"error": f"No download URL in response: {body}"}

        if download_url.startswith("/"):
            encoded_path = quote(download_url)
            download_url = f"{PRESENTON_URL}{encoded_path}"

        req = urllib.request.Request(download_url)
        with urllib.request.urlopen(req) as resp:
            pdf_data = resp.read()

        with open(output_path, "wb") as f:
            f.write(pdf_data)

        file_size = os.path.getsize(output_path)
        logger.info(f"Generated slides: {output_path} ({file_size} bytes)")
        return {"file_path": output_path, "file_size": file_size}

    finally:
        _stop_presenton()


def _render_docs_sync(content: str, output_path: str, ext: str) -> dict:
    """Synchronous Pandoc render — runs inside background thread."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", prefix="doc_content_", dir="/tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)

        cmd = ["pandoc", tmp_path, "-o", output_path]
        if _REFERENCE_DOC.is_file() and ext == "docx":
            cmd.append(f"--reference-doc={_REFERENCE_DOC}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"error": f"pandoc failed (exit {result.returncode}): {result.stderr}"}

        if not os.path.isfile(output_path):
            return {"error": "pandoc did not produce an output file"}

        file_size = os.path.getsize(output_path)
        logger.info(f"Generated document: {output_path} ({file_size} bytes)")
        return {"file_path": output_path, "file_size": file_size}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def render_slides(content: str, filename: str = "") -> dict:
    """Render content to PDF slides via the Presenton API (async/background).

    This tool starts rendering immediately and returns a ``pending`` status
    with the task PID and output path. Because Presenton rendering can take
    minutes, call ``heartbeat_register_monitor`` with the returned ``pid`` and
    ``output_path`` to be notified when the file is ready.

    Args:
        content: The slide content text (Markdown or plain text).
        filename: Optional output filename stem (no extension).

    Returns:
        dict with ``status="pending"``, ``pid``, ``output_path``, ``done_path``,
        or ``status="error"`` if content is empty / Presenton is unavailable.
    """
    if not content.strip():
        return {"status": "error", "error": "Content is empty"}

    # Verify Presenton can start before firing background task
    try:
        _ensure_presenton_running()
        _ensure_presenton_configured()
    except RuntimeError as exc:
        return {"status": "error", "error": str(exc)}

    safe_name = Path(filename).stem if filename else f"presentation_{int(time.time())}"
    output_path = f"/tmp/{safe_name}.pdf"

    pid, log_path = _run_in_background(_render_slides_sync, content, output_path)

    logger.info(f"render_slides: started background task pid={pid} output={output_path}")
    return {
        "status": "pending",
        "pid": pid,
        "output_path": output_path,
        "done_path": f"{log_path}.done",
        "message": (
            "渲染已在后台开始。请调用 heartbeat_register_monitor 监视进度，"
            f"完成后文件保存至 {output_path}"
        ),
    }


@mcp.tool()
def render_docs(content: str, filename: str = "", format: str = "docx") -> dict:
    """Render Markdown content to DOCX (or another format) via Pandoc (async/background).

    Pandoc itself is fast, but may be slow for large documents or when
    generating PDFs via LaTeX. The tool fires the render in the background
    and returns a ``pending`` status immediately. Call
    ``heartbeat_register_monitor`` with the returned ``pid`` and ``output_path``
    to be notified when the document is ready.

    Args:
        content: Markdown content to render.
        filename: Optional output filename stem (no extension).
        format: Output format (default ``docx``). Any Pandoc output format.

    Returns:
        dict with ``status="pending"``, ``pid``, ``output_path``, ``done_path``,
        or ``status="error"`` if content is empty / pandoc not installed.
    """
    if not content.strip():
        return {"status": "error", "error": "Content is empty"}

    if not shutil.which("pandoc"):
        return {
            "status": "error",
            "error": "pandoc not installed. Run: sudo apt install pandoc",
        }

    ext = format if format else "docx"
    safe_name = Path(filename).stem if filename else f"document_{int(time.time())}"
    output_path = f"/tmp/{safe_name}.{ext}"

    pid, log_path = _run_in_background(_render_docs_sync, content, output_path, ext)

    logger.info(f"render_docs: started background task pid={pid} output={output_path}")
    return {
        "status": "pending",
        "pid": pid,
        "output_path": output_path,
        "done_path": f"{log_path}.done",
        "message": (
            "渲染已在后台开始。请调用 heartbeat_register_monitor 监视进度，"
            f"完成后文件保存至 {output_path}"
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Document Render MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host")
    parser.add_argument("--port", type=int, default=8104, help="SSE port")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    logger.info(f"Starting Document Render MCP Server (transport={args.transport})")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
