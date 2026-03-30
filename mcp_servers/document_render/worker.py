"""
Document Render Worker — mcp_servers/document_render/worker.py

Standalone subprocess worker for rendering documents. Launched by
render_slides / render_docs tools via subprocess.Popen so the caller
gets a real PID that heartbeat can monitor.

Usage:
  python -m mcp_servers.document_render.worker --type slides \
      --content-file /tmp/content.txt \
      --output /tmp/presentation.pdf \
      --done-path /tmp/task_123.done

  python -m mcp_servers.document_render.worker --type docs \
      --content-file /tmp/content.txt \
      --output /tmp/document.docx \
      --format docx \
      --done-path /tmp/task_123.done

Exit codes:
  0 — success
  1 — error (details in done-path JSON)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [render_worker] %(levelname)s %(message)s",
)
logger = logging.getLogger("render_worker")

PRESENTON_URL = os.environ.get("PRESENTON_API_URL", "http://localhost:5000")
PRESENTON_CONTAINER = os.environ.get("PRESENTON_CONTAINER_NAME", "presenton")
PRESENTON_API_KEY = os.environ.get("PRESENTON_API_KEY", "")
_TEMPLATE_DIR = _PROJECT_ROOT / "skills" / "pandoc" / "templates"
_REFERENCE_DOC = _TEMPLATE_DIR / "professional.docx"


def _write_done(done_path: str, result: dict) -> None:
    try:
        with open(done_path, "w") as f:
            json.dump(result, f)
    except Exception as e:
        logger.error(f"Failed to write done file: {e}")


def _presenton_ready() -> bool:
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{PRESENTON_URL}/api/v1/ppt/presentation/all", method="GET"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ensure_presenton_running() -> None:
    if _presenton_ready():
        return
    logger.info("Presenton not running. Starting Docker container...")
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    if PRESENTON_CONTAINER not in result.stdout.strip().splitlines():
        raise RuntimeError(
            f"Presenton container '{PRESENTON_CONTAINER}' not found."
        )
    subprocess.run(["docker", "start", PRESENTON_CONTAINER], capture_output=True)
    for _ in range(30):
        if _presenton_ready():
            return
        time.sleep(2)
    raise RuntimeError("Presenton did not become ready within 60 seconds")


def _ensure_presenton_configured() -> None:
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
            f"{PRESENTON_URL}/api/user-config", data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception as exc:
        logger.warning(f"Could not verify/set Presenton config: {exc}")


def _stop_presenton() -> None:
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
        )
        if PRESENTON_CONTAINER in result.stdout.strip().splitlines():
            subprocess.run(["docker", "stop", PRESENTON_CONTAINER], capture_output=True)
    except Exception:
        pass


def render_slides(content_file: str, output_path: str, done_path: str) -> None:
    content = Path(content_file).read_text(encoding="utf-8")
    try:
        _ensure_presenton_running()
        _ensure_presenton_configured()
    except RuntimeError as exc:
        _write_done(done_path, {"status": "error", "error": str(exc)})
        sys.exit(1)

    import urllib.request

    api_url = f"{PRESENTON_URL}/api/v1/ppt/presentation/generate"
    request_body = json.dumps({
        "content": content,
        "n_slides": 10,
        "language": "Chinese",
        "template": "general",
        "export_as": "pdf",
    }).encode()

    try:
        headers = {"Content-Type": "application/json"}
        if PRESENTON_API_KEY:
            headers["Authorization"] = f"Bearer {PRESENTON_API_KEY}"
        req = urllib.request.Request(api_url, data=request_body, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())

        download_url = body.get("path", "")
        if not download_url:
            _write_done(done_path, {"status": "error", "error": f"No download URL: {body}"})
            sys.exit(1)

        if download_url.startswith("/"):
            download_url = f"{PRESENTON_URL}{quote(download_url)}"

        req = urllib.request.Request(download_url)
        with urllib.request.urlopen(req) as resp:
            pdf_data = resp.read()

        with open(output_path, "wb") as f:
            f.write(pdf_data)

        file_size = os.path.getsize(output_path)
        logger.info(f"Slides rendered: {output_path} ({file_size} bytes)")
        _write_done(done_path, {
            "status": "success",
            "file_path": output_path,
            "file_size": file_size,
        })

    except Exception as exc:
        _write_done(done_path, {"status": "error", "error": str(exc)})
        sys.exit(1)
    finally:
        _stop_presenton()


def render_docs(content_file: str, output_path: str, done_path: str, fmt: str = "docx") -> None:
    content = Path(content_file).read_text(encoding="utf-8")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", prefix="doc_worker_", dir="/tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)

        cmd = ["pandoc", tmp_path, "-o", output_path]
        if _REFERENCE_DOC.is_file() and fmt == "docx":
            cmd.append(f"--reference-doc={_REFERENCE_DOC}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _write_done(done_path, {
                "status": "error",
                "error": f"pandoc failed (exit {result.returncode}): {result.stderr}",
            })
            sys.exit(1)

        if not os.path.isfile(output_path):
            _write_done(done_path, {"status": "error", "error": "pandoc produced no output"})
            sys.exit(1)

        file_size = os.path.getsize(output_path)
        logger.info(f"Doc rendered: {output_path} ({file_size} bytes)")
        _write_done(done_path, {
            "status": "success",
            "file_path": output_path,
            "file_size": file_size,
        })

    except Exception as exc:
        _write_done(done_path, {"status": "error", "error": str(exc)})
        sys.exit(1)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Document Render Worker")
    parser.add_argument("--type", choices=["slides", "docs"], required=True)
    parser.add_argument("--content-file", required=True, help="Temp file with render content")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--done-path", required=True, help="Sentinel file path")
    parser.add_argument("--format", default="docx", help="Output format (docs only)")
    args = parser.parse_args()

    if args.type == "slides":
        render_slides(args.content_file, args.output, args.done_path)
    else:
        render_docs(args.content_file, args.output, args.done_path, args.format)


if __name__ == "__main__":
    main()
