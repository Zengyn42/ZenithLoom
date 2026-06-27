"""
share_server.py — Static file server with HTTP Range support for media sharing.

Replaces python -m http.server for the media share pipeline.
Supports video seeking, partial content delivery, and proper MIME types.

Usage:
    python3 -m framework.share_server                # default port 8091
    python3 -m framework.share_server --port 8091
"""

import argparse
import mimetypes
import os
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.responses import Response, FileResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

SHARE_DIR = Path(os.environ.get(
    "ZL_SHARE_DIR", "/home/kingy/Foundation/EdenGateway/share"
))


def create_app(share_dir: Path) -> Starlette:
    """Create a Starlette app that serves static files with Range support."""
    share_dir.mkdir(parents=True, exist_ok=True)

    # StaticFiles from Starlette supports Range requests out of the box
    app = Starlette(
        routes=[
            Mount("/", app=StaticFiles(directory=str(share_dir), html=True)),
        ],
    )
    return app


def main():
    parser = argparse.ArgumentParser(description="Share server with Range support")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = create_app(SHARE_DIR)
    print(f"[share_server] Serving {SHARE_DIR} on {args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
