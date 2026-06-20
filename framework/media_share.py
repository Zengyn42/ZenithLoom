"""
media_share.py — Share local media files via Tailscale Funnel.

Usage:
    python3 -m framework.media_share /path/to/video.mp4 [/path/to/image.png ...]
    python3 -m framework.media_share --title "Results" /path/to/file1 /path/to/file2

Outputs the public URL to stdout. The HTTP server is auto-started if not already running.

Cleanup:
    python3 -m framework.media_share --cleanup          # list all shares
    python3 -m framework.media_share --cleanup --all    # remove all shares
    python3 -m framework.media_share --cleanup <uuid>   # remove specific share
"""

import argparse
import datetime
import os
import shutil
import signal
import socket
import subprocess
import sys
import uuid
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
SHARE_DIR = Path(os.environ.get(
    "ZL_SHARE_DIR", "/home/kingy/Foundation/EdenGateway/share"
))
SHARE_PORT = int(os.environ.get("ZL_SHARE_PORT", "8091"))
PUBLIC_BASE = os.environ.get(
    "ZL_SHARE_URL", "https://kingy.taile5f3af.ts.net/share"
)

# ── Media type detection ─────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def _media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return "file"


def _mime_hint(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    }
    return mapping.get(ext, "application/octet-stream")


# ── HTML generation ──────────────────────────────────────────────────────────
def _generate_html(title: str, media_items: list[dict]) -> str:
    """Generate a responsive HTML page embedding the given media files."""
    cards = []
    for item in media_items:
        name = item["name"]
        mtype = item["type"]
        mime = item["mime"]
        size_mb = item["size_mb"]
        label = f'{name} <span class="meta">({size_mb:.1f} MB)</span>'

        if mtype == "video":
            card = f'''
            <div class="card">
                <div class="label">{label}</div>
                <video controls preload="metadata">
                    <source src="{name}" type="{mime}">
                    Your browser does not support video playback.
                </video>
            </div>'''
        elif mtype == "image":
            card = f'''
            <div class="card">
                <div class="label">{label}</div>
                <img src="{name}" alt="{name}" loading="lazy">
            </div>'''
        elif mtype == "audio":
            card = f'''
            <div class="card">
                <div class="label">{label}</div>
                <audio controls preload="metadata">
                    <source src="{name}" type="{mime}">
                </audio>
            </div>'''
        else:
            card = f'''
            <div class="card">
                <div class="label">{label}</div>
                <a href="{name}" class="download">Download {name}</a>
            </div>'''
        cards.append(card)

    cards_html = "\n".join(cards)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #e6edf3;
    min-height: 100vh; padding: 2rem 1rem;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; color: #58a6ff; }}
  .timestamp {{ font-size: 0.85rem; color: #8b949e; margin-bottom: 2rem; }}
  .card {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 1rem; margin-bottom: 1.5rem;
  }}
  .label {{
    font-size: 0.9rem; color: #c9d1d9; margin-bottom: 0.75rem;
    word-break: break-all;
  }}
  .meta {{ color: #8b949e; font-size: 0.8rem; }}
  video, img {{
    width: 100%; border-radius: 4px; display: block;
    max-height: 80vh; object-fit: contain;
    background: #010409;
  }}
  audio {{ width: 100%; }}
  .download {{
    display: inline-block; padding: 0.5rem 1rem;
    background: #21262d; color: #58a6ff; border-radius: 6px;
    text-decoration: none; border: 1px solid #30363d;
  }}
  .download:hover {{ background: #30363d; }}
  .footer {{
    text-align: center; color: #484f58; font-size: 0.75rem;
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #21262d;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <div class="timestamp">Shared at {now}</div>
  {cards_html}
  <div class="footer">Served by ZenithLoom via Tailscale Funnel</div>
</div>
</body>
</html>'''


# ── HTTP server management ───────────────────────────────────────────────────
def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_server(share_dir: Path, port: int) -> None:
    """Start a background HTTP server if not already running."""
    if _is_port_in_use(port):
        return  # server already running

    share_dir.mkdir(parents=True, exist_ok=True)
    pid_file = share_dir / ".server.pid"

    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port),
         "--directory", str(share_dir), "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    pid_file.write_text(str(proc.pid))
    print(f"[media_share] HTTP server started on port {port} (PID {proc.pid})",
          file=sys.stderr)


# ── Core share function ──────────────────────────────────────────────────────
def share(
    file_paths: list[str],
    title: str = "Shared Media",
) -> str:
    """
    Create a share page for the given files.
    Returns the public URL.
    """
    share_id = uuid.uuid4().hex[:8]
    share_subdir = SHARE_DIR / share_id
    share_subdir.mkdir(parents=True, exist_ok=True)

    media_items = []
    for fpath_str in file_paths:
        fpath = Path(fpath_str).resolve()
        if not fpath.exists():
            print(f"[media_share] WARNING: file not found: {fpath}", file=sys.stderr)
            continue

        # Symlink into share directory
        link_name = fpath.name
        link_path = share_subdir / link_name

        # Handle name collision
        if link_path.exists():
            stem = fpath.stem
            suffix = fpath.suffix
            link_name = f"{stem}_{share_id[:4]}{suffix}"
            link_path = share_subdir / link_name

        os.symlink(fpath, link_path)

        size_mb = fpath.stat().st_size / (1024 * 1024)
        media_items.append({
            "name": link_name,
            "type": _media_type(fpath),
            "mime": _mime_hint(fpath),
            "size_mb": size_mb,
        })

    if not media_items:
        shutil.rmtree(share_subdir, ignore_errors=True)
        raise FileNotFoundError("No valid files to share")

    # Generate HTML
    html = _generate_html(title, media_items)
    (share_subdir / "index.html").write_text(html, encoding="utf-8")

    # Ensure server is running
    _start_server(SHARE_DIR, SHARE_PORT)

    url = f"{PUBLIC_BASE}/{share_id}/"
    return url


# ── Cleanup ──────────────────────────────────────────────────────────────────
def cleanup(share_id: str | None = None, remove_all: bool = False) -> None:
    """List or remove shares."""
    if not SHARE_DIR.exists():
        print("No shares found.")
        return

    subdirs = sorted(
        [d for d in SHARE_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if share_id:
        target = SHARE_DIR / share_id
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed: {share_id}")
        else:
            print(f"Not found: {share_id}")
        return

    if remove_all:
        for d in subdirs:
            shutil.rmtree(d)
        print(f"Removed {len(subdirs)} share(s).")
        return

    # List mode
    if not subdirs:
        print("No shares found.")
        return

    print(f"{'ID':<12} {'Created':<20} {'Files':<6} URL")
    print("-" * 70)
    for d in subdirs:
        mtime = datetime.datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        files = [f for f in d.iterdir() if f.name != "index.html"]
        url = f"{PUBLIC_BASE}/{d.name}/"
        print(f"{d.name:<12} {mtime:<20} {len(files):<6} {url}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Share local media via Tailscale Funnel")
    parser.add_argument("files", nargs="*", help="File paths to share")
    parser.add_argument("--title", default="Shared Media", help="Page title")
    parser.add_argument("--cleanup", action="store_true", help="List or remove shares")
    parser.add_argument("--all", action="store_true", help="Remove all shares (with --cleanup)")
    parser.add_argument("--remove", metavar="ID", help="Remove specific share by ID")

    args = parser.parse_args()

    if args.cleanup or args.remove:
        cleanup(share_id=args.remove, remove_all=args.all)
        return

    if not args.files:
        parser.print_help()
        sys.exit(1)

    try:
        url = share(args.files, title=args.title)
        # Print URL to stdout (this is what the agent captures)
        print(url)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
