"""
media_share.py — Share local media files via Tailscale Funnel.

Usage:
    python3 -m framework.media_share --project GenesisExp /path/to/video.mp4
    python3 -m framework.media_share --project GenesisExp --title "Round 3" --context "..." file1 file2

Outputs the public URL to stdout. The HTTP server is auto-started if not already running.

Directory structure:
    share/
      GenesisExp/
        a1b2c3d4/
          index.html          ← preview page
          context.md           ← background notes
          video.mp4            ← copied media file
          image.png            ← copied media file

Cleanup:
    python3 -m framework.media_share --cleanup                    # list all shares
    python3 -m framework.media_share --cleanup --project X        # list shares in project X
    python3 -m framework.media_share --cleanup --all              # remove all shares
    python3 -m framework.media_share --remove <project/uuid>      # remove specific share
"""

import argparse
import datetime
import os
import shutil
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
def _generate_html(
    title: str,
    media_items: list[dict],
    context: str = "",
    project: str = "",
) -> str:
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
    project_badge = f'<span class="badge">{project}</span> ' if project else ""

    # Context section (background knowledge)
    context_html = ""
    if context:
        # Escape HTML in context, preserve newlines
        import html as html_mod
        escaped = html_mod.escape(context)
        escaped = escaped.replace("\n", "<br>")
        context_html = f'''
        <div class="context">
            <div class="context-title">Background</div>
            <div class="context-body">{escaped}</div>
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{project_badge}{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #e6edf3;
    min-height: 100vh; padding: 2rem 1rem;
  }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; color: #58a6ff; }}
  .badge {{
    display: inline-block; background: #1f6feb; color: #fff;
    padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.8rem;
    vertical-align: middle; margin-right: 0.3rem;
  }}
  .timestamp {{ font-size: 0.85rem; color: #8b949e; margin-bottom: 1.5rem; }}
  .context {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 1rem; margin-bottom: 1.5rem;
  }}
  .context-title {{
    font-size: 0.85rem; color: #58a6ff; font-weight: 600;
    margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .context-body {{
    font-size: 0.9rem; color: #c9d1d9; line-height: 1.6;
  }}
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
  <h1>{project_badge}{title}</h1>
  <div class="timestamp">Shared at {now}</div>
  {context_html}
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
        return

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
    project: str = "",
    context: str = "",
) -> str:
    """
    Create a share page for the given files.

    Args:
        file_paths: Local file paths to share.
        title: Page title.
        project: Project name for directory grouping (e.g. "GenesisExp").
        context: Background knowledge / notes to embed in the page.

    Returns the public URL. Files are COPIED (not symlinked) for persistence.
    """
    share_id = uuid.uuid4().hex[:8]

    # Directory: share/<project>/<uuid>/ or share/<uuid>/
    if project:
        share_subdir = SHARE_DIR / project / share_id
    else:
        share_subdir = SHARE_DIR / share_id
    share_subdir.mkdir(parents=True, exist_ok=True)

    media_items = []
    for fpath_str in file_paths:
        fpath = Path(fpath_str).resolve()
        if not fpath.exists():
            print(f"[media_share] WARNING: file not found: {fpath}", file=sys.stderr)
            continue

        dest_name = fpath.name
        dest_path = share_subdir / dest_name

        # Handle name collision
        if dest_path.exists():
            stem = fpath.stem
            suffix = fpath.suffix
            dest_name = f"{stem}_{share_id[:4]}{suffix}"
            dest_path = share_subdir / dest_name

        # Copy file (not symlink) for persistence
        shutil.copy2(fpath, dest_path)

        size_mb = fpath.stat().st_size / (1024 * 1024)
        media_items.append({
            "name": dest_name,
            "type": _media_type(fpath),
            "mime": _mime_hint(fpath),
            "size_mb": size_mb,
        })

    if not media_items:
        shutil.rmtree(share_subdir, ignore_errors=True)
        raise FileNotFoundError("No valid files to share")

    # Save context as markdown (for future reference outside the HTML)
    if context:
        (share_subdir / "context.md").write_text(context, encoding="utf-8")

    # Generate HTML
    html = _generate_html(title, media_items, context=context, project=project)
    (share_subdir / "index.html").write_text(html, encoding="utf-8")

    # Ensure server is running
    _start_server(SHARE_DIR, SHARE_PORT)

    if project:
        url = f"{PUBLIC_BASE}/{project}/{share_id}/"
    else:
        url = f"{PUBLIC_BASE}/{share_id}/"
    return url


# ── Cleanup ──────────────────────────────────────────────────────────────────
def _iter_shares(project: str = "") -> list[tuple[str, Path]]:
    """Iterate all share directories. Returns list of (display_id, path)."""
    if not SHARE_DIR.exists():
        return []

    results = []
    if project:
        # Only look inside a specific project
        proj_dir = SHARE_DIR / project
        if proj_dir.exists():
            for d in proj_dir.iterdir():
                if d.is_dir():
                    results.append((f"{project}/{d.name}", d))
    else:
        # Look everywhere: top-level UUIDs + project/UUID
        for entry in SHARE_DIR.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            # Check if this is a project dir (contains subdirs with index.html)
            # or a direct share dir (contains index.html itself)
            if (entry / "index.html").exists():
                results.append((entry.name, entry))
            else:
                # Project directory — list its children
                for sub in entry.iterdir():
                    if sub.is_dir() and (sub / "index.html").exists():
                        results.append((f"{entry.name}/{sub.name}", sub))

    results.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    return results


def cleanup(
    share_id: str | None = None,
    remove_all: bool = False,
    project: str = "",
) -> None:
    """List or remove shares."""
    shares = _iter_shares(project=project)

    if share_id:
        # share_id can be "uuid" or "project/uuid"
        target = SHARE_DIR / share_id
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed: {share_id}")
            # Clean up empty project dir
            parent = target.parent
            if parent != SHARE_DIR and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        else:
            print(f"Not found: {share_id}")
        return

    if remove_all:
        for display_id, path in shares:
            shutil.rmtree(path)
        # Clean up empty project dirs
        if SHARE_DIR.exists():
            for entry in SHARE_DIR.iterdir():
                if entry.is_dir() and not entry.name.startswith(".") and not any(entry.iterdir()):
                    entry.rmdir()
        print(f"Removed {len(shares)} share(s).")
        return

    # List mode
    if not shares:
        print("No shares found.")
        return

    print(f"{'ID':<28} {'Created':<20} {'Files':<6} {'Size':<10} URL")
    print("-" * 100)
    for display_id, path in shares:
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        files = [f for f in path.iterdir() if f.name not in ("index.html", "context.md")]
        total_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
        url = f"{PUBLIC_BASE}/{display_id}/"
        print(f"{display_id:<28} {mtime:<20} {len(files):<6} {total_mb:>7.1f} MB {url}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Share local media via Tailscale Funnel")
    parser.add_argument("files", nargs="*", help="File paths to share")
    parser.add_argument("--title", default="Shared Media", help="Page title")
    parser.add_argument("--project", "-p", default="", help="Project name for grouping")
    parser.add_argument("--context", "-c", default="", help="Background notes to embed")
    parser.add_argument("--cleanup", action="store_true", help="List or remove shares")
    parser.add_argument("--all", action="store_true", help="Remove all shares (with --cleanup)")
    parser.add_argument("--remove", metavar="ID", help="Remove specific share (uuid or project/uuid)")

    args = parser.parse_args()

    if args.cleanup or args.remove:
        cleanup(share_id=args.remove, remove_all=args.all, project=args.project)
        return

    if not args.files:
        parser.print_help()
        sys.exit(1)

    try:
        url = share(
            args.files,
            title=args.title,
            project=args.project,
            context=args.context,
        )
        print(url)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
