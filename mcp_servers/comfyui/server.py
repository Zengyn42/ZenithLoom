"""ComfyUI MCP Server — exposes LTX-Video workflows as MCP tools.

Tools:
  - ltx_img2vid:          Single image + prompt → video
  - ltx_keyframe_2:       Start/end frame + prompt → video
  - ltx_keyframe_3:       Start/mid/end frame + prompt → video
  - ltx_digital_human:    Image + audio + prompt → talking head video
  - comfyui_status:       Check ComfyUI server status
  - comfyui_job_status:   Query a running/completed job

Usage:
    python mcp/comfyui/server.py [--comfyui-host HOST] [--comfyui-port PORT]

Or configure in Claude's MCP settings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Import fixup: the local ZenithLoom/mcp/ package shadows the pip "mcp" SDK.
# Solution: remove ZenithLoom root from sys.path before importing SDK,
# then use direct file imports for siblings.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ZENITHLOOM_ROOT = str(_THIS_DIR.parent.parent)

# Remove all paths that would cause local mcp/ to shadow the SDK
_hidden = []
for _p in list(sys.path):
    if os.path.isfile(os.path.join(_p, "mcp", "__init__.py")) and os.path.abspath(_p) == os.path.abspath(_ZENITHLOOM_ROOT):
        sys.path.remove(_p)
        _hidden.append(_p)

# Clear any cached local mcp modules
for _k in list(sys.modules):
    if _k == "mcp" or _k.startswith("mcp."):
        del sys.modules[_k]

from mcp.server.fastmcp import FastMCP  # noqa: E402 — now resolves to pip SDK

# Restore paths (but mcp SDK is already cached in sys.modules)
for _p in _hidden:
    sys.path.insert(0, _p)

# Sibling imports — use direct file import to avoid namespace issues
import importlib.util

def _import_sibling(name: str):
    """Import a .py file from the same directory."""
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_client_mod = _import_sibling("comfyui_client")
_wf_mod = _import_sibling("workflow_manager")
ComfyUIClient = _client_mod.ComfyUIClient
WorkflowManager = _wf_mod.WorkflowManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "localhost")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
OUTPUT_DIR = Path(os.environ.get("COMFYUI_OUTPUT_DIR", tempfile.gettempdir())) / "comfyui_mcp_output"

# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

client = ComfyUIClient(host=COMFYUI_HOST, port=COMFYUI_PORT)
wf_manager = WorkflowManager()

# Track active jobs for status queries
_jobs: dict[str, dict] = {}  # prompt_id -> {status, workflow_type, ...}

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("ComfyUI Video Generation")


# ---------------------------------------------------------------------------
# Shared execution logic
# ---------------------------------------------------------------------------

async def _execute_workflow(
    workflow_type: str,
    prompt: str,
    files_to_upload: dict[str, str],
    width: int = 1280,
    height: int = 720,
    frame_rate: int = 24,
    num_frames: int = 241,
    seed: int | None = None,
) -> dict:
    """Upload files, prepare workflow, submit, wait, return results.

    Args:
        workflow_type: img2vid / keyframe_2 / keyframe_3 / digital_human
        prompt: cinematic prompt (already expanded by the calling LLM)
        files_to_upload: {field: local_path} e.g. {"image": "/tmp/photo.jpg"}
        width, height, frame_rate, num_frames: generation params
        seed: optional seed for reproducibility

    Returns:
        Dict with prompt_id, status, outputs, download_urls, etc.
    """
    # 1. Health check
    try:
        await client.health_check()
    except Exception as e:
        return {"error": f"ComfyUI unreachable: {e}", "status": "error"}

    # 2. Upload files
    uploaded: dict[str, str] = {}
    for field, local_path in files_to_upload.items():
        if local_path:
            try:
                server_name = await client.upload_file(local_path)
                uploaded[field] = server_name
                logger.info(f"Uploaded {field}: {Path(local_path).name} → {server_name}")
            except Exception as e:
                return {"error": f"Upload failed for {field}: {e}", "status": "error"}

    # 3. Prepare workflow
    try:
        workflow = wf_manager.prepare_workflow(
            workflow_type,
            prompt=prompt,
            uploaded_files=uploaded,
            width=width,
            height=height,
            frame_rate=frame_rate,
            num_frames=num_frames,
            seed=seed,
        )
    except Exception as e:
        return {"error": f"Workflow preparation failed: {e}", "status": "error"}

    # 4. Submit
    try:
        prompt_id = await client.submit_workflow(workflow)
        logger.info(f"Submitted {workflow_type}: prompt_id={prompt_id}")
        _jobs[prompt_id] = {"status": "running", "workflow_type": workflow_type}
    except Exception as e:
        return {"error": f"Submit failed: {e}", "status": "error"}

    # 5. Wait for completion
    try:
        await client.wait_for_completion(prompt_id)
        _jobs[prompt_id]["status"] = "completed"
    except Exception as e:
        _jobs[prompt_id]["status"] = "error"
        return {"error": f"Execution failed: {e}", "status": "error", "prompt_id": prompt_id}

    # 6. Get outputs
    try:
        outputs = await client.get_outputs(prompt_id)
    except Exception as e:
        return {"error": f"Output retrieval failed: {e}", "status": "error", "prompt_id": prompt_id}

    # 7. Download outputs to local files
    downloaded = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for out in outputs:
        ext = Path(out["filename"]).suffix or ".mp4"
        local_path = OUTPUT_DIR / f"{prompt_id}_{out['filename']}"
        try:
            await client.download_file(out["download_url"], local_path)
            downloaded.append({
                "media_type": out["media_type"],
                "filename": out["filename"],
                "local_path": str(local_path),
                "download_url": out["download_url"],
            })
        except Exception as e:
            logger.warning(f"Download failed for {out['filename']}: {e}")
            downloaded.append({
                "media_type": out["media_type"],
                "filename": out["filename"],
                "download_url": out["download_url"],
                "download_error": str(e),
            })

    videos = [d for d in downloaded if d["media_type"] == "video"]
    images = [d for d in downloaded if d["media_type"] == "image"]

    return {
        "status": "completed",
        "prompt_id": prompt_id,
        "workflow_type": workflow_type,
        "videos": videos,
        "images": images,
        "summary": f"Generated {len(videos)} video(s), {len(images)} image(s)",
    }


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def ltx_img2vid(
    image_path: str,
    prompt: str,
    width: int = 1280,
    height: int = 720,
    frame_rate: int = 24,
    num_frames: int = 241,
    seed: int | None = None,
) -> str:
    """Generate a video from a single image using LTX-Video 2.3.

    Args:
        image_path: Absolute path to the input image file
        prompt: Cinematic description of the desired video motion and style
        width: Output video width (default 1280)
        height: Output video height (default 720)
        frame_rate: Frames per second (default 24)
        num_frames: Total frames to generate, 241 ≈ 10s at 24fps (default 241)
        seed: Optional seed for reproducibility
    """
    result = await _execute_workflow(
        "img2vid",
        prompt=prompt,
        files_to_upload={"image": image_path},
        width=width, height=height, frame_rate=frame_rate, num_frames=num_frames, seed=seed,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def ltx_keyframe_2(
    image_start_path: str,
    image_end_path: str,
    prompt: str,
    width: int = 1280,
    height: int = 720,
    frame_rate: int = 24,
    num_frames: int = 241,
    seed: int | None = None,
) -> str:
    """Generate a video interpolating between a start and end frame.

    Args:
        image_start_path: Absolute path to the start frame image
        image_end_path: Absolute path to the end frame image
        prompt: Cinematic description of the transition motion and style
        width: Output video width (default 1280)
        height: Output video height (default 720)
        frame_rate: Frames per second (default 24)
        num_frames: Total frames to generate (default 241)
        seed: Optional seed for reproducibility
    """
    result = await _execute_workflow(
        "keyframe_2",
        prompt=prompt,
        files_to_upload={"image": image_start_path, "image_end": image_end_path},
        width=width, height=height, frame_rate=frame_rate, num_frames=num_frames, seed=seed,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def ltx_keyframe_3(
    image_start_path: str,
    image_mid_path: str,
    image_end_path: str,
    prompt: str,
    width: int = 1280,
    height: int = 720,
    frame_rate: int = 24,
    num_frames: int = 241,
    seed: int | None = None,
) -> str:
    """Generate a video interpolating through three keyframes (start, middle, end).

    Args:
        image_start_path: Absolute path to the start frame image
        image_mid_path: Absolute path to the middle frame image
        image_end_path: Absolute path to the end frame image
        prompt: Cinematic description of the transitions and style
        width: Output video width (default 1280)
        height: Output video height (default 720)
        frame_rate: Frames per second (default 24)
        num_frames: Total frames to generate (default 241)
        seed: Optional seed for reproducibility
    """
    result = await _execute_workflow(
        "keyframe_3",
        prompt=prompt,
        files_to_upload={
            "image": image_start_path,
            "image_mid": image_mid_path,
            "image_end": image_end_path,
        },
        width=width, height=height, frame_rate=frame_rate, num_frames=num_frames, seed=seed,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def ltx_digital_human(
    image_path: str,
    audio_path: str,
    prompt: str,
    width: int = 1280,
    height: int = 720,
    frame_rate: int = 24,
    num_frames: int = 241,
    seed: int | None = None,
) -> str:
    """Generate a talking-head video from a portrait image and audio file.

    Args:
        image_path: Absolute path to the portrait image
        audio_path: Absolute path to the audio file (wav/mp3)
        prompt: Cinematic description of the speaking style and background
        width: Output video width (default 1280)
        height: Output video height (default 720)
        frame_rate: Frames per second (default 24)
        num_frames: Total frames to generate (default 241)
        seed: Optional seed for reproducibility
    """
    result = await _execute_workflow(
        "digital_human",
        prompt=prompt,
        files_to_upload={"image": image_path, "audio": audio_path},
        width=width, height=height, frame_rate=frame_rate, num_frames=num_frames, seed=seed,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def comfyui_status() -> str:
    """Check ComfyUI server status, GPU info, and available workflows."""
    try:
        stats = await client.health_check()
        device = stats.get("devices", [{}])[0]
        gpu_name = device.get("name", "unknown")
        vram = device.get("vram_total", 0)
        vram_free = device.get("vram_free", 0)
        vram_gb = vram / (1024**3) if vram else 0
        vram_free_gb = vram_free / (1024**3) if vram_free else 0

        workflows = wf_manager.list_workflows()

        return json.dumps({
            "status": "online",
            "gpu": gpu_name,
            "vram_total_gb": round(vram_gb, 1),
            "vram_free_gb": round(vram_free_gb, 1),
            "workflows": workflows,
            "active_jobs": len([j for j in _jobs.values() if j["status"] == "running"]),
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "offline", "error": str(e)}, indent=2)


@mcp.tool()
async def comfyui_job_status(prompt_id: str) -> str:
    """Query the status of a ComfyUI generation job.

    Args:
        prompt_id: The prompt_id returned when a workflow was submitted
    """
    if prompt_id in _jobs:
        return json.dumps(_jobs[prompt_id], indent=2)

    # Try fetching from ComfyUI history
    try:
        history = await client.get_history(prompt_id)
        if history:
            return json.dumps({"status": "completed", "prompt_id": prompt_id, "has_history": True}, indent=2)
        return json.dumps({"status": "unknown", "prompt_id": prompt_id}, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global client  # noqa: PLW0603
    global OUTPUT_DIR  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="ComfyUI MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="Transport mode")
    parser.add_argument("--host", default="127.0.0.1", help="SSE bind host")
    parser.add_argument("--port", type=int, default=8103, help="SSE bind port")
    parser.add_argument("--comfyui-host", default=COMFYUI_HOST, help="ComfyUI host")
    parser.add_argument("--comfyui-port", type=int, default=COMFYUI_PORT, help="ComfyUI port")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for downloaded outputs")
    args = parser.parse_args()

    client = ComfyUIClient(host=args.comfyui_host, port=args.comfyui_port)
    OUTPUT_DIR = Path(args.output_dir)

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    logger.info(f"Starting ComfyUI MCP Server (transport={args.transport}, comfyui={args.comfyui_host}:{args.comfyui_port})")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
