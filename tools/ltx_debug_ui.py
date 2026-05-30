"""LTX-Video Debug UI — Gradio interface for testing ComfyUI workflows.

Usage:
    python tools/ltx_debug_ui.py [--host HOST] [--port PORT] [--comfyui-host HOST] [--comfyui-port PORT]

Endpoints:
    - img2vid:       single image + prompt → video
    - keyframe_2:    start/end frame + prompt → video
    - keyframe_3:    start/mid/end frame + prompt → video
    - digital_human: image + audio + prompt → talking head video

Each tab allows:
    1. Upload images/audio
    2. Write a short prompt
    3. Submit to ComfyUI
    5. Preview the generated video inline

— technical_architect · 无垠智穹
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure ZenithLoom root is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import gradio as gr

# Import MCP comfyui modules (sibling to this tools/ dir)
_COMFYUI_CLIENT_DIR = _ROOT / "framework" / "clients" / "comfyui"

import importlib.util

def _import_mcp_module(name: str):
    """Import a module from mcp/comfyui/ by direct file path."""
    spec = importlib.util.spec_from_file_location(name, _COMFYUI_CLIENT_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_client_mod = _import_mcp_module("comfyui_client")
_wf_mod = _import_mcp_module("workflow_manager")
ComfyUIClient = _client_mod.ComfyUIClient
WorkflowManager = _wf_mod.WorkflowManager

logger = logging.getLogger("ltx_debug_ui")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

COMFYUI_HOST = "localhost"
COMFYUI_PORT = 8188
client: ComfyUIClient | None = None
wf_manager = WorkflowManager()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


async def _check_comfyui() -> str:
    """Check ComfyUI connectivity."""
    try:
        stats = await client.health_check()
        device = stats.get("devices", [{}])[0]
        gpu = device.get("name", "unknown")
        vram = device.get("vram_total", 0)
        vram_gb = vram / (1024**3) if vram else 0
        return f"✅ ComfyUI online | GPU: {gpu} | VRAM: {vram_gb:.1f} GB"
    except Exception as e:
        return f"❌ ComfyUI unreachable: {e}"


async def _run_workflow(
    workflow_type: str,
    prompt: str,
    image: str | None,
    image_end: str | None = None,
    image_mid: str | None = None,
    audio: str | None = None,
    width: int = 1280,
    height: int = 720,
    frame_rate: int = 24,
    num_frames: int = 241,
) -> tuple[str | None, str]:
    """Run a workflow via ComfyUIClient + WorkflowManager. Returns (video_path, log_text)."""
    log_lines = []

    def log(msg: str):
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        logger.info(msg)

    log(f"Workflow: {workflow_type}")
    log(f"Prompt: {prompt[:100]}...")
    log(f"Resolution: {width}x{height} @ {frame_rate}fps, {num_frames} frames")

    # Health check
    try:
        await client.health_check()
        log("ComfyUI health check passed")
    except Exception as e:
        log(f"ERROR: ComfyUI unreachable: {e}")
        return None, "\n".join(log_lines)

    # Upload files
    files_to_upload = {"image": image, "image_end": image_end, "image_mid": image_mid, "audio": audio}
    uploaded: dict[str, str] = {}
    for field, path in files_to_upload.items():
        if path:
            try:
                server_name = await client.upload_file(path)
                uploaded[field] = server_name
                log(f"Uploaded {field}: {Path(path).name} → {server_name}")
            except Exception as e:
                log(f"ERROR uploading {field}: {e}")
                return None, "\n".join(log_lines)

    # Prepare workflow
    try:
        workflow = wf_manager.prepare_workflow(
            workflow_type,
            prompt=prompt,
            uploaded_files=uploaded,
            width=width,
            height=height,
            frame_rate=frame_rate,
            num_frames=num_frames,
        )
        log("Workflow prepared")
    except Exception as e:
        log(f"ERROR preparing workflow: {e}")
        return None, "\n".join(log_lines)

    # Submit
    try:
        prompt_id = await client.submit_workflow(workflow)
        log(f"Submitted: prompt_id={prompt_id}")
    except Exception as e:
        log(f"ERROR submitting workflow: {e}")
        return None, "\n".join(log_lines)

    # Wait
    log("Waiting for generation...")
    try:
        await client.wait_for_completion(prompt_id)
        log("Generation complete!")
    except Exception as e:
        log(f"ERROR during generation: {e}")
        return None, "\n".join(log_lines)

    # Get outputs
    try:
        outputs = await client.get_outputs(prompt_id)
    except Exception as e:
        log(f"ERROR getting outputs: {e}")
        return None, "\n".join(log_lines)

    if not outputs:
        log("WARNING: No outputs found")
        return None, "\n".join(log_lines)

    for item in outputs:
        log(f"  output: type={item['media_type']} file={item['filename']}")

    videos = [o for o in outputs if o["media_type"] == "video"]
    images = [o for o in outputs if o["media_type"] == "image"]
    log(f"Output: {len(videos)} video(s), {len(images)} image(s)")

    if videos:
        video_url = videos[0]["download_url"]
        log(f"Video URL: {video_url}")

        # Download video to temp file for Gradio preview
        try:
            suffix = Path(videos[0]["filename"]).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            await client.download_file(video_url, Path(tmp.name))
            tmp.close()
            log(f"Downloaded to: {tmp.name}")
            return tmp.name, "\n".join(log_lines)
        except Exception as e:
            log(f"ERROR downloading video: {e}")

    return None, "\n".join(log_lines)


# ---------------------------------------------------------------------------
# Gradio event handlers (sync wrappers)
# ---------------------------------------------------------------------------


def check_comfyui_sync() -> str:
    return asyncio.run(_check_comfyui())


def run_txt2vid(prompt, width, height, fps, frames):
    video, log = asyncio.run(_run_workflow(
        "txt2vid", prompt, image=None,
        width=int(width), height=int(height), frame_rate=int(fps), num_frames=int(frames),
    ))
    return video, log


def run_img2vid(prompt, image, width, height, fps, frames):
    video, log = asyncio.run(_run_workflow(
        "img2vid", prompt, image,
        width=int(width), height=int(height), frame_rate=int(fps), num_frames=int(frames),
    ))
    return video, log


def run_keyframe2(prompt, image_start, image_end, width, height, fps, frames):
    video, log = asyncio.run(_run_workflow(
        "keyframe_2", prompt, image_start, image_end=image_end,
        width=int(width), height=int(height), frame_rate=int(fps), num_frames=int(frames),
    ))
    return video, log


def run_keyframe3(prompt, image_start, image_mid, image_end, width, height, fps, frames):
    video, log = asyncio.run(_run_workflow(
        "keyframe_3", prompt, image_start, image_end=image_end, image_mid=image_mid,
        width=int(width), height=int(height), frame_rate=int(fps), num_frames=int(frames),
    ))
    return video, log


def run_digital_human(prompt, image, audio, width, height, fps, frames):
    video, log = asyncio.run(_run_workflow(
        "digital_human", prompt, image, audio=audio,
        width=int(width), height=int(height), frame_rate=int(fps), num_frames=int(frames),
    ))
    return video, log


def query_job_status(prompt_id: str) -> str:
    """Query ComfyUI job status by prompt_id."""
    if not prompt_id or not prompt_id.strip():
        return "Please enter a prompt_id"
    async def _query():
        try:
            history = await client.get_history(prompt_id.strip())
            if history:
                return json.dumps({"status": "completed", "prompt_id": prompt_id.strip(), "history": history}, ensure_ascii=False, indent=2)
            return json.dumps({"status": "unknown", "prompt_id": prompt_id.strip()}, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)
    return asyncio.run(_query())


# ---------------------------------------------------------------------------
# UI Builder
# ---------------------------------------------------------------------------

def _common_params():
    """Shared parameter components."""
    with gr.Row():
        width = gr.Number(value=1280, label="Width", minimum=128, maximum=2048, step=64)
        height = gr.Number(value=720, label="Height", minimum=128, maximum=2048, step=64)
    with gr.Row():
        fps = gr.Number(value=24, label="FPS", minimum=1, maximum=60)
        frames = gr.Number(value=241, label="Frames", minimum=1, maximum=961, info="241≈10s @24fps")
    return width, height, fps, frames


def _prompt_section(workflow_type: str):
    """Prompt input section."""
    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="一个女孩在雨中奔跑...")
    return prompt


def build_ui() -> gr.Blocks:
    """Build the Gradio debug interface."""
    with gr.Blocks(title="LTX-Video Debug UI") as app:
        gr.Markdown("# 🎬 LTX-Video 2.3 Debug UI")
        gr.Markdown("端到端测试：上传素材 → ComfyUI 生成 → 视频预览")

        # Status bar
        with gr.Row():
            status = gr.Textbox(label="ComfyUI Status", interactive=False, scale=4)
            check_btn = gr.Button("🔄 Check", scale=1)
            check_btn.click(fn=check_comfyui_sync, outputs=status)

        with gr.Tabs():
            # ===== Tab 0: txt2vid =====
            with gr.TabItem("✏️ txt2vid（纯文字）"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2v_prompt = _prompt_section("txt2vid")
                        t2v_w, t2v_h, t2v_fps, t2v_frames = _common_params()
                        t2v_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        t2v_video = gr.Video(label="Generated Video")
                        t2v_log = gr.Textbox(label="Log", lines=12, interactive=False)
                t2v_btn.click(
                    fn=run_txt2vid,
                    inputs=[t2v_prompt, t2v_w, t2v_h, t2v_fps, t2v_frames],
                    outputs=[t2v_video, t2v_log],
                )

            # ===== Tab 1: img2vid =====
            with gr.TabItem("🖼️ img2vid"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_image = gr.Image(label="Input Image", type="filepath")
                        img_prompt = _prompt_section("img2vid")
                        img_w, img_h, img_fps, img_frames = _common_params()
                        img_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        img_video = gr.Video(label="Generated Video")
                        img_log = gr.Textbox(label="Log", lines=12, interactive=False)
                img_btn.click(
                    fn=run_img2vid,
                    inputs=[img_prompt, img_image, img_w, img_h, img_fps, img_frames],
                    outputs=[img_video, img_log],
                )

            # ===== Tab 2: keyframe_2 =====
            with gr.TabItem("🔀 keyframe_2（首尾帧）"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            kf2_start = gr.Image(label="Start Frame", type="filepath")
                            kf2_end = gr.Image(label="End Frame", type="filepath")
                        kf2_prompt = _prompt_section("keyframe_2")
                        kf2_w, kf2_h, kf2_fps, kf2_frames = _common_params()
                        kf2_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        kf2_video = gr.Video(label="Generated Video")
                        kf2_log = gr.Textbox(label="Log", lines=12, interactive=False)
                kf2_btn.click(
                    fn=run_keyframe2,
                    inputs=[kf2_prompt, kf2_start, kf2_end, kf2_w, kf2_h, kf2_fps, kf2_frames],
                    outputs=[kf2_video, kf2_log],
                )

            # ===== Tab 3: keyframe_3 =====
            with gr.TabItem("🔀 keyframe_3（首中尾帧）"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            kf3_start = gr.Image(label="Start Frame", type="filepath")
                            kf3_mid = gr.Image(label="Mid Frame", type="filepath")
                            kf3_end = gr.Image(label="End Frame", type="filepath")
                        kf3_prompt = _prompt_section("keyframe_3")
                        kf3_w, kf3_h, kf3_fps, kf3_frames = _common_params()
                        kf3_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        kf3_video = gr.Video(label="Generated Video")
                        kf3_log = gr.Textbox(label="Log", lines=12, interactive=False)
                kf3_btn.click(
                    fn=run_keyframe3,
                    inputs=[kf3_prompt, kf3_start, kf3_mid, kf3_end, kf3_w, kf3_h, kf3_fps, kf3_frames],
                    outputs=[kf3_video, kf3_log],
                )

            # ===== Tab 4: digital_human =====
            with gr.TabItem("🗣️ digital_human"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dh_image = gr.Image(label="Portrait Image", type="filepath")
                        dh_audio = gr.Audio(label="Audio File", type="filepath")
                        dh_prompt = _prompt_section("digital_human")
                        dh_w, dh_h, dh_fps, dh_frames = _common_params()
                        dh_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        dh_video = gr.Video(label="Generated Video")
                        dh_log = gr.Textbox(label="Log", lines=12, interactive=False)
                dh_btn.click(
                    fn=run_digital_human,
                    inputs=[dh_prompt, dh_image, dh_audio, dh_w, dh_h, dh_fps, dh_frames],
                    outputs=[dh_video, dh_log],
                )

            # ===== Tab 5: Job Status =====
            with gr.TabItem("📋 Job Status"):
                gr.Markdown("查询 ComfyUI 任务状态（输入生成时返回的 prompt_id）")
                with gr.Row():
                    with gr.Column(scale=1):
                        job_prompt_id = gr.Textbox(label="Prompt ID", placeholder="输入 prompt_id...")
                        job_btn = gr.Button("🔍 Query Status", variant="primary")
                    with gr.Column(scale=1):
                        job_result = gr.Textbox(label="Result", lines=15, interactive=False)
                job_btn.click(
                    fn=query_job_status,
                    inputs=job_prompt_id,
                    outputs=job_result,
                )

        # ===== Workflow Inspector =====
        with gr.Accordion("🔧 Workflow Inspector", open=False):
            gr.Markdown("查看实际发送给 ComfyUI 的 workflow JSON（调试用）")
            with gr.Row():
                insp_type = gr.Dropdown(
                    choices=["txt2vid", "img2vid", "keyframe_2", "keyframe_3", "digital_human"],
                    value="txt2vid",
                    label="Workflow Type",
                )
                insp_btn = gr.Button("Load Template")
            insp_json = gr.JSON(label="Workflow Template (API Format)")
            insp_node_ids = gr.JSON(label="Node ID Mapping")

            def inspect_workflow(wf_type):
                template = wf_manager.load_template(wf_type)
                node_ids = _wf_mod.NODE_IDS
                return template, node_ids.get(wf_type, {})

            insp_btn.click(
                fn=inspect_workflow,
                inputs=insp_type,
                outputs=[insp_json, insp_node_ids],
            )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LTX-Video Debug UI")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--comfyui-host", default="localhost", help="ComfyUI host")
    parser.add_argument("--comfyui-port", type=int, default=8188, help="ComfyUI port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    global COMFYUI_HOST, COMFYUI_PORT, client
    COMFYUI_HOST = args.comfyui_host
    COMFYUI_PORT = args.comfyui_port
    client = ComfyUIClient(host=COMFYUI_HOST, port=COMFYUI_PORT)

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
