
import asyncio
import sys
import tempfile
import time
from pathlib import Path
import logging

# Ensure ZenithLoom root is importable
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT.parent))

from framework.clients.comfyui import ComfyUIClient, WorkflowManager

logger = logging.getLogger("generate_random_video")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

COMFYUI_HOST = "localhost"
COMFYUI_PORT = 8188

async def run_workflow(
    workflow_type: str,
    prompt: str,
    width: int = 1280,
    height: int = 720,
    frame_rate: int = 24,
    num_frames: int = 241,
) -> str | None:
    """Run a workflow via ComfyUIClient + WorkflowManager. Returns video_path."""
    log_lines = []
    client = ComfyUIClient(host=COMFYUI_HOST, port=COMFYUI_PORT)
    wf_manager = WorkflowManager()

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
        return None

    # Prepare workflow
    try:
        workflow = wf_manager.prepare_workflow(
            workflow_type,
            prompt=prompt,
            uploaded_files={},
            width=width,
            height=height,
            frame_rate=frame_rate,
            num_frames=num_frames,
        )
        log("Workflow prepared")
    except Exception as e:
        log(f"ERROR preparing workflow: {e}")
        return None

    # Submit
    try:
        prompt_id = await client.submit_workflow(workflow)
        log(f"Submitted: prompt_id={prompt_id}")
    except Exception as e:
        log(f"ERROR submitting workflow: {e}")
        return None

    # Wait
    log("Waiting for generation...")
    try:
        await client.wait_for_completion(prompt_id)
        log("Generation complete!")
    except Exception as e:
        log(f"ERROR during generation: {e}")
        return None

    # Get outputs
    try:
        outputs = await client.get_outputs(prompt_id)
    except Exception as e:
        log(f"ERROR getting outputs: {e}")
        return None

    if not outputs:
        log("WARNING: No outputs found")
        return None

    videos = [o for o in outputs if o["media_type"] == "video"]
    if videos:
        video_url = videos[0]["download_url"]
        log(f"Video URL: {video_url}")

        # Download video to temp file
        try:
            suffix = Path(videos[0]["filename"]).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            await client.download_file(video_url, Path(tmp.name))
            tmp.close()
            log(f"Downloaded to: {tmp.name}")
            return tmp.name
        except Exception as e:
            log(f"ERROR downloading video: {e}")

    return None

async def main():
    prompt = "a dog dancing in the rain, cinematic, 4k"
    video_path = await run_workflow("txt2vid", prompt)
    if video_path:
        print(f"Video generated: {video_path}")
    else:
        print("Failed to generate video.")

if __name__ == "__main__":
    asyncio.run(main())
