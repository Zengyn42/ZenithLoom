"""ComfyUI HTTP/WS client — extracted from ComfyUINode for standalone use.

Handles:
  - Health check
  - Image/audio upload
  - Workflow submission
  - WebSocket progress monitoring
  - Output retrieval & download

No LangGraph dependency.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Async client for ComfyUI API."""

    def __init__(self, host: str = "localhost", port: int = 8188, timeout: float = 600.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex[:12]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}"

    # ── Health ────────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Check ComfyUI status. Returns system_stats or raises."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(f"{self.base_url}/system_stats") as r:
                if r.status != 200:
                    raise RuntimeError(f"ComfyUI returned {r.status}")
                return await r.json()

    # ── Upload ────────────────────────────────────────────────────────────

    async def upload_file(
        self,
        file_path: str | Path,
        *,
        subfolder: str = "",
        overwrite: bool = True,
    ) -> str:
        """Upload an image or audio file to ComfyUI. Returns server-side filename."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = aiohttp.FormData()
        data.add_field(
            "image",
            open(file_path, "rb"),
            filename=file_path.name,
            content_type="application/octet-stream",
        )
        if subfolder:
            data.add_field("subfolder", subfolder)
        if overwrite:
            data.add_field("overwrite", "true")

        async with aiohttp.ClientSession() as s:
            async with s.post(f"{self.base_url}/upload/image", data=data) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Upload failed ({resp.status}): {body}")
                result = await resp.json()
                return result.get("name", file_path.name)

    # ── Submit & Wait ─────────────────────────────────────────────────────

    async def submit_workflow(self, workflow: dict) -> str:
        """Submit a workflow to ComfyUI. Returns prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
            async with s.post(f"{self.base_url}/prompt", json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Submit failed ({resp.status}): {body}")
                result = await resp.json()
                return result["prompt_id"]

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        progress_callback: Any | None = None,
    ) -> None:
        """Monitor via WebSocket until prompt completes or errors."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as ws_session:
            async with ws_session.ws_connect(self.ws_url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "progress" and data.get("data", {}).get("prompt_id") == prompt_id:
                            d = data["data"]
                            if progress_callback:
                                await progress_callback(d.get("value", 0), d.get("max", 0), d.get("node"))

                        elif msg_type == "executing" and data.get("data", {}).get("prompt_id") == prompt_id:
                            if data["data"].get("node") is None:
                                return  # done

                        elif msg_type == "execution_error" and data.get("data", {}).get("prompt_id") == prompt_id:
                            ed = data.get("data", {})
                            raise RuntimeError(
                                f"Execution error: node={ed.get('node_id')} "
                                f"type={ed.get('node_type')} "
                                f"msg={ed.get('exception_message', 'unknown')}"
                            )

                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        raise RuntimeError(f"WebSocket closed: {msg.type}")

    # ── History & Outputs ─────────────────────────────────────────────────

    async def get_history(self, prompt_id: str) -> dict:
        """Fetch execution history for a prompt_id."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
            async with s.get(f"{self.base_url}/history/{prompt_id}") as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"History fetch failed ({resp.status}): {body}")
                history = await resp.json()
                return history.get(prompt_id, {})

    async def get_outputs(self, prompt_id: str) -> list[dict]:
        """Extract output files from history. Returns list of {filename, media_type, download_url, ...}."""
        _VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif", ".webp", ".apng"}
        history = await self.get_history(prompt_id)
        outputs = history.get("outputs", {})
        results = []

        for node_id, node_output in outputs.items():
            for vid in node_output.get("videos", []):
                results.append(self._output_entry(node_id, "video", vid))
            for gif in node_output.get("gifs", []):
                results.append(self._output_entry(node_id, "video", gif))
            for img in node_output.get("images", []):
                fname = img.get("filename", "")
                ext = Path(fname).suffix.lower()
                media = "video" if ext in _VIDEO_EXTS else "image"
                results.append(self._output_entry(node_id, media, img))

        return results

    def _output_entry(self, node_id: str, media_type: str, raw: dict) -> dict:
        filename = raw.get("filename", "")
        subfolder = raw.get("subfolder", "")
        ftype = raw.get("type", "output")
        return {
            "node_id": node_id,
            "media_type": media_type,
            "filename": filename,
            "subfolder": subfolder,
            "download_url": f"{self.base_url}/view?filename={filename}&subfolder={subfolder}&type={ftype}",
        }

    # ── Download ──────────────────────────────────────────────────────────

    async def download_file(self, url: str, dest: str | Path) -> Path:
        """Download a file from ComfyUI to local path."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
            async with s.get(url) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Download failed ({resp.status}): {url}")
                dest.write_bytes(await resp.read())
        return dest
