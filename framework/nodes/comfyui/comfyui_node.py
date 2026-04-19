"""ComfyUINode — COMFYUI node type.

Generic LangGraph node for communicating with any ComfyUI instance.
Handles workflow submission, progress monitoring, and output retrieval.

node_config fields:
  host          str     "localhost"   ComfyUI server hostname
  port          int     8188          ComfyUI server port
  workflow      str     required      Path to workflow JSON template (absolute or relative to blueprint_dir)
  timeout       float   600.0         Max wait for generation (seconds)
  output_field  str     "comfyui_output"  State field to store result
  upload_fields dict    {}            Mapping: {state_field: comfyui_input_name} for image uploads

Lifecycle:
  1. Upload input images (if any) via POST /upload/image
  2. Load workflow template, call prepare_workflow() for field substitution
  3. Submit via POST /prompt
  4. Monitor progress via WebSocket
  5. Retrieve outputs via GET /history/{prompt_id}
  6. Call parse_output() and write to state

Subclasses override:
  prepare_workflow(template, state) -> dict   — fill dynamic fields
  parse_output(history, prompt_id) -> dict    — extract relevant outputs
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from langchain_core.messages import AIMessage

from framework.config import AgentConfig
from framework.debug import is_debug

logger = logging.getLogger(__name__)


class ComfyUINode:
    """Generic ComfyUI API communication node for LangGraph."""

    def __init__(self, config: AgentConfig, node_config: dict) -> None:
        inner = node_config.get("node_config", node_config)
        self._host: str = inner.get("host", "localhost")
        self._port: int = int(inner.get("port", 8188))
        self._timeout: float = float(inner.get("timeout", 600.0))
        self._output_field: str = inner.get("output_field", "comfyui_output")
        self._upload_fields: dict[str, str] = inner.get("upload_fields", {})
        self._client_id: str = uuid.uuid4().hex[:12]

        # Workflow template path
        workflow_path = inner.get("workflow")
        if not workflow_path:
            raise ValueError("ComfyUINode: 'workflow' path is required in node_config")

        self._workflow_path = Path(workflow_path)
        if not self._workflow_path.is_absolute():
            # Relative to blueprint dir
            blueprint_dir = getattr(config, "blueprint_dir", None)
            if blueprint_dir:
                self._workflow_path = Path(blueprint_dir) / self._workflow_path

        self._description: str = inner.get("description", "ComfyUI workflow execution")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self._host}:{self._port}/ws?clientId={self._client_id}"

    # -------------------------------------------------------------------------
    # HTTP helpers
    # -------------------------------------------------------------------------

    async def _health_check(self, session: aiohttp.ClientSession) -> bool:
        """Check if ComfyUI is reachable."""
        try:
            async with session.get(f"{self.base_url}/system_stats", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def upload_image(
        self,
        session: aiohttp.ClientSession,
        image_path: str | Path,
        *,
        subfolder: str = "",
        overwrite: bool = True,
    ) -> str:
        """Upload an image to ComfyUI. Returns the filename as stored on server.

        Args:
            session: aiohttp session
            image_path: local path to image file
            subfolder: optional subfolder in ComfyUI input dir
            overwrite: overwrite existing file with same name

        Returns:
            Filename string usable in workflow JSON.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        data = aiohttp.FormData()
        data.add_field(
            "image",
            open(image_path, "rb"),
            filename=image_path.name,
            content_type="image/png",
        )
        if subfolder:
            data.add_field("subfolder", subfolder)
        if overwrite:
            data.add_field("overwrite", "true")

        async with session.post(f"{self.base_url}/upload/image", data=data) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"ComfyUI upload failed ({resp.status}): {body}")
            result = await resp.json()
            filename = result.get("name", image_path.name)
            if is_debug():
                logger.debug(f"[comfyui] uploaded {image_path.name} -> {filename}")
            return filename

    async def submit_workflow(
        self,
        session: aiohttp.ClientSession,
        workflow: dict,
    ) -> str:
        """Submit a workflow to ComfyUI. Returns prompt_id.

        Args:
            session: aiohttp session
            workflow: complete workflow dict (the "prompt" payload)

        Returns:
            prompt_id string for tracking.
        """
        payload = {
            "prompt": workflow,
            "client_id": self._client_id,
        }
        async with session.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"ComfyUI submit failed ({resp.status}): {body}")
            result = await resp.json()
            prompt_id = result["prompt_id"]
            if is_debug():
                logger.debug(f"[comfyui] submitted workflow, prompt_id={prompt_id}")
            return prompt_id

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        progress_callback: Any | None = None,
    ) -> None:
        """Monitor ComfyUI progress via WebSocket until prompt completes.

        Args:
            prompt_id: the prompt_id returned by submit_workflow
            progress_callback: optional async callable(current, total, node_id) for progress updates
        """
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as ws_session:
            async with ws_session.ws_connect(self.ws_url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "progress" and data.get("data", {}).get("prompt_id") == prompt_id:
                            d = data["data"]
                            current = d.get("value", 0)
                            total = d.get("max", 0)
                            node_id = d.get("node")
                            if is_debug():
                                logger.debug(f"[comfyui] progress: {current}/{total} node={node_id}")
                            if progress_callback:
                                await progress_callback(current, total, node_id)

                        elif msg_type == "executing" and data.get("data", {}).get("prompt_id") == prompt_id:
                            node_id = data["data"].get("node")
                            if node_id is None:
                                # None node means execution complete
                                if is_debug():
                                    logger.debug(f"[comfyui] execution complete: {prompt_id}")
                                return

                        elif msg_type == "execution_error" and data.get("data", {}).get("prompt_id") == prompt_id:
                            error_data = data.get("data", {})
                            raise RuntimeError(
                                f"ComfyUI execution error: node={error_data.get('node_id')} "
                                f"type={error_data.get('node_type')} "
                                f"exception={error_data.get('exception_message', 'unknown')}"
                            )

                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        raise RuntimeError(f"ComfyUI WebSocket closed unexpectedly: {msg.type}")

    async def get_history(
        self,
        session: aiohttp.ClientSession,
        prompt_id: str,
    ) -> dict:
        """Fetch execution history for a prompt_id.

        Returns:
            The history dict for this prompt_id.
        """
        async with session.get(
            f"{self.base_url}/history/{prompt_id}",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"ComfyUI history fetch failed ({resp.status}): {body}")
            history = await resp.json()
            return history.get(prompt_id, {})

    async def get_output_data(
        self,
        session: aiohttp.ClientSession,
        prompt_id: str,
    ) -> list[dict]:
        """Extract output file info from history.

        Returns:
            List of dicts with keys: filename, subfolder, type (e.g. "output").
        """
        _VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif", ".webp", ".apng"}

        history = await self.get_history(session, prompt_id)
        outputs = history.get("outputs", {})
        results = []
        for node_id, node_output in outputs.items():
            # Videos (explicit)
            for vid in node_output.get("videos", []):
                results.append({
                    "node_id": node_id,
                    "media_type": "video",
                    "filename": vid.get("filename"),
                    "subfolder": vid.get("subfolder", ""),
                    "type": vid.get("type", "output"),
                })
            # GIFs / animated outputs (some nodes use "gifs" key)
            for gif in node_output.get("gifs", []):
                results.append({
                    "node_id": node_id,
                    "media_type": "video",
                    "filename": gif.get("filename"),
                    "subfolder": gif.get("subfolder", ""),
                    "type": gif.get("type", "output"),
                })
            # Images — but check if it's actually a video by extension
            for img in node_output.get("images", []):
                fname = img.get("filename", "")
                ext = Path(fname).suffix.lower() if fname else ""
                media = "video" if ext in _VIDEO_EXTS else "image"
                results.append({
                    "node_id": node_id,
                    "media_type": media,
                    "filename": fname,
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
        return results

    def get_download_url(self, filename: str, subfolder: str = "", file_type: str = "output") -> str:
        """Build download URL for an output file."""
        params = f"filename={filename}&subfolder={subfolder}&type={file_type}"
        return f"{self.base_url}/view?{params}"

    # -------------------------------------------------------------------------
    # Workflow template handling — subclasses override these
    # -------------------------------------------------------------------------

    def load_workflow_template(self) -> dict:
        """Load the workflow JSON template from disk.

        Returns:
            Parsed workflow dict.
        """
        if not self._workflow_path.exists():
            raise FileNotFoundError(f"Workflow template not found: {self._workflow_path}")

        with open(self._workflow_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # ComfyUI API format uses the "prompt" key if it's a full workflow export
        # For API-format JSONs, the nodes are keyed by node ID at the top level
        # For UI-format JSONs, nodes are in a "nodes" array — need conversion
        if "nodes" in raw and isinstance(raw["nodes"], list):
            return self._convert_ui_to_api_format(raw)
        return raw

    def _convert_ui_to_api_format(self, ui_workflow: dict) -> dict:
        """Convert ComfyUI UI-format workflow to API-format.

        UI format: {"nodes": [{id, type, widgets_values, ...}], "links": [...]}
        API format: {"node_id": {"class_type": ..., "inputs": {...}}}

        This is a best-effort conversion. Complex workflows may need
        pre-exported API-format JSONs.
        """
        # Build link lookup: link_id -> (source_node_id, source_slot)
        link_map: dict[int, tuple[int, int]] = {}
        for link in ui_workflow.get("links", []):
            # link format: [link_id, source_node, source_slot, target_node, target_slot, type]
            link_id, src_node, src_slot = link[0], link[1], link[2]
            link_map[link_id] = (src_node, src_slot)

        api_format = {}
        for node in ui_workflow.get("nodes", []):
            node_id = str(node["id"])
            inputs = {}

            # Process input connections
            for inp in node.get("inputs", []):
                link_id = inp.get("link")
                if link_id is not None and link_id in link_map:
                    src_node, src_slot = link_map[link_id]
                    inputs[inp["name"]] = [str(src_node), src_slot]

            # Process widget values
            widget_values = node.get("widgets_values", [])
            widget_names = self._get_widget_names(node)
            for i, val in enumerate(widget_values):
                if i < len(widget_names) and widget_names[i] not in inputs:
                    inputs[widget_names[i]] = val

            api_format[node_id] = {
                "class_type": node.get("type", ""),
                "inputs": inputs,
            }
            meta_title = node.get("_meta", {}).get("title") or node.get("title")
            if meta_title:
                api_format[node_id]["_meta"] = {"title": meta_title}

        return api_format

    def _get_widget_names(self, node: dict) -> list[str]:
        """Extract widget parameter names from a UI-format node.

        This is heuristic — ComfyUI doesn't always include widget names in the
        UI export. Subclasses may override with node-type-specific mappings.
        """
        # Some nodes store output names but not widget names in the export
        # Return empty — subclasses or API-format templates are preferred
        return []

    def prepare_workflow(self, template: dict, state: dict) -> dict:
        """Fill dynamic fields in the workflow template from state.

        Override in subclasses for workflow-specific logic.
        Default: simple {field} placeholder replacement in string values.

        Args:
            template: the loaded workflow dict (API format)
            state: current LangGraph state

        Returns:
            Modified workflow dict ready to submit.
        """
        return self._substitute_placeholders(template, state)

    def _substitute_placeholders(self, obj: Any, state: dict) -> Any:
        """Recursively replace {field} placeholders in string values."""
        if isinstance(obj, str):
            import re
            return re.sub(
                r"\{(\w+)\}",
                lambda m: str(state[m.group(1)]) if m.group(1) in state else m.group(0),
                obj,
            )
        elif isinstance(obj, dict):
            return {k: self._substitute_placeholders(v, state) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_placeholders(v, state) for v in obj]
        return obj

    def parse_output(self, output_data: list[dict], prompt_id: str) -> dict:
        """Process raw output data into a structured result.

        Override in subclasses for workflow-specific parsing.

        Args:
            output_data: list of output file dicts from get_output_data()
            prompt_id: the prompt_id

        Returns:
            Dict to merge into LangGraph state.
        """
        videos = [o for o in output_data if o["media_type"] == "video"]
        images = [o for o in output_data if o["media_type"] == "image"]

        result = {
            "prompt_id": prompt_id,
            "videos": [
                {
                    **v,
                    "download_url": self.get_download_url(v["filename"], v["subfolder"], v["type"]),
                }
                for v in videos
            ],
            "images": [
                {
                    **img,
                    "download_url": self.get_download_url(img["filename"], img["subfolder"], img["type"]),
                }
                for img in images
            ],
        }
        return result

    # -------------------------------------------------------------------------
    # LangGraph __call__
    # -------------------------------------------------------------------------

    async def __call__(self, state: dict) -> dict:
        """Execute the full ComfyUI workflow lifecycle.

        Flow: health check -> upload images -> prepare workflow -> submit ->
              wait -> get output -> parse -> return state update.
        """
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1. Health check
            healthy = await self._health_check(session)
            if not healthy:
                error_msg = f"ComfyUI not reachable at {self.base_url}"
                logger.error(f"[comfyui] {error_msg}")
                return {
                    "messages": [AIMessage(content=f"[ComfyUI Error] {error_msg}")],
                }

            # 2. Upload images from state
            uploaded: dict[str, str] = {}
            for state_field, comfyui_name in self._upload_fields.items():
                image_path = state.get(state_field)
                if image_path:
                    filename = await self.upload_image(session, image_path)
                    uploaded[comfyui_name] = filename
                    if is_debug():
                        logger.debug(f"[comfyui] uploaded {state_field}={image_path} -> {filename}")

            # Merge uploaded filenames into state for template substitution
            state_with_uploads = {**state, **uploaded}

            # 3. Load and prepare workflow
            template = self.load_workflow_template()
            workflow = self.prepare_workflow(template, state_with_uploads)

            # 4. Submit
            prompt_id = await self.submit_workflow(session, workflow)
            logger.info(f"[comfyui] workflow submitted: prompt_id={prompt_id}")

            # 5. Wait for completion
            try:
                await self.wait_for_completion(prompt_id)
            except RuntimeError as e:
                error_msg = str(e)
                logger.error(f"[comfyui] execution failed: {error_msg}")
                return {
                    "messages": [AIMessage(content=f"[ComfyUI Error] {error_msg}")],
                }

            # 6. Get outputs
            output_data = await self.get_output_data(session, prompt_id)
            if not output_data:
                logger.warning(f"[comfyui] no outputs found for prompt_id={prompt_id}")
                return {
                    "messages": [AIMessage(content="[ComfyUI] Workflow completed but no outputs found.")],
                }

            # 7. Parse and return
            result = self.parse_output(output_data, prompt_id)
            logger.info(
                f"[comfyui] done: {len(result.get('videos', []))} videos, "
                f"{len(result.get('images', []))} images"
            )

            # Build summary message
            summary_parts = [f"[ComfyUI] Generation complete (prompt_id={prompt_id})"]
            for v in result.get("videos", []):
                summary_parts.append(f"  Video: {v['filename']} ({v['download_url']})")
            for img in result.get("images", []):
                summary_parts.append(f"  Image: {img['filename']} ({img['download_url']})")

            return {
                self._output_field: result,
                "messages": [AIMessage(content="\n".join(summary_parts))],
            }
