"""LTXVideoNode — LTX_VIDEO node type.

Derived from ComfyUINode. Handles LTX-Video 2.3 specific workflows:
  - img2vid:       single image + prompt -> video
  - keyframe_2:    start/end frame + prompt -> video
  - keyframe_3:    start/mid/end frame + prompt -> video
  - digital_human: image + audio + prompt -> talking head video

node_config fields (in addition to ComfyUINode fields):
  workflow_type     str    required   "img2vid" | "keyframe_2" | "keyframe_3" | "digital_human"
  prompt_expand     bool   true       Whether to call LLM for cinematic prompt expansion
  prompt_template   str    ""         Path to prompt expansion system prompt template
  negative_prompt   str    ""         Negative prompt (if workflow supports it)

Expected state fields (depending on workflow_type):
  prompt            str    required   User's short description / idea
  image             str    required   Path to input image (frame 1)
  image_end         str    optional   Path to end frame (keyframe_2, keyframe_3)
  image_mid         str    optional   Path to mid frame (keyframe_3)
  audio             str    optional   Path to audio file (digital_human)
  width             int    1280       Output width
  height            int    720        Output height
  frame_rate        int    24         Frame rate
  num_frames        int    241        Number of frames (~10s at 24fps)
  expanded_prompt   str    ""         Pre-expanded prompt (skips LLM if provided)

IMPORTANT: Workflow templates must be in ComfyUI **API format** (node_id -> {class_type, inputs}).
           Use "Save (API Format)" in ComfyUI, or export via Developer Tools.
           UI-format JSONs (with "nodes" array) will be auto-converted but results may be lossy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.comfyui.comfyui_node import ComfyUINode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node ID mappings per workflow type
# These correspond to the node IDs in the LTX 2.3 workflow templates.
# If your workflow has different IDs, override via node_config["node_ids"].
# ---------------------------------------------------------------------------

_DEFAULT_NODE_IDS = {
    "img2vid": {
        "image": "269",           # LoadImage — input image
        "prompt": "303",          # PrimitiveStringMultiline — main prompt
        "width": "314",           # PrimitiveInt — width
        "height": "299",          # PrimitiveInt — height
        "frame_rate": "300",      # PrimitiveInt — frame rate
        "num_frames": "301",      # PrimitiveInt — length
        "text_mode": "302",       # PrimitiveBoolean — text-to-video switch
    },
    "digital_human": {
        "image": "269",
        "prompt": "303",
        "width": "314",
        "height": "299",
        "frame_rate": "300",
        "num_frames": "301",
        "text_mode": "302",
        "audio": "330",           # LoadAudio — audio input
    },
    "keyframe_2": {
        "image": "269",           # LoadImage — frame 1 (start)
        "image_end": "332",       # LoadImage — frame 2 (end)
        "prompt": "325",          # PrimitiveStringMultiline — actual prompt input (303 is a link to 325)
        "width": "314",
        "height": "299",
        "frame_rate": "300",
        "num_frames": "301",
    },
    "keyframe_3": {
        "image": "269",           # LoadImage — frame 1 (start)
        "image_mid": "342",       # LoadImage — frame 2 (mid)
        "image_end": "332",       # LoadImage — frame 3 (end)
        "prompt": "325",          # PrimitiveStringMultiline — actual prompt input (303 is a link to 325)
        "width": "314",
        "height": "299",
        "frame_rate": "300",
        "num_frames": "301",
    },
}

# Prompt template file mapping per workflow type
_PROMPT_TEMPLATE_FILES = {
    "img2vid": "提示词扩写优化指令.txt",
    "digital_human": "提示词扩写优化指令.txt",
    "keyframe_2": "提示词扩写优化指令 - 首尾帧.txt",
    "keyframe_3": "提示词扩写优化指令 - 首中尾帧.txt",
}


class LTXVideoNode(ComfyUINode):
    """LTX-Video 2.3 workflow node for LangGraph."""

    def __init__(self, config: AgentConfig, node_config: dict) -> None:
        super().__init__(config, node_config)
        inner = node_config.get("node_config", node_config)

        self._workflow_type: str = inner.get("workflow_type", "img2vid")
        if self._workflow_type not in _DEFAULT_NODE_IDS:
            raise ValueError(
                f"LTXVideoNode: unknown workflow_type {self._workflow_type!r}. "
                f"Valid: {list(_DEFAULT_NODE_IDS.keys())}"
            )

        self._prompt_expand: bool = inner.get("prompt_expand", True)
        self._negative_prompt: str = inner.get("negative_prompt", "")

        # Node ID overrides (merge with defaults)
        self._node_ids = {**_DEFAULT_NODE_IDS[self._workflow_type]}
        custom_ids = inner.get("node_ids", {})
        self._node_ids.update(custom_ids)

        # Prompt template for LLM expansion
        prompt_template_path = inner.get("prompt_template", "")
        if prompt_template_path:
            self._prompt_template_path = Path(prompt_template_path)
        else:
            # Auto-select from workflow type
            default_file = _PROMPT_TEMPLATE_FILES.get(self._workflow_type, "")
            if default_file:
                # Look relative to workflow template directory
                self._prompt_template_path = self._workflow_path.parent / default_file
            else:
                self._prompt_template_path = None

        self._prompt_template_cache: str | None = None

    def _load_prompt_template(self) -> str:
        """Load the LLM prompt expansion template."""
        if self._prompt_template_cache is not None:
            return self._prompt_template_cache

        if self._prompt_template_path and self._prompt_template_path.exists():
            self._prompt_template_cache = self._prompt_template_path.read_text(encoding="utf-8")
            return self._prompt_template_cache

        logger.warning(f"[ltx_video] prompt template not found: {self._prompt_template_path}")
        self._prompt_template_cache = ""
        return ""

    async def expand_prompt(self, user_prompt: str) -> str:
        """Expand a short user prompt into a cinematic description using LLM.

        Uses the prompt template specific to this workflow type.
        Currently calls Claude API via anthropic SDK.

        Args:
            user_prompt: short user description

        Returns:
            Expanded cinematic prompt string.
        """
        template = self._load_prompt_template()
        if not template:
            logger.info("[ltx_video] no prompt template, using raw prompt")
            return user_prompt

        system_prompt = template.replace("{user_prompt}", "")
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=system_prompt.strip(),
                messages=[{"role": "user", "content": user_prompt}],
            )
            expanded = response.content[0].text.strip()
            if is_debug():
                logger.debug(f"[ltx_video] expanded prompt ({len(expanded)} chars): {expanded[:200]}...")
            return expanded

        except ImportError:
            logger.warning("[ltx_video] anthropic SDK not installed, using raw prompt")
            return user_prompt
        except Exception as e:
            logger.error(f"[ltx_video] prompt expansion failed: {e}, using raw prompt")
            return user_prompt

    def _set_node_input(self, workflow: dict, node_id: str, field: str, value: Any) -> None:
        """Set an input field on a workflow node (API format).

        Args:
            workflow: API-format workflow dict
            node_id: string node ID
            field: input field name
            value: value to set
        """
        if node_id not in workflow:
            logger.warning(f"[ltx_video] node {node_id} not found in workflow, skipping {field}={value}")
            return
        workflow[node_id].setdefault("inputs", {})[field] = value

    def _set_widget_value(self, workflow: dict, node_id: str, value: Any) -> None:
        """Set the primary widget value for simple Primitive nodes (API format).

        For PrimitiveInt/PrimitiveStringMultiline/PrimitiveBoolean, the main value
        is typically the first (or only) input parameter. We set it on the
        connected output nodes instead since Primitive nodes in API format
        propagate via links.

        In practice, for API-format workflows, we directly set the value on the
        target node's input that references this Primitive. This method sets
        the "value" field which some API-format primitives use.
        """
        if node_id not in workflow:
            logger.warning(f"[ltx_video] node {node_id} not found in workflow")
            return
        # API format primitives often just have a single output
        # We set via the inputs dict
        workflow[node_id].setdefault("inputs", {})
        # Try common field names
        inputs = workflow[node_id]["inputs"]
        if "value" in inputs:
            inputs["value"] = value
        elif len(inputs) == 0:
            inputs["value"] = value
        else:
            # Set first non-link input
            for k, v in inputs.items():
                if not isinstance(v, list):  # list = link reference
                    inputs[k] = value
                    break
            else:
                inputs["value"] = value

    def prepare_workflow(self, template: dict, state: dict) -> dict:
        """Fill LTX-Video specific fields into the workflow template.

        Args:
            template: API-format workflow dict
            state: LangGraph state with user inputs

        Returns:
            Modified workflow ready for submission.
        """
        import copy
        workflow = copy.deepcopy(template)
        ids = self._node_ids

        # --- Prompt ---
        prompt = state.get("expanded_prompt") or state.get("prompt", "")
        if prompt and ids.get("prompt"):
            self._set_widget_value(workflow, ids["prompt"], prompt)
        if prompt and ids.get("prompt_alt"):
            self._set_widget_value(workflow, ids["prompt_alt"], prompt)

        # --- Images ---
        # For LoadImage nodes, set the "image" input to the uploaded filename
        if state.get("image") and ids.get("image"):
            uploaded_name = state.get("_uploaded_image", state["image"])
            self._set_node_input(workflow, ids["image"], "image", uploaded_name)

        if state.get("image_end") and ids.get("image_end"):
            uploaded_name = state.get("_uploaded_image_end", state["image_end"])
            self._set_node_input(workflow, ids["image_end"], "image", uploaded_name)

        if state.get("image_mid") and ids.get("image_mid"):
            uploaded_name = state.get("_uploaded_image_mid", state["image_mid"])
            self._set_node_input(workflow, ids["image_mid"], "image", uploaded_name)

        # --- Audio (digital human) ---
        if state.get("audio") and ids.get("audio"):
            uploaded_name = state.get("_uploaded_audio", state["audio"])
            self._set_node_input(workflow, ids["audio"], "audio", uploaded_name)

        # --- Dimensions & timing ---
        if state.get("width") and ids.get("width"):
            self._set_widget_value(workflow, ids["width"], int(state["width"]))

        if state.get("height") and ids.get("height"):
            self._set_widget_value(workflow, ids["height"], int(state["height"]))

        if state.get("frame_rate") and ids.get("frame_rate"):
            self._set_widget_value(workflow, ids["frame_rate"], int(state["frame_rate"]))

        if state.get("num_frames") and ids.get("num_frames"):
            self._set_widget_value(workflow, ids["num_frames"], int(state["num_frames"]))

        # --- Text-to-video mode switch ---
        if ids.get("text_mode"):
            # False = image-to-video (default), True = text-only
            text_mode = state.get("text_mode", False)
            self._set_widget_value(workflow, ids["text_mode"], bool(text_mode))

        # --- Randomize seeds ---
        # RandomNoise nodes have a fixed noise_seed by default.
        # Randomize them each run so results vary.
        import random
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "RandomNoise":
                new_seed = state.get("seed", random.randint(0, 2**53))
                node_data.setdefault("inputs", {})["noise_seed"] = new_seed
                if is_debug():
                    logger.debug(f"[ltx_video] randomized seed for node {node_id}: {new_seed}")

        if is_debug():
            logger.debug(f"[ltx_video] prepared workflow type={self._workflow_type}")

        return workflow

    async def __call__(self, state: dict) -> dict:
        """Execute LTX-Video generation.

        Adds prompt expansion step before the standard ComfyUI flow.
        """
        # Step 0: Prompt expansion (if enabled and not already expanded)
        if self._prompt_expand and not state.get("expanded_prompt"):
            user_prompt = state.get("prompt", "")
            if user_prompt:
                expanded = await self.expand_prompt(user_prompt)
                state = {**state, "expanded_prompt": expanded}
                logger.info(f"[ltx_video] prompt expanded: {len(user_prompt)} -> {len(expanded)} chars")

        # Delegate to ComfyUINode.__call__ for the rest
        return await super().__call__(state)

    def parse_output(self, output_data: list[dict], prompt_id: str) -> dict:
        """Parse LTX-Video output, prioritizing video files."""
        result = super().parse_output(output_data, prompt_id)
        result["workflow_type"] = self._workflow_type
        return result
