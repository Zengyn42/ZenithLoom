"""Workflow Manager — loads workflow templates and prepares them for submission.

Extracts the LTX-Video workflow logic from LTXVideoNode into a standalone
module with no LangGraph dependency.

Each workflow type has:
  - A JSON template (API format)
  - A node ID mapping (which node controls which parameter)
  - A prepare() method that fills in user parameters
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "blueprints" / "corporations" / "ltx_pipeline" / "templates"

# ---------------------------------------------------------------------------
# Node ID mappings per workflow type
# ---------------------------------------------------------------------------

NODE_IDS = {
    "img2vid": {
        "image": "269",
        "prompt": "303",
        "width": "314",
        "height": "299",
        "frame_rate": "300",
        "num_frames": "301",
        "text_mode": "302",
    },
    "digital_human": {
        "image": "269",
        "prompt": "303",
        "width": "314",
        "height": "299",
        "frame_rate": "300",
        "num_frames": "301",
        "text_mode": "302",
        "audio": "330",
    },
    "keyframe_2": {
        "image": "269",
        "image_end": "332",
        "prompt": "325",
        "width": "314",
        "height": "299",
        "frame_rate": "300",
        "num_frames": "301",
    },
    "keyframe_3": {
        "image": "269",
        "image_mid": "342",
        "image_end": "332",
        "prompt": "325",
        "width": "314",
        "height": "299",
        "frame_rate": "300",
        "num_frames": "301",
    },
}

TEMPLATE_FILES = {
    "img2vid": "ltx_img2vid_api.json",
    "digital_human": "ltx_digital_human_api.json",
    "keyframe_2": "ltx_keyframe2_api.json",
    "keyframe_3": "ltx_keyframe3_api.json",
}


class WorkflowManager:
    """Manages LTX-Video workflow templates."""

    def __init__(self, template_dir: Path | None = None) -> None:
        self.template_dir = template_dir or TEMPLATE_DIR

    def list_workflows(self) -> list[dict]:
        """List available workflow types with metadata."""
        results = []
        for wf_type, filename in TEMPLATE_FILES.items():
            path = self.template_dir / filename
            results.append({
                "type": wf_type,
                "template_file": filename,
                "available": path.exists(),
            })
        return results

    def load_template(self, workflow_type: str) -> dict:
        """Load a workflow template from disk."""
        if workflow_type not in TEMPLATE_FILES:
            raise ValueError(f"Unknown workflow type: {workflow_type}. Valid: {list(TEMPLATE_FILES.keys())}")
        path = self.template_dir / TEMPLATE_FILES[workflow_type]
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_workflow(
        self,
        workflow_type: str,
        *,
        prompt: str,
        uploaded_files: dict[str, str],
        width: int = 1280,
        height: int = 720,
        frame_rate: int = 24,
        num_frames: int = 241,
        seed: int | None = None,
    ) -> dict:
        """Load template and fill in all parameters.

        Args:
            workflow_type: one of img2vid/keyframe_2/keyframe_3/digital_human
            prompt: the prompt text (already expanded by caller)
            uploaded_files: mapping of field -> server-side filename
                           e.g. {"image": "uploaded_001.png", "image_end": "uploaded_002.png"}
            width, height, frame_rate, num_frames: generation parameters
            seed: optional fixed seed (random if None)

        Returns:
            Complete workflow dict ready for submit_workflow().
        """
        template = self.load_template(workflow_type)
        workflow = copy.deepcopy(template)
        ids = NODE_IDS[workflow_type]

        # Prompt
        if prompt and ids.get("prompt"):
            _set_widget_value(workflow, ids["prompt"], prompt)

        # Images & audio
        for field in ("image", "image_end", "image_mid", "audio"):
            if field in uploaded_files and ids.get(field):
                input_name = "audio" if field == "audio" else "image"
                _set_node_input(workflow, ids[field], input_name, uploaded_files[field])

        # Dimensions & timing
        for field, value in [("width", width), ("height", height), ("frame_rate", frame_rate), ("num_frames", num_frames)]:
            if ids.get(field):
                _set_widget_value(workflow, ids[field], int(value))

        # Text mode (img2vid / digital_human only)
        if ids.get("text_mode"):
            _set_widget_value(workflow, ids["text_mode"], False)

        # Randomize seeds
        actual_seed = seed if seed is not None else random.randint(0, 2**53)
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "RandomNoise":
                node_data.setdefault("inputs", {})["noise_seed"] = actual_seed

        return workflow


# ---------------------------------------------------------------------------
# Helpers (extracted from LTXVideoNode)
# ---------------------------------------------------------------------------

def _set_node_input(workflow: dict, node_id: str, field: str, value: Any) -> None:
    if node_id not in workflow:
        logger.warning(f"Node {node_id} not found in workflow, skipping {field}={value}")
        return
    workflow[node_id].setdefault("inputs", {})[field] = value


def _set_widget_value(workflow: dict, node_id: str, value: Any) -> None:
    if node_id not in workflow:
        logger.warning(f"Node {node_id} not found in workflow")
        return
    inputs = workflow[node_id].setdefault("inputs", {})
    if "value" in inputs:
        inputs["value"] = value
    elif len(inputs) == 0:
        inputs["value"] = value
    else:
        for k, v in inputs.items():
            if not isinstance(v, list):
                inputs[k] = value
                break
        else:
            inputs["value"] = value
