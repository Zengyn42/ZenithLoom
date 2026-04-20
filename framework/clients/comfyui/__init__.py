"""ComfyUI client library — HTTP/WebSocket communication with ComfyUI backend.

Exports:
  - ComfyUIClient: async HTTP/WS client for workflow submission and monitoring
  - WorkflowManager: template loading and parameter injection
"""

from framework.clients.comfyui.comfyui_client import ComfyUIClient
from framework.clients.comfyui.workflow_manager import WorkflowManager

__all__ = ["ComfyUIClient", "WorkflowManager"]
