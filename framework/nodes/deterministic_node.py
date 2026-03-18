"""DeterministicNode — DETERMINISTIC node type.

Wraps a pure Python function from {blueprint_dir}/validators.py as a LangGraph node.
Convention: the function name must match the node's id in agent.json.

node_config fields (injected by _build_declarative):
  id          str  required  Node id — used to look up function in validators.py
  agent_dir   str  required  Blueprint directory containing validators.py
"""

import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_validators(agent_dir: str):
    path = Path(agent_dir) / "validators.py"
    if not path.exists():
        raise FileNotFoundError(f"DeterministicNode: validators.py not found at {path}")
    spec = importlib.util.spec_from_file_location("_validators", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeterministicNode:
    """
    DETERMINISTIC node: calls a pure Python function from validators.py.

    The function must be synchronous: (state: dict) -> dict.
    No LLM calls, no I/O — routing and validation logic only.
    """

    def __init__(self, config, node_config: dict):
        node_id = node_config["id"]
        agent_dir = node_config["agent_dir"]
        module = _load_validators(agent_dir)
        self._fn = getattr(module, node_id)
        logger.debug(f"[deterministic] loaded {node_id!r} from {agent_dir}")

    async def __call__(self, state: dict) -> dict:
        return self._fn(state)
