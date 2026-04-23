"""
向后兼容 shim — 请改用 framework.loader。

所有实现已迁移到 framework/loader/ 子包：
  framework.loader.entity_loader  — EntityLoader 类
  framework.loader.graph_builder  — 声明式图构建
  framework.loader.graph_validator — 图验证
  framework.loader.persona        — persona 组装
  framework.loader.topology       — Mermaid 渲染
"""

from framework.loader.entity_loader import (
    EntityLoader,
    AgentLoader,
    _resolve_proxy_class,
    _register_mcp_tools,
)
from framework.loader.graph_builder import (
    _DEFAULT,
    _build_declarative,
    _wrap_node_for_flow_log,
    _maybe_limit,
    _LLM_NODE_TYPES,
    _extract_session_keys_from_json,
    register_state_schema,
    _get_state_schemas,
)
from framework.loader.graph_validator import (
    _collect_all_ids,
    _check_edge_refs,
    _check_reachable,
)
from framework.loader.persona import (
    _load_persona_text,
    _collect_routing_hints,
)
from framework.loader.topology import (
    _mermaid_render,
    _mermaid_agent_ref,
    _mermaid_id,
    _MERMAID_SHAPES,
)

__all__ = [
    "EntityLoader",
    "AgentLoader",
]
