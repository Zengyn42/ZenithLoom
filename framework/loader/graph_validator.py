"""
Pre-build graph validation: node ID uniqueness, edge references, reachability.

Each function accepts either an ``AgentGraph`` instance (preferred) or a raw
``dict`` (backward-compat: auto-wrapped via ``AgentGraph.from_dict()``).
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def _unwrap(graph_spec) -> "AgentGraph":
    """Return an AgentGraph, wrapping a raw dict if necessary."""
    from framework.loader.graph_spec import AgentGraph
    if isinstance(graph_spec, AgentGraph):
        return graph_spec
    return AgentGraph.from_dict(graph_spec)


def _collect_all_ids(graph_spec, seen: set | None = None) -> set:
    """
    递归收集所有节点 ID（包括 SUBGRAPH 内部）。发现重复 ID 立即抛 ValueError。

    Accepts an AgentGraph or a raw dict (backward-compat).
    """
    graph = _unwrap(graph_spec)
    if seen is None:
        seen = set()
    return graph._collect_all_ids(seen)


def _check_edge_refs(graph_spec, all_ids: set) -> None:
    """
    验证所有边引用的节点 ID 存在（__start__ 和 __end__ 为合法虚节点）。

    Accepts an AgentGraph or a raw dict (backward-compat).
    """
    graph = _unwrap(graph_spec)
    graph._check_edge_refs(all_ids)


def _check_reachable(graph_spec, all_ids: set) -> None:
    """
    BFS 验证所有节点可达。支持 __start__ 边和 entry 字段两种入口声明。

    Accepts an AgentGraph or a raw dict (backward-compat).
    """
    graph = _unwrap(graph_spec)
    graph._check_reachable(all_ids)
