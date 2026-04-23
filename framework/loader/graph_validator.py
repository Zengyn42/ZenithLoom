"""
Pre-build graph validation: node ID uniqueness, edge references, reachability.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def _collect_all_ids(graph_spec: dict, seen: set | None = None) -> set:
    """递归收集所有节点 ID（包括 SUBGRAPH 内部）。发现重复 ID 立即抛 ValueError。"""
    if seen is None:
        seen = set()
    for node_def in graph_spec.get("nodes", []):
        nid = node_def.get("id")
        if not nid:
            raise ValueError(f"Node missing 'id': {node_def}")
        if nid in seen:
            raise ValueError(f"Duplicate node ID: {nid!r}")
        seen.add(nid)
        if node_def.get("type") == "SUBGRAPH":
            _collect_all_ids(node_def.get("graph", {}), seen)
    return seen


def _check_edge_refs(graph_spec: dict, all_ids: set) -> None:
    """验证所有边引用的节点 ID 存在（__start__ 和 __end__ 为合法虚节点）。"""
    valid_ids = all_ids | {"__start__", "__end__"}
    for edge in graph_spec.get("edges", []):
        for key in ("from", "to"):
            ref = edge.get(key)
            if ref not in valid_ids:
                raise ValueError(
                    f"Edge references unknown node: {ref!r} "
                    f"(known: {sorted(valid_ids)})"
                )

    entry = graph_spec.get("entry")
    exit_node = graph_spec.get("exit")
    if entry and entry not in all_ids:
        raise ValueError(f"'entry' references unknown node: {entry!r} (known: {sorted(all_ids)})")
    if exit_node and exit_node not in all_ids:
        raise ValueError(f"'exit' references unknown node: {exit_node!r} (known: {sorted(all_ids)})")


def _check_reachable(graph_spec: dict, all_ids: set) -> None:
    """BFS 验证所有节点可达。支持 __start__ 边和 entry 字段两种入口声明。"""
    adjacency: dict[str, set] = defaultdict(set)
    for edge in graph_spec.get("edges", []):
        adjacency[edge["from"]].add(edge["to"])

    start_nodes: set[str] = {"__start__"}
    entry = graph_spec.get("entry")
    if entry:
        start_nodes.add(entry)

    visited = set(start_nodes)
    queue = []
    for s in start_nodes:
        queue.extend(adjacency.get(s, []))

    while queue:
        node = queue.pop()
        if node not in visited:
            visited.add(node)
            queue.extend(adjacency.get(node, []))

    unreachable = all_ids - visited - {"__end__"}
    if unreachable:
        raise ValueError(f"Unreachable nodes from start: {unreachable}")
