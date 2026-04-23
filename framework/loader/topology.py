"""
Mermaid topology rendering for entity graphs.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Node type → Mermaid bracket shapes (open, close)
_MERMAID_SHAPES: dict[str, tuple[str, str]] = {
    "VALIDATE":      ("{{", "}}"),
    "GIT_SNAPSHOT":  ("(", ")"),
    "GIT_ROLLBACK":  ("(", ")"),
    "EXTERNAL_TOOL": ('(["', '"])'),
}


def _mermaid_id(prefix: str, raw: str) -> str:
    """加前缀避免子图内 ID 冲突。__start__/__end__ 去掉双下划线。"""
    if raw in ("__start__", "__end__"):
        stripped = raw.strip("_")
        return (prefix + stripped) if prefix else raw
    return (prefix + raw) if prefix else raw


def _mermaid_render(spec: dict, lines: list, indent: str, prefix: str) -> None:
    """递归将 graph_spec 的节点和边渲染为 Mermaid flowchart 行。"""
    edges = spec.get("edges", [])
    all_refs = {e.get("from") for e in edges} | {e.get("to") for e in edges}

    for node_def in spec.get("nodes", []):
        raw   = node_def["id"]
        ntype = node_def.get("type", "")
        full  = _mermaid_id(prefix, raw)

        if node_def.get("agent_dir") and not ntype:
            _mermaid_agent_ref(node_def, lines, indent, full, raw)
            continue

        if ntype == "SUBGRAPH":
            lines.append(f'{indent}subgraph {full}["{raw}"]')
            lines.append(f'{indent}  direction LR')
            _mermaid_render(node_def.get("graph", {}), lines, indent + "  ", full + "_")
            lines.append(f'{indent}end')
            continue

        open_, close_ = _MERMAID_SHAPES.get(ntype, ('["', '"]'))
        label = f"{raw}\\n{ntype}" if ntype else raw
        lines.append(f'{indent}{full}{open_}{label}{close_}')

    if "__start__" in all_refs:
        sid = _mermaid_id(prefix, "__start__")
        lbl = "start" if not prefix else " "
        lines.append(f'{indent}{sid}(({lbl}))')
    if "__end__" in all_refs:
        eid = _mermaid_id(prefix, "__end__")
        lbl = "end" if not prefix else " "
        lines.append(f'{indent}{eid}(({lbl}))')

    for edge in edges:
        src   = _mermaid_id(prefix, edge["from"])
        dst   = _mermaid_id(prefix, edge["to"])
        etype = edge.get("type", "")
        arrow = f" -->|{etype}| " if etype else " --> "
        lines.append(f"{indent}{src}{arrow}{dst}")


def _mermaid_agent_ref(
    node_def: dict, lines: list, indent: str, full_id: str, raw: str
) -> None:
    """将 external subgraph 节点展开为 Mermaid subgraph，递归加载外部 entity.json。"""
    agent_dir_str = node_def.get("agent_dir", "")
    if not agent_dir_str:
        lines.append(f'{indent}subgraph {full_id}["{raw} ⚠ no agent_dir"]')
        lines.append(f'{indent}  {full_id}_err["⚠ agent_dir missing"]')
        lines.append(f'{indent}end')
        return

    sub_json_path = Path(agent_dir_str) / "entity.json"
    if not sub_json_path.exists():
        lines.append(f'{indent}subgraph {full_id}["{raw}\\n⚠ not found"]')
        lines.append(f'{indent}  {full_id}_err["⚠ {agent_dir_str}"]')
        lines.append(f'{indent}end')
        return

    try:
        sub_json   = json.loads(sub_json_path.read_text(encoding="utf-8"))
        agent_name = sub_json.get("name", Path(agent_dir_str).name)
        sub_spec   = sub_json.get("graph", {})
        sub_prefix = full_id + "_"
        lines.append(f'{indent}subgraph {full_id}["{raw}\\n({agent_name})"]')
        lines.append(f'{indent}  direction LR')
        _mermaid_render(sub_spec, lines, indent + "  ", sub_prefix)
        lines.append(f'{indent}end')
    except Exception as exc:
        lines.append(f'{indent}subgraph {full_id}["{raw} ⚠ load error"]')
        lines.append(f'{indent}  {full_id}_err["⚠ {str(exc)[:60]}"]')
        lines.append(f'{indent}end')
