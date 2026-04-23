"""
Persona text assembly and routing hint collection.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_persona_text(
    persona_files: list[str],
    base_dir: Path,
    prompt: str = "",
    label: str = "",
) -> str:
    """从文件列表 + prompt 组装 persona 文本。

    persona_files 路径相对于 base_dir。
    每段标注来源注释 <!-- [source: label/file] -->。
    """
    parts: list[str] = []
    src_label = label or base_dir.name
    for fname in persona_files:
        p = base_dir / fname
        if p.exists():
            content = p.read_text(encoding="utf-8").strip()
            parts.append(f"<!-- [source: {src_label}/{fname}] -->\n{content}")
        else:
            logger.warning(f"[persona] persona file not found: {p}")
    if prompt and prompt.strip():
        parts.append(prompt.strip())
    return "\n\n---\n\n".join(parts)


def _collect_routing_hints(graph_spec: dict, base_dir: str = "") -> str:
    """
    遍历 graph_spec 中所有含 agent_dir 的子图节点，读取其 entity.json 的 routing_hint 字段，
    构建路由说明字符串，用于注入主节点 system_prompt。
    """
    hints: list[str] = []
    for node_def in graph_spec.get("nodes", []):
        agent_dir = node_def.get("agent_dir", "")
        if not agent_dir or node_def.get("type"):
            continue
        node_id = node_def.get("id", "")
        hint = node_def.get("routing_hint") or ""
        if not hint:
            raw = Path(agent_dir)
            for candidate in [
                raw / "entity.json",
                (Path(base_dir) / raw / "entity.json") if base_dir else None,
            ]:
                if candidate and candidate.exists():
                    try:
                        sub_json = json.loads(candidate.read_text(encoding="utf-8"))
                        hint = sub_json.get("routing_hint", "")
                    except Exception:
                        pass
                    break
        if hint:
            hints.append(f'  - "{node_id}": {hint} <!-- [auto-injected routing_hint] -->')

    if not hints:
        return ""

    lines = [
        "",
        "<!-- [auto-generated section: routing hints collected from subgraph nodes] -->",
        "[可调用子图]",
        "遇到以下情况时，可将任务委托给对应子图。",
        "路由方式：在回复的第一行且仅第一行输出以下 JSON（不加任何前缀或解释）：",
        '{"route": "<节点id>", "context": "<清晰描述议题和相关背景>"}',
        "",
        "可用子图：",
    ] + hints + [
        "",
        "注意：路由是重操作（多轮 LLM 调用），仅在真正有价值时使用，日常任务直接回复。",
    ]
    return "\n".join(lines)
