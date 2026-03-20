"""
框架级 debug 工具 — framework/debug.py

统一 DEBUG 标志，由 awaken.py 通过 set_debug() 在启动时设置。

Debug 模式功能：
  1. is_debug() → True  各模块输出 logger.debug() 级别日志
  2. log_node_thinking()  记录每个 LLM 节点的思考内容到 .md 文件
  3. log_graph_flow()     记录图节点流转到 .md 文件
  4. log_state_snapshot()  记录节点输出后的 state 快照到 state_snapshots.md

日志目录结构（按图层级组织）：
  logs/YYYY-MM-DD/<graph_name>/<subgraph_name>/...
    flow.md              — 节点流转时间线
    thinking.md          — LLM 思考内容
    state_snapshots.md   — 每个节点执行后的 state 快照（含 LLM 完整输出）

图层级由 push_graph_scope() / pop_graph_scope() 管理，
GraphController 和 SubgraphRefNode 自动调用。
"""

import logging
import os
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DEBUG: bool = False

# 图层级栈：("hani",) → ("hani", "debate_brainstorm") → ...
_graph_scope: ContextVar[tuple[str, ...]] = ContextVar("_graph_scope", default=())


def set_debug(value: bool) -> None:
    """由 awaken.py 在解析 --debug 参数后调用。"""
    global _DEBUG
    _DEBUG = bool(value)


def is_debug() -> bool:
    """返回当前进程是否处于 debug 模式。"""
    return _DEBUG


# ── 图层级管理 ────────────────────────────────────────────────────────────


def push_graph_scope(name: str) -> None:
    """进入一个图/子图，将名称压入层级栈。"""
    current = _graph_scope.get()
    _graph_scope.set(current + (name,))
    if _DEBUG:
        logger.debug(f"[debug] push_graph_scope → {current + (name,)}")


def pop_graph_scope() -> None:
    """退出当前图/子图，弹出层级栈顶。"""
    current = _graph_scope.get()
    if current:
        _graph_scope.set(current[:-1])
        if _DEBUG:
            logger.debug(f"[debug] pop_graph_scope → {current[:-1]}")


def get_graph_scope() -> tuple[str, ...]:
    """返回当前图层级栈（只读，供外部诊断用）。"""
    return _graph_scope.get()


# ── 日志目录 ──────────────────────────────────────────────────────────────

# 缓存已创建的目录，避免重复 mkdir
_created_dirs: set[str] = set()


def _get_log_dir() -> Path:
    """
    根据当前日期和图层级栈，计算日志目录并惰性创建。

    例：scope=("hani", "debate_brainstorm") → logs/2026-03-18/hani/debate_brainstorm/
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    scope = _graph_scope.get()
    base = Path(os.getcwd()) / "logs" / date_str
    for part in scope:
        base = base / part
    key = str(base)
    if key not in _created_dirs:
        base.mkdir(parents=True, exist_ok=True)
        _created_dirs.add(key)
    return base


def _append_md(log_dir: Path, filename: str, content: str) -> None:
    """追加写入 markdown 日志文件。"""
    log_file = log_dir / filename
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        logger.warning(f"[debug] 无法写入日志 {log_file}: {e}")


def _ensure_md_header(log_dir: Path, filename: str, header: str) -> None:
    """如果 md 文件不存在或为空，写入 header。"""
    log_file = log_dir / filename
    if not log_file.exists() or log_file.stat().st_size == 0:
        _append_md(log_dir, filename, header)


# ── Thinking 日志（Markdown）─────────────────────────────────────────────


def log_node_thinking(
    node_id: str,
    thinking_text: str = "",
    output_text: str = "",
) -> None:
    """
    记录 LLM 节点的思考和输出内容到 thinking.md。

    仅在 is_debug() == True 时实际写入。
    格式：collapsible details 块，按节点分段。
    """
    if not _DEBUG:
        return
    if not thinking_text and not output_text:
        return

    log_dir = _get_log_dir()
    scope = _graph_scope.get()
    scope_label = " / ".join(scope) if scope else "unknown"

    # 首次写入时添加文件头
    _ensure_md_header(log_dir, "thinking.md", f"# Thinking Log — {scope_label}\n\n")

    timestamp = datetime.now().strftime("%H:%M:%S")
    lines: list[str] = []
    lines.append(f"---\n")
    lines.append(f"## {timestamp} — `{node_id}`\n\n")

    if thinking_text:
        chars = len(thinking_text)
        lines.append(f"<details>\n<summary>Thinking ({chars} chars)</summary>\n")
        lines.append(f"\n```\n{thinking_text.rstrip()}\n```\n")
        lines.append(f"\n</details>\n\n")

    if output_text:
        chars = len(output_text)
        lines.append(f"### Output ({chars} chars)\n\n")
        lines.append(f"```\n{output_text.rstrip()}\n```\n\n")

    _append_md(log_dir, "thinking.md", "".join(lines))

    # 同时输出到 Python logger（便于控制台实时查看）
    if thinking_text:
        preview = thinking_text[:300]
        if len(thinking_text) > 300:
            preview += f"... ({len(thinking_text)} chars total)"
        logger.debug(f"[{node_id}/thinking] {preview}")


# ── Graph flow 日志（Markdown）───────────────────────────────────────────


_FLOW_HEADER = """\
# Flow Log — {scope}

| Time | Event | Node | Detail |
|------|-------|------|--------|
"""

# 事件符号
_FLOW_SYMBOLS = {"enter": "▶ enter", "exit": "◀ exit", "route": "⤳ route", "edge": "→ edge"}


def log_graph_flow(
    event: str,
    node_id: str,
    detail: str = "",
) -> None:
    """
    记录图节点流转事件到 flow.md（Markdown 表格格式）。

    仅在 is_debug() == True 时实际写入。

    Args:
        event: 事件类型 — "enter" | "exit" | "route" | "edge"
        node_id: 节点 ID
        detail: 附加信息（如 routing_target、返回的 state keys 等）
    """
    if not _DEBUG:
        return

    log_dir = _get_log_dir()
    scope = _graph_scope.get()
    scope_label = " / ".join(scope) if scope else "unknown"

    # 首次写入时添加表头
    _ensure_md_header(log_dir, "flow.md", _FLOW_HEADER.format(scope=scope_label))

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    sym = _FLOW_SYMBOLS.get(event, event)
    # 转义 detail 中的 | 符号，防止破坏表格
    safe_detail = detail.replace("|", "\\|") if detail else ""

    row = f"| {timestamp} | {sym} | `{node_id}` | {safe_detail} |\n"
    _append_md(log_dir, "flow.md", row)

    # 同时输出到 Python logger
    logger.debug(f"[flow] {sym} {node_id} {detail}")


# ── State snapshot 日志 ──────────────────────────────────────────────────


def _format_message(msg) -> str:
    """格式化单条 LangChain message，保留完整内容。"""
    role = getattr(msg, "type", "unknown")
    content = getattr(msg, "content", "")
    return f"**[{role}]** ({len(content)} chars)\n\n{content}"


def _format_state_value(key: str, value) -> str:
    """格式化单个 state 字段的值。"""
    if key == "messages":
        # messages 单独处理，每条完整展示
        if not value:
            return "_empty_"
        parts = []
        for i, msg in enumerate(value):
            parts.append(f"#### Message {i}: {_format_message(msg)}")
        return "\n\n".join(parts)
    if isinstance(value, str):
        if not value:
            return "_empty string_"
        return f"```\n{value}\n```"
    if isinstance(value, (list, dict)):
        import json
        try:
            formatted = json.dumps(value, ensure_ascii=False, indent=2)
            return f"```json\n{formatted}\n```"
        except (TypeError, ValueError):
            return f"```\n{repr(value)}\n```"
    return f"`{repr(value)}`"


def log_state_snapshot(
    node_id: str,
    node_output: dict,
    full_state: dict | None = None,
) -> None:
    """
    记录节点输出后的 state 快照到 state_snapshots.md。

    仅在 is_debug() == True 时实际写入。

    Args:
        node_id: 节点 ID
        node_output: 该节点返回的 dict（增量更新）
        full_state: 可选，节点执行前的完整 state（用于展示上下文）
    """
    if not _DEBUG:
        return
    if not node_output:
        return

    log_dir = _get_log_dir()
    scope = _graph_scope.get()
    scope_label = " / ".join(scope) if scope else "unknown"

    _ensure_md_header(
        log_dir, "state_snapshots.md",
        f"# State Snapshots — {scope_label}\n\n"
        "每个节点执行后输出的 state 增量。\n\n"
    )

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    lines: list[str] = []
    lines.append(f"---\n\n## {timestamp} — `{node_id}` output\n\n")

    # 输出该节点返回的每个字段
    for key in sorted(node_output.keys()):
        val = node_output[key]
        # 跳过空值
        if val is None or val == "" or val == [] or val == {}:
            continue
        lines.append(f"### `{key}`\n\n")
        lines.append(_format_state_value(key, val))
        lines.append("\n\n")

    _append_md(log_dir, "state_snapshots.md", "".join(lines))
