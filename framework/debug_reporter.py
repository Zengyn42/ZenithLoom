"""
通用 LangGraph 子图 debug 可视化 — framework/debug_reporter.py

消费 astream(subgraphs=True) 的事件流，输出：
  1. 实时树形缩进 console 输出（带时间戳）
  2. Markdown 日志文件

用法：
    reporter = DebugConsoleReporter("colony_coder")
    async for ns, event in graph.astream(state, stream_mode="updates", subgraphs=True):
        reporter.on_event(ns, event)
    reporter.print_summary()
"""

import sys
from datetime import datetime
from pathlib import Path

INDENT = "  "


class DebugConsoleReporter:
    def __init__(self, graph_name: str, log_dir: Path | None = None):
        self._graph_name = graph_name
        self._prev_namespace: tuple = ()
        self._start_time = datetime.now()
        self._node_count = 0
        self._last_state: dict = {}

        # Markdown log
        date_str = self._start_time.strftime("%Y-%m-%d")
        self._log_dir = log_dir or Path("logs") / date_str / graph_name
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "debug_report.md"
        self._log_file.write_text(
            f"# Debug Report — {graph_name}\n\n"
            f"> Started: {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            encoding="utf-8",
        )

    # ── Scope helpers ──────────────────────────────────────────────

    def _scope_name(self, namespace: tuple) -> str:
        """Return the human-readable scope name for a namespace tuple."""
        if not namespace:
            return self._graph_name
        last = namespace[-1]
        return last.split(":")[0] if ":" in last else last

    def _depth(self, namespace: tuple) -> int:
        """Return nesting depth (0 = top-level graph)."""
        return len(namespace)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _indent(self, depth: int) -> str:
        return INDENT * (depth + 1)

    # ── Value formatting ───────────────────────────────────────────

    def _format_value(self, value) -> str:
        """Format a state value for compact console display."""
        if value is None:
            return "None"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            if len(value) <= 80:
                return repr(value)
            return f"{len(value)} chars"
        if isinstance(value, list):
            return f"{len(value)} items"
        if isinstance(value, dict):
            return f"{{{len(value)} keys}}"
        return repr(value)
