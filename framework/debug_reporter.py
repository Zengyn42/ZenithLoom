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

from langchain_core.messages import AIMessage

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

    # ── Console output ─────────────────────────────────────────────

    def _print(self, text: str) -> None:
        print(text, flush=True)

    # ── Event processing ───────────────────────────────────────────

    def on_event(self, namespace: tuple, update: dict) -> None:
        """Main entry point — process one astream(subgraphs=True) event."""
        for node_id, changes in update.items():
            if node_id in ("__start__", "__end__"):
                continue

            self._node_count += 1
            self._emit_scope_transitions(namespace)
            self._prev_namespace = namespace

            # Merge changes into _last_state
            if isinstance(changes, dict):
                self._last_state.update(changes)

            # Node header
            depth = self._depth(namespace)
            indent = self._indent(depth)
            scope = self._scope_name(namespace)
            self._print(f"{self._timestamp()} {indent}[{scope}] \u25b6 {node_id}")

            if isinstance(changes, dict):
                self._print_messages(changes, indent)
                self._print_state(changes, indent)
                self._write_markdown(namespace, node_id, changes)

    def _emit_scope_transitions(self, namespace: tuple) -> None:
        """Print scope enter/exit markers by comparing with _prev_namespace."""
        prev = self._prev_namespace
        # Find common prefix length
        common = 0
        for i in range(min(len(prev), len(namespace))):
            if prev[i] == namespace[i]:
                common += 1
            else:
                break

        # Exit scopes (deepest first)
        for i in range(len(prev) - 1, common - 1, -1):
            name = prev[i].split(":")[0] if ":" in prev[i] else prev[i]
            depth = i  # depth of the scope being exited
            indent = self._indent(depth)
            self._print(f"{self._timestamp()} {indent}\u25c0 {name}")

        # Enter scopes
        for i in range(common, len(namespace)):
            name = namespace[i].split(":")[0] if ":" in namespace[i] else namespace[i]
            depth = i  # depth of the scope being entered
            indent = self._indent(depth)
            self._print(f"{self._timestamp()} {indent}\u25b6 {name}")

    def _print_messages(self, changes: dict, indent: str) -> None:
        """Print AIMessage content with │ prefix."""
        messages = changes.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                for line in msg.content.splitlines():
                    self._print(f"{indent}\u2502 {line}")

    def _print_state(self, changes: dict, indent: str) -> None:
        """Print routing target and other state changes."""
        routing = changes.get("routing_target")
        if routing:
            self._print(f"{indent}  Route: \u2192 {routing}")

        parts = []
        for key, value in changes.items():
            if key in ("messages", "routing_target"):
                continue
            if value is None or value == "" or value == [] or value == {}:
                continue
            parts.append(f"{key}={self._format_value(value)}")

        if parts:
            self._print(f"{indent}  State: {', '.join(parts)}")

    # ── Markdown logging ───────────────────────────────────────────

    def _write_markdown(self, namespace: tuple, node_id: str, changes: dict) -> None:
        """Append event details to the markdown log file."""
        scope_path = " > ".join(
            [self._graph_name] + [
                s.split(":")[0] if ":" in s else s for s in namespace
            ]
        )
        lines = [
            f"\n## {self._timestamp()} — {scope_path} — {node_id}\n",
        ]

        # AI messages
        messages = changes.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                lines.append("\n```\n")
                lines.append(msg.content)
                lines.append("\n```\n")

        # State changes
        state_items = {
            k: v for k, v in changes.items()
            if k != "messages" and v is not None and v != "" and v != [] and v != {}
        }
        if state_items:
            lines.append("\n**State:**\n")
            for k, v in state_items.items():
                lines.append(f"- `{k}`: {self._format_value(v)}\n")

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.writelines(lines)

    # ── Summary ────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print execution summary to console."""
        elapsed = datetime.now() - self._start_time
        seconds = int(elapsed.total_seconds())
        minutes, secs = divmod(seconds, 60)

        self._print(f"\n{'='*50}")
        self._print(f"  {self._graph_name} — Execution Summary")
        self._print(f"{'='*50}")
        self._print(f"  Nodes executed: {self._node_count}")
        self._print(f"  Elapsed: {minutes}m {secs}s")

        # Success/failure from last state
        success = self._last_state.get("success")
        if success is not None:
            status = "Success" if success else "Failure"
            self._print(f"  Status: {status}")

        # Final files if present
        files = self._last_state.get("files")
        if files:
            self._print(f"  Files: {self._format_value(files)}")

        self._print(f"  Log: {self._log_file}")
        self._print(f"{'='*50}\n")
