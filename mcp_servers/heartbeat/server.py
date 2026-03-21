"""
Heartbeat MCP Server — mcp_servers/heartbeat/server.py

独立进程运行的 MCP 服务，管理 heartbeat 任务。
任何 MCP 客户端（Claude Code、Gemini CLI、框架内 agent）均可连接并：
  - 装载 / 卸载 heartbeat blueprint
  - 查询任务状态
  - 立即执行任务
  - 修改任务频率

启动方式：
  python -m mcp_servers.heartbeat.server                     # stdio（单客户端）
  python -m mcp_servers.heartbeat.server --transport sse     # SSE（多客户端）
  python -m mcp_servers.heartbeat.server --transport sse --port 8100
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 确保项目根在 sys.path 中（支持 python -m 启动）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server import FastMCP

from framework.heartbeat import HeartbeatManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("heartbeat_mcp")

# ---------------------------------------------------------------------------
# 全局状态：多个 blueprint 各自一个 HeartbeatManager
# ---------------------------------------------------------------------------

_managers: dict[str, HeartbeatManager] = {}   # name → manager
_state_dir: Path = _PROJECT_ROOT / "data" / "heartbeat"
_pid_file: Path = _state_dir / "mcp.pid"
_autoload_paths: list[str] = []  # 启动时自动装载的 blueprint 路径
_active_sessions: set = set()  # 活跃的 ServerSession 引用，用于推送告警


def _capture_session():
    """从当前 request context 捕获 ServerSession 并保存。"""
    try:
        ctx = mcp.get_context()
        if ctx._request_context and ctx._request_context.session:
            _active_sessions.add(ctx._request_context.session)
    except Exception:
        pass


async def _broadcast_alert(alert: dict):
    """
    向所有已连接的 Agent 推送告警（通过 SSE LoggingMessageNotification）。
    HeartbeatManager.on_failure 回调指向这里。
    """
    dead_sessions = set()
    for session in _active_sessions:
        try:
            await session.send_log_message(
                level="error",
                data=alert,
                logger="heartbeat",
            )
        except Exception:
            dead_sessions.add(session)
    # 清理已断开的 session
    _active_sessions.difference_update(dead_sessions)

# ---------------------------------------------------------------------------
# Lifespan — 启动时自动装载 blueprint，关闭时停止所有 tasks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(server):
    # startup: write PID file
    _pid_file.parent.mkdir(parents=True, exist_ok=True)
    _pid_file.write_text(str(os.getpid()))
    logger.info(f"PID file written: {_pid_file} (pid={os.getpid()})")

    # autoload blueprints
    for bp_path in _autoload_paths:
        result = await heartbeat_load_blueprint(bp_path)
        logger.info(f"Autoload: {result}")
    yield
    # shutdown: stop all managers, remove PID file
    for name, mgr in list(_managers.items()):
        await mgr.stop()
        logger.info(f"Shutdown: stopped '{name}'")
    _managers.clear()
    _pid_file.unlink(missing_ok=True)
    logger.info("PID file removed, server shutting down")

# ---------------------------------------------------------------------------
# MCP Server 定义
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="heartbeat",
    instructions=(
        "Heartbeat task scheduler. "
        "Load heartbeat blueprints to start periodic tasks (probes, agents). "
        "Query status, trigger tasks, and adjust intervals."
    ),
    lifespan=lifespan,
)


@mcp.tool()
async def heartbeat_load_blueprint(blueprint_path: str, overrides: str = "") -> str:
    """
    装载一个 heartbeat blueprint 并启动其中定义的所有 tasks。
    blueprint_path: heartbeat.json 文件的路径（绝对路径或相对于项目根）。
    overrides: JSON 字符串，按 task_id 覆盖参数。例如:
      {"probe_claude": {"interval_hours": 2, "timeout": 60}}
    如果该 blueprint 已装载，返回提示。
    """
    path = Path(blueprint_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path = path.resolve()

    if not path.exists():
        return f"Blueprint not found: {path}"

    try:
        bp = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"Failed to parse blueprint: {e}"

    name = bp.get("name", path.stem)
    if name in _managers:
        return f"Blueprint '{name}' is already loaded. Use heartbeat_unload_blueprint first."

    state_path = _state_dir / f"{name}_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # 解析 entity 级覆写
    task_overrides = {}
    if overrides:
        try:
            task_overrides = json.loads(overrides) if isinstance(overrides, str) else overrides
        except Exception as e:
            logger.warning(f"Failed to parse overrides: {e}")

    # 捕获当前调用者的 session（用于后续推送告警）
    _capture_session()

    mgr = HeartbeatManager(path, state_path, on_failure=_broadcast_alert,
                           task_overrides=task_overrides)
    await mgr.start()
    _managers[name] = mgr

    task_ids = list(mgr._tasks.keys())
    logger.info(f"Loaded blueprint '{name}' with tasks: {task_ids}")
    return f"Loaded '{name}' — {len(task_ids)} tasks started: {task_ids}"


@mcp.tool()
async def heartbeat_unload_blueprint(name: str) -> str:
    """
    卸载一个已装载的 heartbeat blueprint，停止其所有 tasks。
    name: blueprint 名称（装载时从 heartbeat.json 的 name 字段读取）。
    如果卸载后无任何 blueprint，MCP Server 将自动退出。
    """
    mgr = _managers.get(name)
    if mgr is None:
        return f"Blueprint '{name}' is not loaded. Loaded: {list(_managers.keys())}"

    await mgr.stop()
    del _managers[name]
    logger.info(f"Unloaded blueprint '{name}'")

    msg = f"Unloaded '{name}' — all tasks stopped."

    # 无任何 blueprint → 延迟退出（让响应先发出去）
    if not _managers:
        logger.info("No blueprints remaining — scheduling server shutdown")
        msg += " Server will shut down (no blueprints remaining)."

        async def _delayed_shutdown():
            await asyncio.sleep(1)  # 让 MCP 响应有时间发回客户端
            _pid_file.unlink(missing_ok=True)
            logger.info("Server exiting (empty)")
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.create_task(_delayed_shutdown())

    return msg


@mcp.tool()
def heartbeat_blueprints() -> str:
    """列出当前已装载的所有 heartbeat blueprints 及其 task 数量。"""
    if not _managers:
        return "No blueprints loaded."

    lines = ["Blueprint | Tasks"]
    lines.append("----------|------")
    for name, mgr in _managers.items():
        lines.append(f"{name} | {len(mgr._tasks)}")
    return "\n".join(lines)


@mcp.tool()
def heartbeat_list() -> str:
    """列出所有已装载 blueprint 中的全部 tasks 及其状态。"""
    if not _managers:
        return "No blueprints loaded."

    lines = ["Blueprint | Task ID | Type | Status | Last Run | Interval(h)"]
    lines.append("----------|---------|------|--------|----------|------------")
    for bp_name, mgr in _managers.items():
        for entry in mgr._tasks.values():
            last = entry.last_run.strftime("%Y-%m-%d %H:%M") if entry.last_run else "never"
            lines.append(
                f"{bp_name} | {entry.id} | {entry.type} | "
                f"{entry.status} | {last} | {entry.interval_hours}"
            )
    return "\n".join(lines)


@mcp.tool()
def heartbeat_status(task_id: str) -> str:
    """查看单个 task 的详细状态。在所有已装载的 blueprint 中搜索。"""
    for bp_name, mgr in _managers.items():
        entry = mgr._tasks.get(task_id)
        if entry is not None:
            return f"[{bp_name}]\n{mgr.get_status(task_id)}"
    known = []
    for mgr in _managers.values():
        known.extend(mgr._tasks.keys())
    return f"Unknown task: {task_id!r}. Known: {known}"


@mcp.tool()
async def heartbeat_run(task_id: str) -> str:
    """立即执行指定 task（不等下次 interval）。"""
    for mgr in _managers.values():
        if task_id in mgr._tasks:
            return await mgr.run_now(task_id)
    known = []
    for mgr in _managers.values():
        known.extend(mgr._tasks.keys())
    return f"Unknown task: {task_id!r}. Known: {known}"


@mcp.tool()
async def heartbeat_set_interval(task_id: str, hours: float) -> str:
    """修改 task 的执行频率（小时）。"""
    for mgr in _managers.values():
        if task_id in mgr._tasks:
            return mgr.set_interval(task_id, hours)
    known = []
    for mgr in _managers.values():
        known.extend(mgr._tasks.keys())
    return f"Unknown task: {task_id!r}. Known: {known}"


@mcp.tool()
def heartbeat_alerts() -> str:
    """
    拉取所有未确认的失败告警（拉取即清空）。
    无告警时返回 "No alerts."。
    """
    all_alerts = []
    for bp_name, mgr in _managers.items():
        for alert in mgr.get_alerts():
            alert["blueprint"] = bp_name
            all_alerts.append(alert)

    if not all_alerts:
        return "No alerts."

    lines = [f"⚠ {len(all_alerts)} alert(s):"]
    for a in all_alerts:
        lines.append(
            f"  [{a['blueprint']}] {a['task_id']} FAILED "
            f"(×{a['consecutive_failures']}) at {a['time']}: {a['error']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Heartbeat MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode (default: stdio)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8100, help="SSE port (default: 8100)")
    parser.add_argument(
        "--autoload", nargs="*", default=[],
        help="Blueprint paths to load on startup"
    )
    args = parser.parse_args()

    # 记录 autoload 路径，lifespan 里异步执行
    _autoload_paths.extend(args.autoload)

    logger.info(f"Starting Heartbeat MCP Server (transport={args.transport})")

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
