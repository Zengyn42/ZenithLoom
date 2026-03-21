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
_autoload_paths: list[str] = []  # 启动时自动装载的 blueprint 路径

# ---------------------------------------------------------------------------
# Lifespan — 启动时自动装载 blueprint，关闭时停止所有 tasks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(server):
    # startup: autoload blueprints
    for bp_path in _autoload_paths:
        result = await heartbeat_load_blueprint(bp_path)
        logger.info(f"Autoload: {result}")
    yield
    # shutdown: stop all managers
    for name, mgr in list(_managers.items()):
        await mgr.stop()
        logger.info(f"Shutdown: stopped '{name}'")
    _managers.clear()

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
async def heartbeat_load_blueprint(blueprint_path: str) -> str:
    """
    装载一个 heartbeat blueprint 并启动其中定义的所有 tasks。
    blueprint_path: heartbeat.json 文件的路径（绝对路径或相对于项目根）。
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

    mgr = HeartbeatManager(path, state_path)
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
    """
    mgr = _managers.get(name)
    if mgr is None:
        return f"Blueprint '{name}' is not loaded. Loaded: {list(_managers.keys())}"

    await mgr.stop()
    del _managers[name]
    logger.info(f"Unloaded blueprint '{name}'")
    return f"Unloaded '{name}' — all tasks stopped."


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
