"""
Heartbeat MCP 客户端代理 — framework/nodes/llm/heartbeat_tools.py

通过 MCP 协议连接 Heartbeat MCP Server，将 heartbeat 工具暴露给
框架内所有 LLM 节点（包括无原生 MCP 支持的 Ollama）。

所有 agent 统一走 MCP Server，共享同一份 heartbeat 状态。

生命周期管理：
  - connect() 前先检查 MCP Server 是否运行，未运行则自动启动（detach）
  - 追踪本 proxy 装载的 blueprint 名称
  - cleanup() 卸载本 proxy 装载的所有 blueprint 并断开连接
  - MCP Server 在所有 blueprint 卸载后自动退出

注册模式：TOOL_REGISTRY + TOOL_SCHEMAS 字典，与 tools.py 一致。
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

# PID 文件路径（项目根/data/heartbeat/mcp.pid）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PID_FILE = _PROJECT_ROOT / "data" / "heartbeat" / "mcp.pid"


def _is_server_running() -> bool:
    """检查 Heartbeat MCP Server 进程是否存活（基于 PID 文件）。"""
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 = 存活检测，不发送实际信号
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        # PID 文件残留但进程已死 → 清理
        _PID_FILE.unlink(missing_ok=True)
        return False


def _launch_server(server_url: str) -> bool:
    """
    启动 Heartbeat MCP Server（detach 模式，SSE transport）。
    从 server_url 解析 host/port。
    返回 True 表示启动成功。
    """
    from urllib.parse import urlparse
    parsed = urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8100

    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "mcp_servers.heartbeat",
            "--transport", "sse",
            "--host", host,
            "--port", str(port),
        ],
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # detach from parent process group
    )
    _PID_FILE.write_text(str(proc.pid))
    logger.info(f"[heartbeat_proxy] launched MCP Server pid={proc.pid} at {server_url}")
    return True


class HeartbeatMCPProxy:
    """
    MCP 客户端代理。连接 Heartbeat MCP Server，
    提供 call_tool() 方法供框架 tool loop 转发调用。

    事件推送：
      MCP Server 在 task 执行完成后通过 SSE 发送 LoggingMessageNotification：
        level="info"    — 正常执行报告
        level="warning" — 阈值超标告警
        level="error"   — 执行失败告警
      ClientSession 的 logging_callback 收到后，调用 _alert_callback。
      接口层通过 set_alert_callback() 注册具体处理函数。

    生命周期：
      1. connect() — 检测 server 是否运行，未运行则自动启动；然后建立 SSE 连接
      2. set_alert_callback(fn) — 注册告警回调（SSE push 驱动）
      3. load_blueprint(path) — 装载 blueprint 并记录名称
      4. cleanup() — 卸载所有本 proxy 装载的 blueprint，断开连接
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8100/sse"):
        self._server_url = server_url
        self._session: ClientSession | None = None
        self._read_stream = None
        self._write_stream = None
        self._cm = None  # context manager for sse_client
        self._session_cm = None  # context manager for ClientSession
        self._loaded_blueprints: list[str] = []  # 本 proxy 装载的 blueprint 名称
        self._alert_callback = None  # async callable(alert_dict) | None

    def set_alert_callback(self, callback) -> None:
        """注册事件回调。callback 签名: async def(event: dict) -> None。event 含 level 字段。"""
        self._alert_callback = callback

    async def _on_log_message(self, params) -> None:
        """
        MCP LoggingMessageNotification 回调。
        MCP Server 推送事件：
          level="error"   → 执行失败告警
          level="warning" → 阈值超标告警
          level="info"    → 正常执行报告
        所有级别都转发给 _alert_callback（由上层根据 level 决定处理方式）。
        """
        if self._alert_callback is None:
            return
        # params.level: str, params.data: Any, params.logger: str | None
        if isinstance(params.data, dict):
            # 确保 event 里有 level 字段
            event = params.data
            if "level" not in event:
                event["level"] = params.level
            try:
                await self._alert_callback(event)
            except Exception as e:
                logger.error(f"[heartbeat_proxy] event callback error: {e}")

    def _create_session(self, read_stream, write_stream) -> ClientSession:
        """创建 ClientSession 并注入 logging_callback。"""
        return ClientSession(
            read_stream,
            write_stream,
            logging_callback=self._on_log_message,
        )

    async def connect(self):
        """
        连接到 MCP Server。
        如果 Server 未运行，自动启动（detach）并等待就绪。
        """
        if not _is_server_running():
            _launch_server(self._server_url)
            # 等待 server 就绪（最多 10 秒，每 0.5 秒重试）
            for attempt in range(20):
                await asyncio.sleep(0.5)
                try:
                    self._cm = sse_client(self._server_url)
                    self._read_stream, self._write_stream = await self._cm.__aenter__()
                    self._session_cm = self._create_session(self._read_stream, self._write_stream)
                    self._session = await self._session_cm.__aenter__()
                    await self._session.initialize()
                    logger.info(f"[heartbeat_proxy] connected to {self._server_url} (attempt {attempt + 1})")
                    return
                except Exception:
                    # 连接失败，清理半成品
                    await self._cleanup_partial()
                    continue
            raise ConnectionError(
                f"Heartbeat MCP Server failed to start within 10s at {self._server_url}"
            )

        # Server 已在运行，直接连接
        self._cm = sse_client(self._server_url)
        self._read_stream, self._write_stream = await self._cm.__aenter__()
        self._session_cm = self._create_session(self._read_stream, self._write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        logger.info(f"[heartbeat_proxy] connected to {self._server_url}")

    async def _cleanup_partial(self):
        """清理连接失败的半成品资源。"""
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None
        self._session = None

    async def disconnect(self):
        """断开连接。"""
        await self._cleanup_partial()
        logger.info("[heartbeat_proxy] disconnected")

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用 MCP Server 上的工具，返回文本结果。"""
        if self._session is None:
            return "Heartbeat MCP Server not connected."
        try:
            result = await self._session.call_tool(name, arguments)
            # 提取文本内容
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            return "\n".join(texts) if texts else str(result)
        except Exception as e:
            logger.error(f"[heartbeat_proxy] call_tool({name}) failed: {e}")
            return f"MCP call failed: {e}"

    async def load_blueprint(self, blueprint_path: str, overrides: dict | None = None) -> str:
        """装载 blueprint 并记录名称（供 cleanup 时卸载）。"""
        args = {"blueprint_path": blueprint_path}
        if overrides:
            import json as _json
            args["overrides"] = _json.dumps(overrides)
        result = await self.call_tool("heartbeat_load_blueprint", args)
        # 从返回结果中提取 blueprint 名称（格式: "Loaded '<name>' — ..."）
        if result.startswith("Loaded '"):
            name = result.split("'")[1]
            if name not in self._loaded_blueprints:
                self._loaded_blueprints.append(name)
        logger.info(f"[heartbeat_proxy] load_blueprint: {result}")
        return result

    async def cleanup(self):
        """
        卸载本 proxy 装载的所有 blueprint，然后断开连接。
        MCP Server 会在所有 blueprint 都被卸载后自动退出。
        """
        if self._session is None:
            return

        for name in list(self._loaded_blueprints):
            try:
                result = await self.call_tool("heartbeat_unload_blueprint", {"name": name})
                logger.info(f"[heartbeat_proxy] cleanup unload '{name}': {result}")
            except Exception as e:
                logger.warning(f"[heartbeat_proxy] cleanup unload '{name}' failed: {e}")
        self._loaded_blueprints.clear()

        await self.disconnect()


def make_heartbeat_tools(proxy: HeartbeatMCPProxy):
    """
    给定 MCP 代理实例，返回 (registry, schemas) 元组。
    工具调用通过 MCP 协议转发到 Heartbeat MCP Server。

    registry: dict[name → async callable]
    schemas:  dict[name → OpenAI-style function schema]
    """

    async def heartbeat_load_blueprint(blueprint_path: str) -> dict:
        """装载一个 heartbeat blueprint。"""
        result = await proxy.load_blueprint(blueprint_path)
        return {"result": result}

    async def heartbeat_unload_blueprint(name: str) -> dict:
        """卸载一个 heartbeat blueprint。"""
        result = await proxy.call_tool("heartbeat_unload_blueprint", {"name": name})
        return {"result": result}

    async def heartbeat_blueprints() -> dict:
        """列出已装载的 blueprints。"""
        result = await proxy.call_tool("heartbeat_blueprints", {})
        return {"result": result}

    async def heartbeat_list() -> dict:
        """列出所有 heartbeat 任务的状态。"""
        result = await proxy.call_tool("heartbeat_list", {})
        return {"result": result}

    async def heartbeat_status(task_id: str) -> dict:
        """查看单个任务详情。"""
        result = await proxy.call_tool("heartbeat_status", {"task_id": task_id})
        return {"result": result}

    async def heartbeat_run(task_id: str) -> dict:
        """立即执行指定的 heartbeat 任务。"""
        result = await proxy.call_tool("heartbeat_run", {"task_id": task_id})
        return {"result": result}

    async def heartbeat_set_interval(task_id: str, hours: float) -> dict:
        """修改指定 heartbeat 任务的执行频率。"""
        result = await proxy.call_tool("heartbeat_set_interval", {"task_id": task_id, "hours": hours})
        return {"result": result}

    async def heartbeat_alerts() -> dict:
        """拉取未确认的失败告警（拉取即清空）。"""
        result = await proxy.call_tool("heartbeat_alerts", {})
        return {"result": result}

    registry = {
        "heartbeat_load_blueprint": heartbeat_load_blueprint,
        "heartbeat_unload_blueprint": heartbeat_unload_blueprint,
        "heartbeat_blueprints": heartbeat_blueprints,
        "heartbeat_list": heartbeat_list,
        "heartbeat_status": heartbeat_status,
        "heartbeat_run": heartbeat_run,
        "heartbeat_set_interval": heartbeat_set_interval,
        "heartbeat_alerts": heartbeat_alerts,
    }

    schemas = {
        "heartbeat_load_blueprint": {
            "type": "function",
            "function": {
                "name": "heartbeat_load_blueprint",
                "description": "Load a heartbeat blueprint to start its periodic tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "blueprint_path": {
                            "type": "string",
                            "description": "Path to heartbeat.json blueprint file",
                        },
                    },
                    "required": ["blueprint_path"],
                },
            },
        },
        "heartbeat_unload_blueprint": {
            "type": "function",
            "function": {
                "name": "heartbeat_unload_blueprint",
                "description": "Unload a heartbeat blueprint and stop all its tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Blueprint name to unload",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
        "heartbeat_blueprints": {
            "type": "function",
            "function": {
                "name": "heartbeat_blueprints",
                "description": "List all loaded heartbeat blueprints",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        "heartbeat_list": {
            "type": "function",
            "function": {
                "name": "heartbeat_list",
                "description": "List all heartbeat tasks with status, last run, and interval",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        "heartbeat_status": {
            "type": "function",
            "function": {
                "name": "heartbeat_status",
                "description": "Get detailed status of a specific heartbeat task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID to query",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        },
        "heartbeat_run": {
            "type": "function",
            "function": {
                "name": "heartbeat_run",
                "description": "Immediately execute a heartbeat task by its ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID to execute",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        },
        "heartbeat_set_interval": {
            "type": "function",
            "function": {
                "name": "heartbeat_set_interval",
                "description": "Change the execution interval of a heartbeat task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID to modify",
                        },
                        "hours": {
                            "type": "number",
                            "description": "New interval in hours (must be > 0)",
                        },
                    },
                    "required": ["task_id", "hours"],
                },
            },
        },
        "heartbeat_alerts": {
            "type": "function",
            "function": {
                "name": "heartbeat_alerts",
                "description": "Fetch and clear unacknowledged failure alerts from heartbeat tasks",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    }

    return registry, schemas
