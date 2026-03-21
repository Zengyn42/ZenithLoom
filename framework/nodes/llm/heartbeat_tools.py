"""
Heartbeat MCP 客户端代理 — framework/nodes/llm/heartbeat_tools.py

通过 MCP 协议连接 Heartbeat MCP Server，将 heartbeat 工具暴露给
框架内所有 LLM 节点（包括无原生 MCP 支持的 Ollama）。

所有 agent 统一走 MCP Server，共享同一份 heartbeat 状态。

注册模式：TOOL_REGISTRY + TOOL_SCHEMAS 字典，与 tools.py 一致。
"""

import asyncio
import json
import logging

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class HeartbeatMCPProxy:
    """
    MCP 客户端代理。连接 Heartbeat MCP Server，
    提供 call_tool() 方法供框架 tool loop 转发调用。
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8100/sse"):
        self._server_url = server_url
        self._session: ClientSession | None = None
        self._read_stream = None
        self._write_stream = None
        self._cm = None  # context manager for sse_client
        self._session_cm = None  # context manager for ClientSession

    async def connect(self):
        """连接到 MCP Server。"""
        self._cm = sse_client(self._server_url)
        self._read_stream, self._write_stream = await self._cm.__aenter__()
        self._session_cm = ClientSession(self._read_stream, self._write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        logger.info(f"[heartbeat_proxy] connected to {self._server_url}")

    async def disconnect(self):
        """断开连接。"""
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None
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


def make_heartbeat_tools(proxy: HeartbeatMCPProxy):
    """
    给定 MCP 代理实例，返回 (registry, schemas) 元组。
    工具调用通过 MCP 协议转发到 Heartbeat MCP Server。

    registry: dict[name → async callable]
    schemas:  dict[name → OpenAI-style function schema]
    """

    async def heartbeat_load_blueprint(blueprint_path: str) -> dict:
        """装载一个 heartbeat blueprint。"""
        result = await proxy.call_tool("heartbeat_load_blueprint", {"blueprint_path": blueprint_path})
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

    registry = {
        "heartbeat_load_blueprint": heartbeat_load_blueprint,
        "heartbeat_unload_blueprint": heartbeat_unload_blueprint,
        "heartbeat_blueprints": heartbeat_blueprints,
        "heartbeat_list": heartbeat_list,
        "heartbeat_status": heartbeat_status,
        "heartbeat_run": heartbeat_run,
        "heartbeat_set_interval": heartbeat_set_interval,
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
    }

    return registry, schemas
