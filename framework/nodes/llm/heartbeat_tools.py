"""
Heartbeat 工具函数 — framework/nodes/llm/heartbeat_tools.py

提供 3 个工具函数供 LLM tool-calling 使用：
  heartbeat_list()                     → 列出所有 heartbeat 任务状态
  heartbeat_run(task_id)               → 立即执行指定任务
  heartbeat_set_interval(task_id, h)   → 修改任务执行频率

工具函数通过 EntityLoader 持有的 HeartbeatManager 实例获取 manager 引用。
不使用模块级全局单例。

注册模式：TOOL_REGISTRY + TOOL_SCHEMAS 字典，与 tools.py 一致。
调用方通过 EntityLoader._heartbeat_manager 获取 manager，
再将 manager 注入工具函数的闭包中。
"""

import logging

logger = logging.getLogger(__name__)


def make_heartbeat_tools(manager):
    """
    给定 HeartbeatManager 实例，返回 (registry, schemas) 元组。

    registry: dict[name → async callable]
    schemas:  dict[name → OpenAI-style function schema]

    调用方（EntityLoader）负责将这些工具注册到 LLM 节点的 tool 列表中。
    """

    async def heartbeat_list() -> dict:
        """列出所有 heartbeat 任务的状态。"""
        return {"result": manager.list_tasks()}

    async def heartbeat_run(task_id: str) -> dict:
        """立即执行指定的 heartbeat 任务。"""
        result = await manager.run_now(task_id)
        return {"result": result}

    async def heartbeat_set_interval(task_id: str, hours: float) -> dict:
        """修改指定 heartbeat 任务的执行频率。"""
        result = manager.set_interval(task_id, hours)
        return {"result": result}

    registry = {
        "heartbeat_list": heartbeat_list,
        "heartbeat_run": heartbeat_run,
        "heartbeat_set_interval": heartbeat_set_interval,
    }

    schemas = {
        "heartbeat_list": {
            "type": "function",
            "function": {
                "name": "heartbeat_list",
                "description": "List all heartbeat tasks and their current status (ID, type, status, last run, interval)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        "heartbeat_run": {
            "type": "function",
            "function": {
                "name": "heartbeat_run",
                "description": "Immediately execute a heartbeat task by its ID and return the result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The task ID to execute (e.g. 'probe_ollama')",
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
