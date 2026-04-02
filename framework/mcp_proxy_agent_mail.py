"""
Agent Mail MCP 客户端代理 — framework/mcp_proxy_agent_mail.py

通过 MCP 协议连接 Agent Mail MCP Server，将 send_mail / fetch_inbox /
ack_mail / list_agents / register_agent / unregister_agent 工具暴露给
框架内所有 LLM 节点。

与 HeartbeatMCPProxy 保持相同的接口风格，由 MCPLauncher.ensure_and_connect()
统一管理连接生命周期。
"""

import logging
import os

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class AgentMailProxy:
    """
    Agent Mail MCP 客户端代理。

    connect(server_url) 建立 SSE 连接后，可通过 call_tool() 调用 mail server 工具。
    connect() 由 MCPLauncher.ensure_and_connect() 统一调用，server 已就绪。

    生命周期：
      1. connect()  — 建立 SSE 连接（server 由 MCPLauncher 保证已运行）
      2. call_tool() — 调用 mail server 工具
      3. register()  — 连接成功后注册当前 agent PID（连接即注册）
      4. disconnect() — 断开连接（agent 关闭时注销）
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8200/sse"):
        self._server_url = server_url
        self._session: ClientSession | None = None
        self._read_stream = None
        self._write_stream = None
        self._cm = None          # context manager for sse_client
        self._session_cm = None  # context manager for ClientSession

    async def connect(self):
        """建立 SSE 连接到 Agent Mail MCP Server（server 已由 MCPLauncher 保证就绪）。"""
        self._cm = sse_client(self._server_url)
        try:
            self._read_stream, self._write_stream = await self._cm.__aenter__()
        except Exception:
            self._cm = None
            raise
        self._session_cm = ClientSession(self._read_stream, self._write_stream)
        try:
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()
        except Exception:
            # Roll back the SSE connection on session setup failure
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None
            self._session_cm = None
            self._session = None
            raise
        logger.info(f"[agent_mail_proxy] connected to {self._server_url}")

    async def disconnect(self):
        """断开连接。"""
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
        logger.info("[agent_mail_proxy] disconnected")

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用 Agent Mail MCP Server 上的工具，返回文本结果。"""
        if self._session is None:
            return "Agent Mail MCP Server not connected."
        try:
            result = await self._session.call_tool(name, arguments)
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            return "\n".join(texts) if texts else str(result)
        except Exception as e:
            logger.error(f"[agent_mail_proxy] call_tool({name}) failed: {e}")
            return f"MCP call failed: {e}"

    async def register(self, agent_name: str) -> str:
        """连接后注册当前 agent 的在线状态（连接即注册）。"""
        pid = os.getpid()
        result = await self.call_tool("register_agent", {"name": agent_name, "pid": pid})
        logger.info(f"[agent_mail_proxy] register_agent({agent_name}, pid={pid}): {result}")
        return result

    async def unregister(self, agent_name: str) -> str:
        """agent 关闭时注销在线状态。"""
        result = await self.call_tool("unregister_agent", {"name": agent_name})
        logger.info(f"[agent_mail_proxy] unregister_agent({agent_name}): {result}")
        return result


def make_agent_mail_tools(proxy: AgentMailProxy):
    """
    给定 AgentMailProxy 实例，返回 (registry, schemas) 元组。
    工具调用通过 MCP 协议转发到 Agent Mail MCP Server。

    registry: dict[name → async callable]
    schemas:  dict[name → OpenAI-style function schema]
    """

    async def send_mail(from_agent: str, to: str, subject: str, body: str) -> dict:
        """发送邮件给目标 agent。"""
        import json as _json
        result = await proxy.call_tool("send_mail", {
            "from_agent": from_agent,
            "to": to,
            "subject": subject,
            "body": body,
        })
        try:
            return _json.loads(result)
        except Exception:
            return {"result": result}

    async def fetch_inbox(agent_name: str, unread_only: bool = True) -> dict:
        """查询指定 agent 的收件箱。"""
        import json as _json
        result = await proxy.call_tool("fetch_inbox", {
            "agent_name": agent_name,
            "unread_only": unread_only,
        })
        try:
            return {"mails": _json.loads(result)}
        except Exception:
            return {"result": result}

    async def ack_mail(mail_id: str) -> dict:
        """标记邮件为已读。"""
        import json as _json
        result = await proxy.call_tool("ack_mail", {"mail_id": mail_id})
        try:
            return _json.loads(result)
        except Exception:
            return {"result": result}

    async def list_agents() -> dict:
        """列出所有已知 agent 及在线状态。"""
        import json as _json
        result = await proxy.call_tool("list_agents", {})
        try:
            return {"agents": _json.loads(result)}
        except Exception:
            return {"result": result}

    registry = {
        "send_mail": send_mail,
        "fetch_inbox": fetch_inbox,
        "ack_mail": ack_mail,
        "list_agents": list_agents,
    }

    schemas = {
        "send_mail": {
            "type": "function",
            "function": {
                "name": "send_mail",
                "description": "Send an asynchronous message to another agent via the mail inbox",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_agent": {
                            "type": "string",
                            "description": "Sender agent name",
                        },
                        "to": {
                            "type": "string",
                            "description": "Recipient agent name",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Message subject (e.g. 'monitor_delegate')",
                        },
                        "body": {
                            "type": "string",
                            "description": "Message body as JSON string",
                        },
                    },
                    "required": ["from_agent", "to", "subject", "body"],
                },
            },
        },
        "fetch_inbox": {
            "type": "function",
            "function": {
                "name": "fetch_inbox",
                "description": "Fetch messages from an agent's inbox",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Agent name whose inbox to query",
                        },
                        "unread_only": {
                            "type": "boolean",
                            "description": "If true (default), return only unread messages",
                        },
                    },
                    "required": ["agent_name"],
                },
            },
        },
        "ack_mail": {
            "type": "function",
            "function": {
                "name": "ack_mail",
                "description": "Mark a mail message as read (acknowledged)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mail_id": {
                            "type": "string",
                            "description": "Mail ID to acknowledge",
                        },
                    },
                    "required": ["mail_id"],
                },
            },
        },
        "list_agents": {
            "type": "function",
            "function": {
                "name": "list_agents",
                "description": "List all known agents and their online status",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    }

    return registry, schemas
