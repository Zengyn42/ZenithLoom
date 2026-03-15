"""
框架级 Ollama LLM 节点 — framework/llama/node.py

OllamaNode 继承 AgentNode，实现 call_llm() 接口：
  call_llm(prompt, session_id, tools, cwd) → (text, session_id)

通过 Ollama HTTP API 调用本地模型（llama、qwen 等）。
Ollama 无持久 session，返回传入的 session_id 不变。
keep_alive=-1 确保模型常驻 RAM（防止 5 分钟后自动卸载）。

agent.json 配置：
  node_config["model"]:    "llama3.2:3b"              # 模型名
  node_config["endpoint"]: "http://localhost:11434"    # Ollama endpoint（默认）
  node_config["timeout"]:  120                         # 超时秒数（默认）
"""

import json
import logging

import httpx

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.agent_node import AgentNode

logger = logging.getLogger(__name__)


class OllamaNode(AgentNode):
    """
    Ollama LLM 节点，继承 AgentNode。

    通过 Ollama /api/chat 端点调用本地模型。
    基类 AgentNode.__call__() 处理所有图协议逻辑。
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        super().__init__(config, node_config)
        self._model = node_config.get("model", "llama3")
        self._endpoint = node_config.get("endpoint", "http://localhost:11434")
        self._timeout = node_config.get("timeout", 120)
        self._system_prompt = node_config.get("system_prompt", "")
        self._options = node_config.get("options", {})
        logger.info(f"[ollama] model={self._model} endpoint={self._endpoint} options={self._options}")

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
        history: list | None = None,
    ) -> tuple[str, str]:
        """
        调用 Ollama /api/chat（流式）。返回 (text, session_id)。

        session_id 语义：Ollama 无持久 session，返回传入值不变。
        history: 完整 LangGraph 消息历史，用于重建多轮对话上下文。
          历史中最后一条 HumanMessage 即当前 prompt，跳过以免重复。
        tools/cwd：忽略（Ollama 工具调用留待后续实现）。
        keep_alive=-1：模型常驻 RAM，防止 5 分钟后自动卸载。
        """
        from framework.claude.node import get_stream_callback

        if is_debug():
            logger.debug(f"[ollama] model={self._model} prompt_len={len(prompt)} history_len={len(history) if history else 0}")

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # 多轮对话历史（跳过最后一条，即当前 prompt 的 HumanMessage）
        if history and len(history) > 1:
            for msg in history[:-1]:
                msg_type = getattr(msg, "type", "")
                role = "user" if msg_type == "human" else "assistant"
                content = msg.content if isinstance(msg.content, str) else ""
                if content.strip():
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})

        # "think" is a top-level Ollama /api/chat param, NOT inside "options"
        opts = {k: v for k, v in self._options.items() if k != "think"}
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,
        }
        if opts:
            payload["options"] = opts
        if "think" in self._options:
            payload["think"] = self._options["think"]

        stream_cb = get_stream_callback()
        full_text = ""

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._endpoint}/api/chat",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        try:
                            error = json.loads(body).get("error", f"HTTP {response.status_code}")
                        except Exception:
                            error = f"HTTP {response.status_code}"
                        msg = f"[Ollama 错误] {error}"
                        logger.error(msg)
                        return msg, session_id

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        msg_obj = chunk.get("message", {})
                        thinking = msg_obj.get("thinking", "")
                        token = msg_obj.get("content", "")

                        if thinking and stream_cb is not None:
                            stream_cb(thinking, True)

                        if token:
                            full_text += token
                            if stream_cb is not None:
                                stream_cb(token, False)

                        if chunk.get("done"):
                            break

        except httpx.ConnectError as e:
            msg = (
                f"[Ollama 连接失败] 无法连接到 {self._endpoint}，"
                f"请确认 Ollama 正在运行。({e})"
            )
            logger.error(msg)
            return msg, session_id
        except httpx.TimeoutException:
            msg = f"[Ollama 超时] 模型 {self._model} 响应超时（{self._timeout}s）"
            logger.error(msg)
            return msg, session_id

        return full_text, session_id

    def get_recent_history(self, session_id: str, limit: int = 10) -> list:
        """Ollama 无持久 session，返回空列表。"""
        return []


# Backward compatibility alias — builtins.py imports LlamaNode directly from this module
LlamaNode = OllamaNode
