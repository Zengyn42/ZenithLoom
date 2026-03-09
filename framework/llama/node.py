"""
框架级 Llama LLM 节点 — framework/llama/node.py

LlamaNode 实现与 ClaudeNode 相同的接口（call_llm），
通过 Ollama / vLLM HTTP API 调用本地 Llama 模型。

agent.json 配置：
  "llm": "llama"
  "llama_model": "llama-3.3-70b"      # 模型名
  "llama_endpoint": "http://localhost:11434"  # Ollama endpoint

TODO: 当前为存根，完整实现待 Asa agent 上线时补充。
"""

import logging
import os

from framework.config import AgentConfig
from framework.debug import is_debug

logger = logging.getLogger(__name__)


class LlamaNode:
    """
    Llama LLM 节点（Ollama / vLLM）。

    接口与 ClaudeNode 完全一致：
      call_llm(prompt, session_id, tools, cwd) → (text, session_id)
    """

    def __init__(self, config: AgentConfig, system_prompt: str = ""):
        self.config = config
        self.system_prompt = system_prompt
        self._model = config.llama_model if hasattr(config, "llama_model") else "llama3"
        self._endpoint = (
            getattr(config, "llama_endpoint", None)
            or os.getenv("LLAMA_ENDPOINT", "http://localhost:11434")
        )
        logger.info(f"[llama] model={self._model} endpoint={self._endpoint}")

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> tuple[str, str]:
        """
        调用 Llama（Ollama API）。返回 (text, session_id)。

        session_id 语义：Llama 本身无持久 session，
        此处返回传入的 session_id 不变（无 resume 能力）。
        """
        if is_debug():
            logger.debug(f"[llama] prompt_len={len(prompt)} cwd={cwd!r}")

        # TODO: 实现 Ollama /api/chat HTTP 调用
        # import httpx
        # async with httpx.AsyncClient() as client:
        #     resp = await client.post(f"{self._endpoint}/api/chat", json={...})
        #     ...

        raise NotImplementedError(
            "LlamaNode.call_llm 尚未实现。"
            f"请配置 Ollama 并实现 {self._endpoint}/api/chat 调用。"
        )

    def get_recent_history(self, session_id: str, limit: int = 10) -> list:
        """Llama 无持久 session，返回空列表。"""
        return []
