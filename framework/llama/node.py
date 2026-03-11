"""
框架级 Llama LLM 节点 — framework/llama/node.py

LlamaNode 继承 AgentNode，实现 call_llm() 接口：
  call_llm(prompt, session_id, tools, cwd) → (text, session_id)

通过 Ollama / vLLM HTTP API 调用本地 Llama 模型。
Llama 本身无持久 session，返回传入的 session_id 不变（无 resume 能力）。

agent.json 配置：
  "llm": "llama"
  node_config["model"]: "llama-3.3-70b"          # 模型名
  node_config["endpoint"]: "http://localhost:11434"  # Ollama endpoint
  node_config["resource_lock"]: "GPU_0_VRAM_22GB"   # 可选：GPU 资源锁

TODO: 当前为存根，完整实现待 Asa agent 上线时补充。
"""

import logging

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.agent_node import AgentNode

logger = logging.getLogger(__name__)


class LlamaNode(AgentNode):
    """
    Llama LLM 节点（Ollama / vLLM），继承 AgentNode。

    call_llm() 实现 Ollama API 调用；
    基类 AgentNode.__call__() 处理所有图协议逻辑。
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        super().__init__(config, node_config)
        self._model = node_config.get("model", "llama3")
        self._endpoint = node_config.get("endpoint") or node_config.get("llama_endpoint")
        if not self._endpoint:
            raise ValueError(
                "LlamaNode: 'endpoint' required in node_config "
                "(e.g. {\"endpoint\": \"http://localhost:11434\"})"
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
        返回传入的 session_id 不变（无 resume 能力）。
        """
        if is_debug():
            logger.debug(f"[llama] prompt_len={len(prompt)} cwd={cwd!r}")

        # TODO: 实现 Ollama /api/chat HTTP 调用
        # import httpx
        # async with httpx.AsyncClient() as client:
        #     resp = await client.post(f"{self._endpoint}/api/chat", json={
        #         "model": self._model,
        #         "messages": [{"role": "user", "content": prompt}],
        #         "stream": False,
        #     })
        #     data = resp.json()
        #     return data["message"]["content"], session_id

        raise NotImplementedError(
            "LlamaNode.call_llm 尚未实现。"
            f"请配置 Ollama 并实现 {self._endpoint}/api/chat 调用。"
        )

    def get_recent_history(self, session_id: str, limit: int = 10) -> list:
        """Llama 无持久 session，返回空列表。"""
        return []
