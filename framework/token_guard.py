"""
Token 安全阀 — framework/token_guard.py

在每次 LLM 调用前检查 session 的预估 token 大小。
超过阈值时立即中止调用，返回错误信息，防止失控循环烧光 token。

按节点类型分级：
  CLAUDE_SDK / GEMINI_API / GEMINI_CLI → 50k tokens（云端 API，烧钱）
  OLLAMA / LOCAL_VLLM                  → 1M tokens（本地推理，不烧钱）

优先级：node_config["token_limit"] > 按类型默认值 > 环境变量 BB_TOKEN_LIMIT
"""

import logging
import os

logger = logging.getLogger(__name__)

# 环境变量兜底（未指定节点类型时的默认值）
_ENV_LIMIT: int = int(os.environ.get("BB_TOKEN_LIMIT", "50000"))

# 按节点类型的默认 token 上限
LIMITS_BY_TYPE: dict[str, int] = {
    "CLAUDE_SDK": 50_000,
    "GEMINI_API": 50_000,
    "GEMINI_CLI": 50_000,
    "OLLAMA":     1_000_000,
    "LOCAL_VLLM": 1_000_000,
}

# 保守估算：1 token ≈ 3 chars（混合中英文）
CHARS_PER_TOKEN: float = 3.0


class TokenLimitExceeded(Exception):
    """Session token 数超过安全阈值。"""

    def __init__(self, estimated_tokens: int, limit: int, node_id: str = ""):
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        self.node_id = node_id
        super().__init__(
            f"[Token 安全阀] {node_id}: 预估 {estimated_tokens:,} tokens "
            f"(阈值 {limit:,})，中止 LLM 调用。可能存在死循环。"
        )


def get_default_limit(node_type: str = "") -> int:
    """根据节点类型返回默认 token 上限。"""
    return LIMITS_BY_TYPE.get(node_type, _ENV_LIMIT)


def estimate_tokens(text: str) -> int:
    """粗估文本 token 数。"""
    return int(len(text) / CHARS_PER_TOKEN)


def estimate_tokens_messages(messages: list) -> int:
    """
    粗估消息列表的总 token 数。

    支持：
      - dict 格式 {"role": ..., "content": ...}（Ollama 原生格式）
      - LangChain Message 对象（有 .content 属性）
      - 纯字符串
    """
    total_chars = 0
    for msg in messages:
        if isinstance(msg, dict):
            total_chars += len(str(msg.get("content", "")))
        elif hasattr(msg, "content"):
            total_chars += len(str(msg.content))
        elif isinstance(msg, str):
            total_chars += len(msg)
    return int(total_chars / CHARS_PER_TOKEN)


def check_before_llm(
    *,
    prompt: str = "",
    messages: list | None = None,
    history: list | None = None,
    node_id: str = "",
    limit: int | None = None,
) -> int:
    """
    LLM 调用前的 token 预检。超限抛 TokenLimitExceeded。

    Args:
        prompt:   当前 prompt 文本
        messages: Ollama 格式的完整消息列表（与 prompt 二选一）
        history:  LangGraph state["messages"] 历史（不应包含已作为 prompt 传入的那条消息，避免双重计数）
        node_id:  节点 ID（用于日志）
        limit:    节点级阈值（由 LlmNode.__init__ 从 node_config 读取）

    Returns:
        预估 token 数
    """
    effective_limit = limit or _ENV_LIMIT
    total = 0

    if messages:
        total += estimate_tokens_messages(messages)
    else:
        total += estimate_tokens(prompt)
        if history:
            total += estimate_tokens_messages(history)

    if total > effective_limit:
        logger.error(
            f"[token_guard] {node_id}: estimated {total:,} tokens > limit {effective_limit:,}. BLOCKED."
        )
        raise TokenLimitExceeded(total, effective_limit, node_id)

    logger.debug(f"[token_guard] {node_id}: ~{total:,} tokens (limit {effective_limit:,})")
    return total
