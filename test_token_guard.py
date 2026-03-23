#!/usr/bin/env python3
"""
Token Guard 集成测试 — 验证所有 LLM 节点类型的 token 安全阀。

Mock 所有外部调用（Claude SDK、Ollama HTTP、Gemini），
只验证 token_guard 在超限时正确拦截。

运行：
    pytest test_token_guard.py -v
"""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

# ── 确保 imports ──
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import AIMessage, HumanMessage

from framework.config import AgentConfig
from framework.token_guard import (
    TokenLimitExceeded,
    check_before_llm,
    estimate_tokens,
    get_default_limit,
    LIMITS_BY_TYPE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> AgentConfig:
    """最小 AgentConfig mock。"""
    cfg = MagicMock(spec=AgentConfig)
    cfg.tools = []
    cfg.db_path = "/tmp/test.db"
    cfg.sessions_file = "/tmp/sessions.json"
    cfg.name = "test"
    return cfg


def _make_big_history(char_count: int) -> list:
    """生成一个足够大的 messages 历史。"""
    content = "x" * char_count
    return [HumanMessage(content=content)]


def _make_state(history_chars: int = 100, routing_target: str = "") -> dict:
    return {
        "messages": [HumanMessage(content="a" * history_chars)],
        "node_sessions": {},
        "routing_target": routing_target,
        "routing_context": "",
        "workspace": "",
        "project_root": "",
        "project_meta": {},
        "rollback_reason": "",
        "knowledge_vault": "",
        "project_docs": "",
        "subgraph_call_counts": {},
        "consult_count": 0,
    }


# ---------------------------------------------------------------------------
# Unit tests: token_guard.py
# ---------------------------------------------------------------------------

class TestTokenGuardUnit:

    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("hello") > 0
        # 300 chars ÷ 3 = 100 tokens
        assert estimate_tokens("x" * 300) == 100

    def test_default_limits_by_type(self):
        assert get_default_limit("CLAUDE_SDK") == 50_000
        assert get_default_limit("GEMINI_API") == 50_000
        assert get_default_limit("OLLAMA") == 1_000_000
        assert get_default_limit("LOCAL_VLLM") == 1_000_000

    def test_check_passes_under_limit(self):
        result = check_before_llm(prompt="hello", node_id="test", limit=1000)
        assert result < 1000

    def test_check_raises_over_limit(self):
        big = "x" * 60000  # 60k chars ≈ 20k tokens
        with pytest.raises(TokenLimitExceeded) as exc_info:
            check_before_llm(prompt=big, node_id="test_node", limit=10000)
        assert exc_info.value.estimated_tokens > 10000
        assert exc_info.value.limit == 10000
        assert "test_node" in str(exc_info.value)

    def test_check_with_messages(self):
        msgs = [{"role": "user", "content": "x" * 300000}]  # 300k chars ≈ 100k tokens
        with pytest.raises(TokenLimitExceeded):
            check_before_llm(messages=msgs, node_id="test", limit=50000)

    def test_check_with_history(self):
        history = [HumanMessage(content="x" * 300000)]
        with pytest.raises(TokenLimitExceeded):
            check_before_llm(prompt="hello", history=history, node_id="test", limit=50000)

    def test_ollama_limit_is_1m(self):
        """Ollama 300k chars ≈ 100k tokens，低于 1M 限制，不应触发。"""
        msgs = [{"role": "user", "content": "x" * 300000}]
        result = check_before_llm(messages=msgs, node_id="ollama_test", limit=1_000_000)
        assert result < 1_000_000

    def test_claude_limit_blocks_at_50k(self):
        """Claude 300k chars ≈ 100k tokens，超过 50k 限制。"""
        history = [HumanMessage(content="x" * 300000)]
        with pytest.raises(TokenLimitExceeded) as exc_info:
            check_before_llm(prompt="hi", history=history, node_id="claude", limit=50000)
        assert exc_info.value.estimated_tokens > 50000


# ---------------------------------------------------------------------------
# Integration: LlmNode (Claude) token guard
# ---------------------------------------------------------------------------

class TestClaudeNodeTokenGuard:

    @pytest.mark.asyncio
    async def test_claude_node_blocks_on_big_history(self):
        """ClaudeSDKNode 在 history 超 50k tokens 时应返回 abort，不调 LLM。"""
        from framework.nodes.llm.claude import ClaudeSDKNode

        cfg = _make_config()
        node_config = {
            "id": "test_claude",
            "type": "CLAUDE_SDK",
            "session_key": "test_claude",
        }
        node = ClaudeSDKNode(cfg, node_config)

        # 500k chars ≈ 166k tokens >> 50k limit
        state = _make_state(history_chars=500_000)

        with patch.object(node, "call_llm", new_callable=AsyncMock) as mock_llm:
            result = await node(state)
            mock_llm.assert_not_called()

        assert "Token 安全阀" in result["messages"][0].content
        assert result.get("routing_target") == "__end__"
        logger.info("PASS Claude node blocked on big history")

    @pytest.mark.asyncio
    async def test_claude_node_passes_small_history(self):
        """ClaudeSDKNode 在 history 低于 50k tokens 时正常调 LLM。"""
        from framework.nodes.llm.claude import ClaudeSDKNode

        cfg = _make_config()
        node_config = {
            "id": "test_claude",
            "type": "CLAUDE_SDK",
            "session_key": "test_claude",
        }
        node = ClaudeSDKNode(cfg, node_config)
        state = _make_state(history_chars=100)

        with patch.object(node, "call_llm", new_callable=AsyncMock, return_value=("OK", "sid")) as mock_llm:
            result = await node(state)
            mock_llm.assert_called_once()

        assert result["messages"][0].content == "OK"
        logger.info("PASS Claude node passes on small history")

    @pytest.mark.asyncio
    async def test_claude_custom_token_limit(self):
        """node_config["token_limit"] 覆盖默认值。"""
        from framework.nodes.llm.claude import ClaudeSDKNode

        cfg = _make_config()
        node_config = {
            "id": "test_claude_custom",
            "type": "CLAUDE_SDK",
            "session_key": "test_claude_custom",
            "token_limit": 200_000,  # 自定义 200k
        }
        node = ClaudeSDKNode(cfg, node_config)
        assert node._token_limit == 200_000

        # 300k chars ≈ 100k tokens < 200k limit → 应通过
        state = _make_state(history_chars=300_000)
        with patch.object(node, "call_llm", new_callable=AsyncMock, return_value=("OK", "sid")) as mock_llm:
            result = await node(state)
            mock_llm.assert_called_once()

        assert result["messages"][0].content == "OK"
        logger.info("PASS Claude custom token_limit works")


# ---------------------------------------------------------------------------
# Integration: OllamaNode token guard
# ---------------------------------------------------------------------------

class TestOllamaNodeTokenGuard:

    @pytest.mark.asyncio
    async def test_ollama_tool_mode_blocks_on_huge_session(self):
        """OllamaNode tool 模式在 session 超 1M tokens 时拦截。"""
        from framework.nodes.llm.ollama import OllamaNode

        cfg = _make_config()
        node_config = {
            "id": "test_ollama",
            "type": "OLLAMA",
            "model": "qwen3.5:27b",
            "tools": ["read_file", "write_file"],
            "session_key": "test_ollama",
        }
        node = OllamaNode(cfg, node_config)
        assert node._token_limit == 1_000_000

        # 4M chars ≈ 1.33M tokens >> 1M limit
        state = _make_state(history_chars=100)
        state["ollama_sessions"] = {"existing_uuid": [
            {"role": "user", "content": "x" * 4_000_000}
        ]}
        state["node_sessions"] = {"test_ollama": "existing_uuid"}

        result = await node(state)
        assert "Token 安全阀" in result["messages"][0].content
        logger.info("PASS Ollama tool mode blocked on huge session")

    @pytest.mark.asyncio
    async def test_ollama_tool_mode_passes_normal(self):
        """OllamaNode tool 模式正常 session 应通过。"""
        from framework.nodes.llm.ollama import OllamaNode

        cfg = _make_config()
        node_config = {
            "id": "test_ollama",
            "type": "OLLAMA",
            "model": "qwen3.5:27b",
            "tools": ["read_file"],
            "session_key": "test_ollama",
        }
        node = OllamaNode(cfg, node_config)

        state = _make_state(history_chars=100)

        # Mock _chat_completions to return a simple text response (no tool calls)
        with patch.object(node, "_chat_completions", new_callable=AsyncMock, return_value={
            "role": "assistant", "content": "done",
        }):
            result = await node(state)

        assert "Token 安全阀" not in result.get("messages", [{}])[0].content if result.get("messages") else True
        logger.info("PASS Ollama tool mode passes on normal session")

    @pytest.mark.asyncio
    async def test_ollama_stream_mode_blocks(self):
        """OllamaNode 流式模式（无 tools）在超限时由基类 LlmNode 拦截。"""
        from framework.nodes.llm.ollama import OllamaNode

        cfg = _make_config()
        node_config = {
            "id": "test_ollama_stream",
            "type": "OLLAMA",
            "model": "qwen3.5:27b",
            # no tools → 走 super().__call__ → LlmNode.__call__
        }
        node = OllamaNode(cfg, node_config)

        # 500k chars → ~166k tokens >> Ollama 1M? No, need bigger.
        # 4M chars → ~1.33M tokens > 1M limit
        state = _make_state(history_chars=4_000_000)

        with patch.object(node, "call_llm", new_callable=AsyncMock) as mock_llm:
            result = await node(state)
            mock_llm.assert_not_called()

        assert "Token 安全阀" in result["messages"][0].content
        logger.info("PASS Ollama stream mode blocked on huge history")


# ---------------------------------------------------------------------------
# Integration: GeminiCodeAssistNode token guard
# ---------------------------------------------------------------------------

class TestGeminiNodeTokenGuard:

    @pytest.mark.asyncio
    async def test_gemini_api_blocks_on_big_history(self):
        """GeminiCodeAssistNode 在超 50k tokens 时拦截。"""
        from framework.nodes.llm.gemini import GeminiCodeAssistNode

        cfg = _make_config()
        node_config = {
            "id": "test_gemini",
            "type": "GEMINI_API",
            "session_key": "test_gemini",
        }
        node = GeminiCodeAssistNode(cfg, node_config)
        assert node._token_limit == 50_000

        # 500k chars ≈ 166k tokens >> 50k
        state = _make_state(history_chars=500_000)

        result = await node(state)
        assert "Token 安全阀" in result["messages"][0].content
        logger.info("PASS Gemini API node blocked on big history")

    @pytest.mark.asyncio
    async def test_gemini_cli_blocks_on_big_history(self):
        """GeminiCLINode 在超 50k tokens 时拦截。"""
        from framework.nodes.llm.gemini import GeminiCLINode

        cfg = _make_config()
        node_config = {
            "id": "test_gemini_cli",
            "type": "GEMINI_CLI",
            "session_key": "test_gemini_cli",
        }
        node = GeminiCLINode(cfg, node_config)
        assert node._token_limit == 50_000

        state = _make_state(history_chars=500_000)

        result = await node(state)
        assert "Token 安全阀" in result["messages"][0].content
        logger.info("PASS Gemini CLI node blocked on big history")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTokenGuardEdgeCases:

    def test_env_var_override(self):
        """BB_TOKEN_LIMIT 环境变量兜底生效（未知类型）。
        用 monkeypatch 直接改 _ENV_LIMIT，避免 reload 副作用。"""
        import framework.token_guard as tg
        original = tg._ENV_LIMIT
        try:
            tg._ENV_LIMIT = 99999
            assert tg.get_default_limit("UNKNOWN_TYPE") == 99999
            # 已知类型不受影响
            assert tg.get_default_limit("CLAUDE_SDK") == 50_000
        finally:
            tg._ENV_LIMIT = original

    @pytest.mark.asyncio
    async def test_token_limit_from_node_config(self):
        """node_config["token_limit"] 最高优先。"""
        from framework.nodes.llm.claude import ClaudeSDKNode

        cfg = _make_config()
        # Ollama type 默认 1M，但 token_limit 显式设为 5k
        node_config = {
            "id": "custom_limit",
            "type": "OLLAMA",
            "token_limit": 5000,
        }
        node = ClaudeSDKNode(cfg, node_config)
        assert node._token_limit == 5000

        # 30k chars ≈ 10k tokens > 5k custom limit → 应拦截
        state = _make_state(history_chars=30_000)
        with patch.object(node, "call_llm", new_callable=AsyncMock) as mock_llm:
            result = await node(state)
            mock_llm.assert_not_called()

        assert "Token 安全阀" in result["messages"][0].content
        logger.info("PASS node_config token_limit override works")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
