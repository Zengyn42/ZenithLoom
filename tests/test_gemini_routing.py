"""
test_gemini_routing — Gemini 节点 enable_routing 功能测试

验证 GeminiCLINode 和 GeminiCodeAssistNode 在 enable_routing=true 时：
  - 正确解析输出中的路由信号
  - enable_routing=false（默认）时清除路由信号
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage

from framework.config import AgentConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config() -> AgentConfig:
    return AgentConfig(name="test_gemini")


def _make_state(user_msg: str = "hello") -> dict:
    return {
        "messages": [HumanMessage(content=user_msg)],
        "routing_target": "",
        "routing_context": "",
        "workspace": "/tmp",
        "node_sessions": {},
        "rollback_reason": "",
    }


# ---------------------------------------------------------------------------
# GeminiCLINode tests
# ---------------------------------------------------------------------------

class TestGeminiCLINodeRouting:
    """GeminiCLINode enable_routing 测试。"""

    @pytest.fixture
    def node_with_routing(self):
        from framework.nodes.llm.gemini import GeminiCLINode
        cfg = _make_config()
        node_config = {
            "id": "gemini_main",
            "enable_routing": True,
            "model": "gemini-2.5-pro",
            "timeout": 30,
        }
        node = GeminiCLINode(cfg, node_config)
        return node

    @pytest.fixture
    def node_without_routing(self):
        from framework.nodes.llm.gemini import GeminiCLINode
        cfg = _make_config()
        node_config = {
            "id": "gemini_helper",
            "model": "gemini-2.5-pro",
            "timeout": 30,
        }
        node = GeminiCLINode(cfg, node_config)
        return node

    @pytest.mark.asyncio
    async def test_routing_signal_detected(self, node_with_routing):
        """enable_routing=true 时，输出含路由信号 → 写入 routing_target。"""
        reply = '{"route": "knowledge_shelf", "context": "搜索笔记"}\n'
        with patch.object(node_with_routing, "call_llm", new_callable=AsyncMock, return_value=(reply, "sid_123")):
            result = await node_with_routing(_make_state())
        assert result["routing_target"] == "knowledge_shelf"
        assert result["routing_context"] == "搜索笔记"

    @pytest.mark.asyncio
    async def test_no_routing_signal(self, node_with_routing):
        """enable_routing=true 但输出无路由信号 → 清除 routing_target。"""
        reply = "这是一个普通回复，没有路由信号。"
        with patch.object(node_with_routing, "call_llm", new_callable=AsyncMock, return_value=(reply, "sid_123")):
            result = await node_with_routing(_make_state())
        assert result["routing_target"] == ""
        assert result["routing_context"] == ""

    @pytest.mark.asyncio
    async def test_routing_disabled_by_default(self, node_without_routing):
        """enable_routing 默认 false → 即使输出含路由信号也清除。"""
        reply = '{"route": "knowledge_shelf", "context": "搜索笔记"}\n'
        with patch.object(node_without_routing, "call_llm", new_callable=AsyncMock, return_value=(reply, "sid_123")):
            result = await node_without_routing(_make_state())
        assert result["routing_target"] == ""
        assert result["routing_context"] == ""

    @pytest.mark.asyncio
    async def test_no_routing_resets_state(self, node_with_routing):
        """无路由信号时清空 rollback_reason。"""
        reply = "普通回复"
        with patch.object(node_with_routing, "call_llm", new_callable=AsyncMock, return_value=(reply, "sid_123")):
            result = await node_with_routing(_make_state())
        assert result["rollback_reason"] == ""


# ---------------------------------------------------------------------------
# GeminiCodeAssistNode tests
# ---------------------------------------------------------------------------

class TestGeminiCodeAssistNodeRouting:
    """GeminiCodeAssistNode enable_routing 测试。"""

    @pytest.fixture
    def node_with_routing(self):
        from framework.nodes.llm.gemini import GeminiCodeAssistNode
        cfg = _make_config()
        node_config = {
            "id": "gemini_api_main",
            "enable_routing": True,
            "model": "gemini-2.5-pro",
        }
        node = GeminiCodeAssistNode(cfg, node_config)
        return node

    @pytest.fixture
    def node_without_routing(self):
        from framework.nodes.llm.gemini import GeminiCodeAssistNode
        cfg = _make_config()
        node_config = {
            "id": "gemini_api_helper",
            "model": "gemini-2.5-pro",
        }
        node = GeminiCodeAssistNode(cfg, node_config)
        return node

    @pytest.mark.asyncio
    async def test_routing_signal_detected(self, node_with_routing):
        """enable_routing=true 时检测路由信号。"""
        reply = '{"route": "knowledge_shelf", "context": "查找标签"}\n'
        with patch.object(node_with_routing, "call_llm", new_callable=AsyncMock, return_value=(reply, "sid_456")):
            result = await node_with_routing(_make_state())
        assert result["routing_target"] == "knowledge_shelf"
        assert result["routing_context"] == "查找标签"

    @pytest.mark.asyncio
    async def test_routing_disabled_clears_signal(self, node_without_routing):
        """enable_routing=false 时清除信号。"""
        reply = '{"route": "knowledge_shelf", "context": "test"}\n'
        with patch.object(node_without_routing, "call_llm", new_callable=AsyncMock, return_value=(reply, "sid_456")):
            result = await node_without_routing(_make_state())
        assert result["routing_target"] == ""


# ---------------------------------------------------------------------------
# Jitter multiplier tests
# ---------------------------------------------------------------------------

class TestJitterMultiplier:
    """jitter_multiplier 可控测试。"""

    def test_multiplier_zero_returns_zero(self):
        from framework.nodes.llm.gemini import _jitter_secs
        assert _jitter_secs(1000, multiplier=0) == 0.0

    def test_multiplier_negative_returns_zero(self):
        from framework.nodes.llm.gemini import _jitter_secs
        assert _jitter_secs(1000, multiplier=-1) == 0.0

    def test_multiplier_default_positive(self):
        from framework.nodes.llm.gemini import _jitter_secs
        result = _jitter_secs(1000, multiplier=1.0)
        assert result >= 0.5

    def test_multiplier_double(self):
        """multiplier=2 时延迟至少是 multiplier=1 的最小值的 2 倍。"""
        from framework.nodes.llm.gemini import _jitter_secs
        # 用固定种子不可行（随机），但 multiplier=2 应产生更大值
        results = [_jitter_secs(100, multiplier=2.0) for _ in range(20)]
        # 至少有一些值 > 2.0（multiplier=1 的 base ~= 1.95 for 100 chars）
        assert max(results) > 2.0
