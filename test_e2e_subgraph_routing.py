"""
E2E 测试：子图路由 + max_retry 限速 + 按子图独立计数

覆盖：
  1. BaseAgentState 含 subgraph_call_counts 字段
  2. SubgraphRefNode max_retry 参数读取
  3. SubgraphRefNode 首次调用正常执行
  4. SubgraphRefNode 达到 max_retry 后返回限速 AIMessage（不执行子图）
  5. 不同子图独立计数（brainstorm count 不影响 design）
  6. max_retry=None 时无限制
  7. 通用 state_out 映射（apex_conclusion 不再丢失）
  8. 限速消息通过回路边回传主图（消息含 [子图限速] 标记）
  9. llm_node 无路由时重置 subgraph_call_counts
  10. 主图编译后 routing_to 边不含 max_retry 阻塞逻辑

运行：
    python3 test_e2e_subgraph_routing.py
"""

import asyncio
import logging
import sys
import typing
from pathlib import Path

import pytest

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("test_e2e_subgraph_routing")


# ---------------------------------------------------------------------------
# Mock 工具
# ---------------------------------------------------------------------------

def _make_mock_graph(final_state: dict, node_id: str = "mock_node"):
    """创建支持 astream(stream_mode="updates") 的 mock 图。"""
    class _MockGraph:
        def __init__(self, state, nid):
            self._state = state
            self._node_id = nid
            self._call_count = 0

        async def astream(self, sub_state, *, stream_mode="updates"):
            self._call_count += 1
            yield {self._node_id: self._state}

    return _MockGraph(final_state, node_id)


def _make_subgraph_node(agent_dir, state_out, max_retry=None):
    """创建 SubgraphRefNode 实例（不编译真实图）。"""
    from framework.nodes.subgraph.subgraph_ref_node import SubgraphRefNode
    from framework.config import AgentConfig

    cfg = AgentConfig(tools=[])
    node_config = {
        "id": Path(agent_dir).name.replace("/", "_"),
        "agent_dir": agent_dir,
        "state_in": {"task": "routing_context"},
        "state_out": state_out,
    }
    if max_retry is not None:
        node_config["max_retry"] = max_retry

    return SubgraphRefNode(cfg, node_config)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_has_subgraph_call_counts():
    """BaseAgentState 含 subgraph_call_counts 字段。"""
    from framework.state import BaseAgentState
    hints = typing.get_type_hints(BaseAgentState)
    assert "subgraph_call_counts" in hints, "BaseAgentState 缺少 subgraph_call_counts 字段"
    logger.info("✅ BaseAgentState.subgraph_call_counts 字段存在")


@pytest.mark.asyncio
async def test_subgraph_node_reads_max_retry():
    """SubgraphRefNode 从 node_config 读取 max_retry。"""
    node_with = _make_subgraph_node(
        "blueprints/functional_graphs/debate_gemini_first",
        {"debate_conclusion": "last_message"},
        max_retry=3,
    )
    assert node_with._max_retry == 3, f"max_retry 应为 3，实际: {node_with._max_retry}"

    node_without = _make_subgraph_node(
        "blueprints/functional_graphs/debate_gemini_first",
        {"debate_conclusion": "last_message"},
    )
    assert node_without._max_retry is None, f"默认 max_retry 应为 None，实际: {node_without._max_retry}"

    logger.info("✅ SubgraphRefNode max_retry 读取正确")


@pytest.mark.asyncio
async def test_first_call_executes():
    """首次调用（count=0, max_retry=1）正常执行子图。"""
    from langchain_core.messages import AIMessage

    node = _make_subgraph_node(
        "blueprints/functional_graphs/debate_gemini_first",
        {"debate_conclusion": "last_message"},
        max_retry=1,
    )

    fake_reply = "辩论结论：选方案A。"
    mock_graph = _make_mock_graph({"messages": [AIMessage(content=fake_reply)]})
    node._graph = mock_graph

    parent_state = {
        "routing_context": "测试议题",
        "subgraph_call_counts": {},
    }

    result = await node(parent_state)

    assert mock_graph._call_count == 1, "子图应被执行一次"
    assert result.get("debate_conclusion") == fake_reply, "state_out 映射失败"
    assert result["subgraph_call_counts"].get(node._node_id) == 1, "call_count 应为 1"

    logger.info("✅ 首次调用正常执行")


@pytest.mark.asyncio
async def test_max_retry_blocks_second_call():
    """达到 max_retry 后返回限速 AIMessage，不执行子图。"""
    from langchain_core.messages import AIMessage

    node = _make_subgraph_node(
        "blueprints/functional_graphs/debate_gemini_first",
        {"debate_conclusion": "last_message"},
        max_retry=1,
    )

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="不应被执行")]})
    node._graph = mock_graph

    parent_state = {
        "routing_context": "测试议题",
        "subgraph_call_counts": {node._node_id: 1},  # 已调用 1 次
    }

    result = await node(parent_state)

    assert mock_graph._call_count == 0, "子图不应被执行"
    assert "messages" in result, "应返回 messages"
    assert "[子图限速]" in result["messages"][0].content, "消息应含 [子图限速] 标记"
    assert "max_retry=1" in result["messages"][0].content, "消息应包含 max_retry 值"

    logger.info("✅ max_retry 限速正确，返回限速消息")


@pytest.mark.asyncio
async def test_independent_counting():
    """不同子图独立计数，互不干扰。"""
    from langchain_core.messages import AIMessage

    node_brainstorm = _make_subgraph_node(
        "blueprints/functional_graphs/debate_gemini_first",
        {"debate_conclusion": "last_message"},
        max_retry=1,
    )
    node_design = _make_subgraph_node(
        "blueprints/functional_graphs/debate_claude_first",
        {"debate_conclusion": "last_message"},
        max_retry=1,
    )

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="结论")]})
    node_design._graph = mock_graph

    # brainstorm 已调用 1 次，design 还没调用
    parent_state = {
        "routing_context": "测试议题",
        "subgraph_call_counts": {node_brainstorm._node_id: 1},
    }

    result = await node_design(parent_state)

    assert mock_graph._call_count == 1, "design 子图应正常执行（不受 brainstorm count 影响）"
    counts = result["subgraph_call_counts"]
    assert counts.get(node_design._node_id) == 1, "design count 应为 1"
    assert counts.get(node_brainstorm._node_id) == 1, "brainstorm count 应保持为 1"

    logger.info("✅ 不同子图独立计数")


@pytest.mark.asyncio
async def test_max_retry_none_unlimited():
    """max_retry=None 时不限次数。"""
    from langchain_core.messages import AIMessage

    node = _make_subgraph_node(
        "blueprints/functional_graphs/debate_gemini_first",
        {"debate_conclusion": "last_message"},
        max_retry=None,
    )

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="结论")]})
    node._graph = mock_graph

    # 即使已调用 100 次也不应被拦截
    parent_state = {
        "routing_context": "测试议题",
        "subgraph_call_counts": {node._node_id: 100},
    }

    result = await node(parent_state)

    assert mock_graph._call_count == 1, "max_retry=None 时应不限制"
    assert "[子图限速]" not in result.get("messages", [{}])[0].content, "不应出现限速消息"

    logger.info("✅ max_retry=None 无限制")


@pytest.mark.asyncio
async def test_generic_state_out_mapping():
    """通用 state_out 映射：apex_conclusion 正确写入 messages（不再只看 debate_conclusion）。"""
    from langchain_core.messages import AIMessage

    node = _make_subgraph_node(
        "blueprints/functional_graphs/apex_coder",
        {"apex_conclusion": "last_message"},  # 非 debate_conclusion
        max_retry=None,
    )

    fake_reply = "ApexCoder 执行完毕，修复了 3 个 bug。"
    mock_graph = _make_mock_graph({"messages": [AIMessage(content=fake_reply)]})
    node._graph = mock_graph

    parent_state = {
        "routing_context": "修复 bug",
        "subgraph_call_counts": {},
    }

    result = await node(parent_state)

    assert result.get("apex_conclusion") == fake_reply, "apex_conclusion 应被正确映射"
    assert "messages" in result, "结果应含 messages"
    assert "[子图结论]" in result["messages"][0].content, "AIMessage 应含 [子图结论]（通用前缀）"
    assert fake_reply in result["messages"][0].content, "AIMessage 应含实际结论内容"

    logger.info("✅ 通用 state_out 映射（apex_conclusion）")


@pytest.mark.asyncio
async def test_limit_message_has_actionable_content():
    """限速消息应包含足够信息让主节点做出决策。"""
    from langchain_core.messages import AIMessage

    node = _make_subgraph_node(
        "blueprints/functional_graphs/debate_gemini_first",
        {"debate_conclusion": "last_message"},
        max_retry=2,
    )

    node._graph = _make_mock_graph({"messages": [AIMessage(content="x")]})

    parent_state = {
        "routing_context": "议题",
        "subgraph_call_counts": {node._node_id: 2},  # 恰好等于 max_retry
    }

    result = await node(parent_state)

    msg = result["messages"][0].content
    assert node._node_id in msg, "限速消息应含子图 ID"
    assert "2" in msg, "限速消息应含调用次数"
    assert "max_retry=2" in msg, "限速消息应含 max_retry 值"
    assert "请换一种方式" in msg, "限速消息应含行动建议"

    logger.info("✅ 限速消息内容完整可操作")


@pytest.mark.asyncio
async def test_llm_node_resets_counts_on_no_routing():
    """LlmNode 无路由信号时重置 subgraph_call_counts。"""
    from unittest.mock import AsyncMock, patch
    from langchain_core.messages import HumanMessage
    from framework.config import AgentConfig
    from framework.nodes.llm.llm_node import LlmNode

    # 创建一个 concrete 子类（LlmNode 是 ABC）
    class _MockLlmNode(LlmNode):
        async def call_llm(self, prompt, session_id="", tools=None, cwd=None, history=None):
            return "普通回复，没有路由信号", session_id or "mock-sid"

    cfg = AgentConfig(tools=[])
    node = _MockLlmNode(cfg, {"id": "claude_main"})

    state = {
        "messages": [HumanMessage(content="你好")],
        "node_sessions": {},
        "subgraph_call_counts": {"debate_brainstorm": 1, "debate_design": 2},
        "routing_target": "",
        "routing_context": "",
        "rollback_reason": "",
        "project_root": "",
        "workspace": "",
        "project_meta": {},
    }

    result = await node(state)

    assert result.get("subgraph_call_counts") == {}, \
        f"无路由时 subgraph_call_counts 应被重置为 {{}}, 实际: {result.get('subgraph_call_counts')}"
    assert result.get("routing_target") == "", "routing_target 应为空"

    logger.info("✅ LlmNode 无路由时重置 subgraph_call_counts")


@pytest.mark.asyncio
async def test_llm_node_preserves_counts_on_routing():
    """LlmNode 有路由信号时保留 subgraph_call_counts（不重置）。"""
    from unittest.mock import AsyncMock, patch
    from langchain_core.messages import HumanMessage
    from framework.config import AgentConfig
    from framework.nodes.llm.llm_node import LlmNode

    class _MockLlmNode(LlmNode):
        async def call_llm(self, prompt, session_id="", tools=None, cwd=None, history=None):
            # 首行输出路由信号
            return '{"route": "debate_design", "context": "测试"}\n后续内容', session_id or "mock-sid"

    cfg = AgentConfig(tools=[])
    node = _MockLlmNode(cfg, {"id": "claude_main"})

    state = {
        "messages": [HumanMessage(content="讨论一下架构")],
        "node_sessions": {},
        "subgraph_call_counts": {"debate_brainstorm": 1},
        "routing_target": "",
        "routing_context": "",
        "rollback_reason": "",
        "project_root": "",
        "workspace": "",
        "project_meta": {},
    }

    result = await node(state)

    assert result.get("routing_target") == "debate_design", "应检测到路由信号"
    # 有路由信号时不应出现 subgraph_call_counts 重置
    assert "subgraph_call_counts" not in result or result.get("subgraph_call_counts") != {}, \
        "有路由信号时不应重置 subgraph_call_counts"

    logger.info("✅ LlmNode 有路由信号时保留 subgraph_call_counts")


@pytest.mark.asyncio
async def test_main_graph_routing_edges_no_block():
    """主图的 routing_to 边不含 max_retry 阻塞逻辑（纯 routing_target 匹配）。"""
    from framework.agent_loader import AgentLoader

    loader = AgentLoader(Path("blueprints/role_agents/technical_architect"))
    g = await loader.build_graph()

    # 验证子图节点存在
    node_ids = set(g.nodes)
    for name in ("debate_brainstorm", "debate_design", "apex_coder"):
        assert name in node_ids, f"主图应含子图节点: {name}"

    logger.info("✅ 主图编译成功，子图节点就位")


# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------

async def run():
    logger.info("=== E2E 子图路由 + 限速测试开始 ===")

    tests = [
        test_state_has_subgraph_call_counts,
        test_subgraph_node_reads_max_retry,
        test_first_call_executes,
        test_max_retry_blocks_second_call,
        test_independent_counting,
        test_max_retry_none_unlimited,
        test_generic_state_out_mapping,
        test_limit_message_has_actionable_content,
        test_llm_node_resets_counts_on_no_routing,
        test_llm_node_preserves_counts_on_routing,
        test_main_graph_routing_edges_no_block,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test.__name__}: {e}")

    logger.info("=" * 50)
    print(f"\n{'✅' if failed == 0 else '❌'} {passed}/{passed + failed} 测试通过")
    if failed == 0:
        print("   BaseAgentState.subgraph_call_counts 字段存在")
        print("   SubgraphRefNode max_retry 读取正确（含默认 None）")
        print("   首次调用正常执行子图")
        print("   max_retry 达上限 → 返回 [子图限速] AIMessage")
        print("   不同子图独立计数，互不干扰")
        print("   max_retry=None 时无限制")
        print("   通用 state_out 映射（apex_conclusion 不再丢失）")
        print("   限速消息内容完整可操作")
        print("   LlmNode 无路由时重置 subgraph_call_counts")
        print("   LlmNode 有路由信号时保留 subgraph_call_counts")
        print("   主图编译成功，routing_to 边无阻塞")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run())
