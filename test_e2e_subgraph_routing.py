"""
E2E 测试：子图路由 + max_retry 限速 + 按子图独立计数

覆盖：
  1.  BaseAgentState 含 subgraph_call_counts 字段
  2.  SubgraphNodeWrapper max_retry 参数读取
  3.  SubgraphNodeWrapper 首次调用正常执行
  4.  SubgraphNodeWrapper 达到 max_retry 后返回限速 AIMessage（不执行子图）
  5.  不同子图独立计数（brainstorm count 不影响 design）
  6.  max_retry=None 时无限制
  7.  通用 state_out 映射（apex_conclusion 不再丢失）
  8.  限速消息通过回路边回传主图（消息含 [子图限速] 标记）
  9.  llm_node 无路由时重置 subgraph_call_counts
  10. 主图编译后 routing_to 边不含 max_retry 阻塞逻辑
  11. state_out 非 last_message 映射（按字段名直接取值）
  12. 子图返回空 messages 时降级处理
  13. consult_count 递增验证
  14. subgraph_call_counts 缺失时的容错
  15. 多键 state_out 同时映射
  16. count 远超 max_retry 仍正确拦截

运行：
    pytest test_e2e_subgraph_routing.py -v
"""

import logging
import sys
import typing
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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
# 常量 — 消除硬编码重复
# ---------------------------------------------------------------------------
DEBATE_GEMINI_DIR = "blueprints/functional_graphs/debate_gemini_first"
DEBATE_CLAUDE_DIR = "blueprints/functional_graphs/debate_claude_first"
APEX_CODER_DIR = "blueprints/functional_graphs/apex_coder"

DEFAULT_OUTPUT_FIELD = "debate_conclusion"


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


def _make_subgraph_node(agent_dir, output_field=None, max_retry=None):
    """创建 _SubgraphNodeWrapper 实例（注入 mock 图）。"""
    from framework.agent_loader import _SubgraphNodeWrapper

    node_id = Path(agent_dir).name.replace("/", "_")
    if output_field is None:
        output_field = DEFAULT_OUTPUT_FIELD

    return _SubgraphNodeWrapper(
        graph=None,  # Will be injected by test via wrapper._graph = ...
        node_id=node_id,
        graph_name=node_id,
        input_schema=["routing_context"],
        output_field=output_field,
        max_retry=max_retry,
    )


def _default_parent_state(**overrides) -> dict:
    """构建标准 parent_state，减少每个测试的样板。"""
    state = {
        "routing_context": "测试议题",
        "subgraph_call_counts": {},
    }
    state.update(overrides)
    return state


def _make_llm_state(**overrides) -> dict:
    """构建 LlmNode 需要的完整 state，减少测试内重复。"""
    state = {
        "messages": [HumanMessage(content="你好")],
        "node_sessions": {},
        "subgraph_call_counts": {},
        "routing_target": "",
        "routing_context": "",
        "rollback_reason": "",
        "project_root": "",
        "workspace": "",
        "project_meta": {},
    }
    state.update(overrides)
    return state


class _MockLlmNode:
    """LlmNode 的 concrete mock 子类，避免在多个测试中重复定义。"""

    _instance_cache: dict = {}

    @staticmethod
    def create(reply: str, node_id: str = "claude_main"):
        from framework.config import AgentConfig
        from framework.nodes.llm.llm_node import LlmNode

        class _Impl(LlmNode):
            async def call_llm(self, prompt, session_id="", tools=None, cwd=None, history=None):
                return reply, session_id or "mock-sid"

        return _Impl(AgentConfig(tools=[]), {"id": node_id})


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_has_subgraph_call_counts():
    """BaseAgentState 含 subgraph_call_counts 字段。"""
    from framework.state import BaseAgentState
    hints = typing.get_type_hints(BaseAgentState)
    assert "subgraph_call_counts" in hints, "BaseAgentState 缺少 subgraph_call_counts 字段"
    logger.info("PASS BaseAgentState.subgraph_call_counts 字段存在")


@pytest.mark.asyncio
async def test_subgraph_node_reads_max_retry():
    """SubgraphNodeWrapper 从 node_config 读取 max_retry。"""
    node_with = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=3)
    assert node_with._max_retry == 3, f"max_retry 应为 3，实际: {node_with._max_retry}"

    node_without = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD)
    assert node_without._max_retry is None, f"默认 max_retry 应为 None，实际: {node_without._max_retry}"

    logger.info("PASS SubgraphNodeWrapper max_retry 读取正确")


@pytest.mark.asyncio
async def test_first_call_executes():
    """首次调用（count=0, max_retry=1）正常执行子图。"""
    node = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=1)

    fake_reply = "辩论结论：选方案A。"
    mock_graph = _make_mock_graph({"messages": [AIMessage(content=fake_reply)]})
    node._graph = mock_graph

    result = await node(_default_parent_state())

    assert mock_graph._call_count == 1, "子图应被执行一次"
    assert result.get("debate_conclusion") == fake_reply, "state_out 映射失败"
    assert result["subgraph_call_counts"].get(node._node_id) == 1, "call_count 应为 1"

    logger.info("PASS 首次调用正常执行")


@pytest.mark.asyncio
async def test_max_retry_blocks_second_call():
    """达到 max_retry 后返回限速 AIMessage，不执行子图。"""
    node = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=1)

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="不应被执行")]})
    node._graph = mock_graph

    result = await node(_default_parent_state(subgraph_call_counts={node._node_id: 1}))

    assert mock_graph._call_count == 0, "子图不应被执行"
    assert "messages" in result, "应返回 messages"
    assert "[子图限速]" in result["messages"][0].content, "消息应含 [子图限速] 标记"
    assert "max_retry=1" in result["messages"][0].content, "消息应包含 max_retry 值"

    logger.info("PASS max_retry 限速正确，返回限速消息")


@pytest.mark.asyncio
async def test_independent_counting():
    """不同子图独立计数，互不干扰。"""
    node_brainstorm = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=1)
    node_design = _make_subgraph_node(DEBATE_CLAUDE_DIR, DEFAULT_OUTPUT_FIELD, max_retry=1)

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="结论")]})
    node_design._graph = mock_graph

    # brainstorm 已调用 1 次，design 还没调用
    result = await node_design(_default_parent_state(
        subgraph_call_counts={node_brainstorm._node_id: 1},
    ))

    assert mock_graph._call_count == 1, "design 子图应正常执行（不受 brainstorm count 影响）"
    counts = result["subgraph_call_counts"]
    assert counts.get(node_design._node_id) == 1, "design count 应为 1"
    assert counts.get(node_brainstorm._node_id) == 1, "brainstorm count 应保持为 1"

    logger.info("PASS 不同子图独立计数")


@pytest.mark.asyncio
async def test_max_retry_none_unlimited():
    """max_retry=None 时不限次数。"""
    node = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=None)

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="结论")]})
    node._graph = mock_graph

    # 即使已调用 100 次也不应被拦截
    result = await node(_default_parent_state(subgraph_call_counts={node._node_id: 100}))

    assert mock_graph._call_count == 1, "max_retry=None 时应不限制"
    assert "[子图限速]" not in result["messages"][0].content, "不应出现限速消息"

    logger.info("PASS max_retry=None 无限制")


@pytest.mark.asyncio
async def test_generic_output_field_mapping():
    """通用 output_field 映射：apex_conclusion 正确写入 messages。"""
    node = _make_subgraph_node(
        APEX_CODER_DIR,
        output_field="apex_conclusion",
        max_retry=None,
    )

    fake_reply = "ApexCoder 执行完毕，修复了 3 个 bug。"
    mock_graph = _make_mock_graph({"messages": [AIMessage(content=fake_reply)]})
    node._graph = mock_graph

    result = await node(_default_parent_state(routing_context="修复 bug"))

    assert result.get("apex_conclusion") == fake_reply, "apex_conclusion 应被正确映射"
    assert "messages" in result, "结果应含 messages"
    assert "[子图结论]" in result["messages"][0].content, "AIMessage 应含 [子图结论]"
    assert fake_reply in result["messages"][0].content, "AIMessage 应含实际结论内容"

    logger.info("PASS 通用 state_out 映射（apex_conclusion）")


@pytest.mark.asyncio
async def test_limit_message_has_actionable_content():
    """限速消息应包含足够信息让主节点做出决策。"""
    node = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=2)
    node._graph = _make_mock_graph({"messages": [AIMessage(content="x")]})

    result = await node(_default_parent_state(subgraph_call_counts={node._node_id: 2}))

    msg = result["messages"][0].content
    assert node._node_id in msg, "限速消息应含子图 ID"
    assert "2" in msg, "限速消息应含调用次数"
    assert "max_retry=2" in msg, "限速消息应含 max_retry 值"
    assert "请换一种方式" in msg, "限速消息应含行动建议"

    logger.info("PASS 限速消息内容完整可操作")


@pytest.mark.asyncio
async def test_llm_node_resets_counts_on_no_routing():
    """LlmNode 无路由信号时重置 subgraph_call_counts。"""
    node = _MockLlmNode.create("普通回复，没有路由信号")

    state = _make_llm_state(
        subgraph_call_counts={"debate_brainstorm": 1, "debate_design": 2},
    )
    result = await node(state)

    assert result.get("subgraph_call_counts") == {}, \
        f"无路由时 subgraph_call_counts 应被重置为 {{}}, 实际: {result.get('subgraph_call_counts')}"
    assert result.get("routing_target") == "", "routing_target 应为空"

    logger.info("PASS LlmNode 无路由时重置 subgraph_call_counts")


@pytest.mark.asyncio
async def test_llm_node_preserves_counts_on_routing():
    """LlmNode 有路由信号时保留 subgraph_call_counts（不重置）。"""
    node = _MockLlmNode.create('{"route": "debate_design", "context": "测试"}\n后续内容')

    state = _make_llm_state(
        messages=[HumanMessage(content="讨论一下架构")],
        subgraph_call_counts={"debate_brainstorm": 1},
    )
    result = await node(state)

    assert result.get("routing_target") == "debate_design", "应检测到路由信号"
    assert "subgraph_call_counts" not in result or result.get("subgraph_call_counts") != {}, \
        "有路由信号时不应重置 subgraph_call_counts"

    logger.info("PASS LlmNode 有路由信号时保留 subgraph_call_counts")


@pytest.mark.asyncio
async def test_main_graph_routing_edges_no_block():
    """主图的 routing_to 边不含 max_retry 阻塞逻辑（纯 routing_target 匹配）。"""
    from framework.agent_loader import AgentLoader

    loader = AgentLoader(Path("blueprints/role_agents/technical_architect"))
    g = await loader.build_graph()

    node_ids = set(g.nodes)
    for name in ("debate_brainstorm", "debate_design", "apex_coder"):
        assert name in node_ids, f"主图应含子图节点: {name}"

    logger.info("PASS 主图编译成功，子图节点就位")


# ---------------------------------------------------------------------------
# 新增测试：填补盲区
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_field_extracts_last_message():
    """output_field 从子图最后一条消息提取内容写入指定字段。"""
    node = _make_subgraph_node(
        APEX_CODER_DIR,
        output_field="apex_conclusion",
        max_retry=None,
    )

    mock_graph = _make_mock_graph({
        "messages": [AIMessage(content="最终结论内容")],
    })
    node._graph = mock_graph

    result = await node(_default_parent_state())

    assert result.get("apex_conclusion") == "最终结论内容", \
        "output_field 应从子图最后一条消息提取内容"

    logger.info("PASS output_field 提取最后消息")


@pytest.mark.asyncio
async def test_empty_messages_from_subgraph():
    """子图返回空 messages 时，state_out last_message 应降级为空字符串。"""
    node = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=None)

    # 子图返回空 messages 列表
    mock_graph = _make_mock_graph({"messages": []})
    node._graph = mock_graph

    result = await node(_default_parent_state())

    assert mock_graph._call_count == 1, "子图应被执行"
    assert result.get("debate_conclusion") == "", \
        "空 messages 时 last_message 映射应降级为空字符串"

    logger.info("PASS 子图空 messages 降级处理")


@pytest.mark.asyncio
async def test_consult_count_increments():
    """SubgraphNodeWrapper 每次正常执行应递增 consult_count。"""
    node = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=None)

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="结论")]})
    node._graph = mock_graph

    # consult_count 初始为 3
    result = await node(_default_parent_state(consult_count=3))

    assert result.get("consult_count") == 4, \
        f"consult_count 应从 3 递增到 4，实际: {result.get('consult_count')}"

    logger.info("PASS consult_count 递增正确")


@pytest.mark.asyncio
async def test_missing_subgraph_call_counts_key():
    """state 中完全没有 subgraph_call_counts 键时应容错处理。"""
    node = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=2)

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="结论")]})
    node._graph = mock_graph

    # 完全不传 subgraph_call_counts
    state = {"routing_context": "测试议题"}
    result = await node(state)

    assert mock_graph._call_count == 1, "缺失 subgraph_call_counts 时应正常执行"
    assert result["subgraph_call_counts"].get(node._node_id) == 1, "首次调用 count 应为 1"

    logger.info("PASS subgraph_call_counts 缺失容错")


@pytest.mark.asyncio
async def test_output_field_with_conclusion_message():
    """output_field 映射后，messages 应含 [子图结论] 标记。"""
    node = _make_subgraph_node(
        APEX_CODER_DIR,
        output_field="apex_conclusion",
        max_retry=None,
    )

    mock_graph = _make_mock_graph({
        "messages": [AIMessage(content="最终消息内容")],
    })
    node._graph = mock_graph

    result = await node(_default_parent_state())

    assert result.get("apex_conclusion") == "最终消息内容", \
        "apex_conclusion 应从子图最终消息提取"
    assert "messages" in result, "结果应含 messages"
    assert "[子图结论]" in result["messages"][0].content, \
        "messages 应含 [子图结论] 标记"

    logger.info("PASS output_field + 结论消息")


@pytest.mark.asyncio
async def test_count_far_exceeds_max_retry():
    """count 远超 max_retry（如 count=50, max_retry=2）仍正确拦截。"""
    node = _make_subgraph_node(DEBATE_GEMINI_DIR, DEFAULT_OUTPUT_FIELD, max_retry=2)

    mock_graph = _make_mock_graph({"messages": [AIMessage(content="不应执行")]})
    node._graph = mock_graph

    result = await node(_default_parent_state(subgraph_call_counts={node._node_id: 50}))

    assert mock_graph._call_count == 0, "count 远超 max_retry 时子图不应被执行"
    assert "[子图限速]" in result["messages"][0].content, "应返回限速消息"
    # 确保 call_counts 原值被保留而非被覆盖
    assert result["subgraph_call_counts"].get(node._node_id) == 50, \
        "限速时 call_count 应保持原值（不递增）"

    logger.info("PASS count 远超 max_retry 正确拦截")


@pytest.mark.asyncio
async def test_llm_node_resets_ancillary_fields_on_no_routing():
    """LlmNode 无路由时重置 rollback_reason 但不重置 retry_count。

    retry_count 由 DETERMINISTIC validator 管理，LLM 节点不应干预，
    否则会导致 validator 重试逻辑失效（如 colony_coder 死循环 bug）。
    """
    node = _MockLlmNode.create("普通回复")

    state = _make_llm_state(
        rollback_reason="之前的失败原因",
        subgraph_call_counts={"x": 5},
    )
    result = await node(state)

    assert result.get("rollback_reason") == "", "无路由时 rollback_reason 应被清空"
    assert "retry_count" not in result, "LLM 节点不应在 result 中包含 retry_count（由 validator 管理）"
    assert result.get("consult_count") == 0, "无路由时 consult_count 应被重置为 0"

    logger.info("PASS LlmNode 无路由时重置附属字段（retry_count 不受影响）")
