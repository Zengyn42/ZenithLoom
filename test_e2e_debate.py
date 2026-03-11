"""
E2E 测试：辩论子图架构验证

覆盖：
  1. BaseAgentState 含新字段（knowledge_vault / project_docs / debate_conclusion）
  2. AGENT_REF 节点类型已注册
  3. 两个辩论子图独立编译（debate_gemini_first / debate_claude_first）
  4. Hani 主图含 debate_brainstorm / debate_design 节点
  5. AgentRefNode 实例化 + state_in/state_out 映射逻辑单元测试

运行：
    python3 test_e2e_debate.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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

logger = logging.getLogger("test_e2e_debate")


async def test_state_fields():
    """BaseAgentState 含所有新字段。"""
    import typing
    from framework.state import BaseAgentState
    hints = typing.get_type_hints(BaseAgentState)
    for f in ("knowledge_vault", "project_docs", "debate_conclusion"):
        assert f in hints, f"BaseAgentState 缺少字段: {f}"
    logger.info("✅ BaseAgentState fields OK")


async def test_agent_ref_registered():
    """AGENT_REF 节点类型已在 registry 注册。"""
    import framework.builtins
    from framework.registry import get_node_factory
    get_node_factory("AGENT_REF")
    logger.info("✅ AGENT_REF registered OK")


async def test_debate_graphs_compile():
    """两个辩论子图能独立编译，节点数量正确。"""
    from framework.agent_loader import AgentLoader

    expected = {
        "agents/debate_gemini_first": {
            "gemini_propose", "claude_critique_1", "gemini_revise",
            "claude_critique_2", "gemini_conclusion",
        },
        "agents/debate_claude_first": {
            "claude_propose", "gemini_critique_1", "claude_revise",
            "gemini_critique_2", "claude_conclusion",
        },
    }

    for agent_dir, required_nodes in expected.items():
        g = await AgentLoader(Path(agent_dir)).build_graph()
        node_ids = set(g.nodes) - {"__start__"}
        missing = required_nodes - node_ids
        assert not missing, f"{agent_dir} 缺少节点: {missing}"
        logger.info(f"✅ {agent_dir}: nodes={sorted(node_ids)}")


async def test_hani_graph_with_debate():
    """Hani 主图含 debate_brainstorm / debate_design AGENT_REF 节点。"""
    from framework.agent_loader import AgentLoader

    g = await AgentLoader(Path("agents/hani")).build_graph()
    node_ids = set(g.nodes)

    required = {
        "claude_main", "gemini_advisor", "validate",
        "git_snapshot", "git_rollback",
        "debate_brainstorm", "debate_design",
    }
    missing = required - node_ids
    assert not missing, f"hani 图缺少节点: {missing}"
    logger.info(f"✅ hani nodes: {sorted(node_ids)}")


def _make_mock_graph(final_state: dict, node_id: str = "gemini_conclusion"):
    """创建支持 astream(stream_mode="updates") 的 mock 图。"""
    class _MockGraph:
        def __init__(self, state, nid):
            self._state = state
            self._node_id = nid
            self._call_args = None

        async def astream(self, sub_state, *, stream_mode="updates"):
            self._call_args = sub_state
            # updates 模式返回 {node_id: state_update}
            yield {self._node_id: self._state}

    return _MockGraph(final_state, node_id)


async def test_agent_ref_state_mapping():
    """AgentRefNode state_in/state_out 映射逻辑（不发起真实 LLM 调用）。"""
    from langchain_core.messages import AIMessage
    from framework.nodes.agent_ref_node import AgentRefNode
    from framework.config import AgentConfig

    cfg = AgentConfig(tools=[])
    node_config = {
        "agent_dir": "agents/debate_gemini_first",
        "state_in":  {"task": "routing_context", "knowledge_vault": "knowledge_vault"},
        "state_out": {"debate_conclusion": "last_message"},
    }

    node = AgentRefNode(cfg, node_config)

    # Mock 子图，返回预设 messages
    fake_reply = "最终结论：微服务更适合这个场景。"
    mock_graph = _make_mock_graph({
        "messages": [AIMessage(content=fake_reply)],
    })
    node._graph = mock_graph  # 注入 mock，跳过真实编译

    parent_state = {
        "routing_context": "微服务 vs 单体架构选型",
        "knowledge_vault": "/home/kingy/ObsidianVault",
        "project_docs": "/home/kingy/Projects/Genesis/docs",
        "debate_conclusion": "",
    }

    result = await node(parent_state)

    # 验证 state_in 映射：子图收到的 task 应来自 routing_context
    call_args = mock_graph._call_args
    assert call_args["task"] == "微服务 vs 单体架构选型", "state_in 映射失败"
    assert call_args["knowledge_vault"] == "/home/kingy/ObsidianVault", "knowledge_vault 未透传"

    # 验证 state_out 映射：debate_conclusion 应是最后一条消息内容
    assert result["debate_conclusion"] == fake_reply, "state_out last_message 映射失败"

    # 验证辩论结论被注入为 AIMessage
    assert "messages" in result, "结果应含 messages（AIMessage 注入）"
    assert "[辩论结论]" in result["messages"][0].content, "AIMessage 应含 [辩论结论] 前缀"

    logger.info(f"✅ AgentRefNode mapping OK: conclusion={fake_reply[:40]!r}")


async def test_debate_state_schema():
    """辩论子图使用 DebateState（add_messages），消息列表应累积而非截断。"""
    import typing
    from langgraph.graph.message import add_messages
    from framework.state import DebateState

    hints = typing.get_type_hints(DebateState, include_extras=True)
    assert "messages" in hints, "DebateState 缺少 messages 字段"

    # 验证 reducer 是 add_messages（不是 _keep_last_2）
    messages_annotation = hints["messages"]
    metadata = getattr(messages_annotation, "__metadata__", ())
    assert any(m is add_messages for m in metadata), (
        "DebateState.messages 应使用 add_messages reducer，而非 _keep_last_2"
    )
    logger.info("✅ DebateState uses add_messages OK")


async def test_gemini_node_system_prompt():
    """GeminiNode 从 node_config['system_prompt'] 读取自定义 system prompt。"""
    from framework.gemini.node import GeminiNode, _GEMINI_SYSTEM
    from framework.config import AgentConfig

    cfg = AgentConfig(tools=[])
    custom_prompt = "你是一位专业辩手，请犀利发言。"

    # 有 system_prompt 时用自定义
    node_with = GeminiNode(cfg, {"id": "test_node", "system_prompt": custom_prompt})
    assert node_with._system_prompt == custom_prompt, "system_prompt 未从 node_config 读取"

    # 无 system_prompt 时 fallback 到默认
    node_without = GeminiNode(cfg, {"id": "test_node2"})
    assert node_without._system_prompt == _GEMINI_SYSTEM, "缺少 system_prompt 时应 fallback 到 _GEMINI_SYSTEM"

    logger.info("✅ GeminiNode system_prompt OK")


async def test_no_checkpointer_build():
    """build_graph(checkpointer=None) 编译后无 checkpointer，可直接 ainvoke 无需 thread_id。"""
    from framework.agent_loader import AgentLoader
    from langchain_core.messages import HumanMessage

    loader = AgentLoader(Path("agents/debate_gemini_first"))
    graph = await loader.build_graph(checkpointer=None)

    # 没有 checkpointer 的图可以不传 thread_id 直接调用（通过 mock 快速验证）
    from unittest.mock import AsyncMock, patch, MagicMock
    # 只验证图有 checkpointer=None（即 .checkpointer 属性为 None 或不存在）
    cp = getattr(graph, "checkpointer", None)
    assert cp is None, f"checkpointer=None 应编译无 checkpointer，实际: {cp!r}"
    logger.info("✅ build_graph(checkpointer=None) → no checkpointer OK")


async def test_session_cleanup():
    """AgentRefNode 调用后清理子图产生的孤儿 session。"""
    from langchain_core.messages import AIMessage
    from framework.nodes.agent_ref_node import AgentRefNode
    from framework.config import AgentConfig

    cfg = AgentConfig(tools=[])
    node_config = {
        "agent_dir": "agents/debate_gemini_first",
        "state_in":  {"task": "routing_context", "knowledge_vault": "knowledge_vault"},
        "state_out": {"debate_conclusion": "last_message"},
    }

    node = AgentRefNode(cfg, node_config)

    # Mock 子图，返回预设 messages 和新增的 node_sessions
    fake_reply = "辩论结论：选方案A。"
    mock_graph = _make_mock_graph({
        "messages": [AIMessage(content=fake_reply)],
        "node_sessions": {
            "gemini_propose": "orphan-uuid-gemini",
            "claude_critique_1": "orphan-uuid-claude",
        },
    })
    node._graph = mock_graph

    # 追踪清理调用
    cleanup_calls = []
    original_cleanup = node._cleanup_orphan_sessions
    def tracked_cleanup(result, original_keys):
        cleanup_calls.append((set(result.get("node_sessions", {}).keys()), original_keys))
        # 不执行真实文件删除，只验证调用
    node._cleanup_orphan_sessions = tracked_cleanup

    parent_state = {
        "routing_context": "测试辩题",
        "knowledge_vault": "/tmp/vault",
        "node_sessions": {"claude_main": "existing-uuid"},
    }

    result = await node(parent_state)

    # 验证 cleanup 被调用，且差集正确
    assert len(cleanup_calls) == 1, "cleanup 应被调用一次"
    result_keys, original_keys = cleanup_calls[0]
    new_keys = result_keys - original_keys
    assert "gemini_propose" in new_keys, "gemini_propose 应是新增 key"
    assert "claude_critique_1" in new_keys, "claude_critique_1 应是新增 key"
    assert "claude_main" not in new_keys, "claude_main 是父图 key，不应被清理"

    logger.info("✅ AgentRefNode session cleanup OK")


async def test_gemini_lru_eviction():
    """GeminiNode LRU 淘汰旧 session 缓存。"""
    from framework.gemini.node import GeminiNode, _GeminiSessionMixin
    from framework.config import AgentConfig

    cfg = AgentConfig(tools=[])
    node = GeminiNode(cfg, {"id": "test_lru", "model": "gemini-2.5-flash"})

    # 手动注入两个 session 到缓存
    from framework.gemini.gemini_session import ConversationRecord
    record1 = ConversationRecord(
        sessionId="aaa-111", projectHash="test", startTime="", lastUpdated="", messages=[]
    )
    record2 = ConversationRecord(
        sessionId="bbb-222", projectHash="test", startTime="", lastUpdated="", messages=[]
    )

    node._records["aaa-111"] = record1
    assert len(node._records) == 1

    # 触发 evict（limit=1，已有 1 条 → 应淘汰旧的）
    node._evict_old_sessions()
    assert len(node._records) == 0, "evict 后 cache 应为空"

    # 验证 limit 属性
    assert _GeminiSessionMixin._SESSION_CACHE_LIMIT == 1

    logger.info("✅ GeminiNode LRU eviction OK")


async def test_gemini_delete_session():
    """gemini_session.delete_session 能删除磁盘文件。"""
    import tempfile
    import json
    from unittest.mock import patch
    import framework.gemini.gemini_session as gem_sess

    # 创建临时 session 文件
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(gem_sess, '_GEMINI_DIR', Path(tmpdir)):
            # 创建一个 session 并保存
            record = gem_sess.new_session("test-project", "gemini-2.5-flash")
            sid = record.sessionId
            saved_path = gem_sess.save_session(record, "test-project")
            assert saved_path.exists(), "session 文件应存在"

            # 删除
            deleted = gem_sess.delete_session(sid, "test-project")
            assert deleted, "delete_session 应返回 True"
            assert not saved_path.exists(), "session 文件应已被删除"

            # 重复删除应返回 False
            deleted_again = gem_sess.delete_session(sid, "test-project")
            assert not deleted_again, "重复删除应返回 False"

    logger.info("✅ gemini_session.delete_session OK")


async def run():
    logger.info("=== E2E Debate 架构测试开始 ===")

    await test_state_fields()
    await test_agent_ref_registered()
    await test_debate_graphs_compile()
    await test_hani_graph_with_debate()
    await test_agent_ref_state_mapping()
    await test_debate_state_schema()
    await test_gemini_node_system_prompt()
    await test_no_checkpointer_build()
    await test_session_cleanup()
    await test_gemini_lru_eviction()
    await test_gemini_delete_session()

    logger.info("=" * 50)
    print("\n✅ 全部测试通过")
    print("   state 新字段: knowledge_vault / project_docs / debate_conclusion")
    print("   AGENT_REF 已注册")
    print("   debate_gemini_first / debate_claude_first 图编译成功")
    print("   hani 图含 debate_brainstorm / debate_design")
    print("   AgentRefNode state 映射逻辑正确")
    print("   DebateState 使用 add_messages（消息累积）")
    print("   GeminiNode system_prompt 从 node_config 读取")
    print("   build_graph(checkpointer=None) 无需 thread_id")
    print("   AgentRefNode 孤儿 session 清理正确")
    print("   GeminiNode LRU 缓存淘汰正确")
    print("   gemini_session.delete_session 磁盘删除正确")


if __name__ == "__main__":
    asyncio.run(run())
