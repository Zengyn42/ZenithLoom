"""
Claude AgentNode 简单集成测试

直接调用 ClaudeNode.call_llm() 和 AgentNode.__call__()，不经过 LangGraph 图。

运行：
    cd /home/kingy/Foundation/BootstrapBuilder
    /usr/bin/python3 test_claude_node.py

测试场景：
  1. call_llm — 无 session，英文 system_prompt（跳过 WSL2 Unicode 路径）
  2. call_llm — 无 session，带 system_prompt（两阶段初始化路径）
  3. AgentNode.__call__ — 完整 state dict，验证路由信号解析
"""

import asyncio
import sys
from pathlib import Path


def _make_config():
    from framework.config import AgentConfig
    cfg = AgentConfig()
    cfg.tools = ["Read"]
    cfg.permission_mode = "bypassPermissions"
    cfg.setting_sources = None
    return cfg


def _make_node(system_prompt: str = ""):
    from framework.nodes.llm.claude import ClaudeNode
    cfg = _make_config()
    node_config = {
        "id": "claude_main",
        "type": "CLAUDE_CLI",
        "first_turn_suffix": "",
        "user_msg_prefix": "",
        "tombstone_enabled": False,
    }
    return ClaudeNode(cfg, node_config, system_prompt=system_prompt)


# ── Test 1: 基本 call_llm ──────────────────────────────────────────────────

async def test_call_llm_no_system_prompt():
    """无 system_prompt，直接调用 call_llm，验证 SDK 连通性。"""
    print("--- Test 1: call_llm (no system_prompt) ---")
    node = _make_node(system_prompt="")
    text, sid = await node.call_llm(
        prompt="Reply with just the word OK and nothing else.",
        session_id="",
        tools=["Read"],
        cwd=None,
    )
    assert text, f"Expected non-empty response, got: {text!r}"
    assert sid, f"Expected a session_id back, got: {sid!r}"
    print(f"   response: {text[:80]!r}")
    print(f"   session_id: {sid[:8]}...")
    print("✅ call_llm (no system_prompt) OK\n")
    return sid


# ── Test 2: 两阶段初始化（带 system_prompt）─────────────────────────────────

async def test_call_llm_with_system_prompt():
    """有 system_prompt，触发两阶段初始化，验证 WSL2 Unicode 路径。"""
    print("--- Test 2: call_llm (with system_prompt, 2-phase init) ---")
    node = _make_node(system_prompt="You are a test assistant. Be brief.")
    text, sid = await node.call_llm(
        prompt="Reply with just the word OK and nothing else.",
        session_id="",
        tools=["Read"],
        cwd=None,
    )
    assert text, f"Expected non-empty response, got: {text!r}"
    assert sid, f"Expected a session_id back, got: {sid!r}"
    # Should NOT be an API error
    assert "API Error" not in text, f"API error in response: {text}"
    assert "400" not in text[:50], f"400 error in response: {text[:100]}"
    print(f"   response: {text[:80]!r}")
    print(f"   session_id: {sid[:8]}...")
    print("✅ call_llm (with system_prompt) OK\n")
    return sid


# ── Test 3: resume 已有 session ────────────────────────────────────────────

async def test_call_llm_resume():
    """Resume 已有 session，验证 session 连续性。先建 session 再 resume。"""
    print("--- Test 3: call_llm (resume) ---")
    node = _make_node(system_prompt="You are a test assistant. Be brief.")
    # 先建立 session
    _, existing_sid = await node.call_llm(
        prompt="Remember the word: BANANA.",
        session_id="",
        tools=["Read"],
        cwd=None,
    )
    assert existing_sid, "Failed to get session_id for resume test"
    print(f"   created session: {existing_sid[:8]}...")

    # 再 resume
    text, new_sid = await node.call_llm(
        prompt="What word did I ask you to remember?",
        session_id=existing_sid,
        tools=["Read"],
        cwd=None,
    )
    assert text, f"Expected non-empty response, got: {text!r}"
    assert "API Error" not in text, f"API error on resume: {text}"
    print(f"   response: {text[:80]!r}")
    print(f"   new session_id: {new_sid[:8]}...")
    print("✅ call_llm (resume) OK\n")


# ── Test 4: AgentNode.__call__ ─────────────────────────────────────────────

async def test_agent_node_call():
    """完整的 AgentNode.__call__，验证 state dict 协议。"""
    print("--- Test 4: AgentNode.__call__ (full state protocol) ---")
    from langchain_core.messages import HumanMessage

    node = _make_node(system_prompt="You are a test assistant. Be brief.")
    state = {
        "messages": [HumanMessage(content="Reply with just the word OK and nothing else.")],
        "routing_target": "",
        "routing_context": "",
        "workspace": "",
        "project_root": "",
        "project_meta": {},
        "last_stable_commit": "",
        "retry_count": 0,
        "rollback_reason": "",
        "node_sessions": {},
        "knowledge_vault": "",
        "project_docs": "",
        "debate_conclusion": "",
    }

    result = await node(state)

    assert "messages" in result, "Result missing 'messages'"
    assert result["messages"], "Result messages list is empty"
    msg = result["messages"][-1]
    assert hasattr(msg, "content"), "Last message has no content"
    assert msg.content, f"Empty message content: {msg.content!r}"
    assert "API Error" not in msg.content, f"API error in response: {msg.content}"

    assert "node_sessions" in result, "Result missing 'node_sessions'"
    assert result["node_sessions"].get("claude_main"), "claude_main session_id not set"

    print(f"   content: {msg.content[:80]!r}")
    print(f"   node_sessions: {result['node_sessions']}")
    print("✅ AgentNode.__call__ OK\n")


# ── runner ─────────────────────────────────────────────────────────────────

async def run_all():
    results = {"pass": 0, "fail": 0}

    async def run(coro, name):
        try:
            val = await coro
            results["pass"] += 1
            return val
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["fail"] += 1
            return None

    await run(test_call_llm_no_system_prompt(), "Test 1")
    sid = await run(test_call_llm_with_system_prompt(), "Test 2")
    if sid:
        await run(test_call_llm_resume(sid), "Test 3")
    else:
        print("⏭  Test 3 skipped (Test 2 failed)\n")
    await run(test_agent_node_call(), "Test 4")

    print(f"\n{'='*40}")
    print(f"结果：{results['pass']} 通过 / {results['fail']} 失败")
    if results["fail"]:
        sys.exit(1)
    else:
        print("🎉 全部通过")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    asyncio.run(run_all())
