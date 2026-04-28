"""
Gemini session compact (!compact → /compress) 单元测试

覆盖：
  GeminiCLINode.compress_session()
    1. 成功，session_id 不变（原地压缩）
    2. 成功，session_id 改变（CLI 新建 session）
    3. 超时 → 返回错误信息，原 sid 不变
    4. CLI 非零退出码 → 返回错误信息
    5. JSON 解析失败 → 返回错误信息
    6. session_id 为空 → 立即返回 skip

  GraphController.compact_gemini_session()
    7. 图中无 GeminiCLINode → 返回提示
    8. node_sessions 为空 → 返回提示
    9. 成功，session_id 不变 → 不调用 aupdate_state
   10. 成功，session_id 改变 → 调用 aupdate_state 写回

运行：
    python3 -m pytest test_gemini_compact.py -v
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from framework.config import AgentConfig
from framework.nodes.llm.gemini import GeminiCLINode


# ─────────────────────────── helpers ───────────────────────────────────────

def _make_node(model: str = "gemini-2.5-pro") -> GeminiCLINode:
    config = AgentConfig(
        tools=[],
        permission_mode="bypassPermissions",
        setting_sources=None,
        settings_override={"enabledPlugins": []},
    )
    return GeminiCLINode(
        config=config,
        node_config={"id": "gemini_main", "session_key": "jei_main",
                     "model": model, "system_prompt": "test"},
    )


def _fake_proc(stdout: bytes, returncode: int = 0):
    """构造模拟 subprocess，communicate() 返回固定 stdout。"""
    proc = MagicMock()
    proc.returncode = returncode

    async def _communicate(input=None):
        return stdout, b""

    proc.communicate = _communicate

    async def _wait():
        return returncode

    proc.wait = _wait
    proc.kill = MagicMock()
    return proc


# ─────────── GeminiCLINode.compress_session ────────────────────────────────

@pytest.mark.asyncio
async def test_compress_session_same_sid():
    """成功压缩，CLI 返回原 session_id（原地压缩）。"""
    sid = "421a68ea-0000-0000-0000-000000000000"
    response = json.dumps({"session_id": sid, "response": "Context compressed."})
    proc = _fake_proc(response.encode())

    node = _make_node()
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        msg, new_sid = await node.compress_session(sid, cwd="/tmp")

    assert new_sid == sid
    assert "✅" in msg
    assert "压缩完成" in msg
    assert "421a68ea" in msg


@pytest.mark.asyncio
async def test_compress_session_new_sid():
    """成功压缩，CLI 返回新 session_id。"""
    old_sid = "421a68ea-0000-0000-0000-000000000000"
    new_sid_val = "deadbeef-1111-1111-1111-111111111111"
    response = json.dumps({"session_id": new_sid_val, "response": "Summarized."})
    proc = _fake_proc(response.encode())

    node = _make_node()
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        msg, new_sid = await node.compress_session(old_sid, cwd="/tmp")

    assert new_sid == new_sid_val
    assert "✅" in msg
    assert "421a68ea" in msg   # old
    assert "deadbeef" in msg   # new


@pytest.mark.asyncio
async def test_compress_session_timeout():
    """/compress 超时 → 返回错误，保持原 sid。"""
    sid = "421a68ea-0000-0000-0000-000000000000"
    proc = MagicMock()
    proc.kill = MagicMock()

    async def _slow_communicate(input=None):
        await asyncio.sleep(999)

    proc.communicate = _slow_communicate

    node = _make_node()
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            msg, new_sid = await node.compress_session(sid, cwd="/tmp")

    assert new_sid == sid
    assert "❌" in msg
    assert "超时" in msg


@pytest.mark.asyncio
async def test_compress_session_nonzero_returncode():
    """CLI 退出码非零 → 返回错误信息。"""
    sid = "421a68ea-0000-0000-0000-000000000000"
    proc = _fake_proc(b"some error output", returncode=1)

    node = _make_node()
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        msg, new_sid = await node.compress_session(sid, cwd="/tmp")

    assert new_sid == sid
    assert "❌" in msg
    assert "退出码" in msg


@pytest.mark.asyncio
async def test_compress_session_json_parse_error():
    """CLI 输出无法解析为 JSON → 返回错误信息。"""
    sid = "421a68ea-0000-0000-0000-000000000000"
    proc = _fake_proc(b"not valid json at all", returncode=0)

    node = _make_node()
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        msg, new_sid = await node.compress_session(sid, cwd="/tmp")

    assert new_sid == sid
    assert "❌" in msg
    assert "JSON" in msg


@pytest.mark.asyncio
async def test_compress_session_empty_sid():
    """session_id 为空 → 立即 skip，不调用 CLI。"""
    node = _make_node()
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        msg, new_sid = await node.compress_session("", cwd="/tmp")

    mock_exec.assert_not_called()
    assert new_sid == ""
    assert "跳过" in msg


# ─────────── GraphController.compact_gemini_session ────────────────────────

def _make_controller(gemini_node=None):
    """构造最小化 GraphController mock。"""
    from framework.graph_controller import GraphController

    graph = MagicMock()
    graph._llm_node_instances = {}
    if gemini_node is not None:
        graph._llm_node_instances["gemini_main"] = gemini_node

    session_mgr = MagicMock()
    config = MagicMock()
    ctrl = GraphController.__new__(GraphController)
    ctrl._graph = graph
    ctrl._session_mgr = session_mgr
    ctrl._config = config
    ctrl.sync_node_sessions = MagicMock()
    return ctrl


@pytest.mark.asyncio
async def test_compact_gemini_no_nodes():
    """图中无 GeminiCLINode → 返回提示字符串。"""
    ctrl = _make_controller(gemini_node=None)
    result = await ctrl.compact_gemini_session("thread-001")
    assert "无 Gemini CLI session 可压缩" in result


@pytest.mark.asyncio
async def test_compact_gemini_empty_node_sessions():
    """node_sessions 为空 → 返回提示，不调用 compress_session。"""
    node = _make_node()
    node.compress_session = AsyncMock()

    ctrl = _make_controller(gemini_node=node)
    snapshot = MagicMock()
    snapshot.values = {"node_sessions": {}, "workspace": "/tmp"}
    ctrl._graph.aget_state = AsyncMock(return_value=snapshot)

    result = await ctrl.compact_gemini_session("thread-001")
    node.compress_session.assert_not_called()
    assert "node_sessions 为空" in result


@pytest.mark.asyncio
async def test_compact_gemini_session_unchanged():
    """成功压缩，session_id 不变 → 不调用 aupdate_state。"""
    sid = "421a68ea-0000-0000-0000-000000000000"
    node = _make_node()
    node.compress_session = AsyncMock(return_value=("✅ 压缩完成 (sid=421a68ea): ok", sid))

    ctrl = _make_controller(gemini_node=node)
    snapshot = MagicMock()
    snapshot.values = {"node_sessions": {"jei_main": sid}, "workspace": "/tmp"}
    ctrl._graph.aget_state = AsyncMock(return_value=snapshot)
    ctrl._graph.aupdate_state = AsyncMock()

    result = await ctrl.compact_gemini_session("thread-001")

    node.compress_session.assert_called_once_with(sid, cwd="/tmp")
    ctrl._graph.aupdate_state.assert_not_called()
    assert "✅" in result


@pytest.mark.asyncio
async def test_compact_gemini_session_changed_writes_back():
    """session_id 改变 → 调用 aupdate_state 写回新 sid。"""
    old_sid = "421a68ea-0000-0000-0000-000000000000"
    new_sid = "deadbeef-1111-1111-1111-111111111111"
    node = _make_node()
    node.compress_session = AsyncMock(
        return_value=(f"✅ 压缩完成 (421a68ea → deadbeef): ok", new_sid)
    )

    ctrl = _make_controller(gemini_node=node)
    snapshot = MagicMock()
    snapshot.values = {"node_sessions": {"jei_main": old_sid}, "workspace": "/tmp"}
    ctrl._graph.aget_state = AsyncMock(return_value=snapshot)
    ctrl._graph.aupdate_state = AsyncMock()

    result = await ctrl.compact_gemini_session("thread-001")

    ctrl._graph.aupdate_state.assert_called_once()
    call_kwargs = ctrl._graph.aupdate_state.call_args
    updated_ns = call_kwargs[0][1]["node_sessions"]
    assert updated_ns["jei_main"] == new_sid

    ctrl.sync_node_sessions.assert_called_once()
    assert "✅" in result
