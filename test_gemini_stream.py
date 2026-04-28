"""
GeminiCLINode stream-json 模式测试

覆盖：
  1. [unit] 有 stream callback 时自动切换 -o stream-json，增量文本实时回调
  2. [unit] 无 stream callback 时走原 -o json 路径（向后兼容）
  3. [unit] stream 模式下 init 事件提取 session_id
  4. [unit] stream 模式下 result 失败时抛 RuntimeError
  5. [e2e]  真实 Gemini CLI stream 调用（默认 skip）

运行：
    python3 -m pytest test_gemini_stream.py -v
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from framework.config import AgentConfig
from framework.nodes.llm.gemini import GeminiCLINode
from framework.nodes.llm.llm_node import _stream_cb, set_stream_callback


# ─────────────────────────── helpers ───────────────────────────────────────

def _make_node(model: str = "gemini-2.5-flash") -> GeminiCLINode:
    config = AgentConfig(
        tools=[],
        permission_mode="bypassPermissions",
        setting_sources=None,
        settings_override={"enabledPlugins": []},
    )
    return GeminiCLINode(
        config=config,
        node_config={"id": "test_node", "model": model, "system_prompt": "You are a helpful assistant."},
    )


class _FakeAsyncIterator:
    """逐行异步迭代器，模拟 asyncio.StreamReader。"""

    def __init__(self, lines: list[str]):
        self._lines = [l.encode() for l in lines]
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._lines):
            raise StopAsyncIteration
        val = self._lines[self._idx]
        self._idx += 1
        return val


class _FakeProcess:
    """模拟 asyncio.subprocess.Process。"""

    def __init__(self, stdout_lines: list[str], returncode: int = 0):
        self.stdout = _FakeAsyncIterator(stdout_lines)
        self.stderr = _FakeAsyncIterator([])
        self.returncode = returncode

        self.stdin = AsyncMock()
        self.stdin.write = MagicMock()
        self.stdin.drain = AsyncMock()
        self.stdin.close = MagicMock()
        self.stdin.wait_closed = AsyncMock()

    async def wait(self):
        return self.returncode

    def kill(self):
        pass


# ─────────────────────── unit tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_mode_triggers_when_callback_set():
    """有 stream callback → 使用 stream-json，增量文本实时回调。"""
    stream_lines = [
        json.dumps({"type": "init", "session_id": "sid-abc123", "model": "gemini-2.5-flash"}),
        json.dumps({"type": "message", "role": "user", "content": "say hi"}),
        json.dumps({"type": "message", "role": "assistant", "content": "Hello", "delta": True}),
        json.dumps({"type": "message", "role": "assistant", "content": " world", "delta": True}),
        json.dumps({"type": "result", "status": "success", "stats": {}}),
    ]
    fake_proc = _FakeProcess(stream_lines)

    received_chunks: list[str] = []

    def on_stream(text: str, is_thinking: bool):
        received_chunks.append(text)

    node = _make_node()

    with patch("asyncio.create_subprocess_exec", return_value=fake_proc) as mock_exec:
        set_stream_callback(on_stream)
        try:
            reply, sid = await node._run_cli("say hi", model="gemini-2.5-flash")
        finally:
            set_stream_callback(None)

    # 验证使用了 stream-json
    call_args = mock_exec.call_args[0]
    assert "-o" in call_args
    idx = list(call_args).index("-o")
    assert call_args[idx + 1] == "stream-json", f"expected stream-json, got {call_args[idx+1]}"

    # 验证增量回调
    assert received_chunks == ["Hello", " world"], f"chunks: {received_chunks}"

    # 验证最终文本
    assert reply == "Hello world"

    # 验证 session_id 来自 init 事件
    assert sid == "sid-abc123"


@pytest.mark.asyncio
async def test_non_stream_mode_when_no_callback():
    """无 stream callback → 使用 json，走 communicate() 路径。"""
    json_response = json.dumps({"response": "pong", "session_id": "sid-xyz"})
    fake_proc = _FakeProcess([])  # communicate() 路径不走 __aiter__

    async def fake_communicate(input=None):
        return json_response.encode(), b""

    fake_proc.communicate = fake_communicate

    node = _make_node()

    # 确保没有 stream callback
    set_stream_callback(None)

    with patch("asyncio.create_subprocess_exec", return_value=fake_proc) as mock_exec:
        reply, sid = await node._run_cli("ping", model="gemini-2.5-flash")

    # 验证使用了 json（非 stream）
    call_args = mock_exec.call_args[0]
    idx = list(call_args).index("-o")
    assert call_args[idx + 1] == "json", f"expected json, got {call_args[idx+1]}"

    assert reply == "pong"
    assert sid == "sid-xyz"


@pytest.mark.asyncio
async def test_stream_session_id_from_init():
    """stream 模式：session_id 从 init 事件提取，而非 result。"""
    stream_lines = [
        json.dumps({"type": "init", "session_id": "init-session-999", "model": "gemini-2.5-flash"}),
        json.dumps({"type": "message", "role": "assistant", "content": "hi", "delta": True}),
        json.dumps({"type": "result", "status": "success", "stats": {}}),
    ]
    fake_proc = _FakeProcess(stream_lines)

    node = _make_node()

    with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
        set_stream_callback(lambda t, b: None)
        try:
            reply, sid = await node._run_cli("hi", model="gemini-2.5-flash")
        finally:
            set_stream_callback(None)

    assert sid == "init-session-999"
    assert reply == "hi"


@pytest.mark.asyncio
async def test_stream_result_failure_raises():
    """stream 模式：result status != success → 抛 RuntimeError。"""
    stream_lines = [
        json.dumps({"type": "init", "session_id": "sid-err", "model": "gemini-2.5-flash"}),
        json.dumps({"type": "result", "status": "error", "error": "something went wrong"}),
    ]
    fake_proc = _FakeProcess(stream_lines)

    node = _make_node()

    with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
        set_stream_callback(lambda t, b: None)
        try:
            with pytest.raises(RuntimeError, match="something went wrong"):
                await node._run_cli("hi", model="gemini-2.5-flash")
        finally:
            set_stream_callback(None)


# ─────────────────────── e2e test (skip by default) ────────────────────────

@pytest.mark.skip(reason="requires live Gemini CLI")
@pytest.mark.asyncio
async def test_e2e_gemini_cli_stream():
    """E2E：真实 Gemini CLI stream 调用，验证增量回调触发。"""
    node = _make_node(model="gemini-2.5-flash")

    chunks: list[str] = []

    def on_stream(text: str, is_thinking: bool):
        chunks.append(text)

    set_stream_callback(on_stream)
    try:
        reply, sid = await node.call_llm("say 'hello' in exactly one word")
    finally:
        set_stream_callback(None)

    assert reply, "E2E: GeminiCLI stream 返回空文本"
    assert sid, "E2E: GeminiCLI stream 未返回 session_id"
    assert len(chunks) > 0, "E2E: stream callback 从未被调用"
    assert "".join(chunks).strip() == reply, "E2E: chunks 拼接结果与 reply 不一致"
