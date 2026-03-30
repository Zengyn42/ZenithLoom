"""Tests for framework/mcp_manager.py — MCPManager lifecycle."""

import asyncio
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from framework.mcp_manager import MCPManager, _ServerEntry


# Always start each test with a fresh MCPManager instance
@pytest.fixture(autouse=True)
def reset_manager():
    """Reset MCPManager singleton between tests."""
    MCPManager._instance = None
    yield
    MCPManager._instance = None


SPEC = {
    "name": "test-server",
    "module": "mcp_servers.fake.server",
    "module_args": ["--transport", "sse", "--port", "9999"],
    "url": "http://localhost:9999/sse",
    "shared": True,
}


# ---------------------------------------------------------------------------
# get_instance
# ---------------------------------------------------------------------------

class TestGetInstance:
    def test_singleton(self):
        a = MCPManager.get_instance()
        b = MCPManager.get_instance()
        assert a is b

    def test_fresh_instance_has_no_servers(self):
        mgr = MCPManager.get_instance()
        assert mgr.running_servers() == []


# ---------------------------------------------------------------------------
# acquire — server already running
# ---------------------------------------------------------------------------

class TestAcquireAlreadyRunning:
    @pytest.mark.asyncio
    async def test_uses_external_server_without_starting_process(self):
        mgr = MCPManager.get_instance()
        with patch.object(mgr, "_is_reachable", return_value=True):
            with patch.object(mgr, "_start_process") as mock_start:
                ok = await mgr.acquire(SPEC, "agent_a")
        assert ok is True
        mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_external_server_has_no_proc(self):
        mgr = MCPManager.get_instance()
        with patch.object(mgr, "_is_reachable", return_value=True):
            await mgr.acquire(SPEC, "agent_a")
        entry = mgr._servers["test-server"]
        assert entry.proc is None

    @pytest.mark.asyncio
    async def test_second_acquire_adds_agent(self):
        mgr = MCPManager.get_instance()
        with patch.object(mgr, "_is_reachable", return_value=True):
            await mgr.acquire(SPEC, "agent_a")
            await mgr.acquire(SPEC, "agent_b")
        assert "agent_a" in mgr._servers["test-server"].agents
        assert "agent_b" in mgr._servers["test-server"].agents


# ---------------------------------------------------------------------------
# acquire — needs to start server
# ---------------------------------------------------------------------------

class TestAcquireStartServer:
    @pytest.mark.asyncio
    async def test_spawns_process_when_not_running(self):
        mgr = MCPManager.get_instance()
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        with patch.object(mgr, "_is_reachable", return_value=False):
            with patch.object(mgr, "_start_process", return_value=mock_proc):
                with patch.object(mgr, "_wait_ready", return_value=True):
                    ok = await mgr.acquire(SPEC, "agent_a")
        assert ok is True
        assert mgr._servers["test-server"].proc is mock_proc

    @pytest.mark.asyncio
    async def test_returns_false_when_not_ready(self):
        mgr = MCPManager.get_instance()
        mock_proc = MagicMock(spec=subprocess.Popen)
        with patch.object(mgr, "_is_reachable", return_value=False):
            with patch.object(mgr, "_start_process", return_value=mock_proc):
                with patch.object(mgr, "_wait_ready", return_value=False):
                    ok = await mgr.acquire(SPEC, "agent_a")
        assert ok is False
        assert "test-server" not in mgr._servers

    @pytest.mark.asyncio
    async def test_kills_proc_on_timeout(self):
        mgr = MCPManager.get_instance()
        mock_proc = MagicMock(spec=subprocess.Popen)
        with patch.object(mgr, "_is_reachable", return_value=False):
            with patch.object(mgr, "_start_process", return_value=mock_proc):
                with patch.object(mgr, "_wait_ready", return_value=False):
                    await mgr.acquire(SPEC, "agent_a")
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_spawn_error(self):
        mgr = MCPManager.get_instance()
        with patch.object(mgr, "_is_reachable", return_value=False):
            with patch.object(mgr, "_start_process", side_effect=OSError("spawn failed")):
                ok = await mgr.acquire(SPEC, "agent_a")
        assert ok is False


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

class TestRelease:
    @pytest.mark.asyncio
    async def test_removes_agent_from_refs(self):
        mgr = MCPManager.get_instance()
        with patch.object(mgr, "_is_reachable", return_value=True):
            await mgr.acquire(SPEC, "agent_a")
            await mgr.acquire(SPEC, "agent_b")
        await mgr.release("test-server", "agent_a")
        assert "agent_a" not in mgr._servers["test-server"].agents
        assert "agent_b" in mgr._servers["test-server"].agents

    @pytest.mark.asyncio
    async def test_server_removed_when_last_agent_releases(self):
        mgr = MCPManager.get_instance()
        with patch.object(mgr, "_is_reachable", return_value=True):
            await mgr.acquire(SPEC, "agent_a")
        await mgr.release("test-server", "agent_a")
        assert "test-server" not in mgr._servers

    @pytest.mark.asyncio
    async def test_terminates_owned_proc_on_last_release(self):
        mgr = MCPManager.get_instance()
        mock_proc = MagicMock(spec=subprocess.Popen)
        with patch.object(mgr, "_is_reachable", return_value=False):
            with patch.object(mgr, "_start_process", return_value=mock_proc):
                with patch.object(mgr, "_wait_ready", return_value=True):
                    await mgr.acquire(SPEC, "agent_a")
        await mgr.release("test-server", "agent_a")
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_terminate_external_proc(self):
        mgr = MCPManager.get_instance()
        with patch.object(mgr, "_is_reachable", return_value=True):
            await mgr.acquire(SPEC, "agent_a")
        # External server (proc=None) — should not call terminate
        await mgr.release("test-server", "agent_a")
        # No error = pass

    @pytest.mark.asyncio
    async def test_release_unknown_server_is_noop(self):
        mgr = MCPManager.get_instance()
        await mgr.release("nonexistent", "agent_a")  # should not raise

    @pytest.mark.asyncio
    async def test_release_all(self):
        mgr = MCPManager.get_instance()
        spec2 = {**SPEC, "name": "server-2", "url": "http://localhost:9998/sse"}
        with patch.object(mgr, "_is_reachable", return_value=True):
            await mgr.acquire(SPEC, "agent_a")
            await mgr.acquire(spec2, "agent_a")
        assert len(mgr.running_servers()) == 2
        await mgr.release_all("agent_a")
        assert len(mgr.running_servers()) == 0


# ---------------------------------------------------------------------------
# get_sse_configs / get_all_configs
# ---------------------------------------------------------------------------

class TestGetConfigs:
    @pytest.mark.asyncio
    async def test_get_sse_configs_returns_running_servers(self):
        mgr = MCPManager.get_instance()
        with patch.object(mgr, "_is_reachable", return_value=True):
            await mgr.acquire(SPEC, "agent_a")
        configs = mgr.get_sse_configs(["test-server"])
        assert configs == {"test-server": {"type": "sse", "url": "http://localhost:9999/sse"}}

    @pytest.mark.asyncio
    async def test_get_sse_configs_skips_not_running(self):
        mgr = MCPManager.get_instance()
        configs = mgr.get_sse_configs(["nonexistent"])
        assert configs == {}

    @pytest.mark.asyncio
    async def test_get_all_configs(self):
        mgr = MCPManager.get_instance()
        spec2 = {**SPEC, "name": "server-2", "url": "http://localhost:9998/sse"}
        with patch.object(mgr, "_is_reachable", return_value=True):
            await mgr.acquire(SPEC, "agent_a")
            await mgr.acquire(spec2, "agent_b")
        configs = mgr.get_all_configs()
        assert len(configs) == 2
        assert configs["test-server"]["type"] == "sse"
        assert configs["server-2"]["url"] == "http://localhost:9998/sse"

    def test_get_all_configs_empty(self):
        mgr = MCPManager.get_instance()
        assert mgr.get_all_configs() == {}


# ---------------------------------------------------------------------------
# _is_reachable
# ---------------------------------------------------------------------------

class TestIsReachable:
    def test_returns_true_on_200(self):
        mgr = MCPManager.get_instance()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert mgr._is_reachable("http://localhost:8101/sse") is True

    def test_returns_false_on_connection_error(self):
        mgr = MCPManager.get_instance()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert mgr._is_reachable("http://localhost:8101/sse") is False

    def test_strips_sse_suffix_for_check(self):
        mgr = MCPManager.get_instance()
        captured_urls = []

        def fake_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            raise OSError("refused")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            mgr._is_reachable("http://localhost:8101/sse")

        assert captured_urls[0] == "http://localhost:8101/"
