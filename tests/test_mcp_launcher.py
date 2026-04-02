"""
Tests for framework/mcp_launcher.py — MCPLauncher core behaviour.

All external dependencies (subprocess, aiohttp, proxy connect) are mocked.
No real processes are started; no real network connections are made.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from framework.mcp_launcher import MCPLauncher, _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pid(pid_file: Path, pid: int) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


# ===========================================================================
# is_running()
# ===========================================================================

class TestIsRunning:
    def test_pid_file_not_exists_returns_false(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        assert MCPLauncher.is_running(pid_file) is False

    def test_pid_file_exists_but_process_dead_returns_false_and_removes_file(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        # Use a PID that almost certainly doesn't exist (large number)
        _write_pid(pid_file, 999999999)
        with patch("os.kill", side_effect=ProcessLookupError):
            result = MCPLauncher.is_running(pid_file)
        assert result is False
        assert not pid_file.exists()

    def test_pid_file_exists_process_alive_returns_true(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        current_pid = os.getpid()
        _write_pid(pid_file, current_pid)
        # os.kill(pid, 0) for our own pid should succeed — no mock needed
        result = MCPLauncher.is_running(pid_file)
        assert result is True

    def test_relative_pid_file_resolved_from_project_root(self, tmp_path):
        # Relative path should be resolved relative to _PROJECT_ROOT
        # We patch the resolved absolute path to avoid touching real filesystem
        abs_pid_file = _PROJECT_ROOT / "data" / "test_rel.pid"
        with patch.object(Path, "exists", return_value=False):
            result = MCPLauncher.is_running(Path("data/test_rel.pid"))
        assert result is False

    def test_permission_error_treated_as_dead_cleans_up(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        _write_pid(pid_file, 12345)
        with patch("os.kill", side_effect=PermissionError):
            result = MCPLauncher.is_running(pid_file)
        # PermissionError means we can't signal it — treated as not-running
        assert result is False
        assert not pid_file.exists()


# ===========================================================================
# launch()
# ===========================================================================

class TestLaunch:
    def test_normal_launch_calls_popen_and_writes_pid(self, tmp_path):
        pid_file = tmp_path / "server.pid"
        mock_proc = MagicMock()
        mock_proc.pid = 42

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            MCPLauncher.launch(
                module="mcp_servers.test",
                host="127.0.0.1",
                port=9000,
                pid_file=pid_file,
            )

        mock_popen.assert_called_once()
        assert pid_file.exists()
        assert pid_file.read_text().strip() == "42"

    def test_launch_popen_called_with_correct_args(self, tmp_path):
        pid_file = tmp_path / "server.pid"
        mock_proc = MagicMock()
        mock_proc.pid = 99

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            MCPLauncher.launch("my_module", "0.0.0.0", 8100, pid_file)

        args = mock_popen.call_args[0][0]
        assert "--transport" in args
        assert "sse" in args
        assert "--host" in args
        assert "0.0.0.0" in args
        assert "--port" in args
        assert "8100" in args
        assert "my_module" in args

    def test_double_check_already_running_skips_launch(self, tmp_path):
        """Inside launch(), if is_running() returns True after acquiring lock, skip _do_launch.

        The double-check in launch() calls is_running() once after acquiring the lock.
        If it returns True, _do_launch (and therefore Popen) must not be called.
        """
        pid_file = tmp_path / "server.pid"
        mock_proc = MagicMock()
        mock_proc.pid = 55

        # is_running returns True → double-check decides server is already up
        with patch.object(MCPLauncher, "is_running", return_value=True):
            with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                MCPLauncher.launch("mod", "127.0.0.1", 9000, pid_file)

        # Popen must NOT be called because double-check found server running
        mock_popen.assert_not_called()

    def test_lock_contention_skips_launch(self, tmp_path):
        """If lock is held by another process (BlockingIOError), skip launch."""
        pid_file = tmp_path / "server.pid"

        import fcntl as _fcntl

        mock_proc = MagicMock()
        mock_proc.pid = 77

        with patch("fcntl.flock", side_effect=BlockingIOError):
            with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                # Should not raise; should skip silently
                MCPLauncher.launch("mod", "127.0.0.1", 9000, pid_file)

        mock_popen.assert_not_called()


# ===========================================================================
# wait_ready()
# ===========================================================================

class TestWaitReady:
    @pytest.mark.asyncio
    async def test_server_ready_first_attempt_returns_true(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await MCPLauncher.wait_ready("http://127.0.0.1:9000/sse", timeout=5.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_server_ready_second_attempt_returns_true(self):
        """First attempt raises, second attempt succeeds."""
        call_count = [0]

        class FakeResp:
            status = 200
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                pass

        class FakeSession:
            def get(self, url, timeout=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise OSError("not ready yet")
                return FakeResp()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                pass

        with patch("aiohttp.ClientSession", return_value=FakeSession()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await MCPLauncher.wait_ready("http://127.0.0.1:9000/sse", timeout=5.0)

        assert result is True
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        """If server never becomes ready, return False."""
        import asyncio

        # Simulate a loop where time always exceeds deadline immediately
        real_loop = asyncio.get_event_loop()

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status = 503
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_cls.return_value = mock_session

            with patch("asyncio.sleep", new_callable=AsyncMock):
                # Use a very short timeout so the loop expires quickly
                result = await MCPLauncher.wait_ready(
                    "http://127.0.0.1:9000/sse", timeout=0.0
                )

        assert result is False


# ===========================================================================
# ensure_and_connect()
# ===========================================================================

class TestEnsureAndConnect:
    _BASE_CONF = {
        "name": "test_mcp",
        "module": "mcp_servers.test",
        "transport": "sse",
        "host": "127.0.0.1",
        "port": 9999,
        "pid_file": "data/test_mcp/test.pid",
    }

    @pytest.mark.asyncio
    async def test_non_sse_transport_returns_none(self):
        conf = {**self._BASE_CONF, "transport": "stdio"}
        proxy_class = MagicMock()
        result = await MCPLauncher.ensure_and_connect(conf, proxy_class)
        assert result is None
        proxy_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_server_already_running_skips_launch(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        conf = {**self._BASE_CONF, "pid_file": str(pid_file)}

        mock_proxy = AsyncMock()
        proxy_class = MagicMock(return_value=mock_proxy)

        with patch.object(MCPLauncher, "is_running", return_value=True) as mock_running:
            with patch.object(MCPLauncher, "launch") as mock_launch:
                with patch.object(MCPLauncher, "wait_ready", new_callable=AsyncMock, return_value=True):
                    result = await MCPLauncher.ensure_and_connect(conf, proxy_class)

        mock_launch.assert_not_called()
        assert result is mock_proxy

    @pytest.mark.asyncio
    async def test_server_not_running_calls_launch_then_connect(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        conf = {**self._BASE_CONF, "pid_file": str(pid_file)}

        mock_proxy = AsyncMock()
        proxy_class = MagicMock(return_value=mock_proxy)

        with patch.object(MCPLauncher, "is_running", return_value=False):
            with patch.object(MCPLauncher, "launch") as mock_launch:
                with patch.object(MCPLauncher, "wait_ready", new_callable=AsyncMock, return_value=True):
                    result = await MCPLauncher.ensure_and_connect(conf, proxy_class)

        mock_launch.assert_called_once()
        mock_proxy.connect.assert_called_once()
        assert result is mock_proxy

    @pytest.mark.asyncio
    async def test_connect_failure_returns_none(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        conf = {**self._BASE_CONF, "pid_file": str(pid_file)}

        mock_proxy = AsyncMock()
        mock_proxy.connect.side_effect = ConnectionRefusedError("refused")
        proxy_class = MagicMock(return_value=mock_proxy)

        with patch.object(MCPLauncher, "is_running", return_value=True):
            with patch.object(MCPLauncher, "wait_ready", new_callable=AsyncMock, return_value=True):
                result = await MCPLauncher.ensure_and_connect(conf, proxy_class)

        assert result is None

    @pytest.mark.asyncio
    async def test_server_not_ready_returns_none(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        conf = {**self._BASE_CONF, "pid_file": str(pid_file)}

        proxy_class = MagicMock()

        with patch.object(MCPLauncher, "is_running", return_value=False):
            with patch.object(MCPLauncher, "launch"):
                with patch.object(MCPLauncher, "wait_ready", new_callable=AsyncMock, return_value=False):
                    result = await MCPLauncher.ensure_and_connect(conf, proxy_class)

        assert result is None
        proxy_class.assert_not_called()
