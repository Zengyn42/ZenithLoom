"""
Tests for EntityLoader.start_mcps() and stop_mcps() — framework/agent_loader.py.

EntityLoader reads entity.json at construction time, so we use a minimal real
blueprint (technical_architect) for the agent_dir, and point data_dir at a
tmp_path containing a synthetic identity.json.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from framework.agent_loader import EntityLoader

# ---------------------------------------------------------------------------
# Blueprint path (needs a real entity.json to exist for EntityLoader.__init__)
# ---------------------------------------------------------------------------

_BLUEPRINT_DIR = (
    Path(__file__).resolve().parent.parent
    / "blueprints" / "role_agents" / "technical_architect"
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_loader(tmp_path: Path, identity: dict | None = None) -> EntityLoader:
    """Create an EntityLoader with a synthetic identity.json in tmp_path."""
    if identity is not None:
        (tmp_path / "identity.json").write_text(
            json.dumps(identity), encoding="utf-8"
        )
    return EntityLoader(_BLUEPRINT_DIR, data_dir=tmp_path)


# ===========================================================================
# start_mcps — edge cases
# ===========================================================================

class TestStartMcpsEdgeCases:
    @pytest.mark.asyncio
    async def test_no_identity_json_skips_silently(self, tmp_path):
        """If identity.json does not exist, start_mcps should return without error."""
        loader = _make_loader(tmp_path)  # no identity.json written
        # No error should be raised
        await loader.start_mcps()
        assert loader._mcp_proxies == {}

    @pytest.mark.asyncio
    async def test_empty_mcps_list_skips_silently(self, tmp_path):
        """If mcps field is empty list, nothing is launched."""
        loader = _make_loader(tmp_path, identity={"name": "test_agent", "mcps": []})
        await loader.start_mcps()
        assert loader._mcp_proxies == {}

    @pytest.mark.asyncio
    async def test_no_mcps_field_skips_silently(self, tmp_path):
        """If identity.json has no 'mcps' key, nothing is launched."""
        loader = _make_loader(tmp_path, identity={"name": "test_agent"})
        await loader.start_mcps()
        assert loader._mcp_proxies == {}

    @pytest.mark.asyncio
    async def test_unknown_mcp_name_is_skipped(self, tmp_path):
        """MCP with unrecognised name logs a warning and is skipped."""
        loader = _make_loader(tmp_path, identity={
            "name": "test_agent",
            "mcps": [{"name": "unknown_mcp", "transport": "sse"}],
        })
        await loader.start_mcps()
        assert "unknown_mcp" not in loader._mcp_proxies


# ===========================================================================
# start_mcps — normal flow
# ===========================================================================

class TestStartMcpsNormalFlow:
    @pytest.mark.asyncio
    async def test_successful_connect_stores_proxy(self, tmp_path):
        """ensure_and_connect returns a proxy → stored in _mcp_proxies."""
        mock_proxy = AsyncMock()

        loader = _make_loader(tmp_path, identity={
            "name": "test_agent",
            "mcps": [{"name": "agent_mail", "transport": "sse",
                      "host": "127.0.0.1", "port": 8200,
                      "pid_file": "data/agent_mail/mail.pid"}],
        })

        with patch("framework.mcp_launcher.MCPLauncher.ensure_and_connect",
                   new_callable=AsyncMock, return_value=mock_proxy):
            await loader.start_mcps()

        assert "agent_mail" in loader._mcp_proxies
        assert loader._mcp_proxies["agent_mail"] is mock_proxy

    @pytest.mark.asyncio
    async def test_ensure_and_connect_called_with_correct_conf(self, tmp_path):
        """ensure_and_connect is invoked with the correct mcp_conf dict."""
        mcp_conf = {
            "name": "agent_mail",
            "transport": "sse",
            "host": "127.0.0.1",
            "port": 8200,
            "pid_file": "data/agent_mail/mail.pid",
        }
        mock_proxy = AsyncMock()

        loader = _make_loader(tmp_path, identity={"name": "test_agent", "mcps": [mcp_conf]})

        with patch("framework.mcp_launcher.MCPLauncher.ensure_and_connect",
                   new_callable=AsyncMock, return_value=mock_proxy) as mock_eac:
            await loader.start_mcps()

        call_args = mock_eac.call_args
        assert call_args[0][0] == mcp_conf

    @pytest.mark.asyncio
    async def test_failed_connect_not_stored_in_proxies(self, tmp_path):
        """If ensure_and_connect returns None, proxy is NOT stored."""
        loader = _make_loader(tmp_path, identity={
            "name": "test_agent",
            "mcps": [{"name": "agent_mail", "transport": "sse"}],
        })

        with patch("framework.mcp_launcher.MCPLauncher.ensure_and_connect",
                   new_callable=AsyncMock, return_value=None):
            await loader.start_mcps()

        assert "agent_mail" not in loader._mcp_proxies

    @pytest.mark.asyncio
    async def test_agent_mail_proxy_register_called_after_connect(self, tmp_path):
        """After connecting agent_mail, register() must be called with agent name."""
        mock_proxy = AsyncMock()
        mock_proxy.register = AsyncMock(return_value="ok")

        loader = _make_loader(tmp_path, identity={
            "name": "my_agent",
            "mcps": [{"name": "agent_mail", "transport": "sse"}],
        })

        with patch("framework.mcp_launcher.MCPLauncher.ensure_and_connect",
                   new_callable=AsyncMock, return_value=mock_proxy):
            await loader.start_mcps()

        mock_proxy.register.assert_called_once_with("my_agent")

    @pytest.mark.asyncio
    async def test_non_agent_mail_proxy_register_not_called(self, tmp_path):
        """For non-agent_mail MCPs, register() should not be called."""
        mock_proxy = AsyncMock()
        mock_proxy.register = AsyncMock()

        # Patch _resolve_proxy_class to return something for a custom name
        with patch("framework.agent_loader._resolve_proxy_class", return_value=MagicMock()):
            loader = _make_loader(tmp_path, identity={
                "name": "my_agent",
                "mcps": [{"name": "heartbeat", "transport": "sse"}],
            })

            with patch("framework.mcp_launcher.MCPLauncher.ensure_and_connect",
                       new_callable=AsyncMock, return_value=mock_proxy):
                await loader.start_mcps()

        # register should NOT have been called (heartbeat is not agent_mail)
        mock_proxy.register.assert_not_called()


# ===========================================================================
# stop_mcps
# ===========================================================================

class TestStopMcps:
    @pytest.mark.asyncio
    async def test_stop_mcps_calls_disconnect_on_all_proxies(self, tmp_path):
        loader = _make_loader(tmp_path, identity={"name": "test_agent"})

        proxy_a = AsyncMock()
        proxy_b = AsyncMock()
        loader._mcp_proxies = {"mcp_a": proxy_a, "mcp_b": proxy_b}

        await loader.stop_mcps()

        proxy_a.disconnect.assert_called_once()
        proxy_b.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_mcps_clears_proxies_dict(self, tmp_path):
        loader = _make_loader(tmp_path, identity={"name": "test_agent"})

        proxy_a = AsyncMock()
        loader._mcp_proxies = {"mcp_a": proxy_a}

        await loader.stop_mcps()

        assert loader._mcp_proxies == {}

    @pytest.mark.asyncio
    async def test_stop_mcps_agent_mail_calls_unregister_before_disconnect(self, tmp_path):
        loader = _make_loader(tmp_path, identity={"name": "my_agent"})

        mock_proxy = AsyncMock()
        mock_proxy.unregister = AsyncMock(return_value="ok")
        loader._mcp_proxies = {"agent_mail": mock_proxy}

        await loader.stop_mcps()

        mock_proxy.unregister.assert_called_once_with("my_agent")
        mock_proxy.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_mcps_empty_proxies_no_error(self, tmp_path):
        loader = _make_loader(tmp_path, identity={"name": "test_agent"})
        # _mcp_proxies already empty by default
        await loader.stop_mcps()  # Should not raise
        assert loader._mcp_proxies == {}

    @pytest.mark.asyncio
    async def test_stop_mcps_disconnect_error_does_not_prevent_cleanup(self, tmp_path):
        """Even if disconnect() raises, _mcp_proxies is still cleared."""
        loader = _make_loader(tmp_path, identity={"name": "test_agent"})

        proxy_a = AsyncMock()
        proxy_a.disconnect.side_effect = RuntimeError("boom")
        loader._mcp_proxies = {"mcp_a": proxy_a}

        await loader.stop_mcps()

        assert loader._mcp_proxies == {}

    @pytest.mark.asyncio
    async def test_stop_mcps_non_agent_mail_no_unregister(self, tmp_path):
        """For non-agent_mail proxies, unregister should not be called."""
        loader = _make_loader(tmp_path, identity={"name": "test_agent"})

        proxy = AsyncMock()
        proxy.unregister = AsyncMock()
        loader._mcp_proxies = {"heartbeat": proxy}

        await loader.stop_mcps()

        proxy.unregister.assert_not_called()
        proxy.disconnect.assert_called_once()
