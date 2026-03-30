"""Tests for mcp_servers/vault_sync MCP Server."""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from mcp_servers.vault_sync.server import (
    _run_rsync,
    vault_sync_pull,
    vault_sync_push,
    vault_sync_status,
)


# ---------------------------------------------------------------------------
# _run_rsync
# ---------------------------------------------------------------------------

class TestRunRsync:
    @patch("mcp_servers.vault_sync.server.subprocess.run")
    def test_success_no_changes(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="sending incremental file list\n\nsent 100 bytes\ntotal size is 1000\n",
            stderr="",
        )
        result = _run_rsync(["rsync", "-a", "--delete", "/src/", "/dst/"])
        assert result["success"] is True
        assert result["files_changed"] == 0

    @patch("mcp_servers.vault_sync.server.subprocess.run")
    def test_success_with_changes(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="sending incremental file list\nfile1.md\ndir/file2.txt\n\nsent 200 bytes\ntotal size is 2000\n",
            stderr="",
        )
        result = _run_rsync(["rsync", "-a", "/src/", "/dst/"])
        assert result["success"] is True
        assert result["files_changed"] == 2
        assert "file1.md" in result["changed"]
        assert "dir/file2.txt" in result["changed"]

    @patch("mcp_servers.vault_sync.server.subprocess.run")
    def test_rsync_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=23,
            stdout="",
            stderr="rsync: some error occurred\n",
        )
        result = _run_rsync(["rsync", "-a", "/src/", "/dst/"])
        assert result["success"] is False
        assert result["return_code"] == 23
        assert "failed" in result["summary"]

    @patch("mcp_servers.vault_sync.server.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="rsync", timeout=60)
        result = _run_rsync(["rsync", "-a", "/src/", "/dst/"])
        assert result["success"] is False
        assert "timed out" in result["summary"]

    @patch("mcp_servers.vault_sync.server.subprocess.run")
    def test_rsync_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = _run_rsync(["rsync", "-a", "/src/", "/dst/"])
        assert result["success"] is False
        assert "not found" in result["summary"]

    @patch("mcp_servers.vault_sync.server.subprocess.run")
    def test_filters_rsync_metadata_lines(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "sending incremental file list\n"
                "./\n"
                "real_file.md\n"
                "building file list ... done\n"
                "receiving file list ... done\n"
                "sent 100 bytes  received 50 bytes\n"
                "total size is 1000  speedup is 5.0\n"
            ),
            stderr="",
        )
        result = _run_rsync(["rsync", "-a", "/src/", "/dst/"])
        assert result["success"] is True
        assert result["files_changed"] == 1
        assert result["changed"] == ["real_file.md"]

    @patch("mcp_servers.vault_sync.server.subprocess.run")
    def test_caps_changed_at_100(self, mock_run):
        files = "\n".join([f"file_{i}.txt" for i in range(200)])
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"sending incremental file list\n{files}\nsent 100\ntotal size is 1\n",
            stderr="",
        )
        result = _run_rsync(["rsync", "-a", "/src/", "/dst/"])
        assert result["success"] is True
        assert len(result["changed"]) == 100


# ---------------------------------------------------------------------------
# vault_sync_pull
# ---------------------------------------------------------------------------

class TestVaultSyncPull:
    @patch("mcp_servers.vault_sync.server._run_rsync")
    def test_pull_calls_rsync_correct_direction(self, mock_rsync):
        mock_rsync.return_value = {
            "success": True, "return_code": 0,
            "files_changed": 0, "changed": [],
            "summary": "0 file(s) changed", "stderr": None,
        }
        result = vault_sync_pull()
        assert result["direction"] == "pull (Windows -> WSL)"
        # Verify rsync args: source is Windows path, dest is WSL path
        args = mock_rsync.call_args[0][0]
        assert args[0] == "rsync"
        assert "--delete" in args


# ---------------------------------------------------------------------------
# vault_sync_push
# ---------------------------------------------------------------------------

class TestVaultSyncPush:
    @patch("mcp_servers.vault_sync.server._run_rsync")
    def test_push_calls_rsync_correct_direction(self, mock_rsync):
        mock_rsync.return_value = {
            "success": True, "return_code": 0,
            "files_changed": 0, "changed": [],
            "summary": "0 file(s) changed", "stderr": None,
        }
        result = vault_sync_push()
        assert result["direction"] == "push (WSL -> Windows)"


# ---------------------------------------------------------------------------
# vault_sync_status
# ---------------------------------------------------------------------------

class TestVaultSyncStatus:
    @patch("mcp_servers.vault_sync.server._run_rsync")
    def test_status_dry_run_both_directions(self, mock_rsync):
        mock_rsync.return_value = {
            "success": True, "return_code": 0,
            "files_changed": 3, "changed": ["a.md", "b.md", "c.md"],
            "summary": "3 file(s) changed", "stderr": None,
        }
        result = vault_sync_status()
        assert "pull_preview" in result
        assert "push_preview" in result
        assert "Dry-run" in result["note"]
        # Should have been called twice (pull + push)
        assert mock_rsync.call_count == 2
        # Both calls should include --dry-run
        for call in mock_rsync.call_args_list:
            assert "--dry-run" in call[0][0]
