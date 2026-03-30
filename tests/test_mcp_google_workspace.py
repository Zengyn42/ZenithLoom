"""Tests for mcp_servers/google_workspace MCP Server."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest

from mcp_servers.google_workspace.server import (
    _validate_command,
    _run_gws,
    gws_gmail_read,
    gws_drive_list,
    gws_slides_exec,
    gws_docs_exec,
)


# ---------------------------------------------------------------------------
# _validate_command
# ---------------------------------------------------------------------------

class TestValidateCommand:
    def test_valid_gmail_command(self):
        assert _validate_command("gws gmail list --query is:unread", "gws gmail") is None

    def test_valid_slides_command(self):
        assert _validate_command("gws slides list", "gws slides") is None

    def test_wrong_prefix(self):
        error = _validate_command("gws docs list", "gws slides")
        assert error is not None
        assert "must start with" in error

    def test_dangerous_pipe(self):
        error = _validate_command("gws gmail list | cat", "gws gmail")
        assert error is not None
        assert "metacharacters" in error

    def test_dangerous_semicolon(self):
        error = _validate_command("gws gmail list; rm -rf /", "gws gmail")
        assert error is not None

    def test_dangerous_dollar(self):
        error = _validate_command("gws gmail list $HOME", "gws gmail")
        assert error is not None

    def test_dangerous_backtick(self):
        error = _validate_command("gws gmail list `whoami`", "gws gmail")
        assert error is not None

    def test_dangerous_redirect(self):
        error = _validate_command("gws gmail list > /tmp/out", "gws gmail")
        assert error is not None

    def test_dangerous_ampersand(self):
        error = _validate_command("gws gmail list & echo x", "gws gmail")
        assert error is not None


# ---------------------------------------------------------------------------
# _run_gws
# ---------------------------------------------------------------------------

class TestRunGws:
    @patch("mcp_servers.google_workspace.server.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="ok\n", stderr=""
        )
        result = _run_gws(["gws", "gmail", "list"])
        assert result["success"] is True
        assert result["stdout"] == "ok\n"

    @patch("mcp_servers.google_workspace.server.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error msg"
        )
        result = _run_gws(["gws", "gmail", "list"])
        assert result["success"] is False
        assert result["returncode"] == 1

    @patch("mcp_servers.google_workspace.server.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gws", timeout=60)
        result = _run_gws(["gws", "gmail", "list"])
        assert result["success"] is False
        assert "timed out" in result["error"]

    @patch("mcp_servers.google_workspace.server.subprocess.run")
    def test_gws_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = _run_gws(["gws", "gmail", "list"])
        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# gws_gmail_read
# ---------------------------------------------------------------------------

class TestGwsGmailRead:
    @patch("mcp_servers.google_workspace.server._run_gws")
    def test_default_query(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "messages"}
        result = gws_gmail_read()
        mock_run.assert_called_once_with(["gws", "gmail", "list", "--query", "is:unread"])

    @patch("mcp_servers.google_workspace.server._run_gws")
    def test_custom_query(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "messages"}
        result = gws_gmail_read("from:boss@example.com")
        mock_run.assert_called_once_with(
            ["gws", "gmail", "list", "--query", "from:boss@example.com"]
        )

    def test_dangerous_query_rejected(self):
        result = gws_gmail_read("is:unread | cat /etc/passwd")
        assert result["success"] is False
        assert "metacharacters" in result["error"]


# ---------------------------------------------------------------------------
# gws_drive_list
# ---------------------------------------------------------------------------

class TestGwsDriveList:
    @patch("mcp_servers.google_workspace.server._run_gws")
    def test_default_no_folder(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "files"}
        result = gws_drive_list()
        mock_run.assert_called_once_with(["gws", "drive", "list"])

    @patch("mcp_servers.google_workspace.server._run_gws")
    def test_with_folder(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "files"}
        result = gws_drive_list("MyFolder")
        mock_run.assert_called_once_with(["gws", "drive", "list", "--folder", "MyFolder"])

    def test_dangerous_folder_rejected(self):
        result = gws_drive_list("folder; rm -rf /")
        assert result["success"] is False
        assert "metacharacters" in result["error"]


# ---------------------------------------------------------------------------
# gws_slides_exec / gws_docs_exec
# ---------------------------------------------------------------------------

class TestGwsSlidesExec:
    @patch("mcp_servers.google_workspace.server._run_gws")
    def test_valid_command(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "ok"}
        result = gws_slides_exec("gws slides list")
        assert mock_run.called

    def test_wrong_prefix_rejected(self):
        result = gws_slides_exec("gws docs list")
        assert result["success"] is False
        assert "must start with" in result["error"]

    def test_injection_rejected(self):
        result = gws_slides_exec("gws slides list; rm -rf /")
        assert result["success"] is False


class TestGwsDocsExec:
    @patch("mcp_servers.google_workspace.server._run_gws")
    def test_valid_command(self, mock_run):
        mock_run.return_value = {"success": True, "stdout": "ok"}
        result = gws_docs_exec("gws docs list")
        assert mock_run.called

    def test_wrong_prefix_rejected(self):
        result = gws_docs_exec("gws slides list")
        assert result["success"] is False

    def test_injection_rejected(self):
        result = gws_docs_exec("gws docs get --id x | cat")
        assert result["success"] is False
