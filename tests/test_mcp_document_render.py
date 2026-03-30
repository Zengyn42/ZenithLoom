"""Tests for mcp_servers/document_render MCP Server."""

import json
import os
import subprocess
import time
from unittest.mock import patch, MagicMock

import pytest

from mcp_servers.document_render.server import (
    _presenton_ready,
    _ensure_presenton_running,
    _stop_presenton,
    _render_slides_sync,
    _render_docs_sync,
    render_slides,
    render_docs,
)


# ---------------------------------------------------------------------------
# _presenton_ready
# ---------------------------------------------------------------------------

class TestPrestonReady:
    @patch("urllib.request.urlopen")
    def test_ready_returns_true(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        assert _presenton_ready() is True

    @patch("urllib.request.urlopen")
    def test_not_ready_returns_false(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        assert _presenton_ready() is False


# ---------------------------------------------------------------------------
# _stop_presenton
# ---------------------------------------------------------------------------

class TestStopPresenton:
    @patch("mcp_servers.document_render.server.subprocess.run")
    def test_stop_running_container(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout="presenton\nother\n"),
            MagicMock(),
        ]
        _stop_presenton()
        assert mock_run.call_count == 2

    @patch("mcp_servers.document_render.server.subprocess.run")
    def test_stop_not_running(self, mock_run):
        mock_run.return_value = MagicMock(stdout="other\n")
        _stop_presenton()
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# render_slides — now fire-and-forget, returns pending
# ---------------------------------------------------------------------------

class TestRenderSlides:
    def test_empty_content(self):
        result = render_slides("")
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()

    def test_whitespace_only_content(self):
        result = render_slides("   \n  ")
        assert result["status"] == "error"

    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_presenton_not_available(self, mock_ensure, mock_config):
        mock_ensure.side_effect = RuntimeError("Container not found")
        result = render_slides("some content")
        assert result["status"] == "error"
        assert "Container" in result["error"]

    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_returns_pending_immediately(self, mock_ensure, mock_config):
        """render_slides should return pending without waiting for render."""
        result = render_slides("My presentation content", filename="test_pending")
        assert result["status"] == "pending"
        assert "pid" in result
        assert result["output_path"] == "/tmp/test_pending.pdf"
        assert "done_path" in result
        assert result["done_path"].endswith(".done")
        assert "heartbeat_register_monitor" in result["message"]

    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_pending_pid_is_valid(self, mock_ensure, mock_config):
        """Returned PID should be the current process (thread-based background)."""
        result = render_slides("content", filename="test_pid")
        assert result["status"] == "pending"
        assert result["pid"] == os.getpid()


# ---------------------------------------------------------------------------
# render_docs — now fire-and-forget, returns pending
# ---------------------------------------------------------------------------

class TestRenderDocs:
    def test_empty_content(self):
        result = render_docs("")
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()

    @patch("mcp_servers.document_render.server.shutil.which")
    def test_pandoc_not_installed(self, mock_which):
        mock_which.return_value = None
        result = render_docs("# Hello")
        assert result["status"] == "error"
        assert "pandoc" in result["error"].lower()

    @patch("mcp_servers.document_render.server.shutil.which")
    def test_returns_pending_immediately(self, mock_which):
        """render_docs should return pending without waiting for pandoc."""
        mock_which.return_value = "/usr/bin/pandoc"
        result = render_docs("# Hello World", filename="test_doc_pending")
        assert result["status"] == "pending"
        assert "pid" in result
        assert result["output_path"] == "/tmp/test_doc_pending.docx"
        assert "done_path" in result
        assert "heartbeat_register_monitor" in result["message"]

    @patch("mcp_servers.document_render.server.shutil.which")
    def test_custom_format_in_output_path(self, mock_which):
        mock_which.return_value = "/usr/bin/pandoc"
        result = render_docs("# Hello", filename="test_html", format="html")
        assert result["status"] == "pending"
        assert result["output_path"].endswith(".html")


# ---------------------------------------------------------------------------
# _render_slides_sync — tests the actual sync rendering logic
# ---------------------------------------------------------------------------

class TestRenderSlicesSync:
    @patch("mcp_servers.document_render.server._stop_presenton")
    @patch("urllib.request.urlopen")
    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_successful_sync_render(self, mock_ensure, mock_config, mock_urlopen, mock_stop):
        api_resp = MagicMock()
        api_resp.read.return_value = json.dumps({"path": "/output/slides.pdf"}).encode()
        api_resp.__enter__ = MagicMock(return_value=api_resp)
        api_resp.__exit__ = MagicMock(return_value=False)

        pdf_resp = MagicMock()
        pdf_resp.read.return_value = b"%PDF-1.4 fake pdf content"
        pdf_resp.__enter__ = MagicMock(return_value=pdf_resp)
        pdf_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [api_resp, pdf_resp]

        output_path = "/tmp/test_slides_sync.pdf"
        result = _render_slides_sync("My content", output_path)
        assert result["file_path"] == output_path
        assert result["file_size"] > 0

        if os.path.exists(output_path):
            os.unlink(output_path)


# ---------------------------------------------------------------------------
# _render_docs_sync — tests the actual pandoc logic
# ---------------------------------------------------------------------------

class TestRenderDocsSync:
    @patch("mcp_servers.document_render.server.subprocess.run")
    def test_successful_sync_render(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output_path = "/tmp/test_docs_sync.docx"
        with open(output_path, "wb") as f:
            f.write(b"fake docx content")
        try:
            result = _render_docs_sync("# Hello", output_path, "docx")
            assert result["file_path"] == output_path
            assert result["file_size"] > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("mcp_servers.document_render.server.subprocess.run")
    def test_pandoc_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="pandoc error")
        result = _render_docs_sync("# Hello", "/tmp/fail.docx", "docx")
        assert "error" in result
        assert "pandoc failed" in result["error"]


# ---------------------------------------------------------------------------
# Background task writes done_path sentinel
# ---------------------------------------------------------------------------

class TestBackgroundSentinel:
    @patch("mcp_servers.document_render.server.shutil.which")
    @patch("mcp_servers.document_render.server.subprocess.run")
    def test_done_file_written_on_success(self, mock_run, mock_which):
        """Background thread should write .done sentinel file when complete."""
        mock_which.return_value = "/usr/bin/pandoc"
        result = render_docs("# Test", filename="test_sentinel")
        output_path = result["output_path"]
        done_path = result["done_path"]

        # Create fake output file for pandoc
        with open(output_path, "wb") as f:
            f.write(b"fake docx")
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # Wait up to 3s for background thread to complete
        for _ in range(30):
            if os.path.exists(done_path):
                break
            time.sleep(0.1)

        # Cleanup regardless
        for path in [output_path, done_path]:
            try:
                os.unlink(path)
            except OSError:
                pass
