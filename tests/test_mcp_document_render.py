"""Tests for mcp_servers/document_render MCP Server."""

import json
import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from mcp_servers.document_render.server import (
    _presenton_ready,
    _ensure_presenton_running,
    _stop_presenton,
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
        # First call: docker ps (container is running)
        mock_run.side_effect = [
            MagicMock(stdout="presenton\nother\n"),
            MagicMock(),  # docker stop
        ]
        _stop_presenton()
        assert mock_run.call_count == 2

    @patch("mcp_servers.document_render.server.subprocess.run")
    def test_stop_not_running(self, mock_run):
        mock_run.return_value = MagicMock(stdout="other\n")
        _stop_presenton()
        assert mock_run.call_count == 1  # only docker ps, no stop


# ---------------------------------------------------------------------------
# render_slides
# ---------------------------------------------------------------------------

class TestRenderSlides:
    def test_empty_content(self):
        result = render_slides("")
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()

    def test_whitespace_only_content(self):
        result = render_slides("   \n  ")
        assert result["status"] == "error"

    @patch("mcp_servers.document_render.server._stop_presenton")
    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_presenton_not_available(self, mock_ensure, mock_config, mock_stop):
        mock_ensure.side_effect = RuntimeError("Container not found")
        result = render_slides("some content")
        assert result["status"] == "error"
        assert "Container" in result["error"]

    @patch("mcp_servers.document_render.server._stop_presenton")
    @patch("urllib.request.urlopen")
    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_successful_render(self, mock_ensure, mock_config, mock_urlopen, mock_stop):
        # First urlopen: API generate call
        api_resp = MagicMock()
        api_resp.read.return_value = json.dumps({"path": "/output/slides.pdf"}).encode()
        api_resp.__enter__ = MagicMock(return_value=api_resp)
        api_resp.__exit__ = MagicMock(return_value=False)

        # Second urlopen: download PDF
        pdf_resp = MagicMock()
        pdf_resp.read.return_value = b"%PDF-1.4 fake pdf content"
        pdf_resp.__enter__ = MagicMock(return_value=pdf_resp)
        pdf_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [api_resp, pdf_resp]

        result = render_slides("My presentation content", filename="test_slides")
        assert result["status"] == "success"
        assert result["file_path"] == "/tmp/test_slides.pdf"
        assert result["file_size"] > 0

        # Cleanup
        if os.path.exists("/tmp/test_slides.pdf"):
            os.unlink("/tmp/test_slides.pdf")


# ---------------------------------------------------------------------------
# render_docs
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

    @patch("mcp_servers.document_render.server.subprocess.run")
    @patch("mcp_servers.document_render.server.shutil.which")
    def test_pandoc_failure(self, mock_which, mock_run):
        mock_which.return_value = "/usr/bin/pandoc"
        mock_run.return_value = MagicMock(returncode=1, stderr="pandoc error")
        result = render_docs("# Hello", filename="test_fail")
        assert result["status"] == "error"
        assert "pandoc failed" in result["error"]

    @patch("mcp_servers.document_render.server.subprocess.run")
    @patch("mcp_servers.document_render.server.shutil.which")
    def test_successful_render(self, mock_which, mock_run):
        mock_which.return_value = "/usr/bin/pandoc"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        output_path = "/tmp/test_doc_render.docx"
        # Create fake output file that pandoc would produce
        with open(output_path, "wb") as f:
            f.write(b"fake docx content")

        try:
            result = render_docs("# Hello World", filename="test_doc_render")
            assert result["status"] == "success"
            assert result["file_path"] == output_path
            assert result["file_size"] > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("mcp_servers.document_render.server.subprocess.run")
    @patch("mcp_servers.document_render.server.shutil.which")
    def test_custom_format(self, mock_which, mock_run):
        mock_which.return_value = "/usr/bin/pandoc"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        output_path = "/tmp/test_html_render.html"
        with open(output_path, "w") as f:
            f.write("<html>test</html>")

        try:
            result = render_docs("# Hello", filename="test_html_render", format="html")
            assert result["status"] == "success"
            assert result["file_path"].endswith(".html")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
