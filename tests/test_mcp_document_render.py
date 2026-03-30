"""Tests for mcp_servers/document_render MCP Server."""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from mcp_servers.document_render.server import (
    _presenton_ready,
    _stop_presenton,
    _render_slides_sync,
    _render_docs_sync,
    _launch_worker,
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
# _launch_worker — subprocess.Popen based, returns real pid
# ---------------------------------------------------------------------------

class TestLaunchWorker:
    @patch("mcp_servers.document_render.server.subprocess.Popen")
    def test_returns_real_pid(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=99999)
        pid, done_path = _launch_worker("docs", "# Hello", "/tmp/out.docx")
        assert pid == 99999
        assert done_path.endswith(".json")
        assert done_path.startswith("/tmp/")

    @patch("mcp_servers.document_render.server.subprocess.Popen")
    def test_worker_type_in_cmd(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=1)
        _launch_worker("slides", "content", "/tmp/slides.pdf")
        cmd = mock_popen.call_args[0][0]
        assert sys.executable in cmd
        assert "--type" in cmd
        assert "slides" in cmd
        assert "--output" in cmd
        assert "/tmp/slides.pdf" in cmd

    @patch("mcp_servers.document_render.server.subprocess.Popen")
    def test_both_types_work(self, mock_popen):
        mock_popen.return_value = MagicMock(pid=1)
        for t in ("slides", "docs"):
            _launch_worker(t, "c", f"/tmp/out.{t}")
        assert mock_popen.call_count == 2


# ---------------------------------------------------------------------------
# render_slides
# ---------------------------------------------------------------------------

class TestRenderSlides:
    def test_empty_content(self):
        result = render_slides("", agent_id="jei")
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()

    def test_empty_agent_id(self):
        result = render_slides("content", agent_id="")
        assert result["status"] == "error"
        assert "agent_id" in result["error"]

    def test_whitespace_content(self):
        result = render_slides("   \n", agent_id="jei")
        assert result["status"] == "error"

    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_presenton_unavailable(self, mock_ensure, mock_config):
        mock_ensure.side_effect = RuntimeError("Container not found")
        result = render_slides("content", agent_id="jei")
        assert result["status"] == "error"
        assert "Container" in result["error"]

    @patch("mcp_servers.document_render.server._launch_worker")
    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_returns_pending_with_pid_and_agent_id(self, mock_ensure, mock_config, mock_launch):
        mock_launch.return_value = (54321, "/tmp/render_done_123.json")
        result = render_slides("My content", agent_id="jei", filename="test_slides")

        assert result["status"] == "pending"
        assert result["pid"] == 54321
        assert result["agent_id"] == "jei"
        assert result["output_path"] == "/tmp/test_slides.pdf"
        assert result["done_path"] == "/tmp/render_done_123.json"
        assert "heartbeat_register_monitor" in result["message"]
        assert "54321" in result["message"]
        assert "jei" in result["message"]

    @patch("mcp_servers.document_render.server._launch_worker")
    @patch("mcp_servers.document_render.server._ensure_presenton_configured")
    @patch("mcp_servers.document_render.server._ensure_presenton_running")
    def test_launches_slides_worker(self, mock_ensure, mock_config, mock_launch):
        mock_launch.return_value = (1, "/tmp/done.json")
        render_slides("content", agent_id="jei")
        assert mock_launch.call_args[0][0] == "slides"


# ---------------------------------------------------------------------------
# render_docs
# ---------------------------------------------------------------------------

class TestRenderDocs:
    def test_empty_content(self):
        result = render_docs("", agent_id="jei")
        assert result["status"] == "error"

    def test_empty_agent_id(self):
        result = render_docs("# Hello", agent_id="")
        assert result["status"] == "error"
        assert "agent_id" in result["error"]

    @patch("mcp_servers.document_render.server.shutil.which")
    def test_pandoc_not_installed(self, mock_which):
        mock_which.return_value = None
        result = render_docs("# Hello", agent_id="jei")
        assert result["status"] == "error"
        assert "pandoc" in result["error"].lower()

    @patch("mcp_servers.document_render.server._launch_worker")
    @patch("mcp_servers.document_render.server.shutil.which")
    def test_returns_pending_with_pid_and_agent_id(self, mock_which, mock_launch):
        mock_which.return_value = "/usr/bin/pandoc"
        mock_launch.return_value = (67890, "/tmp/render_done_456.json")
        result = render_docs("# Hello", agent_id="hani", filename="test_doc")

        assert result["status"] == "pending"
        assert result["pid"] == 67890
        assert result["agent_id"] == "hani"
        assert result["output_path"] == "/tmp/test_doc.docx"
        assert result["done_path"] == "/tmp/render_done_456.json"
        assert "heartbeat_register_monitor" in result["message"]

    @patch("mcp_servers.document_render.server._launch_worker")
    @patch("mcp_servers.document_render.server.shutil.which")
    def test_custom_format(self, mock_which, mock_launch):
        mock_which.return_value = "/usr/bin/pandoc"
        mock_launch.return_value = (1, "/tmp/done.json")
        result = render_docs("# Hello", agent_id="jei", filename="test_html", format="html")
        assert result["output_path"].endswith(".html")

    @patch("mcp_servers.document_render.server._launch_worker")
    @patch("mcp_servers.document_render.server.shutil.which")
    def test_launches_docs_worker(self, mock_which, mock_launch):
        mock_which.return_value = "/usr/bin/pandoc"
        mock_launch.return_value = (1, "/tmp/done.json")
        render_docs("# Hello", agent_id="jei")
        assert mock_launch.call_args[0][0] == "docs"


# ---------------------------------------------------------------------------
# _render_slides_sync (sync core, used by worker.py)
# ---------------------------------------------------------------------------

class TestRenderSlidesSync:
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
        pdf_resp.read.return_value = b"%PDF-1.4 fake pdf"
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
# _render_docs_sync (sync core, used by worker.py)
# ---------------------------------------------------------------------------

class TestRenderDocsSync:
    @patch("mcp_servers.document_render.server.subprocess.run")
    def test_successful_sync_render(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        output_path = "/tmp/test_docs_sync.docx"
        with open(output_path, "wb") as f:
            f.write(b"fake docx")
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
