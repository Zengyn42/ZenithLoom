"""Tests for mcp_servers/ollama_basics MCP Server."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mcp_servers.ollama_basics.server import (
    read_file,
    write_file,
    list_directory,
    run_command,
    search_files,
    grep_content,
    web_fetch,
    _strip_html,
    _fetch_url,
    _run_cmd,
    MAX_FILE_CHARS,
    MAX_TIMEOUT,
)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await read_file(str(f))
        assert result["ok"] is True
        assert "line1" in result["content"]
        assert result["line_count"] == 3
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        result = await read_file("/nonexistent/path/file.txt")
        assert result["ok"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_read_truncation(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * (MAX_FILE_CHARS + 100))
        result = await read_file(str(f))
        assert result["ok"] is True
        assert result["truncated"] is True
        assert len(result["content"]) == MAX_FILE_CHARS

    @pytest.mark.asyncio
    async def test_read_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = await read_file(str(f))
        assert result["ok"] is True
        assert result["content"] == ""
        assert result["line_count"] == 0

    @pytest.mark.asyncio
    async def test_read_directory_returns_error(self, tmp_path):
        result = await read_file(str(tmp_path))
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        f = tmp_path / "output.txt"
        result = await write_file(str(f), "hello world")
        assert result["ok"] is True
        assert result["bytes_written"] == len("hello world".encode("utf-8"))
        assert f.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "file.txt"
        result = await write_file(str(f), "nested")
        assert result["ok"] is True
        assert f.read_text() == "nested"

    @pytest.mark.asyncio
    async def test_write_overwrite_existing(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content")
        result = await write_file(str(f), "new content")
        assert result["ok"] is True
        assert f.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_write_unicode(self, tmp_path):
        f = tmp_path / "unicode.txt"
        content = "你好世界 🌍"
        result = await write_file(str(f), content)
        assert result["ok"] is True
        assert f.read_text() == content


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

class TestListDirectory:
    @pytest.mark.asyncio
    async def test_list_with_files_and_dirs(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "subdir").mkdir()
        result = await list_directory(str(tmp_path))
        assert result["ok"] is True
        assert result["count"] == 2
        names = {e["name"] for e in result["entries"]}
        assert "a.txt" in names
        assert "subdir" in names
        types = {e["name"]: e["type"] for e in result["entries"]}
        assert types["a.txt"] == "file"
        assert types["subdir"] == "dir"

    @pytest.mark.asyncio
    async def test_list_nonexistent_dir(self):
        result = await list_directory("/nonexistent/dir")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_list_empty_dir(self, tmp_path):
        result = await list_directory(str(tmp_path))
        assert result["ok"] is True
        assert result["count"] == 0


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------

class TestRunCommand:
    @pytest.mark.asyncio
    async def test_simple_command(self):
        result = await run_command("echo hello")
        assert result["ok"] is True
        assert "hello" in result["stdout"]
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_command_failure(self):
        result = await run_command("false")
        assert result["ok"] is True  # command ran, just returned nonzero
        assert result["exit_code"] != 0

    @pytest.mark.asyncio
    async def test_timeout_capped(self):
        # timeout should be capped at MAX_TIMEOUT
        result = await run_command("echo ok", timeout=999)
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_timeout_min_1(self):
        result = await run_command("echo ok", timeout=-5)
        assert result["ok"] is True

    def test_run_cmd_timeout(self):
        result = _run_cmd("sleep 10", timeout=1)
        assert result["ok"] is False
        assert "timed out" in result["stderr"]


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------

class TestSearchFiles:
    @pytest.mark.asyncio
    async def test_search_glob(self, tmp_path):
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.txt").write_text("hello")
        (tmp_path / "c.py").write_text("pass")
        result = await search_files("*.py", str(tmp_path))
        assert result["ok"] is True
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_search_max_results(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file_{i}.txt").write_text("x")
        result = await search_files("*.txt", str(tmp_path), max_results=3)
        assert result["ok"] is True
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_search_nonexistent_dir(self):
        result = await search_files("*.py", "/nonexistent")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# grep_content
# ---------------------------------------------------------------------------

class TestGrepContent:
    @pytest.mark.asyncio
    async def test_grep_finds_matches(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello world\nfoo bar\nhello again")
        result = await grep_content("hello", str(tmp_path), file_glob="*.txt")
        assert result["ok"] is True
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, tmp_path):
        (tmp_path / "a.txt").write_text("nothing here")
        result = await grep_content("zzz_nonexistent", str(tmp_path))
        assert result["ok"] is True
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_grep_invalid_regex(self, tmp_path):
        result = await grep_content("[invalid", str(tmp_path))
        assert result["ok"] is False
        assert "Invalid regex" in result["error"]

    @pytest.mark.asyncio
    async def test_grep_max_results(self, tmp_path):
        (tmp_path / "a.txt").write_text("\n".join(["match"] * 50))
        result = await grep_content("match", str(tmp_path), max_results=5)
        assert result["ok"] is True
        assert result["count"] == 5

    @pytest.mark.asyncio
    async def test_grep_line_truncation(self, tmp_path):
        (tmp_path / "a.txt").write_text("x" * 1000)
        result = await grep_content("x+", str(tmp_path))
        assert result["ok"] is True
        # Line should be truncated to 500 chars
        if result["count"] > 0:
            assert len(result["results"][0]["line"]) <= 500


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------

class TestWebFetch:
    @pytest.mark.asyncio
    async def test_invalid_scheme(self):
        result = await web_fetch("ftp://example.com")
        assert result["ok"] is False
        assert "http" in result["error"].lower()

    def test_fetch_url_invalid_scheme(self):
        result = _fetch_url("ftp://example.com")
        assert result["ok"] is False

    def test_strip_html_basic(self):
        assert "hello" in _strip_html("<p>hello</p>")
        assert "<p>" not in _strip_html("<p>hello</p>")

    def test_strip_html_script_removal(self):
        html = "<script>alert('x')</script><p>safe</p>"
        result = _strip_html(html)
        assert "alert" not in result
        assert "safe" in result

    def test_strip_html_style_removal(self):
        html = "<style>.x{color:red}</style><p>text</p>"
        result = _strip_html(html)
        assert "color" not in result
        assert "text" in result
