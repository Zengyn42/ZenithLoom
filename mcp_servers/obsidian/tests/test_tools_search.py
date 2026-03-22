"""
测试 tools/search.py — obsidian_search_files, obsidian_get_links
"""

import pytest

from mcp_servers.obsidian.tools.search import register


def _get_tools(vault):
    class FakeMCP:
        def __init__(self):
            self.tools = {}
        def tool(self):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator
    fake = FakeMCP()
    register(fake, vault)
    return fake.tools


class TestSearchFiles:
    """obsidian_search_files"""

    @pytest.mark.asyncio
    async def test_content_search(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_search_files"]("background")
        assert result["status"] == "success"
        assert result["data"]["count"] >= 1
        paths = [r["path"] for r in result["data"]["results"]]
        assert any("design-doc" in p for p in paths)

    @pytest.mark.asyncio
    async def test_filename_search(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_search_files"](
            "design", search_type="filename",
        )
        assert result["status"] == "success"
        assert result["data"]["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_in_directory(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_search_files"](
            "meeting", directory="daily",
        )
        assert result["status"] == "success"
        assert result["data"]["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_search_files"]("xyznonexistent123")
        assert result["status"] == "success"
        assert result["data"]["count"] == 0

    @pytest.mark.asyncio
    async def test_search_empty_query(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_search_files"]("")
        assert result["status"] == "error"
        assert result["error_code"] == "validation_error"

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_search_files"]("BACKGROUND")
        # 应该能找到（content 里是小写的 background）
        assert result["data"]["count"] >= 1

    @pytest.mark.asyncio
    async def test_search_max_results(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_search_files"](
            "the", max_results=1,
        )
        assert result["data"]["count"] <= 1

    @pytest.mark.asyncio
    async def test_content_search_returns_matches(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_search_files"]("background")
        for r in result["data"]["results"]:
            if r["match_type"] == "content":
                assert "matches" in r
                assert len(r["matches"]) > 0
                assert "line" in r["matches"][0]
                assert "text" in r["matches"][0]


class TestGetLinks:
    """obsidian_get_links"""

    @pytest.mark.asyncio
    async def test_outgoing_links(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_get_links"](paths["design_doc"])
        assert result["status"] == "success"
        outgoing = result["data"]["outgoing"]
        assert "meeting-notes" in outgoing
        assert "research/paper-a" in outgoing

    @pytest.mark.asyncio
    async def test_incoming_links(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_get_links"](paths["design_doc"])
        incoming = result["data"]["incoming"]
        # meeting-notes.md 和 old-note.md 都链接到 design-doc
        assert len(incoming) >= 2

    @pytest.mark.asyncio
    async def test_note_with_no_links(self, tmp_vault):
        (tmp_vault.base_dir / "lonely.md").write_text("No links here.", encoding="utf-8")
        tools = _get_tools(tmp_vault)
        result = await tools["obsidian_get_links"]("lonely.md")
        assert result["status"] == "success"
        assert result["data"]["outgoing"] == []

    @pytest.mark.asyncio
    async def test_links_nonexistent_note(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        result = await tools["obsidian_get_links"]("ghost.md")
        assert result["error_code"] == "not_found"

    @pytest.mark.asyncio
    async def test_bidirectional_consistency(self, populated_vault):
        """如果 A 链接到 B，那么 B 的 incoming 应包含 A。"""
        vault, paths = populated_vault
        tools = _get_tools(vault)

        # design-doc 链接到 meeting-notes
        result_dd = await tools["obsidian_get_links"](paths["design_doc"])
        assert "meeting-notes" in result_dd["data"]["outgoing"]

        # meeting-notes 的 incoming 应包含 design-doc
        result_mn = await tools["obsidian_get_links"](paths["meeting_notes"])
        incoming_paths = result_mn["data"]["incoming"]
        assert any("design-doc" in p for p in incoming_paths)
