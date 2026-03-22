"""
测试 tools/read.py — obsidian_read_note, obsidian_list_files
"""

import pytest

from mcp_servers.obsidian.core.cas import compute_file_hash
from mcp_servers.obsidian.tools.read import register


def _get_tools(vault):
    """注册 tools 并返回 tool 函数字典。"""
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


class TestReadNote:
    """obsidian_read_note"""

    @pytest.mark.asyncio
    async def test_read_existing_note(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_read_note"](paths["design_doc"])
        assert result["status"] == "success"
        assert "Design Doc" in result["data"]["content"]
        assert result["data"]["frontmatter"]["tags"] == ["project", "design"]
        assert len(result["data"]["cas_hash"]) == 64
        assert result["data"]["mtime_ms"] > 0

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_read_note"]("nonexistent.md")
        assert result["status"] == "error"
        assert result["error_code"] == "not_found"

    @pytest.mark.asyncio
    async def test_read_note_no_frontmatter(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_read_note"](paths["plain"])
        assert result["status"] == "success"
        assert result["data"]["frontmatter"] == {}

    @pytest.mark.asyncio
    async def test_read_blocked_path(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_read_note"](".obsidian/config.json")
        assert result["status"] == "error"
        assert result["error_code"] == "permission_denied"

    @pytest.mark.asyncio
    async def test_read_traversal_attack(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_read_note"]("../../etc/passwd")
        assert result["status"] == "error"
        assert result["error_code"] == "path_traversal"

    @pytest.mark.asyncio
    async def test_read_directory_not_file(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_read_note"]("projects")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_cas_hash_consistency(self, populated_vault):
        """read 返回的 hash 应该和 compute_file_hash 一致。"""
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_read_note"](paths["design_doc"])
        abs_path = vault.resolve_path(paths["design_doc"])
        assert result["data"]["cas_hash"] == compute_file_hash(abs_path)


class TestListFiles:
    """obsidian_list_files"""

    @pytest.mark.asyncio
    async def test_list_root(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_list_files"]()
        assert result["status"] == "success"
        # 根目录只有 plain.md（子目录文件不在非递归列表中）
        assert result["data"]["count"] >= 1

    @pytest.mark.asyncio
    async def test_list_recursive(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_list_files"](recursive=True)
        assert result["status"] == "success"
        assert result["data"]["count"] == 4  # 4 个笔记文件

    @pytest.mark.asyncio
    async def test_list_subdir(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_list_files"](directory="projects")
        assert result["status"] == "success"
        paths = [f["path"] for f in result["data"]["files"]]
        assert any("design-doc.md" in p for p in paths)

    @pytest.mark.asyncio
    async def test_list_nonexistent_dir(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_list_files"](directory="nonexistent")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_list_blocked_dir(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_list_files"](directory=".obsidian")
        assert result["status"] == "error"
        assert result["error_code"] == "permission_denied"

    @pytest.mark.asyncio
    async def test_file_info_fields(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_list_files"](directory="projects")
        assert result["status"] == "success"
        for f in result["data"]["files"]:
            assert "path" in f
            assert "size_bytes" in f
            assert "mtime_ms" in f
            assert f["size_bytes"] > 0
