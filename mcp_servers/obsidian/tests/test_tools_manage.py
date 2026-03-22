"""
测试 tools/manage.py — move, delete, frontmatter, tags
"""

import pytest

from mcp_servers.obsidian.core.cas import compute_file_hash
from mcp_servers.obsidian.tools.manage import register


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


class TestMoveNote:
    """obsidian_move_note"""

    @pytest.mark.asyncio
    async def test_move_note(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_move_note"](
            paths["plain"], "archive/plain-moved.md",
        )
        assert result["status"] == "success"
        assert result["data"]["new_path"] == "archive/plain-moved.md"
        # 源文件不存在了
        src = vault.resolve_path(paths["plain"])
        assert not src.exists()
        # 目标文件存在
        dst = vault.resolve_path("archive/plain-moved.md")
        assert dst.exists()

    @pytest.mark.asyncio
    async def test_move_source_not_found(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_move_note"]("ghost.md", "new.md")
        assert result["error_code"] == "not_found"

    @pytest.mark.asyncio
    async def test_move_dest_already_exists(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_move_note"](
            paths["plain"], paths["design_doc"],
        )
        assert result["error_code"] == "already_exists"

    @pytest.mark.asyncio
    async def test_move_to_blocked_dir(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_move_note"](
            paths["plain"], ".obsidian/evil.md",
        )
        assert result["error_code"] == "permission_denied"

    @pytest.mark.asyncio
    async def test_move_creates_parent_dirs(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_move_note"](
            paths["plain"], "new/nested/dir/note.md",
        )
        assert result["status"] == "success"
        dst = vault.resolve_path("new/nested/dir/note.md")
        assert dst.exists()


class TestDeleteNote:
    """obsidian_delete_note"""

    @pytest.mark.asyncio
    async def test_soft_delete(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_delete_note"](paths["plain"])
        assert result["status"] == "success"
        assert ".trash/" in result["data"]["deleted_to"]
        # 原文件不存在
        src = vault.resolve_path(paths["plain"])
        assert not src.exists()
        # .trash 目录有文件
        trash = vault.base_dir / ".trash"
        assert any(trash.iterdir())

    @pytest.mark.asyncio
    async def test_permanent_delete(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_delete_note"](
            paths["plain"], permanent=True,
        )
        assert result["status"] == "success"
        assert "permanently" in result["data"]["deleted_to"]
        src = vault.resolve_path(paths["plain"])
        assert not src.exists()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        result = await tools["obsidian_delete_note"]("ghost.md")
        assert result["error_code"] == "not_found"

    @pytest.mark.asyncio
    async def test_delete_with_cas_ok(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["plain"])
        h = compute_file_hash(abs_path)
        result = await tools["obsidian_delete_note"](paths["plain"], cas_hash=h)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_with_cas_conflict(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_delete_note"](
            paths["plain"], cas_hash="wrong",
        )
        assert result["error_code"] == "conflict"

    @pytest.mark.asyncio
    async def test_delete_blocked_path(self, populated_vault):
        vault, _ = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_delete_note"](".git/config")
        assert result["error_code"] == "permission_denied"


class TestGetFrontmatter:
    """obsidian_get_frontmatter"""

    @pytest.mark.asyncio
    async def test_get_frontmatter(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_get_frontmatter"](paths["design_doc"])
        assert result["status"] == "success"
        assert result["data"]["frontmatter"]["tags"] == ["project", "design"]

    @pytest.mark.asyncio
    async def test_get_frontmatter_no_fm(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_get_frontmatter"](paths["plain"])
        assert result["status"] == "success"
        assert result["data"]["frontmatter"] == {}

    @pytest.mark.asyncio
    async def test_get_frontmatter_not_found(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        result = await tools["obsidian_get_frontmatter"]("ghost.md")
        assert result["error_code"] == "not_found"


class TestUpdateFrontmatter:
    """obsidian_update_frontmatter"""

    @pytest.mark.asyncio
    async def test_update_frontmatter(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_update_frontmatter"](
            paths["design_doc"], {"status": "completed"},
        )
        assert result["status"] == "success"
        # 验证
        abs_path = vault.resolve_path(paths["design_doc"])
        content = abs_path.read_text(encoding="utf-8")
        assert "status: completed" in content
        assert "tags:" in content  # 原有字段保留

    @pytest.mark.asyncio
    async def test_update_with_cas_ok(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        h = compute_file_hash(abs_path)
        result = await tools["obsidian_update_frontmatter"](
            paths["design_doc"], {"new_field": "yes"}, cas_hash=h,
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_update_with_cas_conflict(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_update_frontmatter"](
            paths["design_doc"], {"x": "y"}, cas_hash="wrong",
        )
        assert result["error_code"] == "conflict"


class TestManageTags:
    """obsidian_manage_tags"""

    @pytest.mark.asyncio
    async def test_add_tags(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_manage_tags"](
            paths["design_doc"], add=["new-tag", "another"],
        )
        assert result["status"] == "success"
        assert "new-tag" in result["data"]["tags"]
        assert "another" in result["data"]["tags"]
        # 原有 tags 保留
        assert "project" in result["data"]["tags"]

    @pytest.mark.asyncio
    async def test_remove_tags(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_manage_tags"](
            paths["design_doc"], remove=["project"],
        )
        assert result["status"] == "success"
        assert "project" not in result["data"]["tags"]
        assert "design" in result["data"]["tags"]

    @pytest.mark.asyncio
    async def test_add_and_remove(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_manage_tags"](
            paths["design_doc"],
            add=["new-tag"],
            remove=["design"],
        )
        assert result["status"] == "success"
        assert "new-tag" in result["data"]["tags"]
        assert "design" not in result["data"]["tags"]

    @pytest.mark.asyncio
    async def test_add_with_hash_prefix(self, populated_vault):
        """tags 前有 # 号应该被去掉。"""
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_manage_tags"](
            paths["design_doc"], add=["#hashed-tag"],
        )
        assert "hashed-tag" in result["data"]["tags"]

    @pytest.mark.asyncio
    async def test_add_duplicate_tag(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_manage_tags"](
            paths["design_doc"], add=["project"],  # already exists
        )
        assert result["data"]["tags"].count("project") == 1

    @pytest.mark.asyncio
    async def test_manage_tags_on_no_fm_note(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_manage_tags"](
            paths["plain"], add=["new"],
        )
        assert result["status"] == "success"
        assert "new" in result["data"]["tags"]
        # 验证 frontmatter 已添加
        abs_path = vault.resolve_path(paths["plain"])
        content = abs_path.read_text(encoding="utf-8")
        assert "---" in content
