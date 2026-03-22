"""
测试 tools/write.py — obsidian_write_note, obsidian_patch_note
"""

import pytest

from mcp_servers.obsidian.core.cas import compute_file_hash, compute_hash
from mcp_servers.obsidian.tools.write import register


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


class TestWriteNote:
    """obsidian_write_note"""

    @pytest.mark.asyncio
    async def test_create_new_note(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        content = "---\ntitle: New\n---\n\n# New Note\n\nContent."
        result = await tools["obsidian_write_note"]("new-note.md", content)
        assert result["status"] == "success"
        assert len(result["data"]["cas_hash"]) == 64

        # 验证文件已写入
        abs_path = tmp_vault.resolve_path("new-note.md")
        assert abs_path.exists()
        assert abs_path.read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_create_with_nested_dir(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        result = await tools["obsidian_write_note"](
            "a/b/c/deep.md", "# Deep"
        )
        assert result["status"] == "success"
        abs_path = tmp_vault.resolve_path("a/b/c/deep.md")
        assert abs_path.exists()

    @pytest.mark.asyncio
    async def test_create_already_exists(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_write_note"](
            paths["design_doc"], "# New Content"
        )
        assert result["status"] == "error"
        assert result["error_code"] == "already_exists"

    @pytest.mark.asyncio
    async def test_overwrite_with_correct_cas(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        current_hash = compute_file_hash(abs_path)

        new_content = "# Rewritten\n\nAll new."
        result = await tools["obsidian_write_note"](
            paths["design_doc"], new_content, cas_hash=current_hash,
        )
        assert result["status"] == "success"
        assert abs_path.read_text(encoding="utf-8") == new_content

    @pytest.mark.asyncio
    async def test_overwrite_with_wrong_cas(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_write_note"](
            paths["design_doc"], "# Bad", cas_hash="wrong_hash",
        )
        assert result["status"] == "error"
        assert result["error_code"] == "conflict"

    @pytest.mark.asyncio
    async def test_write_blocked_path(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        result = await tools["obsidian_write_note"](
            ".obsidian/evil.md", "# Evil"
        )
        assert result["status"] == "error"
        assert result["error_code"] == "permission_denied"

    @pytest.mark.asyncio
    async def test_write_traversal_attack(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        result = await tools["obsidian_write_note"](
            "../../etc/evil.md", "# Evil"
        )
        assert result["status"] == "error"
        assert result["error_code"] == "path_traversal"

    @pytest.mark.asyncio
    async def test_hash_returned_matches_content(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        content = "# Test\n\nContent."
        result = await tools["obsidian_write_note"]("test.md", content)
        assert result["data"]["cas_hash"] == compute_hash(content)


class TestPatchNote:
    """obsidian_patch_note"""

    @pytest.mark.asyncio
    async def test_replace_section(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [{"action": "replace_section", "target_heading": "## 背景", "content": "\nNew background content.\n"}],
        )
        assert result["status"] == "success"
        assert "## 背景" in result["data"]["sections_affected"]

        new_content = abs_path.read_text(encoding="utf-8")
        assert "New background content." in new_content
        # 其他 section 不受影响
        assert "## 方案" in new_content

    @pytest.mark.asyncio
    async def test_append_to_section(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [{"action": "append_to_section", "target_heading": "## 时间线", "content": "- Phase 3: TBD"}],
        )
        assert result["status"] == "success"
        new_content = abs_path.read_text(encoding="utf-8")
        assert "Phase 3: TBD" in new_content
        assert "Phase 1: 1 week" in new_content  # 原内容保留

    @pytest.mark.asyncio
    async def test_delete_section(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [{"action": "delete_section", "target_heading": "## 时间线"}],
        )
        assert result["status"] == "success"
        new_content = abs_path.read_text(encoding="utf-8")
        assert "## 时间线" not in new_content
        assert "## 背景" in new_content  # 其他 section 不受影响

    @pytest.mark.asyncio
    async def test_update_frontmatter_via_patch(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [{"action": "update_frontmatter", "content": {"status": "done", "priority": 1}}],
        )
        assert result["status"] == "success"
        new_content = abs_path.read_text(encoding="utf-8")
        assert "status:" in new_content

    @pytest.mark.asyncio
    async def test_cas_conflict(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            "wrong_hash_value",
            [{"action": "replace_section", "target_heading": "## 背景", "content": "new"}],
        )
        assert result["status"] == "error"
        assert result["error_code"] == "conflict"

    @pytest.mark.asyncio
    async def test_section_not_found(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [{"action": "replace_section", "target_heading": "## Nonexistent", "content": "x"}],
        )
        assert result["status"] == "error"
        assert result["error_code"] == "validation_error"

    @pytest.mark.asyncio
    async def test_multiple_operations(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [
                {"action": "replace_section", "target_heading": "## 背景", "content": "\nUpdated background.\n"},
                {"action": "replace_section", "target_heading": "## 方案", "content": "\nUpdated solution.\n"},
            ],
        )
        assert result["status"] == "success"
        assert len(result["data"]["sections_affected"]) == 2

    @pytest.mark.asyncio
    async def test_insert_after_section(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [{"action": "insert_after_section", "target_heading": "## 背景", "content": "## New Section\n\nInserted content."}],
        )
        assert result["status"] == "success"
        new_content = abs_path.read_text(encoding="utf-8")
        # "New Section" should appear between 背景 and 方案
        bg_pos = new_content.index("## 背景")
        new_pos = new_content.index("Inserted content")
        sol_pos = new_content.index("## 方案")
        assert bg_pos < new_pos < sol_pos

    @pytest.mark.asyncio
    async def test_patch_nonexistent_file(self, tmp_vault):
        tools = _get_tools(tmp_vault)
        result = await tools["obsidian_patch_note"](
            "ghost.md", "fake_hash",
            [{"action": "replace_section", "target_heading": "## X", "content": "y"}],
        )
        assert result["status"] == "error"
        assert result["error_code"] == "not_found"

    @pytest.mark.asyncio
    async def test_unknown_action(self, populated_vault):
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [{"action": "unknown_action", "target_heading": "## 背景", "content": "x"}],
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_partial_success_with_warnings(self, populated_vault):
        """Some ops succeed, some fail → success with warnings."""
        vault, paths = populated_vault
        tools = _get_tools(vault)
        abs_path = vault.resolve_path(paths["design_doc"])
        cas_hash = compute_file_hash(abs_path)

        result = await tools["obsidian_patch_note"](
            paths["design_doc"],
            cas_hash,
            [
                {"action": "replace_section", "target_heading": "## 背景", "content": "\nOK.\n"},
                {"action": "replace_section", "target_heading": "## Nonexistent", "content": "fail"},
            ],
        )
        assert result["status"] == "success"
        assert "warnings" in result
