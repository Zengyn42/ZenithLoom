"""
端到端集成测试 — 模拟完整的 MCP tool 调用链路
"""

import pytest

from mcp_servers.obsidian.core.cas import compute_file_hash
from mcp_servers.obsidian.core.vault import Vault
from mcp_servers.obsidian.tools import manage, read, search, write


def _register_all(vault):
    """注册所有 tool 模块，返回 tool 函数字典。"""
    class FakeMCP:
        def __init__(self):
            self.tools = {}
        def tool(self):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn
            return decorator

    fake = FakeMCP()
    read.register(fake, vault)
    write.register(fake, vault)
    manage.register(fake, vault)
    search.register(fake, vault)
    return fake.tools


class TestE2EWorkflow:
    """完整工作流：创建 → 读取 → patch → 搜索 → 链接 → 标签 → 移动 → 删除"""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, tmp_vault):
        tools = _register_all(tmp_vault)

        # 1. 创建笔记
        content = (
            "---\ntags:\n  - project\ncreated: 2026-03-21\n---\n\n"
            "# My Project\n\n"
            "## Overview\n\n"
            "This is the overview.\n\n"
            "## Details\n\n"
            "Some details here.\n"
            "Link to [[related-note]].\n"
        )
        r = await tools["obsidian_write_note"]("project/my-note.md", content)
        assert r["status"] == "success"
        hash1 = r["data"]["cas_hash"]

        # 2. 读取笔记
        r = await tools["obsidian_read_note"]("project/my-note.md")
        assert r["status"] == "success"
        assert r["data"]["cas_hash"] == hash1
        assert r["data"]["frontmatter"]["tags"] == ["project"]
        assert "My Project" in r["data"]["content"]

        # 3. Patch — replace section
        r = await tools["obsidian_patch_note"](
            "project/my-note.md",
            hash1,
            [{"action": "replace_section", "target_heading": "## Details",
              "content": "\nUpdated details with [[another-note]].\n"}],
        )
        assert r["status"] == "success"
        hash2 = r["data"]["cas_hash"]
        assert hash2 != hash1

        # 4. 搜索
        r = await tools["obsidian_search_files"]("Updated details")
        assert r["data"]["count"] >= 1

        # 5. 创建 related-note 用于链接测试
        r2 = await tools["obsidian_write_note"](
            "related-note.md",
            "# Related\n\nLinks to [[my-note]].\n",
        )
        assert r2["status"] == "success"

        # 6. 查询链接
        r = await tools["obsidian_get_links"]("project/my-note.md")
        assert r["status"] == "success"
        # After patch, Details section was replaced — only another-note remains
        assert "another-note" in r["data"]["outgoing"]

        # 7. 管理标签
        r = await tools["obsidian_manage_tags"](
            "project/my-note.md", add=["v2", "important"], remove=["project"],
        )
        assert r["status"] == "success"
        assert "v2" in r["data"]["tags"]
        assert "project" not in r["data"]["tags"]

        # 8. 获取 frontmatter
        r = await tools["obsidian_get_frontmatter"]("project/my-note.md")
        assert "v2" in r["data"]["frontmatter"]["tags"]

        # 9. 列出文件
        r = await tools["obsidian_list_files"](recursive=True)
        assert r["data"]["count"] == 2

        # 10. 移动笔记
        r = await tools["obsidian_move_note"](
            "project/my-note.md", "archive/my-note.md",
        )
        assert r["status"] == "success"

        # 11. 验证移动后
        r = await tools["obsidian_read_note"]("project/my-note.md")
        assert r["error_code"] == "not_found"

        r = await tools["obsidian_read_note"]("archive/my-note.md")
        assert r["status"] == "success"

        # 12. 删除（soft）
        r = await tools["obsidian_delete_note"]("archive/my-note.md")
        assert r["status"] == "success"
        assert ".trash/" in r["data"]["deleted_to"]

    @pytest.mark.asyncio
    async def test_cas_chain_integrity(self, tmp_vault):
        """每次写入后的 hash 可以用于下一次写入 — CAS 链条完整性。"""
        tools = _register_all(tmp_vault)

        # Create
        r = await tools["obsidian_write_note"]("chain.md", "# V1")
        h1 = r["data"]["cas_hash"]

        # Write V2 using h1
        r = await tools["obsidian_write_note"]("chain.md", "# V2", cas_hash=h1)
        h2 = r["data"]["cas_hash"]

        # Write V3 using h2
        r = await tools["obsidian_write_note"]("chain.md", "# V3", cas_hash=h2)
        h3 = r["data"]["cas_hash"]

        # Write V4 using stale h1 → should fail
        r = await tools["obsidian_write_note"]("chain.md", "# V4", cas_hash=h1)
        assert r["error_code"] == "conflict"

        # Verify final content
        r = await tools["obsidian_read_note"]("chain.md")
        assert r["data"]["content"] == "# V3"
        assert r["data"]["cas_hash"] == h3


class TestSecurityE2E:
    """端到端安全测试。"""

    @pytest.mark.asyncio
    async def test_all_tools_reject_traversal(self, tmp_vault):
        tools = _register_all(tmp_vault)
        evil = "../../etc/passwd"

        r = await tools["obsidian_read_note"](evil)
        assert r["error_code"] == "path_traversal"

        r = await tools["obsidian_write_note"](evil, "x")
        assert r["error_code"] == "path_traversal"

        r = await tools["obsidian_get_frontmatter"](evil)
        assert r["error_code"] == "path_traversal"

        r = await tools["obsidian_get_links"](evil)
        assert r["error_code"] == "path_traversal"

        r = await tools["obsidian_search_files"]("test", directory=evil)
        # resolve_dir 的穿越检查
        assert r["error_code"] == "path_traversal"

    @pytest.mark.asyncio
    async def test_all_tools_reject_blocked_dirs(self, tmp_vault):
        tools = _register_all(tmp_vault)
        blocked = ".obsidian/workspace.json"

        r = await tools["obsidian_read_note"](blocked)
        assert r["error_code"] == "permission_denied"

        r = await tools["obsidian_write_note"](blocked, "x")
        assert r["error_code"] == "permission_denied"

        r = await tools["obsidian_delete_note"](blocked)
        assert r["error_code"] == "permission_denied"
