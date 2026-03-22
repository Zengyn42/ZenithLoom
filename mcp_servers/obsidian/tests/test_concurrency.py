"""
测试并发场景 — 多个异步任务同时操作同一文件
"""

import asyncio

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


class TestConcurrency:
    """并发写入测试。"""

    @pytest.mark.asyncio
    async def test_concurrent_writes_same_file(self, tmp_vault):
        """多个并发写入同一文件：只有一个成功（CAS 保护）。"""
        tools = _get_tools(tmp_vault)

        # 先创建文件
        result = await tools["obsidian_write_note"]("shared.md", "# Original")
        assert result["status"] == "success"
        original_hash = result["data"]["cas_hash"]

        # 并发 5 个写入
        async def write_variant(i):
            return await tools["obsidian_write_note"](
                "shared.md", f"# Version {i}", cas_hash=original_hash,
            )

        results = await asyncio.gather(*[write_variant(i) for i in range(5)])

        success_count = sum(1 for r in results if r["status"] == "success")
        conflict_count = sum(1 for r in results if r.get("error_code") == "conflict")

        # 只有一个能成功，其余 conflict
        assert success_count == 1
        assert conflict_count == 4

    @pytest.mark.asyncio
    async def test_concurrent_writes_different_files(self, tmp_vault):
        """不同文件并发写入：全部成功。"""
        tools = _get_tools(tmp_vault)

        async def create_file(i):
            return await tools["obsidian_write_note"](
                f"file-{i}.md", f"# File {i}",
            )

        results = await asyncio.gather(*[create_file(i) for i in range(10)])
        assert all(r["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_patch_same_file(self, tmp_vault):
        """并发 patch 同一文件：CAS 保护。"""
        tools = _get_tools(tmp_vault)

        content = "# Doc\n\n## Section A\n\nOriginal A.\n\n## Section B\n\nOriginal B.\n"
        result = await tools["obsidian_write_note"]("doc.md", content)
        original_hash = result["data"]["cas_hash"]

        from mcp_servers.obsidian.tools.write import register as wr
        # patch tools are in same register

        async def patch_variant(i):
            return await tools["obsidian_patch_note"](
                "doc.md",
                original_hash,
                [{"action": "replace_section", "target_heading": "## Section A", "content": f"\nVersion {i}\n"}],
            )

        results = await asyncio.gather(*[patch_variant(i) for i in range(5)])
        success_count = sum(1 for r in results if r["status"] == "success")
        assert success_count == 1
