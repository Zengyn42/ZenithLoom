"""
Obsidian Vault MCP — 读取工具

obsidian_read_note: 读取笔记内容 + frontmatter + CAS hash
obsidian_list_files: 列出目录下的笔记文件
"""

from pathlib import Path

from mcp_servers.obsidian.core.cas import compute_file_hash, get_mtime_ms
from mcp_servers.obsidian.core.errors import VaultErrorCode, fail, ok
from mcp_servers.obsidian.core.markdown_ops import parse_frontmatter
from mcp_servers.obsidian.core.vault import Vault


def register(mcp, vault: Vault):
    """注册读取相关的 MCP tools。"""

    @mcp.tool()
    async def obsidian_read_note(path: str) -> dict:
        """
        读取 Obsidian 笔记。返回内容、frontmatter、CAS hash。

        参数：
          path: Vault 内相对路径（如 "projects/my-note.md"）

        返回：
          data.content: 完整 Markdown 内容
          data.frontmatter: 解析好的 YAML frontmatter 字典
          data.cas_hash: SHA-256 摘要（用于后续写入的乐观锁）
          data.mtime_ms: 最后修改时间戳（毫秒）
        """
        resolved = vault.resolve_path(path)
        if isinstance(resolved, dict):
            return resolved  # 错误响应

        if not resolved.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"笔记不存在: {path}")

        if not resolved.is_file():
            return fail(VaultErrorCode.VALIDATION_ERROR, f"不是文件: {path}")

        content = resolved.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(content)
        cas_hash = compute_file_hash(resolved)
        mtime_ms = get_mtime_ms(resolved)

        return ok(data={
            "content": content,
            "frontmatter": fm,
            "cas_hash": cas_hash,
            "mtime_ms": mtime_ms,
            "path": path,
        })

    @mcp.tool()
    async def obsidian_list_files(
        directory: str = "",
        pattern: str = "*.md",
        recursive: bool = False,
    ) -> dict:
        """
        列出 Vault 目录下的文件。

        参数：
          directory: 相对目录路径（空 = vault 根目录）
          pattern: glob 模式（默认 "*.md"）
          recursive: 是否递归子目录

        返回：
          data.files: 文件信息列表 [{path, size_bytes, mtime_ms}]
        """
        resolved = vault.resolve_dir(directory)
        if isinstance(resolved, dict):
            return resolved

        if not resolved.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"目录不存在: {directory or '/'}")

        if not resolved.is_dir():
            return fail(VaultErrorCode.VALIDATION_ERROR, f"不是目录: {directory}")

        glob_method = resolved.rglob if recursive else resolved.glob
        files = []
        for p in sorted(glob_method(pattern)):
            if p.is_file() and vault.is_note(p):
                try:
                    stat = p.stat()
                    files.append({
                        "path": vault.relative_path(p),
                        "size_bytes": stat.st_size,
                        "mtime_ms": int(stat.st_mtime * 1000),
                    })
                except OSError:
                    continue

        return ok(data={"files": files, "count": len(files)})
