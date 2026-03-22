"""
Obsidian Vault MCP — 搜索与链接工具

obsidian_search_files: 文件名/内容搜索
obsidian_get_links: 查询双向链接（outgoing + incoming）
"""

import re
from pathlib import Path

from mcp_servers.obsidian.core.errors import VaultErrorCode, fail, ok
from mcp_servers.obsidian.core.markdown_ops import extract_wikilinks
from mcp_servers.obsidian.core.vault import Vault


def register(mcp, vault: Vault):
    """注册搜索相关的 MCP tools。"""

    @mcp.tool()
    async def obsidian_search_files(
        query: str,
        search_type: str = "content",
        directory: str = "",
        max_results: int = 20,
    ) -> dict:
        """
        搜索 Vault 中的笔记。

        参数：
          query: 搜索关键词
          search_type: "content"（搜索内容）| "filename"（搜索文件名）
          directory: 限定搜索目录（空 = 全 vault）
          max_results: 最大返回数量（默认 20）

        返回：
          data.results: 匹配结果列表 [{path, matches}]
        """
        if not query.strip():
            return fail(VaultErrorCode.VALIDATION_ERROR, "搜索关键词不能为空")

        base = vault.resolve_dir(directory)
        if isinstance(base, dict):
            return base

        if not base.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"目录不存在: {directory or '/'}")

        results: list[dict] = []
        query_lower = query.lower()
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        for p in sorted(base.rglob("*.md")):
            if not p.is_file() or not vault.is_note(p):
                continue

            rel = vault.relative_path(p)

            if search_type == "filename":
                if query_lower in p.name.lower():
                    results.append({
                        "path": rel,
                        "match_type": "filename",
                    })
            else:
                # content search
                try:
                    content = p.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue

                matches = []
                for i, line in enumerate(content.split("\n"), 1):
                    if pattern.search(line):
                        matches.append({
                            "line": i,
                            "text": line.strip()[:200],  # 截断长行
                        })

                if matches:
                    results.append({
                        "path": rel,
                        "match_type": "content",
                        "matches": matches[:5],  # 每个文件最多 5 条匹配
                        "total_matches": len(matches),
                    })

            if len(results) >= max_results:
                break

        return ok(data={
            "results": results,
            "count": len(results),
            "query": query,
            "search_type": search_type,
        })

    @mcp.tool()
    async def obsidian_get_links(path: str) -> dict:
        """
        查询笔记的双向链接。

        参数：
          path: Vault 内相对路径

        返回：
          data.outgoing: 当前笔记指向的 [[wikilink]] 目标列表
          data.incoming: 链接到当前笔记的其他笔记列表
        """
        resolved = vault.resolve_path(path)
        if isinstance(resolved, dict):
            return resolved

        if not resolved.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"笔记不存在: {path}")

        # Outgoing links
        content = resolved.read_text(encoding="utf-8")
        outgoing = extract_wikilinks(content)

        # Incoming links: 扫描 vault 中所有笔记，找引用了当前笔记的
        # 用文件名（不含后缀）作为匹配目标
        note_name = resolved.stem
        incoming: list[str] = []

        for p in vault.base_dir.rglob("*.md"):
            if p == resolved or not p.is_file():
                continue
            try:
                other_content = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            other_links = extract_wikilinks(other_content)
            # 检查是否链接到当前笔记
            for link in other_links:
                # wikilink 可能是 "note" 或 "folder/note"
                link_name = link.split("/")[-1] if "/" in link else link
                if link_name == note_name or link == vault.relative_path(resolved).replace(".md", ""):
                    incoming.append(vault.relative_path(p))
                    break

        return ok(data={
            "outgoing": outgoing,
            "incoming": incoming,
            "path": path,
        })
