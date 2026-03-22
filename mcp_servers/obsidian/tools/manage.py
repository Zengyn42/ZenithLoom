"""
Obsidian Vault MCP — 管理工具

obsidian_move_note: 移动/重命名笔记
obsidian_delete_note: 删除笔记（默认移至 .trash/）
obsidian_manage_tags: 管理标签
obsidian_get_frontmatter: 获取 frontmatter
obsidian_update_frontmatter: 更新 frontmatter
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

from mcp_servers.obsidian.core.audit_log import log_operation
from mcp_servers.obsidian.core.cas import compute_file_hash, get_file_lock
from mcp_servers.obsidian.core.errors import VaultErrorCode, fail, ok
from mcp_servers.obsidian.core.markdown_ops import (
    extract_tags,
    parse_frontmatter,
    serialize_frontmatter,
    update_frontmatter,
)
from mcp_servers.obsidian.core.vault import Vault


def register(mcp, vault: Vault):
    """注册管理相关的 MCP tools。"""

    @mcp.tool()
    async def obsidian_move_note(
        source: str,
        destination: str,
    ) -> dict:
        """
        移动或重命名笔记。

        参数：
          source: 源文件相对路径
          destination: 目标相对路径
        """
        src = vault.resolve_path(source)
        if isinstance(src, dict):
            return src

        dst = vault.resolve_path(destination)
        if isinstance(dst, dict):
            return dst

        if not src.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"源文件不存在: {source}")

        if dst.exists():
            return fail(VaultErrorCode.ALREADY_EXISTS, f"目标已存在: {destination}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        new_hash = compute_file_hash(dst)

        log_operation(
            tool="obsidian_move_note",
            target=source,
            action="move",
            status="success",
            destination=destination,
        )

        return ok(data={
            "new_path": destination,
            "cas_hash": new_hash,
        })

    @mcp.tool()
    async def obsidian_delete_note(
        path: str,
        cas_hash: str = "",
        permanent: bool = False,
    ) -> dict:
        """
        删除笔记。默认移至 .trash/ 目录（安全删除）。

        参数：
          path: Vault 内相对路径
          cas_hash: 可选 CAS 校验
          permanent: True = 永久删除（慎用）；False = 移至 .trash/
        """
        resolved = vault.resolve_path(path)
        if isinstance(resolved, dict):
            return resolved

        if not resolved.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"笔记不存在: {path}")

        # CAS 校验（如果提供了 hash）
        if cas_hash:
            actual = compute_file_hash(resolved)
            if actual != cas_hash:
                return fail(
                    VaultErrorCode.CONFLICT,
                    f"CAS 冲突: 文件已被修改",
                    expected_hash=cas_hash,
                    actual_hash=actual,
                )

        if permanent:
            resolved.unlink()
            action = "permanent_delete"
            deleted_to = "(permanently deleted)"
        else:
            # 移至 .trash/
            trash_dir = vault.base_dir / ".trash"
            trash_dir.mkdir(exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            trash_name = f"{resolved.stem}_{ts}{resolved.suffix}"
            trash_path = trash_dir / trash_name
            shutil.move(str(resolved), str(trash_path))
            action = "trash"
            deleted_to = f".trash/{trash_name}"

        log_operation(
            tool="obsidian_delete_note",
            target=path,
            action=action,
            status="success",
        )

        return ok(data={"deleted_to": deleted_to})

    @mcp.tool()
    async def obsidian_get_frontmatter(path: str) -> dict:
        """
        获取笔记的 YAML frontmatter。

        参数：
          path: Vault 内相对路径

        返回：
          data.frontmatter: 解析好的字典
        """
        resolved = vault.resolve_path(path)
        if isinstance(resolved, dict):
            return resolved

        if not resolved.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"笔记不存在: {path}")

        content = resolved.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(content)
        return ok(data={"frontmatter": fm, "path": path})

    @mcp.tool()
    async def obsidian_update_frontmatter(
        path: str,
        updates: dict,
        cas_hash: str = "",
    ) -> dict:
        """
        更新笔记的 frontmatter 字段（合并，不覆盖其他字段）。

        参数：
          path: Vault 内相对路径
          updates: 要更新的键值对
          cas_hash: 可选 CAS 校验
        """
        resolved = vault.resolve_path(path)
        if isinstance(resolved, dict):
            return resolved

        if not resolved.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"笔记不存在: {path}")

        lock = get_file_lock(resolved)
        async with lock:
            if cas_hash:
                actual = compute_file_hash(resolved)
                if actual != cas_hash:
                    return fail(
                        VaultErrorCode.CONFLICT,
                        "CAS 冲突",
                        expected_hash=cas_hash,
                        actual_hash=actual,
                    )

            content = resolved.read_text(encoding="utf-8")
            new_content = update_frontmatter(content, updates)
            resolved.write_text(new_content, encoding="utf-8")
            from mcp_servers.obsidian.core.cas import compute_hash
            new_hash = compute_hash(new_content)

            log_operation(
                tool="obsidian_update_frontmatter",
                target=path,
                action="update_frontmatter",
                status="success",
                cas_after=new_hash,
            )

            return ok(data={"cas_hash": new_hash, "path": path})

    @mcp.tool()
    async def obsidian_manage_tags(
        path: str,
        add: list[str] | None = None,
        remove: list[str] | None = None,
    ) -> dict:
        """
        管理笔记的标签（通过 frontmatter tags 字段）。

        参数：
          path: Vault 内相对路径
          add: 要添加的标签列表
          remove: 要移除的标签列表

        返回：
          data.tags: 更新后的标签列表
        """
        resolved = vault.resolve_path(path)
        if isinstance(resolved, dict):
            return resolved

        if not resolved.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"笔记不存在: {path}")

        lock = get_file_lock(resolved)
        async with lock:
            content = resolved.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(content)

            tags = list(fm.get("tags", []) or [])

            if add:
                for tag in add:
                    t = tag.lstrip("#").strip()
                    if t and t not in tags:
                        tags.append(t)

            if remove:
                remove_set = {r.lstrip("#").strip() for r in remove}
                tags = [t for t in tags if t not in remove_set]

            fm["tags"] = tags
            new_content = serialize_frontmatter(fm, body)
            resolved.write_text(new_content, encoding="utf-8")
            from mcp_servers.obsidian.core.cas import compute_hash
            new_hash = compute_hash(new_content)

            log_operation(
                tool="obsidian_manage_tags",
                target=path,
                action="manage_tags",
                status="success",
                cas_after=new_hash,
            )

            return ok(data={"tags": tags, "cas_hash": new_hash, "path": path})
