"""
Obsidian Vault MCP — 写入工具

obsidian_write_note: 全量写入（新建或完全覆盖）
obsidian_patch_note: Section-based 局部修改
"""

from pathlib import Path

from mcp_servers.obsidian.core.audit_log import log_operation
from mcp_servers.obsidian.core.cas import (
    compute_hash,
    get_file_lock,
    verify_cas,
)
from mcp_servers.obsidian.core.errors import VaultErrorCode, fail, ok
from mcp_servers.obsidian.core.markdown_ops import (
    find_section,
    parse_frontmatter,
    reassemble_sections,
    serialize_frontmatter,
    split_sections,
    update_frontmatter,
)
from mcp_servers.obsidian.core.vault import Vault


def register(mcp, vault: Vault):
    """注册写入相关的 MCP tools。"""

    @mcp.tool()
    async def obsidian_write_note(
        path: str,
        content: str,
        cas_hash: str = "",
    ) -> dict:
        """
        全量写入笔记。用于新建文件或完全重构内容。

        参数：
          path: Vault 内相对路径
          content: 完整 Markdown 内容（含 frontmatter）
          cas_hash: 乐观锁。空字符串 = 新建（文件已存在则 conflict）。
                    非空 = 覆盖（hash 不匹配则 conflict）。

        返回：
          data.cas_hash: 写入后的新 hash
          metadata.index_status: pending（预留 RAG 索引状态）
        """
        resolved = vault.resolve_path(path)
        if isinstance(resolved, dict):
            return resolved

        lock = get_file_lock(resolved)
        async with lock:
            expected = cas_hash if cas_hash else None
            is_valid, actual = verify_cas(resolved, expected)

            if not is_valid:
                if expected is None:
                    return fail(
                        VaultErrorCode.ALREADY_EXISTS,
                        f"文件已存在: {path}。如需覆盖，请先 read_note 获取 cas_hash。",
                        actual_hash=actual,
                    )
                return fail(
                    VaultErrorCode.CONFLICT,
                    f"CAS 冲突: 文件已被修改。expected={expected[:12]}... actual={actual[:12]}...",
                    expected_hash=expected,
                    actual_hash=actual,
                )

            # 确保父目录存在
            resolved.parent.mkdir(parents=True, exist_ok=True)

            # 写入
            resolved.write_text(content, encoding="utf-8")
            new_hash = compute_hash(content)

            log_operation(
                tool="obsidian_write_note",
                target=path,
                action="create" if expected is None else "overwrite",
                status="success",
                cas_before=cas_hash,
                cas_after=new_hash,
            )

            return ok(
                data={"cas_hash": new_hash, "path": path},
                index_status="pending",
            )

    @mcp.tool()
    async def obsidian_patch_note(
        path: str,
        cas_hash: str,
        operations: list[dict],
    ) -> dict:
        """
        基于 Heading Section 的局部修改。

        参数：
          path: Vault 内相对路径
          cas_hash: 必须提供，用于 CAS 校验
          operations: 操作列表，每个 operation 包含：
            - action: "update_frontmatter" | "replace_section" | "append_to_section" |
                      "insert_after_section" | "delete_section"
            - target_heading: Section 标题（如 "## 背景"），update_frontmatter 时可省略
            - content: 新内容（delete_section 时可省略）

        返回：
          data.cas_hash: 修改后的新 hash
          data.sections_affected: 受影响的 section 标题列表
        """
        resolved = vault.resolve_path(path)
        if isinstance(resolved, dict):
            return resolved

        if not resolved.exists():
            return fail(VaultErrorCode.NOT_FOUND, f"笔记不存在: {path}")

        lock = get_file_lock(resolved)
        async with lock:
            is_valid, actual = verify_cas(resolved, cas_hash)
            if not is_valid:
                return fail(
                    VaultErrorCode.CONFLICT,
                    f"CAS 冲突: expected={cas_hash[:12]}... actual={actual[:12]}...",
                    expected_hash=cas_hash,
                    actual_hash=actual,
                )

            content = resolved.read_text(encoding="utf-8")
            sections_affected: list[str] = []
            errors: list[str] = []

            for i, op in enumerate(operations):
                action = op.get("action", "")
                target = op.get("target_heading", "")
                new_content = op.get("content", "")

                if action == "update_frontmatter":
                    # Frontmatter 更新
                    if not isinstance(new_content, dict):
                        # content 可能是 JSON 字符串或 dict
                        try:
                            import json
                            new_content = json.loads(new_content) if isinstance(new_content, str) else {}
                        except (json.JSONDecodeError, TypeError):
                            errors.append(f"operation[{i}]: update_frontmatter content 必须是 dict 或 JSON 字符串")
                            continue

                    content = update_frontmatter(content, new_content)
                    sections_affected.append("(frontmatter)")

                elif action in ("replace_section", "append_to_section",
                                "insert_after_section", "delete_section"):
                    if not target:
                        errors.append(f"operation[{i}]: {action} 需要 target_heading")
                        continue

                    sections = split_sections(content)
                    idx = find_section(sections, target)

                    if idx is None:
                        errors.append(f"operation[{i}]: 未找到 section '{target}'")
                        continue

                    if action == "replace_section":
                        sections[idx].content = new_content
                    elif action == "append_to_section":
                        sections[idx].content = sections[idx].content.rstrip("\n") + "\n\n" + new_content
                    elif action == "insert_after_section":
                        # 在目标 section 之后插入新 section
                        from mcp_servers.obsidian.core.markdown_ops import Section
                        new_sec = Section(
                            heading="",
                            level=0,
                            title="(inserted)",
                            content=new_content,
                            start_line=0,
                            end_line=0,
                        )
                        sections.insert(idx + 1, new_sec)
                    elif action == "delete_section":
                        sections.pop(idx)

                    content = reassemble_sections(sections)
                    sections_affected.append(target)

                else:
                    errors.append(f"operation[{i}]: 未知 action '{action}'")

            if errors and not sections_affected:
                return fail(
                    VaultErrorCode.VALIDATION_ERROR,
                    f"所有操作失败: {'; '.join(errors)}",
                )

            # 写入
            resolved.write_text(content, encoding="utf-8")
            new_hash = compute_hash(content)

            log_operation(
                tool="obsidian_patch_note",
                target=path,
                action="patch",
                status="success" if not errors else "partial",
                cas_before=cas_hash,
                cas_after=new_hash,
                sections_affected=sections_affected,
            )

            result = ok(
                data={
                    "cas_hash": new_hash,
                    "path": path,
                    "sections_affected": sections_affected,
                },
                index_status="pending",
            )

            if errors:
                result["warnings"] = errors

            return result
