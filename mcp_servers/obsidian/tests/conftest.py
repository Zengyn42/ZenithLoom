"""
共享测试 fixtures — 每个测试用例使用独立的临时 vault 目录。
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# 确保项目根在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp_servers.obsidian.core.vault import Vault


@pytest.fixture
def tmp_vault(tmp_path):
    """创建一个临时 vault 目录并返回 Vault 实例。"""
    vault_dir = tmp_path / "test_vault"
    vault_dir.mkdir()
    return Vault(vault_dir)


@pytest.fixture
def populated_vault(tmp_vault):
    """
    创建一个包含示例笔记的 vault。
    返回 (vault, {name: relative_path}) 方便测试引用。
    """
    vault = tmp_vault
    base = vault.base_dir

    # 创建目录结构
    (base / "projects").mkdir()
    (base / "daily").mkdir()
    (base / "archive").mkdir()

    # 笔记 1: 带 frontmatter 和 wikilinks
    note1 = base / "projects" / "design-doc.md"
    note1.write_text(
        "---\ntags:\n  - project\n  - design\ncreated: 2026-03-21\n---\n\n"
        "# Design Doc\n\n"
        "## 背景\n\n"
        "This is the background section.\n"
        "See also [[meeting-notes]] and [[research/paper-a]].\n\n"
        "## 方案\n\n"
        "Proposed solution goes here.\n\n"
        "## 时间线\n\n"
        "- Phase 1: 1 week\n"
        "- Phase 2: 2 weeks\n",
        encoding="utf-8",
    )

    # 笔记 2: 引用笔记 1
    note2 = base / "daily" / "meeting-notes.md"
    note2.write_text(
        "---\ntags:\n  - meeting\ncreated: 2026-03-20\n---\n\n"
        "# Meeting Notes\n\n"
        "Discussed the [[design-doc]] today.\n"
        "Action items tracked in [[todo-list]].\n",
        encoding="utf-8",
    )

    # 笔记 3: 无 frontmatter
    note3 = base / "archive" / "old-note.md"
    note3.write_text(
        "# Old Note\n\n"
        "This is an old note without frontmatter.\n"
        "It has a #legacy tag and links to [[design-doc]].\n",
        encoding="utf-8",
    )

    # 笔记 4: 纯文本（无 heading）
    note4 = base / "plain.md"
    note4.write_text(
        "Just some plain text without any headings.\n"
        "Contains a #misc tag.\n",
        encoding="utf-8",
    )

    return vault, {
        "design_doc": "projects/design-doc.md",
        "meeting_notes": "daily/meeting-notes.md",
        "old_note": "archive/old-note.md",
        "plain": "plain.md",
    }
