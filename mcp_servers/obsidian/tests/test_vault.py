"""
测试 core/vault.py — 路径沙箱与安全护栏
"""

from pathlib import Path

import pytest

from mcp_servers.obsidian.core.vault import Vault


class TestVaultInit:
    """Vault 初始化。"""

    def test_init_valid_dir(self, tmp_path):
        v = Vault(tmp_path)
        assert v.base_dir == tmp_path.resolve()

    def test_init_nonexistent_dir(self, tmp_path):
        with pytest.raises(ValueError, match="不存在"):
            Vault(tmp_path / "nonexistent")

    def test_init_string_path(self, tmp_path):
        v = Vault(str(tmp_path))
        assert v.base_dir == tmp_path.resolve()


class TestResolvePath:
    """路径解析 + 安全校验。"""

    def test_simple_relative_path(self, tmp_vault):
        result = tmp_vault.resolve_path("notes/test.md")
        assert isinstance(result, Path)
        assert result == tmp_vault.base_dir / "notes" / "test.md"

    def test_leading_slash_stripped(self, tmp_vault):
        result = tmp_vault.resolve_path("/notes/test.md")
        assert isinstance(result, Path)
        assert result == tmp_vault.base_dir / "notes" / "test.md"

    def test_empty_path_rejected(self, tmp_vault):
        result = tmp_vault.resolve_path("")
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["error_code"] == "validation_error"

    def test_whitespace_only_rejected(self, tmp_vault):
        result = tmp_vault.resolve_path("   ")
        assert isinstance(result, dict)
        assert result["error_code"] == "validation_error"

    # -- 路径穿越攻击 --

    def test_traversal_dotdot(self, tmp_vault):
        result = tmp_vault.resolve_path("../../etc/passwd")
        assert isinstance(result, dict)
        assert result["error_code"] == "path_traversal"

    def test_traversal_encoded(self, tmp_vault):
        result = tmp_vault.resolve_path("notes/../../../etc/passwd")
        assert isinstance(result, dict)
        assert result["error_code"] == "path_traversal"

    def test_traversal_absolute(self, tmp_vault):
        """绝对路径被 lstrip 后可能仍然穿越。"""
        result = tmp_vault.resolve_path("/tmp/evil.md")
        # 由于 lstrip("/")，变成 "tmp/evil.md"，在 vault 内
        # 这不是穿越，而是合法的 vault 内路径
        assert isinstance(result, Path)

    # -- 敏感目录黑名单 --

    def test_blocked_obsidian_dir(self, tmp_vault):
        result = tmp_vault.resolve_path(".obsidian/config.json")
        assert isinstance(result, dict)
        assert result["error_code"] == "permission_denied"

    def test_blocked_git_dir(self, tmp_vault):
        result = tmp_vault.resolve_path(".git/config")
        assert isinstance(result, dict)
        assert result["error_code"] == "permission_denied"

    def test_blocked_trash_dir(self, tmp_vault):
        result = tmp_vault.resolve_path(".trash/deleted-note.md")
        assert isinstance(result, dict)
        assert result["error_code"] == "permission_denied"

    def test_blocked_node_modules(self, tmp_vault):
        result = tmp_vault.resolve_path("node_modules/package/index.js")
        assert isinstance(result, dict)
        assert result["error_code"] == "permission_denied"

    def test_blocked_nested_obsidian(self, tmp_vault):
        result = tmp_vault.resolve_path("notes/.obsidian/config")
        assert isinstance(result, dict)
        assert result["error_code"] == "permission_denied"

    def test_allowed_normal_dir(self, tmp_vault):
        result = tmp_vault.resolve_path("projects/design.md")
        assert isinstance(result, Path)


class TestResolveDir:
    """目录路径解析。"""

    def test_empty_returns_base(self, tmp_vault):
        result = tmp_vault.resolve_dir("")
        assert result == tmp_vault.base_dir

    def test_dot_returns_base(self, tmp_vault):
        result = tmp_vault.resolve_dir(".")
        assert result == tmp_vault.base_dir

    def test_subdir(self, tmp_vault):
        result = tmp_vault.resolve_dir("projects")
        assert isinstance(result, Path)

    def test_traversal_blocked(self, tmp_vault):
        result = tmp_vault.resolve_dir("../../")
        assert isinstance(result, dict)
        assert result["error_code"] == "path_traversal"

    def test_blocked_dir(self, tmp_vault):
        result = tmp_vault.resolve_dir(".obsidian")
        assert isinstance(result, dict)
        assert result["error_code"] == "permission_denied"


class TestRelativePath:
    """绝对路径 → 相对路径转换。"""

    def test_relative_path(self, tmp_vault):
        abs_p = tmp_vault.base_dir / "notes" / "test.md"
        assert tmp_vault.relative_path(abs_p) == "notes/test.md"


class TestIsNote:
    """文件类型判断。"""

    def test_md_is_note(self, tmp_vault):
        assert tmp_vault.is_note(Path("test.md"))

    def test_markdown_is_note(self, tmp_vault):
        assert tmp_vault.is_note(Path("test.markdown"))

    def test_txt_is_note(self, tmp_vault):
        assert tmp_vault.is_note(Path("test.txt"))

    def test_canvas_is_note(self, tmp_vault):
        assert tmp_vault.is_note(Path("test.canvas"))

    def test_py_is_not_note(self, tmp_vault):
        assert not tmp_vault.is_note(Path("test.py"))

    def test_jpg_is_not_note(self, tmp_vault):
        assert not tmp_vault.is_note(Path("image.jpg"))
