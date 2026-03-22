"""
测试 core/cas.py — CAS 乐观锁机制
"""

import asyncio
from pathlib import Path

import pytest

from mcp_servers.obsidian.core.cas import (
    compute_file_hash,
    compute_hash,
    get_file_lock,
    get_mtime_ms,
    verify_cas,
)


class TestComputeHash:
    """Hash 计算。"""

    def test_string_hash(self):
        h = compute_hash("hello")
        assert len(h) == 64  # SHA-256 hex
        assert h == compute_hash("hello")  # 确定性

    def test_bytes_hash(self):
        h = compute_hash(b"hello")
        assert h == compute_hash("hello")  # str 和 bytes 一致

    def test_different_content_different_hash(self):
        assert compute_hash("a") != compute_hash("b")

    def test_empty_string(self):
        h = compute_hash("")
        assert len(h) == 64


class TestComputeFileHash:
    """文件 hash。"""

    def test_file_hash(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("hello", encoding="utf-8")
        assert compute_file_hash(f) == compute_hash("hello")

    def test_utf8_file(self, tmp_path):
        f = tmp_path / "chinese.md"
        content = "你好世界"
        f.write_text(content, encoding="utf-8")
        assert compute_file_hash(f) == compute_hash(content)


class TestGetMtimeMs:
    """mtime 毫秒。"""

    def test_mtime_positive(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("x")
        ms = get_mtime_ms(f)
        assert isinstance(ms, int)
        assert ms > 0


class TestVerifyCas:
    """CAS 验证矩阵。"""

    def test_create_new_file_not_exists(self, tmp_path):
        """cas_hash=None, 文件不存在 → 新建，valid"""
        f = tmp_path / "new.md"
        is_valid, actual = verify_cas(f, None)
        assert is_valid is True
        assert actual == ""

    def test_create_file_already_exists(self, tmp_path):
        """cas_hash=None, 文件已存在 → conflict"""
        f = tmp_path / "exist.md"
        f.write_text("existing content")
        is_valid, actual = verify_cas(f, None)
        assert is_valid is False
        assert len(actual) == 64  # 返回 actual hash

    def test_update_hash_matches(self, tmp_path):
        """cas_hash 匹配 → valid"""
        f = tmp_path / "test.md"
        content = "hello"
        f.write_text(content)
        expected = compute_hash(content)
        is_valid, actual = verify_cas(f, expected)
        assert is_valid is True
        assert actual == expected

    def test_update_hash_mismatch(self, tmp_path):
        """cas_hash 不匹配 → conflict"""
        f = tmp_path / "test.md"
        f.write_text("original")
        is_valid, actual = verify_cas(f, "wrong_hash")
        assert is_valid is False
        assert actual == compute_hash("original")

    def test_update_file_not_exists(self, tmp_path):
        """cas_hash 非 None 但文件不存在 → invalid"""
        f = tmp_path / "ghost.md"
        is_valid, actual = verify_cas(f, "some_hash")
        assert is_valid is False
        assert actual == ""


class TestGetFileLock:
    """文件级锁。"""

    def test_same_path_same_lock(self, tmp_path):
        f = tmp_path / "test.md"
        lock1 = get_file_lock(f)
        lock2 = get_file_lock(f)
        assert lock1 is lock2

    def test_different_path_different_lock(self, tmp_path):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        assert get_file_lock(f1) is not get_file_lock(f2)

    def test_lock_is_asyncio_lock(self, tmp_path):
        f = tmp_path / "test.md"
        lock = get_file_lock(f)
        assert isinstance(lock, asyncio.Lock)
