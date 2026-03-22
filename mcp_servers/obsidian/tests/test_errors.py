"""
测试 core/errors.py — 错误码与响应构造
"""

from mcp_servers.obsidian.core.errors import VaultErrorCode, fail, ok


class TestOk:
    def test_basic_ok(self):
        r = ok()
        assert r["status"] == "success"
        assert "data" not in r

    def test_ok_with_data(self):
        r = ok(data={"key": "val"})
        assert r["data"]["key"] == "val"

    def test_ok_with_metadata(self):
        r = ok(data={"a": 1}, index_status="pending")
        assert r["metadata"]["index_status"] == "pending"


class TestFail:
    def test_basic_fail(self):
        r = fail(VaultErrorCode.NOT_FOUND, "oops")
        assert r["status"] == "error"
        assert r["error_code"] == "not_found"
        assert r["message"] == "oops"

    def test_fail_with_metadata(self):
        r = fail(VaultErrorCode.CONFLICT, "conflict", actual_hash="abc")
        assert r["metadata"]["actual_hash"] == "abc"


class TestErrorCodes:
    def test_all_codes_are_strings(self):
        for code in VaultErrorCode:
            assert isinstance(code.value, str)
