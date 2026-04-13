"""Tests for DebugConsoleReporter — scope tracking and value formatting."""
import pytest
from pathlib import Path


# --- Scope tracking ---

def test_scope_name_top_level():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("myapp")
    assert r._scope_name(()) == "myapp"

def test_scope_name_one_level():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("myapp")
    assert r._scope_name(("plan:abc123",)) == "plan"

def test_scope_name_nested():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("myapp")
    assert r._scope_name(("plan:abc123", "design_debate:def456")) == "design_debate"

def test_depth():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("myapp")
    assert r._depth(()) == 0
    assert r._depth(("a:1",)) == 1
    assert r._depth(("a:1", "b:2")) == 2


# --- Value formatting ---

def test_format_value_short_string():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value("hello") == "'hello'"

def test_format_value_long_string():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    long_s = "x" * 100
    result = r._format_value(long_s)
    assert "100 chars" in result

def test_format_value_list():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value([1, 2, 3]) == "3 items"

def test_format_value_dict():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value({"a": 1, "b": 2}) == "{2 keys}"

def test_format_value_int():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value(42) == "42"

def test_format_value_bool():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value(True) == "True"

def test_format_value_none():
    from framework.debug_reporter import DebugConsoleReporter
    r = DebugConsoleReporter("test")
    assert r._format_value(None) == "None"
