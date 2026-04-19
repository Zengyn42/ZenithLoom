import pytest
from framework.string_utils import longest_common_substring

def test_lcs_basic():
    assert longest_common_substring("abcde", "ace") == "ace"
    assert longest_common_substring("abc", "def") == ""
    assert longest_common_substring("abc", "abc") == "abc"
    assert longest_common_substring("abcdxyz", "xyzabcd") == "abcd"
    assert longest_common_substring("zxabcdezy", "yzabcdezx") == "abcdez"

def test_lcs_empty_strings():
    assert longest_common_substring("", "abc") == ""
    assert longest_common_substring("abc", "") == ""
    assert longest_common_substring("", "") == ""

def test_lcs_single_character():
    assert longest_common_substring("a", "a") == "a"
    assert longest_common_substring("a", "b") == ""

def test_lcs_multiple_common_substrings():
    # If multiple LCS exist, any one is acceptable.
    # The current implementation returns the one ending earliest in s1 if lengths are equal.
    assert longest_common_substring("banana", "bandana") in ["bana", "ana"] 
    assert longest_common_substring("climbing", "diving") == "ing"

def test_lcs_no_common_substring():
    assert longest_common_substring("hello", "world") == ""
    assert longest_common_substring("123", "456") == ""
