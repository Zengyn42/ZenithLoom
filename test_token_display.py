"""Unit test for framework.token_display process-global toggle."""

from framework import token_display


def test_default_is_enabled():
    import importlib
    importlib.reload(token_display)
    assert token_display.is_token_display_enabled() is True


def test_set_and_read_false():
    token_display.set_token_display(False)
    assert token_display.is_token_display_enabled() is False


def test_set_and_read_true():
    token_display.set_token_display(True)
    assert token_display.is_token_display_enabled() is True


def test_set_coerces_truthy_falsy():
    token_display.set_token_display(0)
    assert token_display.is_token_display_enabled() is False
    token_display.set_token_display("yes")
    assert token_display.is_token_display_enabled() is True


if __name__ == "__main__":
    test_default_is_enabled()
    test_set_and_read_false()
    test_set_and_read_true()
    test_set_coerces_truthy_falsy()
    print("✅ token_display OK")
