"""
Process-level toggle for the inline token usage line emitted after each
Claude reply.

Pattern mirrors framework/debug.py: a single module-level boolean with
get/set helpers. Process-global scope is intentional — each agent
(hani/asa/jei) runs its own process, so the toggle naturally scopes to
"the current agent". Per-session granularity would double the surface
area with no observed benefit.

Default: True. The original intent expressed by the user was "every
message shows the token line"; the toggle exists so it can be silenced
when noisy, not because it should start off.
"""

_enabled: bool = True


def is_token_display_enabled() -> bool:
    """Return whether the inline [tokens: …] line should be emitted."""
    return _enabled


def set_token_display(value) -> None:
    """Enable or disable the inline token line (bool-coerced)."""
    global _enabled
    _enabled = bool(value)
