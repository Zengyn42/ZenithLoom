"""
Shared LangGraph state reducers for ZenithLoom schemas.
"""


def _merge_dict(a: dict | None, b: dict | None) -> dict:
    """Merge reducer: b's values overwrite a's for shared keys. Safe for parallel node writes."""
    return {**(a or {}), **(b or {})}
