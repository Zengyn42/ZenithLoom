"""
Shared LangGraph state reducers for BootstrapBuilder schemas.
"""


def _merge_dict(a: dict, b: dict) -> dict:
    """Merge reducer: b's values overwrite a's for shared keys. Safe for parallel node writes."""
    return {**a, **b}
