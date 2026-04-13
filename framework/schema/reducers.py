"""
Shared LangGraph state reducers for ZenithLoom schemas.
"""

# Sentinel: when a node returns CLEAR_DICT as node_sessions value,
# _merge_dict replaces the entire dict instead of merging.
CLEAR_DICT = {"__clear__": True}


def _merge_dict(a: dict | None, b: dict | None) -> dict:
    """Merge reducer: b's values overwrite a's for shared keys. Safe for parallel node writes.

    If b is the CLEAR_DICT sentinel, returns {} (full replacement) so that
    _subgraph_init can clear node_sessions for fresh_per_call / isolated modes.
    """
    if b is not None and b.get("__clear__") is True:
        # Strip the sentinel and return only non-sentinel keys from b
        return {k: v for k, v in b.items() if k != "__clear__"}
    return {**(a or {}), **(b or {})}
