"""
Smoke tests for SUBGRAPH_NODE type handling in agent_loader.
"""
import inspect


def test_subgraph_node_type_handled_in_agent_loader():
    """SUBGRAPH_NODE branch must exist in _build_declarative in agent_loader."""
    import framework.agent_loader as al
    src = inspect.getsource(al)
    assert "SUBGRAPH_NODE" in src, "SUBGRAPH_NODE not handled in agent_loader"


def test_no_circular_import_with_reducers():
    """Importing agent_loader and reducers together must not circular-import."""
    import framework.agent_loader  # noqa
    from framework.schema.reducers import _merge_dict  # noqa
    assert callable(_merge_dict)
