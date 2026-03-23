"""
Tests for SUBGRAPH_NODE type handling in agent_loader.
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


def test_subgraph_node_uses_correct_entityloader_kwarg():
    """SUBGRAPH_NODE branch must NOT use blueprint_dir kwarg (was wrong in original implementation)."""
    import framework.agent_loader as al
    src = inspect.getsource(al)
    # Must NOT use wrong kwarg
    assert "EntityLoader(blueprint_dir=" not in src, \
        "SUBGRAPH_NODE must use positional agent_dir, not blueprint_dir kwarg"
    # Must use correct positional call
    assert "EntityLoader(inner_dir)" in src or "EntityLoader(agent_dir=inner_dir)" in src


def test_entity_loader_accepts_positional_agent_dir():
    """EntityLoader must accept positional agent_dir (regression for SUBGRAPH_NODE blueprint_dir bug)."""
    import tempfile
    from pathlib import Path
    from framework.agent_loader import EntityLoader
    with tempfile.TemporaryDirectory() as tmp:
        # Create minimal entity.json so EntityLoader doesn't fail on missing config
        agent_json = Path(tmp) / "entity.json"
        agent_json.write_text('{"graph": {"nodes": [], "edges": []}}')
        loader = EntityLoader(Path(tmp))
        assert loader is not None
