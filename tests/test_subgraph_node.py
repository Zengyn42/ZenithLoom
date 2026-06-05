"""
Tests for external subgraph (agent_dir) handling in graph_builder.
"""
import inspect


def test_subgraph_node_type_handled_in_graph_builder():
    """external subgraph branch must exist in _build_declarative in graph_builder.

    After the AgentGraph refactor, subgraph detection is expressed via
    ``node_spec.is_subgraph`` (a property on NodeSpec) rather than the old
    inline ``node_def.get("agent_dir") and not node_type`` pattern.
    """
    import framework.loader.graph_builder as gb
    src = inspect.getsource(gb)
    # The new implementation delegates to NodeSpec.is_subgraph
    assert "is_subgraph" in src, "subgraph detection (is_subgraph) not found in graph_builder"


def test_no_circular_import_with_reducers():
    """Importing framework.loader and reducers together must not circular-import."""
    import framework.loader  # noqa
    from framework.schema.reducers import _merge_dict  # noqa
    assert callable(_merge_dict)


def test_subgraph_node_uses_correct_entityloader_kwarg():
    """external subgraph branch must NOT use blueprint_dir kwarg (was wrong in original implementation)."""
    import framework.loader.graph_builder as gb
    src = inspect.getsource(gb)
    # Must NOT use wrong kwarg
    assert "EntityLoader(blueprint_dir=" not in src, \
        "external subgraph must use positional agent_dir, not blueprint_dir kwarg"
    # Must use correct positional call
    assert "EntityLoader(inner_dir)" in src or "EntityLoader(agent_dir=inner_dir)" in src


def test_entity_loader_accepts_positional_agent_dir():
    """EntityLoader must accept positional agent_dir (regression for external subgraph blueprint_dir bug)."""
    import tempfile
    from pathlib import Path
    from framework.loader import EntityLoader
    with tempfile.TemporaryDirectory() as tmp:
        # Create minimal entity.json so EntityLoader doesn't fail on missing config
        agent_json = Path(tmp) / "entity.json"
        agent_json.write_text('{"graph": {"nodes": [], "edges": []}}')
        loader = EntityLoader(Path(tmp))
        assert loader is not None
