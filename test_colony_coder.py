import pytest
import tempfile
from pathlib import Path


def _write_validators(tmp_dir: str, content: str):
    Path(tmp_dir, "validators.py").write_text(content)


@pytest.mark.asyncio
async def test_deterministic_node_calls_function():
    with tempfile.TemporaryDirectory() as tmp:
        _write_validators(tmp, "def my_node(state): return {'result': state['x'] + 1}")
        from framework.nodes.deterministic_node import DeterministicNode
        node = DeterministicNode(config={}, node_config={"id": "my_node", "agent_dir": tmp})
        result = await node({"x": 5})
    assert result == {"result": 6}


@pytest.mark.asyncio
async def test_deterministic_node_missing_fn_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _write_validators(tmp, "def other_node(state): return {}")
        from framework.nodes.deterministic_node import DeterministicNode
        with pytest.raises(AttributeError):
            DeterministicNode(config={}, node_config={"id": "missing", "agent_dir": tmp})


def test_deterministic_registered():
    import framework.builtins
    from framework.registry import get_node_factory
    factory = get_node_factory("DETERMINISTIC")
    assert factory is not None
