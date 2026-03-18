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


@pytest.mark.asyncio
async def test_code_execution_success():
    from framework.nodes.external_tool_node import ExternalToolNode
    node = ExternalToolNode(
        config={},
        node_config={"id": "execute", "backend": "code_execution", "timeout": 10},
    )
    result = await node({"execution_command": "echo hello", "working_directory": ""})
    assert result["execution_stdout"].strip() == "hello"
    assert result["execution_returncode"] == 0


@pytest.mark.asyncio
async def test_code_execution_nonzero_exit():
    from framework.nodes.external_tool_node import ExternalToolNode
    node = ExternalToolNode(
        config={},
        node_config={"id": "execute", "backend": "code_execution", "timeout": 10},
    )
    result = await node({"execution_command": "false", "working_directory": ""})
    assert result["execution_returncode"] != 0


def test_tool_registry_has_all_tools():
    from framework.nodes.llm.tools import TOOL_REGISTRY, TOOL_SCHEMAS
    expected = {"read_file", "write_file", "bash_exec", "list_dir", "submit_validation"}
    assert set(TOOL_REGISTRY.keys()) == expected
    assert set(TOOL_SCHEMAS.keys()) == expected


def test_build_tool_schemas_subset():
    from framework.nodes.llm.tools import build_tool_schemas
    schemas = build_tool_schemas(["read_file", "submit_validation"])
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"read_file", "submit_validation"}


def test_submit_validation_has_required_fields():
    from framework.nodes.llm.tools import TOOL_SCHEMAS
    props = TOOL_SCHEMAS["submit_validation"]["function"]["parameters"]["properties"]
    for f in ("status", "category", "severity", "rationale"):
        assert f in props, f"submit_validation missing field: {f}"


from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_ollama_no_tools_uses_base_path():
    """When no tools configured, __call__ is the base class path (no _call_with_tools)."""
    from framework.nodes.llm.ollama import OllamaNode
    node = OllamaNode(config={}, node_config={"id": "code_gen", "model": "qwen3.5:27b"})
    assert node._tools == []


@pytest.mark.asyncio
async def test_ollama_tool_loop_terminates_on_submit_validation():
    """Tool loop ends when submit_validation (_terminal=True) is returned."""
    from framework.nodes.llm.ollama import OllamaNode

    submit_call = {
        "function": {
            "name": "submit_validation",
            "arguments": {
                "status": "pass",
                "category": "correctness",
                "severity": "low",
                "rationale": "looks good",
            },
        }
    }
    tool_call_response = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [submit_call],
        }
    }

    node = OllamaNode(
        config={},
        node_config={"id": "soft_validate", "model": "qwen3.5:27b", "tools": ["submit_validation"]},
    )
    with patch.object(node, "_post_chat", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = tool_call_response
        result = await node._call_with_tools({
            "messages": [{"role": "user", "content": "validate this code"}],
            "node_sessions": {},
            "ollama_sessions": {},
        })

    assert result["validation_output"]["status"] == "pass"
    assert "ollama_sessions" in result


@pytest.mark.asyncio
async def test_ollama_tool_loop_text_response():
    """Tool loop ends on text response with no tool_calls."""
    from framework.nodes.llm.ollama import OllamaNode

    text_response = {
        "message": {"role": "assistant", "content": "here is the code", "tool_calls": []}
    }

    node = OllamaNode(
        config={},
        node_config={"id": "code_gen", "model": "qwen3.5:27b", "tools": ["read_file", "write_file"]},
    )
    with patch.object(node, "_post_chat", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = text_response
        result = await node._call_with_tools({
            "messages": [{"role": "user", "content": "write hello.py"}],
            "node_sessions": {},
            "ollama_sessions": {},
        })

    assert "messages" in result or "ollama_sessions" in result


def test_executor_state_has_required_fields():
    from blueprints.functional_graphs.colony_coder_executor.state import ColonyCoderExecutorState
    import typing
    hints = typing.get_type_hints(ColonyCoderExecutorState, include_extras=True)
    for f in ("tasks", "ollama_sessions", "validation_output", "success", "abort_reason"):
        assert f in hints, f"ColonyCoderExecutorState missing field: {f}"


def test_merge_dict_reducer():
    from blueprints.functional_graphs.colony_coder_executor.state import _merge_dict
    a = {"k1": [1, 2], "k2": [3]}
    b = {"k2": [4], "k3": [5]}
    assert _merge_dict(a, b) == {"k1": [1, 2], "k2": [4], "k3": [5]}


def test_colony_executor_schema_registered():
    from framework.agent_loader import register_state_schema, _get_state_schemas
    # Importing state.py should auto-register the schema
    import blueprints.functional_graphs.colony_coder_executor.state  # noqa: F401
    schemas = _get_state_schemas()
    assert "colony_executor" in schemas
