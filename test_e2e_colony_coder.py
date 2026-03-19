# test_e2e_colony_coder.py
"""E2E tests for Colony Coder with mocked LLM backends."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)


def _ollama_text(content: str) -> dict:
    return {"message": {"role": "assistant", "content": content, "tool_calls": []}}


def _ollama_tool_call(name: str, args: dict) -> dict:
    return {"message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": args}}],
    }}


@pytest.mark.asyncio
async def test_decomposition_validator_flow():
    """Planner validator routing: valid state → __end__ (exits planner subgraph)."""
    from blueprints.functional_graphs.colony_coder_planner.validators import decomposition_validator
    result = decomposition_validator({
        "tasks": [{"id": "t1", "description": "hello", "dependencies": []}],
        "execution_order": ["t1"],
        "retry_count": 0,
    })
    assert result["routing_target"] == "__end__"


@pytest.mark.asyncio
async def test_executor_happy_path_routing():
    """Executor validator chain: pass → execute → __end__ (happy path, no LLM)."""
    from blueprints.functional_graphs.colony_coder_executor.validators import (
        hard_validate, error_classifier,
    )
    state = {"validation_output": {"status": "pass"}, "transient_retry_count": 0, "retry_count": 0}
    hv = hard_validate(state)
    assert hv["routing_target"] == "execute"


@pytest.mark.asyncio
async def test_executor_self_fix_routing():
    """Soft fail → hard_validate → error_classifier → self_fix."""
    from blueprints.functional_graphs.colony_coder_executor.validators import (
        hard_validate, error_classifier,
    )
    fail_state = {
        "validation_output": {"status": "fail", "category": "syntax_error", "severity": "low"},
        "transient_retry_count": 0, "retry_count": 0,
    }
    hv = hard_validate(fail_state)
    assert hv["routing_target"] == "error_classifier"

    ec = error_classifier({**fail_state, **hv})
    assert ec["routing_target"] == "self_fix"
    assert ec["transient_retry_count"] == 1


@pytest.mark.asyncio
async def test_executor_cross_task_routing():
    """Cross-task failure → rescue_router → rollback_state."""
    from blueprints.functional_graphs.colony_coder_executor.validators import (
        hard_validate, error_classifier, rescue_router, rollback_state, cascade_rollback,
    )
    fail_state = {
        "validation_output": {
            "status": "fail", "category": "cross_task",
            "severity": "high", "rationale": "interface broken",
            "affected_scope": "t1,t2",
        },
        "transient_retry_count": 0, "retry_count": 0,
        "cross_task_issues": [], "current_task_id": "t3",
    }
    hv = hard_validate(fail_state)
    assert hv["routing_target"] == "error_classifier"

    ec = error_classifier({**fail_state, **hv})
    assert ec["routing_target"] == "rescue_router"

    rr = rescue_router({**fail_state, **hv, **ec})
    assert rr["routing_target"] == "rollback_state"
    assert len(rr["cross_task_issues"]) == 1
    assert set(rr["affected_task_ids"]) == {"t1", "t2"}

    tasks = [
        {"id": "t1", "dependencies": []},
        {"id": "t2", "dependencies": ["t1"]},
        {"id": "t3", "dependencies": ["t2"]},
    ]
    affected = cascade_rollback(tasks, ["t1"])
    assert affected == {"t1", "t2", "t3"}


@pytest.mark.asyncio
async def test_planner_graph_compiles_e2e():
    import blueprints.functional_graphs.colony_coder.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_planner")).build_graph()
    assert set(g.nodes) - {"__start__"} >= {
        "plan", "design_debate", "claude_swarm", "task_decompose", "decomposition_validator",
    }


@pytest.mark.asyncio
async def test_executor_graph_compiles_e2e():
    import blueprints.functional_graphs.colony_coder.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_executor")).build_graph()
    assert set(g.nodes) - {"__start__"} >= {
        "code_gen", "soft_validate", "self_fix", "apply_patch", "execute",
        "hard_validate", "error_classifier", "rescue_router", "rollback_state", "claude_rescue",
    }


@pytest.mark.asyncio
async def test_integrator_graph_compiles_e2e():
    import blueprints.functional_graphs.colony_coder.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_integrator")).build_graph()
    assert set(g.nodes) - {"__start__"} >= {
        "integration_test", "integration_rescue", "apply_patch", "integration_route",
    }


@pytest.mark.asyncio
async def test_master_graph_compiles_e2e():
    import blueprints.functional_graphs.colony_coder.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder")).build_graph()
    assert set(g.nodes) - {"__start__"} >= {"plan", "execute", "integrate"}
