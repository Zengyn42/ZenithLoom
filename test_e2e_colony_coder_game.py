# test_e2e_colony_coder_game.py
"""E2E test: Colony Coder full chain (Planner → Executor → Integrator)
builds a CLI number guessing game.

Strategy:
  - Planner subgraph runs for real (ClaudeSDKNode.__call__ mocked).
  - Executor and Integrator mocked at the AGENT_REF level
    to avoid the hard_validate→execute loop (execute node doesn't
    reset validation_output, so hard_validate re-routes to execute
    infinitely in the current design).
  - Real file I/O: guess_game.py is created via pathlib in the executor mock
    (simulates the write_file tool that code_gen would invoke).
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

# ── Must import before any graph building to register "colony_coder_schema" ──
import blueprints.functional_graphs.colony_coder.state  # noqa: F401

from framework.agent_loader import AgentLoader
from framework.nodes.llm.claude import ClaudeSDKNode
from framework.nodes.subgraph.subgraph_ref_node import SubgraphRefNode
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Game source code that the "executor" will write
# ---------------------------------------------------------------------------

GAME_CODE = '''\
import random


def play_game():
    """Play one round of the guessing game."""
    number = random.randint(1, 100)
    attempts = 0

    while True:
        try:
            guess = int(input("Guess a number (1-100): "))
        except ValueError:
            print("Please enter a valid number.")
            continue

        attempts += 1

        if guess < number:
            print("Too low!")
        elif guess > number:
            print("Too high!")
        else:
            print(f"Correct! You guessed it in {attempts} attempts.")
            print(f"Score: {attempts}")
            return attempts


def main():
    """Main game loop with replay support."""
    print("Welcome to the Number Guessing Game!")

    while True:
        score = play_game()
        print(f"Your score: {score}")
        again = input("Play again? (y/n): ").strip().lower()
        if again != "y":
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Task decomposition (Planner output)
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "game_core",
        "description": "Create guess_game.py with random number generation, input loop, and hints",
        "dependencies": [],
    },
    {
        "id": "scoring",
        "description": "Add attempt counting and score display",
        "dependencies": ["game_core"],
    },
    {
        "id": "replay",
        "description": "Add play again functionality",
        "dependencies": ["scoring"],
    },
]

EXECUTION_ORDER = ["game_core", "scoring", "replay"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai(text: str) -> AIMessage:
    return AIMessage(content=text)


def _base_return(state: dict, content: str, **extra) -> dict:
    """Build a minimal ClaudeSDKNode-style return dict."""
    result = {
        "messages": [_ai(content)],
        "routing_target": "",
        "routing_context": "",
        "consult_count": 0,
        "subgraph_call_counts": {},
        "rollback_reason": "",
        "retry_count": 0,
        "node_sessions": dict(state.get("node_sessions") or {}),
    }
    result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_colony_coder_game(tmp_path):
    """Full-chain E2E: master graph drives Planner → Executor → Integrator."""

    tmpdir = str(tmp_path)

    # ── 1. ClaudeSDKNode mock (planner's 4 CLAUDE_SDK nodes) ─────────────
    claude_call_count = 0

    async def _mock_claude_call(self, state):
        nonlocal claude_call_count
        claude_call_count += 1
        idx = claude_call_count

        if idx == 1:  # plan
            return _base_return(
                state,
                "High-level plan: Build a CLI number guessing game with "
                "random number generation, input handling, scoring, and replay.",
            )
        elif idx == 2:  # design_debate
            return _base_return(
                state,
                "Risk analysis: Input validation needed. Random range "
                "[1,100] is standard. Edge cases: non-integer input, "
                "immediate correct guess.",
            )
        elif idx == 3:  # claude_swarm
            return _base_return(
                state,
                "Multi-perspective review: Code is simple, maintainable, "
                "and testable. All three reviewers approve. Proceed.",
            )
        elif idx == 4:  # task_decompose
            decomposition = {
                "tasks": TASKS,
                "execution_order": EXECUTION_ORDER,
                "refined_plan": "Build a CLI number guessing game",
                "working_directory": tmpdir,
            }
            return _base_return(
                state,
                json.dumps(decomposition, indent=2),
                # Write parsed fields directly into state so
                # decomposition_validator can read them.
                tasks=TASKS,
                execution_order=EXECUTION_ORDER,
                refined_plan=(
                    "Build a CLI number guessing game with random number "
                    "generation, input loop, hints, scoring, and replay."
                ),
                working_directory=tmpdir,
            )
        else:
            # claude_rescue / integration_rescue — should not be reached
            # in the happy-path test.
            return _base_return(state, "Rescue response (unexpected)")

    # ── 2. SubgraphRefNode mock (planner real, executor/integrator mocked) ──
    _original_subgraph_call = SubgraphRefNode.__call__

    async def _mock_subgraph_call(self, state):
        name = self._loader.name

        if name == "colony_coder_planner":
            # Delegate to real implementation — ClaudeSDKNode.__call__
            # is class-patched so the planner subgraph gets our mocks.
            return await _original_subgraph_call(self, state)

        # ── Shared bookkeeping (mirrors SubgraphRefNode internals) ────
        call_counts = dict(state.get("subgraph_call_counts") or {})
        my_count = call_counts.get(self._node_id, 0)
        call_counts[self._node_id] = my_count + 1

        if name == "colony_coder_executor":
            # Simulate code_gen writing the game file.
            game_path = Path(tmpdir) / "guess_game.py"
            game_path.write_text(GAME_CODE, encoding="utf-8")

            return {
                "completed_tasks": list(EXECUTION_ORDER),
                "final_files": ["guess_game.py"],
                "success": True,
                "abort_reason": None,
                "messages": [_ai(
                    "All 3 tasks executed successfully. "
                    "guess_game.py written to working directory."
                )],
                "subgraph_call_counts": call_counts,
                "consult_count": state.get("consult_count", 0) + 1,
            }

        if name == "colony_coder_integrator":
            return {
                "success": True,
                "abort_reason": None,
                "messages": [_ai(
                    "Integration tests passed. guess_game.py imports "
                    "cleanly, all functions defined, replay loop works."
                )],
                "subgraph_call_counts": call_counts,
                "consult_count": state.get("consult_count", 0) + 1,
            }

        # Unknown subgraph — fall through to original
        return await _original_subgraph_call(self, state)

    # ── 3. Build & run master graph ──────────────────────────────────────
    with (
        patch.object(ClaudeSDKNode, "__call__", _mock_claude_call),
        patch.object(SubgraphRefNode, "__call__", _mock_subgraph_call),
    ):
        loader = AgentLoader(Path("blueprints/functional_graphs/colony_coder"))
        graph = await loader.build_graph(checkpointer=None)

        initial_state = {
            "messages": [HumanMessage(
                content="Build a CLI number guessing game called guess_game.py"
            )],
        }

        # Use astream(updates) to accumulate the full state.
        result: dict = {}
        async for event in graph.astream(initial_state, stream_mode="updates"):
            for node_id, update in event.items():
                if node_id not in ("__start__", "__end__"):
                    result.update(update)

    # ── 4. Assertions ────────────────────────────────────────────────────

    # Success flag
    assert result.get("success") is True, (
        f"Expected success=True, got {result.get('success')!r}"
    )

    # File exists and has expected content
    game_path = Path(tmpdir) / "guess_game.py"
    assert game_path.exists(), f"guess_game.py not found in {tmpdir}"
    game_content = game_path.read_text(encoding="utf-8")
    assert "import random" in game_content
    assert "play_game" in game_content
    assert "def main" in game_content
    assert "Play again" in game_content or "play again" in game_content

    # No abort
    abort = result.get("abort_reason")
    assert not abort, f"Expected no abort_reason, got {abort!r}"

    # Planner was fully exercised (4 Claude SDK calls)
    assert claude_call_count == 4, (
        f"Expected 4 planner Claude calls, got {claude_call_count}"
    )

    # Tasks propagated correctly through state mapping
    assert result.get("completed_tasks") == EXECUTION_ORDER
    assert result.get("final_files") == ["guess_game.py"]


@pytest.mark.asyncio
async def test_planner_decomposition_validator_rejects_empty():
    """Planner validator rejects empty tasks and retries task_decompose."""
    from blueprints.functional_graphs.colony_coder_planner.validators import (
        decomposition_validator,
    )

    # Empty tasks → retry
    r = decomposition_validator({"tasks": [], "execution_order": [], "retry_count": 0})
    assert r["routing_target"] == "task_decompose"
    assert r["retry_count"] == 1

    # After RETRY_CAP → abort
    r = decomposition_validator({"tasks": [], "execution_order": [], "retry_count": 2})
    assert r["routing_target"] == "__end__"
    assert r["success"] is False


@pytest.mark.asyncio
async def test_planner_decomposition_validator_accepts_valid():
    """Planner validator accepts valid decomposition."""
    from blueprints.functional_graphs.colony_coder_planner.validators import (
        decomposition_validator,
    )

    r = decomposition_validator({
        "tasks": TASKS,
        "execution_order": EXECUTION_ORDER,
        "retry_count": 0,
    })
    assert r["routing_target"] == "__end__"


@pytest.mark.asyncio
async def test_executor_happy_path_validators():
    """Executor validators: pass → execute, fail → error routes."""
    from blueprints.functional_graphs.colony_coder_executor.validators import (
        hard_validate,
        error_classifier,
    )

    # Pass → execute
    hv = hard_validate({
        "validation_output": {"status": "pass"},
        "retry_count": 0,
    })
    assert hv["routing_target"] == "execute"

    # Fail (transient) → self_fix
    hv = hard_validate({
        "validation_output": {"status": "fail", "category": "syntax_error", "severity": "low"},
        "retry_count": 0,
    })
    assert hv["routing_target"] == "error_classifier"
    ec = error_classifier({
        "validation_output": {"status": "fail", "category": "syntax_error", "severity": "low"},
        "transient_retry_count": 0,
    })
    assert ec["routing_target"] == "self_fix"

    # Fail (cross_task) → rescue_router
    ec = error_classifier({
        "validation_output": {"status": "fail", "category": "cross_task", "severity": "high"},
        "transient_retry_count": 0,
    })
    assert ec["routing_target"] == "rescue_router"


@pytest.mark.asyncio
async def test_integrator_route_pass():
    """Integrator route: pass → __end__ with success."""
    from blueprints.functional_graphs.colony_coder_integrator.validators import (
        integration_route,
    )

    r = integration_route({
        "validation_output": {"status": "pass"},
        "retry_count": 0,
    })
    assert r["routing_target"] == "__end__"
    assert r["success"] is True


@pytest.mark.asyncio
async def test_integrator_route_fail_retry():
    """Integrator route: fail → rescue then retry."""
    from blueprints.functional_graphs.colony_coder_integrator.validators import (
        integration_route,
    )

    r = integration_route({
        "validation_output": {"status": "fail", "category": "test_failure"},
        "retry_count": 0,
    })
    assert r["routing_target"] == "integration_rescue"
    assert r["retry_count"] == 1


@pytest.mark.asyncio
async def test_game_file_is_syntactically_valid(tmp_path):
    """The game code we inject must be valid Python."""
    game_path = tmp_path / "guess_game.py"
    game_path.write_text(GAME_CODE, encoding="utf-8")
    compile(GAME_CODE, str(game_path), "exec")  # raises SyntaxError if invalid


@pytest.mark.asyncio
async def test_master_graph_node_wiring():
    """Master graph has plan → execute → integrate wiring."""
    loader = AgentLoader(Path("blueprints/functional_graphs/colony_coder"))
    graph = await loader.build_graph(checkpointer=None)
    assert set(graph.nodes) - {"__start__"} >= {"plan", "execute", "integrate"}
