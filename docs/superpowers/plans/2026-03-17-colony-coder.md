# Colony Coder Implementation Plan

> **⚠️ SUPERSEDED (2026-03-22)**
> 本文档是历史规划文档，记录了 Colony Coder 的原始设计。实际实现已做重大调整：
> - Executor 从 10 节点简化为 4 节点 (inject_task_context/code_gen/run_tests/test_route)
> - Master 第三阶段从 `integrate` 改为 `qa` (独立 QA 子图)
> - Master 节点类型从 AGENT_REF 改为 SUBGRAPH_NODE
> - Schema 名称从 `colony_executor` 改为 `colony_coder_schema`
> - 配置文件从 `agent.json` 改为 `entity.json`
> - `_post_chat` 方法已被 `_chat_completions` 替代
>
> **当前架构请参阅: `docs/vault/architecture/colony-coder.md`**

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Colony Coder — a 17-node collaborative coding subgraph using Qwen3.5-27B (Ollama) for local code execution and ApexCoder (Claude SDK) for planning/rescue, organized as three independent LangGraph subgraphs.

**Architecture:** Phase 1 adds four framework primitives (DeterministicNode, ExternalToolNode code_execution backend, OllamaNode tool registry, OllamaNode tool loop). Phases 2-6 build the three subgraphs (planner/executor/integrator) plus a thin master wrapper. Subgraphs are tested standalone before integration.

**Tech Stack:** LangGraph 0.2+, Ollama `/api/chat` (Qwen3.5-27B), Claude SDK (ApexCoder), httpx, pytest-asyncio, Python 3.11+

---

## File Map

### New Files

| File | Responsibility |
|------|----------------|
| `framework/nodes/deterministic_node.py` | Pure-Python validator/router node; loads `validators.py` from blueprint dir by convention |
| `framework/nodes/llm/tools.py` | Tool registry + schemas for OllamaNode (read_file, write_file, bash_exec, list_dir, submit_validation) |
| `blueprints/functional_graphs/colony_coder/entity.json` | Master: 3 AGENT_REF nodes (plan → execute → integrate) |
| `blueprints/functional_graphs/colony_coder_planner/entity.json` | Planner: 5 nodes (plan, design_debate, claude_swarm, task_decompose, decomposition_validator) |
| `blueprints/functional_graphs/colony_coder_planner/validators.py` | `decomposition_validator(state)` |
| `blueprints/functional_graphs/colony_coder_planner/system_prompt.md` | Shared planner system prompt (stub) |
| `blueprints/functional_graphs/colony_coder_executor/entity.json` | Executor: 9 nodes |
| `blueprints/functional_graphs/colony_coder_executor/state.py` | `ColonyCoderExecutorState` with merge-dict ollama_sessions |
| `blueprints/functional_graphs/colony_coder_executor/validators.py` | hard_validate, error_classifier, rescue_router, rollback_state |
| `blueprints/functional_graphs/colony_coder_executor/system_prompt.md` | Code gen system prompt (stub) |
| `blueprints/functional_graphs/colony_coder_integrator/entity.json` | Integrator: 4 nodes |
| `blueprints/functional_graphs/colony_coder_integrator/validators.py` | `integration_route(state)` |
| `blueprints/functional_graphs/colony_coder_integrator/system_prompt.md` | Integration test system prompt (stub) |
| `test_colony_coder.py` | Unit tests: framework components + validators |
| `test_e2e_colony_coder.py` | E2E: full graph with mocked Ollama/Claude |

### Modified Files

| File | Change |
|------|--------|
| `framework/nodes/external_tool_node.py` | Add `code_execution` backend: reads `execution_command` from state, runs as subprocess |
| `framework/nodes/llm/ollama.py` | Add `_tools` list + `_call_with_tools` + `_post_chat` methods |
| `framework/agent_loader.py` | (1) Add `blueprint_dir` param to `_build_declarative`; (2) inject `agent_dir` for DETERMINISTIC nodes; (3) make `_STATE_SCHEMAS` module-level; (4) register `colony_executor` |
| `framework/builtins.py` | Register `DETERMINISTIC` node type |

---

## Phase 1: Framework Primitives

### Task 1: DeterministicNode

**Files:**
- Create: `framework/nodes/deterministic_node.py`
- Modify: `framework/builtins.py`
- Modify: `framework/agent_loader.py` (lines ~384, ~196, ~277, ~395, ~401)
- Test: `test_colony_coder.py`

- [ ] **Step 1: Write the failing tests**

```python
# test_colony_coder.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/kingy/Foundation/ZenithLoom
pytest test_colony_coder.py::test_deterministic_node_calls_function -v
```
Expected: `ModuleNotFoundError: No module named 'framework.nodes.deterministic_node'`

- [ ] **Step 3: Create `framework/nodes/deterministic_node.py`**

```python
"""DeterministicNode — DETERMINISTIC node type.

Wraps a pure Python function from {blueprint_dir}/validators.py as a LangGraph node.
Convention: the function name must match the node's id in entity.json.

node_config fields (injected by _build_declarative):
  id          str  required  Node id — used to look up function in validators.py
  agent_dir   str  required  Blueprint directory containing validators.py
"""

import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_validators(agent_dir: str):
    path = Path(agent_dir) / "validators.py"
    if not path.exists():
        raise FileNotFoundError(f"DeterministicNode: validators.py not found at {path}")
    spec = importlib.util.spec_from_file_location("_validators", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeterministicNode:
    """
    DETERMINISTIC node: calls a pure Python function from validators.py.

    The function must be synchronous: (state: dict) -> dict.
    No LLM calls, no I/O — routing and validation logic only.
    """

    def __init__(self, config, node_config: dict):
        node_id = node_config["id"]
        agent_dir = node_config["agent_dir"]
        module = _load_validators(agent_dir)
        self._fn = getattr(module, node_id)
        logger.debug(f"[deterministic] loaded {node_id!r} from {agent_dir}")

    async def __call__(self, state: dict) -> dict:
        return self._fn(state)
```

- [ ] **Step 4: Register DETERMINISTIC in `framework/builtins.py`**

Add after the last `@register_node` block:

```python
@register_node("DETERMINISTIC")
def _(config, node_config):
    from framework.nodes.deterministic_node import DeterministicNode
    return DeterministicNode(config, node_config)
```

- [ ] **Step 5: Modify `framework/agent_loader.py` — make `_STATE_SCHEMAS` module-level and register `colony_executor`**

Find the local `_STATE_SCHEMAS` dict inside `_build_declarative` (around line 384):

```python
    # ── Step 2: 构建图节点 ────────────────────────────────────────────────
    _STATE_SCHEMAS = {"debate": DebateState, "base": BaseAgentState}
    state_schema = _STATE_SCHEMAS.get(graph_spec.get("state_schema", "base"), BaseAgentState)
```

Extract `_STATE_SCHEMAS` to **module level** (before `_build_declarative`), and expose a registration helper:

```python
# Module-level state schema registry (add custom schemas before build_graph is called)
_STATE_SCHEMAS: dict = {}  # populated after imports resolve; see _get_state_schemas()


def _get_state_schemas() -> dict:
    """Lazy-load default schemas + any registered custom schemas."""
    from framework.state import BaseAgentState, DebateState
    base = {"debate": DebateState, "base": BaseAgentState}
    return {**base, **_STATE_SCHEMAS}


def register_state_schema(name: str, cls):
    """Register a custom TypedDict as a named state schema for declarative graphs."""
    _STATE_SCHEMAS[name] = cls
```

Then inside `_build_declarative`, replace the two-line block with:

```python
    state_schema = _get_state_schemas().get(graph_spec.get("state_schema", "base"), BaseAgentState)
```

(Remove the old import of `BaseAgentState, DebateState` from inside `_build_declarative` — they're now in `_get_state_schemas`.)

- [ ] **Step 6: Modify `framework/agent_loader.py` — inject `agent_dir` for DETERMINISTIC nodes**

Add `blueprint_dir: str = ""` parameter to `_build_declarative`:

```python
async def _build_declarative(
    graph_spec: dict,
    config: AgentConfig,
    system_prompt: str,
    checkpointer,
    blueprint_dir: str = "",          # ← add this
) -> object:
```

Inside the node-building loop, replace:

```python
            effective_def = (
                {**node_def, "system_prompt": system_prompt}
                if system_prompt and "main" in node_id
                else node_def
            )
```

with:

```python
            _base = (
                {**node_def, "system_prompt": system_prompt}
                if system_prompt and "main" in node_id
                else dict(node_def)
            )
            if node_type == "DETERMINISTIC" and blueprint_dir:
                _base["agent_dir"] = blueprint_dir
            effective_def = _base
```

Update **all three call sites** to pass `blueprint_dir`:

1. In `EntityLoader.build_graph()` (~line 196):
```python
            return await _build_declarative(
                graph_dict, config, system_prompt, checkpointer,
                blueprint_dir=str(self._dir),    # ← add
            )
```

2. In `EntityLoader.build_heartbeat_graph()` (~line 277):
```python
        graph = await _build_declarative(
            graph_spec, config, system_prompt="", checkpointer=None,
            blueprint_dir=str(self._dir),    # ← add
        )
```

3. In the recursive SUBGRAPH branch inside `_build_declarative` (~line 394):
```python
            inner = await _build_declarative(
                node_def["graph"], config, system_prompt, None,
                blueprint_dir=blueprint_dir,    # ← add (pass through)
            )
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest test_colony_coder.py::test_deterministic_node_calls_function \
       test_colony_coder.py::test_deterministic_node_missing_fn_raises \
       test_colony_coder.py::test_deterministic_registered -v
```
Expected: 3 PASSED

- [ ] **Step 8: Verify no regressions**

```bash
pytest test_cli.py test_e2e_debate.py -v
```
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add framework/nodes/deterministic_node.py framework/builtins.py \
        framework/agent_loader.py test_colony_coder.py
git commit -m "feat: add DeterministicNode + DETERMINISTIC registry + agent_dir injection"
```

---

### Task 2: ExternalToolNode `code_execution` Backend

**Files:**
- Modify: `framework/nodes/external_tool_node.py`
- Test: `test_colony_coder.py`

Read `framework/nodes/external_tool_node.py` first to understand the current `__init__` guard and `_run_subprocess` helper.

- [ ] **Step 1: Write the failing tests**

```python
# Add to test_colony_coder.py

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test_colony_coder.py::test_code_execution_success -v
```
Expected: Error — `ExternalToolNode: 'command' must be a non-empty list` (no `command` field when `backend=code_execution`)

- [ ] **Step 3: Modify `ExternalToolNode.__init__` to allow `code_execution` backend**

```python
    def __init__(self, config: AgentConfig, node_config: dict) -> None:
        inner = node_config.get("node_config", node_config)
        self._backend: str = inner.get("backend", "cli")
        raw = inner.get("command")
        if self._backend != "code_execution" and not raw:
            raise ValueError("ExternalToolNode: 'command' must be a non-empty list")
        self._command: list[str] = shlex.split(raw) if isinstance(raw, str) else list(raw or [])
        self._timeout = float(inner.get("timeout", 30.0))
        self._inject_as: str = inner.get("inject_as", "message")
        self._description: str = inner.get("description", "")
```

- [ ] **Step 4: Add `_run_code_execution` and update `__call__`**

Add a new helper following the same `asyncio.to_thread` + `subprocess.run` pattern as `_run_subprocess`:

```python
def _run_code_exec(cmd: list[str], timeout: float, cwd: str | None) -> subprocess.CompletedProcess:
    """Run cmd (arg-list, no shell) with timeout and optional cwd."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd or None,
    )
```

Add `_run_code_execution` instance method:

```python
    async def _run_code_execution(self, state: dict) -> dict:
        cmd_str: str = state.get("execution_command", "")
        working_dir: str = state.get("working_directory") or ""
        cmd = shlex.split(cmd_str)
        cwd = working_dir if working_dir else None
        try:
            result = await asyncio.to_thread(_run_code_exec, cmd, self._timeout, cwd)
        except subprocess.TimeoutExpired:
            return {
                "execution_stdout": "",
                "execution_stderr": f"timeout after {self._timeout}s",
                "execution_returncode": -1,
            }
        return {
            "execution_stdout": result.stdout,
            "execution_stderr": result.stderr,
            "execution_returncode": result.returncode,
        }
```

Update `__call__` to dispatch:

```python
    async def __call__(self, state: dict) -> dict:
        if self._backend == "code_execution":
            return await self._run_code_execution(state)
        # ... existing CLI logic unchanged ...
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest test_colony_coder.py::test_code_execution_success \
       test_colony_coder.py::test_code_execution_nonzero_exit -v
```
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add framework/nodes/external_tool_node.py test_colony_coder.py
git commit -m "feat: add code_execution backend to ExternalToolNode"
```

---

### Task 3: OllamaNode Tool Registry

**Files:**
- Create: `framework/nodes/llm/tools.py`
- Test: `test_colony_coder.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to test_colony_coder.py

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test_colony_coder.py::test_tool_registry_has_all_tools -v
```
Expected: `ModuleNotFoundError: No module named 'framework.nodes.llm.tools'`

- [ ] **Step 3: Create `framework/nodes/llm/tools.py`**

```python
"""Tool registry for OllamaNode tool-calling loop.

Tools:
  read_file          — read a file from the filesystem
  write_file         — write content to a file
  bash_exec          — run a command (arg-list form via shlex.split, no shell)
  list_dir           — list directory entries
  submit_validation  — structured validation output (terminates the tool loop)

TOOL_REGISTRY   dict[name → async callable]
TOOL_SCHEMAS    dict[name → OpenAI-style function schema]
build_tool_schemas(names) → list of schemas for use in /api/chat payload
"""

import asyncio
import json
import shlex
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def read_file(path: str) -> dict:
    try:
        return {"content": Path(path).read_text(encoding="utf-8")}
    except Exception as exc:
        return {"error": str(exc)}


async def write_file(path: str, content: str) -> dict:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return {"written": True}
    except Exception as exc:
        return {"error": str(exc)}


def _exec_cmd(args: list[str], timeout: int) -> dict:
    """Synchronous helper: run args (no shell) and capture output."""
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"timeout after {timeout}s", "returncode": -1}
    except FileNotFoundError as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": 127}


async def bash_exec(command: str, timeout: int = 30) -> dict:
    """Execute a command by splitting it with shlex (no shell expansion)."""
    args = shlex.split(command)
    return await asyncio.to_thread(_exec_cmd, args, timeout)


async def list_dir(path: str) -> dict:
    try:
        entries = sorted(str(p) for p in Path(path).iterdir())
        return {"entries": entries}
    except Exception as exc:
        return {"error": str(exc)}


async def submit_validation(
    status: str,
    category: str,
    severity: str,
    rationale: str,
    affected_scope: str = "",
    is_regression: bool = False,
    raw_stderr: str = "",
) -> dict:
    """Structured validation output — signals the tool loop to terminate."""
    return {
        "status": status,
        "category": category,
        "severity": severity,
        "rationale": rationale,
        "affected_scope": affected_scope,
        "is_regression": is_regression,
        "raw_stderr": raw_stderr,
        "_terminal": True,
    }


# ---------------------------------------------------------------------------
# Registry + schemas
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict = {
    "read_file": read_file,
    "write_file": write_file,
    "bash_exec": bash_exec,
    "list_dir": list_dir,
    "submit_validation": submit_validation,
}

TOOL_SCHEMAS: dict = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute or relative file path"}},
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates parent dirs if needed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "bash_exec": {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Execute a shell command and return stdout/stderr/returncode",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    "list_dir": {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List all entries in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "submit_validation": {
        "type": "function",
        "function": {
            "name": "submit_validation",
            "description": "Submit a structured validation result (terminates the tool loop)",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["pass", "fail", "abort"]},
                    "category": {"type": "string", "description": "Error category (e.g. syntax_error, cross_task)"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "rationale": {"type": "string", "description": "Human-readable explanation"},
                    "affected_scope": {"type": "string", "description": "Comma-separated task IDs affected"},
                    "is_regression": {"type": "boolean"},
                    "raw_stderr": {"type": "string"},
                },
                "required": ["status", "category", "severity", "rationale"],
            },
        },
    },
}


def build_tool_schemas(tool_names: list) -> list:
    """Return a list of tool schemas for the given tool names (order preserved)."""
    return [TOOL_SCHEMAS[n] for n in tool_names if n in TOOL_SCHEMAS]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test_colony_coder.py::test_tool_registry_has_all_tools \
       test_colony_coder.py::test_build_tool_schemas_subset \
       test_colony_coder.py::test_submit_validation_has_required_fields -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add framework/nodes/llm/tools.py test_colony_coder.py
git commit -m "feat: add OllamaNode tool registry (tools.py)"
```

---

### Task 4: OllamaNode Tool-Calling Loop

**Files:**
- Modify: `framework/nodes/llm/ollama.py`
- Test: `test_colony_coder.py`

Read `framework/nodes/llm/ollama.py` first — understand `__init__` and `call_llm` before modifying.

- [ ] **Step 1: Write the failing tests**

```python
# Add to test_colony_coder.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test_colony_coder.py::test_ollama_tool_loop_terminates_on_submit_validation -v
```
Expected: `AttributeError: 'OllamaNode' object has no attribute '_tools'`

- [ ] **Step 3: Add `_tools` and `_node_config` to `OllamaNode.__init__`**

In `OllamaNode.__init__`, after `self._options = ...`, add:

```python
        self._tools: list = node_config.get("tools", [])
        self._node_config = node_config  # used by _call_with_tools for session_key/id lookup
```

- [ ] **Step 4: Override `__call__` to dispatch to tool loop**

In `OllamaNode`, add:

```python
    async def __call__(self, state: dict) -> dict:
        if self._tools:
            return await self._call_with_tools(state)
        return await super().__call__(state)
```

- [ ] **Step 5: Add `_post_chat`**

```python
    async def _post_chat(self, messages: list, tools: list | None = None) -> dict:
        """Non-streaming POST to /api/chat. Returns parsed response dict."""
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._endpoint}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
```

- [ ] **Step 6: Add `_call_with_tools`**

```python
    async def _call_with_tools(self, state: dict) -> dict:
        """Multi-turn Ollama tool-calling loop.

        Session management:
          - ollama_sessions: dict[uuid → messages list] (merged into state, not overwritten)
          - node_sessions:   dict[session_key → uuid]   (maps logical key to session uuid)
          - Max 200 messages per session (prune oldest non-system messages).
        """
        import uuid as _uuid
        from framework.nodes.llm.tools import TOOL_REGISTRY, build_tool_schemas

        tool_schemas = build_tool_schemas(self._tools)
        session_key = self._node_config.get("session_key", self._node_config.get("id", ""))

        # Load or init session messages
        ollama_sessions: dict = dict(state.get("ollama_sessions") or {})
        node_sessions: dict = dict(state.get("node_sessions") or {})
        session_uuid = node_sessions.get(session_key)
        messages: list = list(ollama_sessions.get(session_uuid, [])) if session_uuid else []

        # System prompt
        if self._system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self._system_prompt})

        # Append current user message
        lm = (state.get("messages") or [])
        if lm:
            last = lm[-1]
            content = getattr(last, "content", None) or (last.get("content", "") if isinstance(last, dict) else "")
            if content:
                messages.append({"role": "user", "content": content})

        MAX_MESSAGES = 200
        MAX_ITERATIONS = 10
        terminal_result = None
        last_msg = {}

        for _ in range(MAX_ITERATIONS):
            response = await self._post_chat(messages, tools=tool_schemas)
            last_msg = response.get("message", {})
            messages.append(last_msg)

            tool_calls = last_msg.get("tool_calls") or []
            if not tool_calls:
                break  # text response — loop ends

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"].get("arguments", {})
                if isinstance(fn_args, str):
                    import json as _json
                    fn_args = _json.loads(fn_args)

                tool_fn = TOOL_REGISTRY.get(fn_name)
                tool_result = await tool_fn(**fn_args) if tool_fn else {"error": f"unknown tool: {fn_name}"}

                messages.append({"role": "tool", "content": str(tool_result)})

                if tool_result.get("_terminal"):
                    terminal_result = {k: v for k, v in tool_result.items() if k != "_terminal"}
                    break

            if terminal_result:
                break

        # Prune session to MAX_MESSAGES
        if len(messages) > MAX_MESSAGES:
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            non_sys  = [m for m in messages if m.get("role") != "system"]
            messages = sys_msgs + non_sys[-(MAX_MESSAGES - len(sys_msgs)):]

        # Persist session
        if not session_uuid:
            session_uuid = str(_uuid.uuid4())
        node_sessions[session_key] = session_uuid
        ollama_sessions[session_uuid] = messages

        updates: dict = {
            "node_sessions": node_sessions,
            "ollama_sessions": ollama_sessions,
        }
        if terminal_result:
            updates["validation_output"] = terminal_result
        else:
            text = last_msg.get("content", "")
            if text:
                from langchain_core.messages import AIMessage
                updates["messages"] = [AIMessage(content=text)]

        return updates
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest test_colony_coder.py::test_ollama_no_tools_uses_base_path \
       test_colony_coder.py::test_ollama_tool_loop_terminates_on_submit_validation \
       test_colony_coder.py::test_ollama_tool_loop_text_response -v
```
Expected: 3 PASSED

- [ ] **Step 8: Verify no regressions**

```bash
pytest test_cli.py test_e2e_debate.py -v
```
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add framework/nodes/llm/ollama.py test_colony_coder.py
git commit -m "feat: add tool-calling loop to OllamaNode (_call_with_tools, _post_chat)"
```

---

## Phase 2: State Schema

### Task 5: ColonyCoderExecutorState

**Files:**
- Create: `blueprints/functional_graphs/colony_coder_executor/__init__.py`
- Create: `blueprints/functional_graphs/colony_coder_executor/state.py`
- Test: `test_colony_coder.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to test_colony_coder.py

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test_colony_coder.py::test_executor_state_has_required_fields -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `blueprints/functional_graphs/colony_coder_executor/__init__.py`**

Empty file.

- [ ] **Step 4: Create `blueprints/functional_graphs/colony_coder_executor/state.py`**

```python
"""ColonyCoderExecutorState — LangGraph state schema for colony_coder_executor.

Auto-registers as "colony_executor" schema on import.
"""

from __future__ import annotations

from typing import Annotated, Optional

from framework.state import BaseAgentState
from framework.agent_loader import register_state_schema


def _merge_dict(a: dict, b: dict) -> dict:
    """Merge reducer: b's values overwrite a's for shared keys. Safe for parallel node writes."""
    return {**a, **b}


class ColonyCoderExecutorState(BaseAgentState):
    # Task management
    tasks: list                        # list of {"id", "description", "dependencies"}
    execution_order: list              # ordered list of task ids
    refined_plan: str
    working_directory: str
    current_task_index: int
    current_task_id: str
    retry_count: int
    transient_retry_count: int
    error_history: list
    completed_tasks: list

    # Cross-task issues accumulate across tasks
    cross_task_issues: list

    # Validation output from soft_validate / hard_validate
    validation_output: Optional[dict]

    # Rescue context
    rescue_scope: str
    rescue_rationale: str
    affected_task_ids: list

    # Ollama sessions stored in state (not files) for LangGraph checkpoint compatibility.
    # merge_dict reducer prevents parallel node writes from clobbering each other.
    ollama_sessions: Annotated[dict, _merge_dict]

    # Final output
    final_files: list
    abort_reason: Optional[str]
    success: bool


# Auto-register on import
register_state_schema("colony_executor", ColonyCoderExecutorState)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest test_colony_coder.py::test_executor_state_has_required_fields \
       test_colony_coder.py::test_merge_dict_reducer \
       test_colony_coder.py::test_colony_executor_schema_registered -v
```
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add blueprints/functional_graphs/colony_coder_executor/ test_colony_coder.py
git commit -m "feat: add ColonyCoderExecutorState with merge-dict ollama_sessions"
```

---

## Phase 3: Planner Subgraph

### Task 6: colony_coder_planner

**Files:**
- Create: `blueprints/functional_graphs/colony_coder_planner/__init__.py`
- Create: `blueprints/functional_graphs/colony_coder_planner/validators.py`
- Create: `blueprints/functional_graphs/colony_coder_planner/entity.json`
- Create: `blueprints/functional_graphs/colony_coder_planner/system_prompt.md`
- Test: `test_colony_coder.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to test_colony_coder.py

def test_decomposition_validator_pass():
    from blueprints.functional_graphs.colony_coder_planner.validators import decomposition_validator
    result = decomposition_validator({
        "tasks": [{"id": "t1", "description": "write hello.py", "dependencies": []}],
        "execution_order": ["t1"],
        "retry_count": 0,
    })
    # "__end__" exits the planner subgraph; master graph routes to executor via fixed edge
    assert result["routing_target"] == "__end__"


def test_decomposition_validator_fail_retry():
    from blueprints.functional_graphs.colony_coder_planner.validators import decomposition_validator
    result = decomposition_validator({
        "tasks": [],
        "execution_order": [],
        "retry_count": 1,
    })
    assert result["routing_target"] == "task_decompose"
    assert result["retry_count"] == 2


def test_decomposition_validator_abort_at_cap():
    from blueprints.functional_graphs.colony_coder_planner.validators import decomposition_validator
    result = decomposition_validator({
        "tasks": [],
        "execution_order": [],
        "retry_count": 2,
    })
    assert result["routing_target"] == "__end__"
    assert result["success"] is False


@pytest.mark.asyncio
async def test_planner_graph_compiles():
    import blueprints.functional_graphs.colony_coder_executor.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    from pathlib import Path
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_planner")).build_graph()
    node_ids = set(g.nodes) - {"__start__"}
    required = {"plan", "design_debate", "claude_swarm", "task_decompose", "decomposition_validator"}
    assert required <= node_ids, f"Missing nodes: {required - node_ids}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test_colony_coder.py::test_decomposition_validator_pass -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `blueprints/functional_graphs/colony_coder_planner/__init__.py`**

Empty file.

- [ ] **Step 4: Create `blueprints/functional_graphs/colony_coder_planner/validators.py`**

```python
"""Deterministic validator nodes for colony_coder_planner.

Node: decomposition_validator
  Routes to: execute (pass) | task_decompose (retry) | __end__ (abort)
"""

RETRY_CAP = 2


def decomposition_validator(state: dict) -> dict:
    """Validate task decomposition output from task_decompose.

    Valid if: tasks non-empty, execution_order non-empty, all order IDs exist in tasks.
    """
    tasks = state.get("tasks") or []
    execution_order = state.get("execution_order") or []
    retry_count = state.get("retry_count", 0)

    def _is_valid() -> bool:
        if not tasks or not execution_order:
            return False
        task_ids = {t["id"] for t in tasks if isinstance(t, dict) and "id" in t}
        return bool(task_ids) and all(oid in task_ids for oid in execution_order)

    if _is_valid():
        # "__end__" exits the planner subgraph; master graph continues to executor via fixed edge
        return {"routing_target": "__end__"}

    if retry_count >= RETRY_CAP:
        return {
            "routing_target": "__end__",
            "success": False,
            "abort_reason": "decomposition_failed_after_retries",
        }

    return {"routing_target": "task_decompose", "retry_count": retry_count + 1}
```

- [ ] **Step 5: Create `blueprints/functional_graphs/colony_coder_planner/entity.json`**

Note: Uses `"colony_executor"` schema so internal nodes can write `tasks`, `execution_order`,
`refined_plan`, `working_directory`, and `retry_count` into state. `BaseAgentState` lacks these
fields and LangGraph channels are defined per-schema — unknown keys are dropped. Import
`blueprints.functional_graphs.colony_coder_executor.state` before calling `build_graph()`.

```json
{
  "name": "colony_coder_planner",
  "persona_files": [],
  "graph": {
    "state_schema": "colony_executor",
    "nodes": [
      {
        "id": "plan",
        "type": "CLAUDE_SDK",
        "session_key": "session_a",
        "system_prompt": "You are the Colony Coder planning layer. Analyze the coding task and produce an initial technical approach."
      },
      {
        "id": "design_debate",
        "type": "CLAUDE_SDK",
        "session_key": "session_a",
        "system_prompt": "You are reviewing the technical plan. Identify risks, gaps, and alternative approaches."
      },
      {
        "id": "claude_swarm",
        "type": "CLAUDE_SDK",
        "session_key": "session_a",
        "system_prompt": "You are a multi-perspective reviewer. Evaluate the plan from three angles: correctness, maintainability, and testability. Synthesize a final recommendation."
      },
      {
        "id": "task_decompose",
        "type": "CLAUDE_SDK",
        "session_key": "session_a",
        "system_prompt": "Decompose the approved plan into atomic coding tasks. Output JSON with: tasks (list of {id, description, dependencies}), execution_order (list of task ids), refined_plan (prose summary), working_directory (absolute path)."
      },
      {
        "id": "decomposition_validator",
        "type": "DETERMINISTIC"
      }
    ],
    "edges": [
      {"from": "__start__",               "to": "plan"},
      {"from": "plan",                    "to": "design_debate"},
      {"from": "design_debate",           "to": "claude_swarm"},
      {"from": "claude_swarm",            "to": "task_decompose"},
      {"from": "task_decompose",          "to": "decomposition_validator"},
      {"from": "decomposition_validator", "to": "task_decompose", "type": "routing_to"},
      {"from": "decomposition_validator", "to": "__end__",        "type": "routing_to"}
    ]
  }
}
```

- [ ] **Step 6: Create `blueprints/functional_graphs/colony_coder_planner/system_prompt.md`** (stub)

```markdown
# Colony Coder — Planner

You are the planning layer of Colony Coder.
Your goal is to analyze a coding task and produce a structured, atomic task decomposition.
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest test_colony_coder.py::test_decomposition_validator_pass \
       test_colony_coder.py::test_decomposition_validator_fail_retry \
       test_colony_coder.py::test_decomposition_validator_abort_at_cap -v
```
Expected: 3 PASSED

- [ ] **Step 8: Run compile test**

```bash
pytest test_colony_coder.py::test_planner_graph_compiles -v
```
Expected: PASS. If it fails with routing edge error, check entity.json edge type — the `"routing_to"` edge needs a matching node in the graph. Adjust if needed.

- [ ] **Step 9: Commit**

```bash
git add blueprints/functional_graphs/colony_coder_planner/ test_colony_coder.py
git commit -m "feat: add colony_coder_planner subgraph"
```

---

## Phase 4: Executor Subgraph

### Task 7: Executor Skeleton (OLLAMA + EXTERNAL_TOOL nodes)

**Files:**
- Create: `blueprints/functional_graphs/colony_coder_executor/entity.json`
- Create: `blueprints/functional_graphs/colony_coder_executor/system_prompt.md`

Note: validators.py and claude_rescue come in Task 8. This task creates the skeleton.

- [ ] **Step 1: Create `blueprints/functional_graphs/colony_coder_executor/system_prompt.md`** (stub)

```markdown
# Colony Coder — Executor

You are a code generation agent. For the current task, write clean, working code.
Use the provided tools to read context, write files, and run validation.
```

- [ ] **Step 2: Create `blueprints/functional_graphs/colony_coder_executor/entity.json`**

```json
{
  "name": "colony_coder_executor",
  "persona_files": [],
  "graph": {
    "state_schema": "colony_executor",
    "nodes": [
      {
        "id": "code_gen",
        "type": "OLLAMA",
        "model": "qwen3.5:27b",
        "tools": ["read_file", "write_file", "list_dir"],
        "session_key": "executor_session"
      },
      {
        "id": "soft_validate",
        "type": "OLLAMA",
        "model": "qwen3.5:27b",
        "tools": ["read_file", "bash_exec", "submit_validation"],
        "session_key": "executor_session"
      },
      {
        "id": "self_fix",
        "type": "OLLAMA",
        "model": "qwen3.5:27b",
        "tools": ["read_file", "write_file", "bash_exec"],
        "session_key": "executor_session"
      },
      {
        "id": "apply_patch",
        "type": "EXTERNAL_TOOL",
        "backend": "code_execution",
        "timeout": 30
      },
      {
        "id": "execute",
        "type": "EXTERNAL_TOOL",
        "backend": "code_execution",
        "timeout": 60
      },
      {
        "id": "hard_validate",
        "type": "DETERMINISTIC"
      },
      {
        "id": "error_classifier",
        "type": "DETERMINISTIC"
      },
      {
        "id": "rescue_router",
        "type": "DETERMINISTIC"
      },
      {
        "id": "rollback_state",
        "type": "DETERMINISTIC"
      },
      {
        "id": "claude_rescue",
        "type": "CLAUDE_SDK",
        "session_key": "session_b",
        "system_prompt": "You are ApexCoder, a rescue agent. Analyze the error and produce a corrected implementation."
      }
    ],
    "edges": [
      {"from": "__start__",        "to": "code_gen"},
      {"from": "code_gen",         "to": "apply_patch"},
      {"from": "apply_patch",      "to": "soft_validate"},
      {"from": "soft_validate",    "to": "hard_validate"},
      {"from": "execute",          "to": "hard_validate"},
      {"from": "self_fix",         "to": "apply_patch"},
      {"from": "claude_rescue",    "to": "apply_patch"},
      {"from": "hard_validate",    "to": "execute",          "type": "routing_to"},
      {"from": "hard_validate",    "to": "error_classifier", "type": "routing_to"},
      {"from": "hard_validate",    "to": "__end__",           "type": "routing_to"},
      {"from": "error_classifier", "to": "self_fix",          "type": "routing_to"},
      {"from": "error_classifier", "to": "claude_rescue",     "type": "routing_to"},
      {"from": "error_classifier", "to": "rescue_router",     "type": "routing_to"},
      {"from": "rescue_router",    "to": "rollback_state",    "type": "routing_to"},
      {"from": "rollback_state",   "to": "claude_rescue",     "type": "routing_to"}
    ]
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add blueprints/functional_graphs/colony_coder_executor/
git commit -m "feat: add colony_coder_executor entity.json skeleton"
```

---

### Task 8: Executor Validators

**Files:**
- Create: `blueprints/functional_graphs/colony_coder_executor/validators.py`
- Test: `test_colony_coder.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to test_colony_coder.py

def test_hard_validate_pass():
    from blueprints.functional_graphs.colony_coder_executor.validators import hard_validate
    result = hard_validate({
        "validation_output": {"status": "pass", "severity": "low"},
        "transient_retry_count": 0, "retry_count": 0,
    })
    assert result["routing_target"] == "execute"


def test_hard_validate_routes_to_error_classifier():
    from blueprints.functional_graphs.colony_coder_executor.validators import hard_validate
    result = hard_validate({
        "validation_output": {"status": "fail", "severity": "medium"},
        "transient_retry_count": 0, "retry_count": 0,
    })
    assert result["routing_target"] == "error_classifier"


def test_hard_validate_abort_at_cap():
    from blueprints.functional_graphs.colony_coder_executor.validators import hard_validate
    result = hard_validate({
        "validation_output": {"status": "fail", "severity": "high"},
        "transient_retry_count": 0, "retry_count": 3,
    })
    assert result["routing_target"] == "__end__"
    assert result["success"] is False


def test_error_classifier_transient():
    from blueprints.functional_graphs.colony_coder_executor.validators import error_classifier
    result = error_classifier({
        "validation_output": {"category": "syntax_error", "severity": "low"},
        "transient_retry_count": 0,
    })
    assert result["routing_target"] == "self_fix"
    assert result["transient_retry_count"] == 1


def test_error_classifier_escalates_to_claude():
    from blueprints.functional_graphs.colony_coder_executor.validators import error_classifier
    result = error_classifier({
        "validation_output": {"category": "syntax_error", "severity": "low"},
        "transient_retry_count": 2,  # at TRANSIENT_RETRY_CAP
    })
    assert result["routing_target"] == "claude_rescue"


def test_rescue_router_dual_write():
    from blueprints.functional_graphs.colony_coder_executor.validators import rescue_router
    result = rescue_router({
        "validation_output": {
            "status": "fail", "category": "cross_task",
            "severity": "high", "rationale": "shared interface broken",
            "affected_scope": "t1,t2",
        },
        "cross_task_issues": [],
        "current_task_id": "t3",
    })
    assert result["routing_target"] == "rollback_state"
    assert len(result["cross_task_issues"]) == 1
    assert result["affected_task_ids"] == ["t1", "t2"]


def test_cascade_rollback_transitive():
    from blueprints.functional_graphs.colony_coder_executor.validators import cascade_rollback
    tasks = [
        {"id": "t1", "dependencies": []},
        {"id": "t2", "dependencies": ["t1"]},
        {"id": "t3", "dependencies": ["t2"]},
        {"id": "t4", "dependencies": []},
    ]
    affected = cascade_rollback(tasks, ["t1"])
    assert "t2" in affected
    assert "t3" in affected
    assert "t4" not in affected
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test_colony_coder.py::test_hard_validate_pass -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `blueprints/functional_graphs/colony_coder_executor/validators.py`**

```python
"""Deterministic validator nodes for colony_coder_executor.

Nodes (each function = one DETERMINISTIC node):
  hard_validate     — routes pass/fail/abort based on validation_output
  error_classifier  — classifies error: self_fix vs claude_rescue
  rescue_router     — cross-task issues: dual-write to cross_task_issues + routing_target
  rollback_state    — cascade-rollback affected tasks, route to claude_rescue

Helper (not a node):
  cascade_rollback  — transitively find all tasks dependent on affected_task_ids
"""

RETRY_CAP = 3
TRANSIENT_RETRY_CAP = 2
TRANSIENT_CATEGORIES = {"syntax_error", "import_error", "test_failure", "lint_error"}
CROSS_TASK_CATEGORIES = {"cross_task", "interface_mismatch", "dependency_break"}


def hard_validate(state: dict) -> dict:
    """Route based on validation_output.status."""
    vo = state.get("validation_output") or {}
    status = vo.get("status", "fail")
    retry_count = state.get("retry_count", 0)

    if status == "pass":
        return {"routing_target": "execute"}

    if status == "abort" or retry_count >= RETRY_CAP:
        return {
            "routing_target": "__end__",
            "success": False,
            "abort_reason": vo.get("rationale", "hard_validate_abort"),
        }

    return {"routing_target": "error_classifier"}


def error_classifier(state: dict) -> dict:
    """Classify error type and route to self_fix or claude_rescue."""
    vo = state.get("validation_output") or {}
    category = vo.get("category", "unknown")
    severity = vo.get("severity", "medium")
    transient_retry = state.get("transient_retry_count", 0)

    if category in CROSS_TASK_CATEGORIES:
        return {"routing_target": "rescue_router"}

    if category in TRANSIENT_CATEGORIES and transient_retry < TRANSIENT_RETRY_CAP:
        return {"routing_target": "self_fix", "transient_retry_count": transient_retry + 1}

    return {"routing_target": "claude_rescue"}


def rescue_router(state: dict) -> dict:
    """Handle cross-task failures.

    Dual-write: appends to cross_task_issues (accumulator) AND sets routing_target.
    """
    vo = state.get("validation_output") or {}
    cross_task_issues = list(state.get("cross_task_issues") or [])
    current_task_id = state.get("current_task_id", "")

    issue_record = {
        "task_id": current_task_id,
        "category": vo.get("category"),
        "severity": vo.get("severity"),
        "rationale": vo.get("rationale"),
        "affected_scope": vo.get("affected_scope", ""),
    }
    cross_task_issues.append(issue_record)

    raw_scope = vo.get("affected_scope", "")
    affected_task_ids = [s.strip() for s in raw_scope.split(",") if s.strip()]

    return {
        "routing_target": "rollback_state",
        "cross_task_issues": cross_task_issues,
        "affected_task_ids": affected_task_ids,
        "rescue_scope": "cross_task",
        "rescue_rationale": vo.get("rationale", ""),
    }


def cascade_rollback(tasks: list, affected_task_ids: list) -> set:
    """Helper (not a graph node): transitively find all tasks dependent on affected_task_ids."""
    affected = set(affected_task_ids)
    changed = True
    while changed:
        changed = False
        for task in tasks:
            tid = task.get("id")
            deps = task.get("dependencies") or []
            if tid not in affected and any(d in affected for d in deps):
                affected.add(tid)
                changed = True
    return affected


def rollback_state(state: dict) -> dict:
    """Mark affected tasks for re-execution; route to claude_rescue."""
    tasks = state.get("tasks") or []
    affected_task_ids = state.get("affected_task_ids") or []
    all_affected = cascade_rollback(tasks, affected_task_ids)
    completed_tasks = [t for t in (state.get("completed_tasks") or []) if t not in all_affected]
    return {
        "completed_tasks": completed_tasks,
        "current_task_index": 0,
        "routing_target": "claude_rescue",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest test_colony_coder.py::test_hard_validate_pass \
       test_colony_coder.py::test_hard_validate_routes_to_error_classifier \
       test_colony_coder.py::test_hard_validate_abort_at_cap \
       test_colony_coder.py::test_error_classifier_transient \
       test_colony_coder.py::test_error_classifier_escalates_to_claude \
       test_colony_coder.py::test_rescue_router_dual_write \
       test_colony_coder.py::test_cascade_rollback_transitive -v
```
Expected: 7 PASSED

- [ ] **Step 5: Compile test for executor graph**

```bash
pytest test_colony_coder.py::test_executor_graph_compiles -v
```

Add this compile test first:

```python
# Add to test_colony_coder.py

@pytest.mark.asyncio
async def test_executor_graph_compiles():
    import blueprints.functional_graphs.colony_coder_executor.state  # register schema
    from framework.agent_loader import AgentLoader
    from pathlib import Path
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_executor")).build_graph()
    node_ids = set(g.nodes) - {"__start__"}
    required = {
        "code_gen", "soft_validate", "self_fix",
        "apply_patch", "execute",
        "hard_validate", "error_classifier", "rescue_router", "rollback_state",
        "claude_rescue",
    }
    assert required <= node_ids, f"Missing: {required - node_ids}"
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add blueprints/functional_graphs/colony_coder_executor/validators.py test_colony_coder.py
git commit -m "feat: add colony_coder_executor validators + compile test"
```

---

## Phase 5: Integrator Subgraph

### Task 9: colony_coder_integrator

**Files:**
- Create: `blueprints/functional_graphs/colony_coder_integrator/__init__.py`
- Create: `blueprints/functional_graphs/colony_coder_integrator/validators.py`
- Create: `blueprints/functional_graphs/colony_coder_integrator/entity.json`
- Create: `blueprints/functional_graphs/colony_coder_integrator/system_prompt.md`
- Test: `test_colony_coder.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to test_colony_coder.py

def test_integration_route_pass():
    from blueprints.functional_graphs.colony_coder_integrator.validators import integration_route
    result = integration_route({"validation_output": {"status": "pass"}, "retry_count": 0})
    assert result["routing_target"] == "__end__"
    assert result["success"] is True


def test_integration_route_fail_rescue():
    from blueprints.functional_graphs.colony_coder_integrator.validators import integration_route
    result = integration_route({"validation_output": {"status": "fail"}, "retry_count": 0})
    assert result["routing_target"] == "integration_rescue"
    assert result["retry_count"] == 1


def test_integration_route_abort_at_cap():
    from blueprints.functional_graphs.colony_coder_integrator.validators import integration_route
    result = integration_route({"validation_output": {"status": "fail"}, "retry_count": 2})
    assert result["routing_target"] == "__end__"
    assert result["success"] is False


@pytest.mark.asyncio
async def test_integrator_graph_compiles():
    import blueprints.functional_graphs.colony_coder_executor.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    from pathlib import Path
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_integrator")).build_graph()
    node_ids = set(g.nodes) - {"__start__"}
    required = {"integration_test", "integration_rescue", "apply_patch", "integration_route"}
    assert required <= node_ids, f"Missing: {required - node_ids}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test_colony_coder.py::test_integration_route_pass -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `blueprints/functional_graphs/colony_coder_integrator/__init__.py`**

Empty file.

- [ ] **Step 4: Create `blueprints/functional_graphs/colony_coder_integrator/validators.py`**

```python
"""Deterministic validator nodes for colony_coder_integrator.

Node: integration_route
  Routes to: __end__ (pass/abort) | integration_rescue (retry)
"""

RETRY_CAP = 2


def integration_route(state: dict) -> dict:
    """Route integration test results."""
    vo = state.get("validation_output") or {}
    status = vo.get("status", "fail")
    retry_count = state.get("retry_count", 0)

    if status == "pass":
        return {"routing_target": "__end__", "success": True}

    if status == "abort" or retry_count >= RETRY_CAP:
        return {
            "routing_target": "__end__",
            "success": False,
            "abort_reason": vo.get("rationale", "integration_abort"),
        }

    return {"routing_target": "integration_rescue", "retry_count": retry_count + 1}
```

- [ ] **Step 5: Create `blueprints/functional_graphs/colony_coder_integrator/entity.json`**

Note: Uses `"colony_executor"` schema so `integration_test` can write `validation_output` and
`integration_route` can write `success`/`abort_reason` into state. Import
`blueprints.functional_graphs.colony_coder_executor.state` before calling `build_graph()`.

```json
{
  "name": "colony_coder_integrator",
  "persona_files": [],
  "graph": {
    "state_schema": "colony_executor",
    "nodes": [
      {
        "id": "integration_test",
        "type": "OLLAMA",
        "model": "qwen3.5:27b",
        "tools": ["read_file", "bash_exec", "submit_validation"],
        "session_key": "integrator_session"
      },
      {
        "id": "integration_rescue",
        "type": "CLAUDE_SDK",
        "session_key": "session_b",
        "system_prompt": "You are ApexCoder. The integration tests failed. Analyze the failures and produce a fix."
      },
      {
        "id": "apply_patch",
        "type": "EXTERNAL_TOOL",
        "backend": "code_execution",
        "timeout": 30
      },
      {
        "id": "integration_route",
        "type": "DETERMINISTIC"
      }
    ],
    "edges": [
      {"from": "__start__",          "to": "integration_test"},
      {"from": "integration_test",   "to": "integration_route"},
      {"from": "integration_rescue", "to": "apply_patch"},
      {"from": "apply_patch",        "to": "integration_test"},
      {"from": "integration_route",  "to": "__end__",             "type": "routing_to"},
      {"from": "integration_route",  "to": "integration_rescue",  "type": "routing_to"}
    ]
  }
}
```

- [ ] **Step 6: Create `blueprints/functional_graphs/colony_coder_integrator/system_prompt.md`** (stub)

```markdown
# Colony Coder — Integrator

Run integration tests across all completed tasks.
Use the provided tools to execute test suites and submit structured validation results.
```

- [ ] **Step 7: Run all tests to verify they pass**

```bash
pytest test_colony_coder.py::test_integration_route_pass \
       test_colony_coder.py::test_integration_route_fail_rescue \
       test_colony_coder.py::test_integration_route_abort_at_cap \
       test_colony_coder.py::test_integrator_graph_compiles -v
```
Expected: 4 PASSED

- [ ] **Step 8: Commit**

```bash
git add blueprints/functional_graphs/colony_coder_integrator/ test_colony_coder.py
git commit -m "feat: add colony_coder_integrator subgraph"
```

---

## Phase 6: Master Graph + E2E

### Task 10: Master Colony Coder Graph

**Files:**
- Create: `blueprints/functional_graphs/colony_coder/__init__.py`
- Create: `blueprints/functional_graphs/colony_coder/entity.json`
- Test: `test_colony_coder.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to test_colony_coder.py

@pytest.mark.asyncio
async def test_master_graph_compiles():
    # Must import state.py first to register "colony_executor" schema
    import blueprints.functional_graphs.colony_coder_executor.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    from pathlib import Path
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder")).build_graph()
    node_ids = set(g.nodes) - {"__start__"}
    assert {"plan", "execute", "integrate"} <= node_ids
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test_colony_coder.py::test_master_graph_compiles -v
```
Expected: Error — blueprint dir missing

- [ ] **Step 3: Create `blueprints/functional_graphs/colony_coder/__init__.py`**

Empty file.

- [ ] **Step 4: Create `blueprints/functional_graphs/colony_coder/entity.json`**

Note: Uses `"colony_executor"` state schema because the master graph passes inter-subgraph fields
(`tasks`, `execution_order`, `completed_tasks`, `final_files`, etc.) that are not in `BaseAgentState`.
The `ColonyCoderExecutorState` is a superset that includes all needed fields.
Import `blueprints.functional_graphs.colony_coder_executor.state` before calling `build_graph()`
to ensure the schema is registered.

```json
{
  "name": "colony_coder",
  "persona_files": [],
  "graph": {
    "state_schema": "colony_executor",
    "nodes": [
      {
        "id": "plan",
        "type": "AGENT_REF",
        "agent_dir": "blueprints/functional_graphs/colony_coder_planner",
        "state_in": {
          "messages": "messages",
          "node_sessions": "node_sessions"
        },
        "state_out": {
          "tasks": "tasks",
          "execution_order": "execution_order",
          "refined_plan": "refined_plan",
          "working_directory": "working_directory",
          "messages": "messages"
        }
      },
      {
        "id": "execute",
        "type": "AGENT_REF",
        "agent_dir": "blueprints/functional_graphs/colony_coder_executor",
        "state_in": {
          "tasks": "tasks",
          "execution_order": "execution_order",
          "refined_plan": "refined_plan",
          "working_directory": "working_directory",
          "messages": "messages"
        },
        "state_out": {
          "completed_tasks": "completed_tasks",
          "final_files": "final_files",
          "success": "success",
          "abort_reason": "abort_reason",
          "messages": "messages"
        }
      },
      {
        "id": "integrate",
        "type": "AGENT_REF",
        "agent_dir": "blueprints/functional_graphs/colony_coder_integrator",
        "state_in": {
          "completed_tasks": "completed_tasks",
          "final_files": "final_files",
          "working_directory": "working_directory",
          "messages": "messages"
        },
        "state_out": {
          "success": "success",
          "abort_reason": "abort_reason",
          "messages": "messages"
        }
      }
    ],
    "edges": [
      {"from": "__start__", "to": "plan"},
      {"from": "plan",      "to": "execute"},
      {"from": "execute",   "to": "integrate"},
      {"from": "integrate", "to": "__end__"}
    ]
  }
}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest test_colony_coder.py::test_master_graph_compiles -v
```
Expected: PASS

- [ ] **Step 6: Run all unit tests**

```bash
pytest test_colony_coder.py -v
```
Expected: All PASSED

- [ ] **Step 7: Commit**

```bash
git add blueprints/functional_graphs/colony_coder/ test_colony_coder.py
git commit -m "feat: add colony_coder master graph"
```

---

### Task 11: E2E Tests

**Files:**
- Create: `test_e2e_colony_coder.py`

These tests mock Ollama + Claude to verify routing logic and graph flow without real LLM calls.

- [ ] **Step 1: Write E2E tests**

```python
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
    import blueprints.functional_graphs.colony_coder_executor.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_planner")).build_graph()
    assert set(g.nodes) - {"__start__"} >= {
        "plan", "design_debate", "claude_swarm", "task_decompose", "decomposition_validator",
    }


@pytest.mark.asyncio
async def test_executor_graph_compiles_e2e():
    import blueprints.functional_graphs.colony_coder_executor.state
    from framework.agent_loader import AgentLoader
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_executor")).build_graph()
    assert set(g.nodes) - {"__start__"} >= {
        "code_gen", "soft_validate", "self_fix", "apply_patch", "execute",
        "hard_validate", "error_classifier", "rescue_router", "rollback_state", "claude_rescue",
    }


@pytest.mark.asyncio
async def test_integrator_graph_compiles_e2e():
    import blueprints.functional_graphs.colony_coder_executor.state  # noqa: F401
    from framework.agent_loader import AgentLoader
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder_integrator")).build_graph()
    assert set(g.nodes) - {"__start__"} >= {
        "integration_test", "integration_rescue", "apply_patch", "integration_route",
    }


@pytest.mark.asyncio
async def test_master_graph_compiles_e2e():
    import blueprints.functional_graphs.colony_coder_executor.state  # noqa: F401 — registers "colony_executor"
    from framework.agent_loader import AgentLoader
    g = await AgentLoader(Path("blueprints/functional_graphs/colony_coder")).build_graph()
    assert set(g.nodes) - {"__start__"} >= {"plan", "execute", "integrate"}
```

- [ ] **Step 2: Run routing-only tests (no network)**

```bash
pytest test_e2e_colony_coder.py::test_decomposition_validator_flow \
       test_e2e_colony_coder.py::test_executor_happy_path_routing \
       test_e2e_colony_coder.py::test_executor_self_fix_routing \
       test_e2e_colony_coder.py::test_executor_cross_task_routing -v
```
Expected: 4 PASSED (pure Python, no LLM)

- [ ] **Step 3: Run all E2E compile tests**

```bash
pytest test_e2e_colony_coder.py -v
```
Expected: All pass. Fix any import path or schema registration issues that arise.

- [ ] **Step 4: Run full test suite — verify no regressions**

```bash
pytest test_cli.py test_e2e_debate.py test_colony_coder.py test_e2e_colony_coder.py -v
```
Expected: All PASSED, same count as before on existing tests.

- [ ] **Step 5: Commit**

```bash
git add test_e2e_colony_coder.py
git commit -m "feat: add Colony Coder E2E tests"
```

---

## Summary

| Phase | Tasks | Deliverable |
|-------|-------|-------------|
| 1 | 1–4 | DeterministicNode, ExternalToolNode code_execution, tools.py, OllamaNode tool loop |
| 2 | 5 | ColonyCoderExecutorState + schema registration |
| 3 | 6 | colony_coder_planner (5 nodes) |
| 4 | 7–8 | colony_coder_executor (10 nodes incl. claude_rescue) |
| 5 | 9 | colony_coder_integrator (4 nodes) |
| 6 | 10–11 | Master graph + E2E tests |

**Spec:** `docs/superpowers/specs/2026-03-17-colony-coder-design.md`

**After plan approved:** Use `superpowers:subagent-driven-development` to execute — one fresh subagent per task, two-stage review after each.
