# ApexCoder TDD Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild ApexCoder from a single Claude SDK node into a 4-node TDD pipeline (splitter → ClaudeQA → reset_for_coder → ClaudeCoder) where QA writes tests first and Coder must pass them.

**Architecture:** Two DETERMINISTIC nodes (splitter, reset_for_coder) handle input normalization and message isolation. Two CLAUDE_SDK nodes (ClaudeQA, ClaudeCoder) run as independent sessions — QA writes tests from user requirements, Coder implements code that passes those tests. A PreToolUse hook prevents Coder from modifying QA test files.

**Tech Stack:** LangGraph, Claude Agent SDK, Python 3.12+, pytest

---

## File Map

### New Files

| File | Responsibility |
|------|----------------|
| `blueprints/functional_graphs/apex_coder/state.py` | `ApexCoderState` TypedDict + auto-register |
| `blueprints/functional_graphs/apex_coder/validators.py` | `splitter()` + `reset_for_coder()` DETERMINISTIC nodes |
| `blueprints/functional_graphs/apex_coder/CODER_ROLE.md` | Coder persona (TDD, must pass QA tests) |
| `blueprints/functional_graphs/apex_coder/QA_ROLE.md` | QA persona (test-first, real env, 5-10 tests) |
| `blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py` | PreToolUse hook blocking Write/Edit to qa_tests/ |
| `tests/test_apex_coder.py` | Unit tests for state, validators, graph compilation |

### Modified Files

| File | Change |
|------|--------|
| `blueprints/functional_graphs/apex_coder/entity.json` | Rewrite: 1 node → 4 nodes |

### Deleted Files

| File | Reason |
|------|--------|
| `blueprints/functional_graphs/apex_coder/ROLE.md` | Split into CODER_ROLE.md + QA_ROLE.md |

### Unchanged Files

| File | Reason |
|------|--------|
| `blueprints/functional_graphs/apex_coder/PROTOCOL.md` | Shared by both Coder and QA |
| `blueprints/functional_graphs/apex_coder/.claude/agents/*` | Coder can still spawn sub-agents |
| `blueprints/functional_graphs/apex_coder/.claude/skills/*` | Coder can still use skills |

---

## Task 1: ApexCoderState schema

**Files:**
- Create: `blueprints/functional_graphs/apex_coder/state.py`
- Create: `tests/test_apex_coder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_apex_coder.py
import pytest


def test_apex_coder_schema_registered():
    import blueprints.functional_graphs.apex_coder.state  # noqa: F401
    from framework.registry import get_all_schemas
    schemas = get_all_schemas()
    assert "apex_coder_schema" in schemas, f"missing apex_coder_schema, got: {list(schemas.keys())}"


def test_apex_coder_schema_has_required_fields():
    import typing
    from blueprints.functional_graphs.apex_coder.state import ApexCoderState
    hints = typing.get_type_hints(ApexCoderState, include_extras=True)
    for field in ("user_requirements", "working_directory", "qa_bypass",
                  "qa_tests_dir", "run_qa_script", "qa_summary", "apex_conclusion"):
        assert field in hints, f"ApexCoderState missing field: {field}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/kingy/Foundation/ZenithLoom && python3 -m pytest tests/test_apex_coder.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write the implementation**

```python
# blueprints/functional_graphs/apex_coder/state.py
"""ApexCoderState — apex_coder TDD pipeline state schema.

splitter / ClaudeQA / reset_for_coder / ClaudeCoder 共享此 schema。

Auto-registers as "apex_coder_schema" on import.
"""

from __future__ import annotations

from typing import Annotated, Optional

from framework.schema.base import BaseAgentState
from framework.schema.reducers import _merge_dict
from framework.registry import register_schema


class ApexCoderState(BaseAgentState):
    # Splitter output
    user_requirements: str          # 用户需求文本（从 messages[0] 或 plan.md 提取）
    working_directory: str          # 工作目录路径

    # QA output
    qa_bypass: bool                 # QA 是否跳过（任务不需要测试时为 True）
    qa_tests_dir: str               # QA 测试目录路径
    run_qa_script: str              # run_qa.sh 路径
    qa_summary: str                 # QA 输出摘要

    # Coder output
    apex_conclusion: str            # Coder 最终报告

    # Override node_sessions with merge reducer
    node_sessions: Annotated[dict, _merge_dict]


# Auto-register on import
register_schema("apex_coder_schema", ApexCoderState)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/state.py tests/test_apex_coder.py
git commit -m "feat(apex_coder): add ApexCoderState schema for TDD pipeline"
```

---

## Task 2: Validators — splitter + reset_for_coder

**Files:**
- Create: `blueprints/functional_graphs/apex_coder/validators.py`
- Modify: `tests/test_apex_coder.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_apex_coder.py`:

```python
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
import tempfile


def test_splitter_text_input():
    from blueprints.functional_graphs.apex_coder.validators import splitter
    result = splitter({
        "messages": [HumanMessage(content="Build a snake game\n\n## 工作目录: /tmp/test_splitter_apex")]
    })
    assert result["user_requirements"] == "Build a snake game\n\n## 工作目录: /tmp/test_splitter_apex"
    assert result["working_directory"] == "/tmp/test_splitter_apex"
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == result["user_requirements"]


def test_splitter_file_input():
    from blueprints.functional_graphs.apex_coder.validators import splitter
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("Build a todo app")
        f.flush()
        result = splitter({"messages": [HumanMessage(content=f.name)]})
    assert result["user_requirements"] == "Build a todo app"
    assert result["working_directory"].startswith("/tmp/apex_")


def test_splitter_auto_generates_working_dir():
    from blueprints.functional_graphs.apex_coder.validators import splitter
    result = splitter({"messages": [HumanMessage(content="Build something")]})
    assert result["working_directory"].startswith("/tmp/apex_")
    assert Path(result["working_directory"]).exists()
    assert Path(result["working_directory"], "test_tool", "qa_tests").exists()


def test_splitter_creates_directories():
    from blueprints.functional_graphs.apex_coder.validators import splitter
    result = splitter({
        "messages": [HumanMessage(content="Task\n\n## 工作目录: /tmp/test_splitter_dirs")]
    })
    assert Path("/tmp/test_splitter_dirs/test_tool/qa_tests").is_dir()


def test_reset_for_coder_clears_qa_messages():
    from blueprints.functional_graphs.apex_coder.validators import reset_for_coder
    result = reset_for_coder({
        "messages": [
            HumanMessage(content="user task", id="h1"),
            AIMessage(content="QA reasoning blah blah", id="a1"),
        ],
        "user_requirements": "user task",
        "working_directory": "/tmp/test_reset",
        "qa_bypass": False,
        "run_qa_script": "/tmp/test_reset/test_tool/run_qa.sh",
    })
    # Should have RemoveMessages for old messages + 1 new HumanMessage
    msgs = result["messages"]
    # New HumanMessage should contain user requirements
    human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 1
    assert "user task" in human_msgs[0].content
    assert "run_qa.sh" in human_msgs[0].content


def test_reset_for_coder_bypass_mode():
    from blueprints.functional_graphs.apex_coder.validators import reset_for_coder
    result = reset_for_coder({
        "messages": [HumanMessage(content="task", id="h1")],
        "user_requirements": "simple task",
        "working_directory": "/tmp/test_bypass",
        "qa_bypass": True,
        "run_qa_script": "",
    })
    human_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
    assert "BYPASSED" in human_msgs[0].content
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 2 pass, 6 fail

- [ ] **Step 3: Write the implementation**

```python
# blueprints/functional_graphs/apex_coder/validators.py
"""DETERMINISTIC nodes for apex_coder TDD pipeline.

Nodes:
  splitter          — extract user_requirements, create working_directory
  reset_for_coder   — clear QA messages, build clean prompt for Coder
"""

import os
import re
import uuid
from pathlib import Path


def splitter(state: dict) -> dict:
    """Extract user_requirements and set up working_directory."""
    from langchain_core.messages import HumanMessage

    msg = state["messages"][0].content.strip()

    # Smart input: file path or raw text
    if msg.startswith("/") and Path(msg).is_file():
        user_requirements = Path(msg).read_text(encoding="utf-8")
    else:
        user_requirements = msg

    # Parse working_directory from task or auto-generate
    wd_match = re.search(
        r"[#]*\s*(?:工作目录|working.?dir(?:ectory)?)[:：]\s*(\S+)",
        user_requirements,
        re.IGNORECASE,
    )
    if wd_match:
        working_directory = wd_match.group(1)
    else:
        working_directory = f"/tmp/apex_{uuid.uuid4().hex[:8]}"

    # Create directory structure
    Path(working_directory).mkdir(parents=True, exist_ok=True)
    Path(working_directory, "test_tool", "qa_tests").mkdir(parents=True, exist_ok=True)

    return {
        "user_requirements": user_requirements,
        "working_directory": working_directory,
        "messages": [HumanMessage(content=user_requirements)],
    }


def reset_for_coder(state: dict) -> dict:
    """Clear QA messages, build clean prompt for ClaudeCoder."""
    from langchain_core.messages import HumanMessage, RemoveMessage

    user_req = state.get("user_requirements", "")
    working_dir = state.get("working_directory", "")
    qa_bypass = state.get("qa_bypass", False)
    run_qa_script = state.get("run_qa_script", "")

    # Remove all previous messages (QA's conversation history)
    msgs = state.get("messages", [])
    removals = [RemoveMessage(id=m.id) for m in msgs]

    # Build clean prompt for Coder
    lines = [f"## User Requirements\n\n{user_req}"]
    lines.append(f"\n## Working Directory: `{working_dir}`")

    if qa_bypass:
        lines.append("\n## QA: BYPASSED (no tests to pass)")
    else:
        lines.append(f"\n## QA Tests")
        lines.append(f"- Runner script: `{run_qa_script}`")
        lines.append(f"- **Read the QA tests FIRST to understand what's expected.**")
        lines.append(f"- **Run `bash {run_qa_script}` and ensure ALL tests pass before finishing.**")
        lines.append(f"- **DO NOT modify any files in `test_tool/qa_tests/`.**")

        # List QA test files
        qa_dir = os.path.join(working_dir, "test_tool", "qa_tests")
        if os.path.isdir(qa_dir):
            test_files = [f for f in os.listdir(qa_dir) if f.endswith(".py") and f != "__init__.py"]
            if test_files:
                lines.append(f"\nQA test files ({len(test_files)}):")
                for f in sorted(test_files):
                    lines.append(f"  - `{qa_dir}/{f}`")

    prompt = "\n".join(lines)
    return {
        "messages": removals + [HumanMessage(content=prompt)],
    }
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/validators.py tests/test_apex_coder.py
git commit -m "feat(apex_coder): add splitter and reset_for_coder validators"
```

---

## Task 3: Persona files — CODER_ROLE.md + QA_ROLE.md

**Files:**
- Create: `blueprints/functional_graphs/apex_coder/CODER_ROLE.md`
- Create: `blueprints/functional_graphs/apex_coder/QA_ROLE.md`
- Delete: `blueprints/functional_graphs/apex_coder/ROLE.md`

- [ ] **Step 1: Create CODER_ROLE.md**

```markdown
# Coder Role

你是 P8 级别工程师。端到端实现功能。

## 你的输入

1. **User Requirements** — 用户需求（在消息中）
2. **QA Tests** — 独立 QA 工程师写的自动化测试（在 test_tool/qa_tests/ 目录）

如果 QA 标记为 BYPASSED，你只需满足 User Requirements，不需要跑测试。

## 你的工作流程

1. **先读 QA 测试** — 理解 QA 期望的行为和验收标准
2. **再读 User Requirements** — 确认完整需求
3. **实现** — 写代码满足需求
4. **自测** — 运行 `bash <run_qa_script>`
5. **修复** — 如果测试失败，读错误输出，修代码，重跑
6. **重复 4-5 直到全部通过**
7. **完成报告**

## 铁律

- ❌ 禁止修改 `test_tool/qa_tests/` 下的任何文件（有 hook 强制阻止）
- ❌ 禁止删除或重命名 QA 测试
- ✅ 如果你认为 QA 测试有问题，在最终报告里说明，但不要改它
- ✅ 所有 Bash 命令必须带 timeout（参见 PROTOCOL.md）

## 你的团队

你有 Agent 工具可以 spawn 专家子 agent：

| Agent | 用途 | 何时 spawn |
|-------|------|-----------|
| planner | 需求分析 + 实现计划 | 复杂任务开始前 |
| architect | 系统设计 + ADR | 架构决策 |
| code-reviewer | 代码审查 | 实现完成后 |
| build-error-resolver | 构建错误修复 | build 失败时 |
| pua-debugger | 极限调试 | 反复失败时 |

## 报告格式

完成后输出：
- 创建/修改了哪些文件
- QA 测试结果（全部通过 or 哪些仍失败 + 原因）
- 代码架构简述
```

- [ ] **Step 2: Create QA_ROLE.md**

```markdown
# QA Engineer Role

你是独立的 QA 工程师。你的任务是根据用户需求写自动化测试。
你不知道代码会怎么实现，也不应该关心。你只关心：用户要什么功能？怎么验证？

## 你的工作

1. 读取用户需求
2. 判断是否需要 QA 测试：
   - 不需要时（纯重构无行为变化、文档编辑、配置改动、代码风格调整、修改大型代码库中难以做 E2E 测试的部分）：输出 `QA_BYPASS: <原因>`，不写测试
   - 需要时：继续下面的步骤
3. 在 `<working_directory>/test_tool/qa_tests/` 目录写 5-10 个测试
4. 写 `<working_directory>/test_tool/run_qa.sh` 执行脚本
5. 输出测试摘要

## 测试规则

- 测试从**用户视角**验证功能，不测内部实现
- 每个测试 < 10 秒，总测试 < 90 秒
- curses/终端程序：必须用 pty 模块在**真实终端环境**测试
  - ❌ 禁止 `python3 -c "import X; X.Game().tick()"` 这种 headless 测试
  - ✅ 必须用 pty + 24x80 终端启动真实进程
- 覆盖：核心功能 + 关键边界 + 退出行为
- 不要测试实现细节（内部类名、函数签名等）

## run_qa.sh 模板

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")/.."
timeout 120 python3 -m pytest test_tool/qa_tests/ -v 2>&1
```

## 输出格式

如果写了测试：
  QA_READY: <测试数量> tests written to <path>

如果跳过：
  QA_BYPASS: <原因>
```

- [ ] **Step 3: Delete old ROLE.md**

```bash
git rm blueprints/functional_graphs/apex_coder/ROLE.md
```

- [ ] **Step 4: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/CODER_ROLE.md \
       blueprints/functional_graphs/apex_coder/QA_ROLE.md
git commit -m "feat(apex_coder): split ROLE.md into CODER_ROLE.md + QA_ROLE.md"
```

---

## Task 4: PreToolUse hook

**Files:**
- Create: `blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py`
- Modify: `tests/test_apex_coder.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_apex_coder.py`:

```python
import subprocess
import json


def test_hook_blocks_qa_test_write():
    hook_path = "blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py"
    data = {"tool_input": {"file_path": "/tmp/game/test_tool/qa_tests/test_foo.py"}}
    result = subprocess.run(
        ["python3", hook_path],
        input=json.dumps(data),
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["decision"] == "block"


def test_hook_allows_source_write():
    hook_path = "blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py"
    data = {"tool_input": {"file_path": "/tmp/game/main.py"}}
    result = subprocess.run(
        ["python3", hook_path],
        input=json.dumps(data),
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["decision"] == "allow"


def test_hook_allows_unit_test_write():
    hook_path = "blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py"
    data = {"tool_input": {"file_path": "/tmp/game/test_tool/unit_tests/test_main.py"}}
    result = subprocess.run(
        ["python3", hook_path],
        input=json.dumps(data),
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["decision"] == "allow"
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python3 -m pytest tests/test_apex_coder.py::test_hook_blocks_qa_test_write -v`
Expected: FAIL — file not found or JSON error

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""PreToolUse hook: block Write/Edit to test_tool/qa_tests/ directory.

Reads tool_input from stdin (JSON), checks if file_path targets QA tests.
Outputs JSON decision: {"decision": "allow"} or {"decision": "block", "reason": "..."}.
"""
import json
import sys

data = json.load(sys.stdin)
tool_input = data.get("tool_input", {})
path = tool_input.get("file_path", "") or tool_input.get("path", "")

if "test_tool/qa_tests" in path:
    print(json.dumps({
        "decision": "block",
        "reason": f"BLOCKED: Cannot modify QA test file '{path}'. Fix your source code instead.",
    }))
else:
    print(json.dumps({"decision": "allow"}))
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py tests/test_apex_coder.py
git commit -m "feat(apex_coder): add PreToolUse hook to protect QA test files"
```

---

## Task 5: entity.json rewrite + graph compilation test

**Files:**
- Modify: `blueprints/functional_graphs/apex_coder/entity.json`
- Modify: `tests/test_apex_coder.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_apex_coder.py`:

```python
@pytest.mark.asyncio
async def test_apex_coder_graph_compiles():
    import blueprints.functional_graphs.apex_coder.state  # noqa: F401
    from framework.agent_loader import EntityLoader
    g = await EntityLoader(Path("blueprints/functional_graphs/apex_coder")).build_graph(checkpointer=None)
    node_ids = set(g.nodes) - {"__start__", "__end__"}
    required = {"splitter", "claude_qa", "reset_for_coder", "claude_coder"}
    assert required <= node_ids, f"Missing nodes: {required - node_ids}, got: {node_ids}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_apex_coder.py::test_apex_coder_graph_compiles -v`
Expected: FAIL — old entity.json has `apex_main` not 4 nodes

- [ ] **Step 3: Rewrite entity.json**

```json
{
  "name": "apex_coder",
  "routing_hint": "当编程问题极其复杂、需要全力以赴穷举所有方案时使用。适用：多次失败仍未解决的 bug、完整架构重构、跨多文件系统变更、需要深度 debug 的环境/配置问题。",
  "llm": "claude",
  "graph": {
    "state_schema": "apex_coder_schema",
    "entry": "splitter",
    "exit": "claude_coder",
    "nodes": [
      {
        "id": "splitter",
        "type": "DETERMINISTIC"
      },
      {
        "id": "claude_qa",
        "type": "CLAUDE_SDK",
        "model": "claude-opus-4-6",
        "session_key": "apex_qa",
        "tools": ["Read", "Write", "Bash"],
        "permission_mode": "bypassPermissions",
        "persona_files": ["./QA_ROLE.md", "./PROTOCOL.md"],
        "output_field": "qa_summary",
        "setting_sources": null
      },
      {
        "id": "reset_for_coder",
        "type": "DETERMINISTIC"
      },
      {
        "id": "claude_coder",
        "type": "CLAUDE_SDK",
        "model": "claude-opus-4-6",
        "extra_persona": true,
        "session_key": "apex_coder",
        "tools": [
          "Read", "Write", "Edit", "Bash",
          "Glob", "Grep", "NotebookEdit", "Agent"
        ],
        "permission_mode": "bypassPermissions",
        "max_turns": 50,
        "persona_files": ["./CODER_ROLE.md", "./PROTOCOL.md"],
        "output_field": "apex_conclusion",
        "setting_sources": null,
        "settings_override": {
          "enabledPlugins": [],
          "hooks": {
            "PreToolUse": [{
              "matcher": "Write|Edit",
              "hooks": [{
                "type": "command",
                "command": "python3 blueprints/functional_graphs/apex_coder/hooks/protect_qa_tests.py"
              }]
            }]
          }
        },
        "add_dirs": [
          "blueprints/functional_graphs/apex_coder"
        ]
      }
    ],
    "edges": [
      {"from": "splitter",         "to": "claude_qa"},
      {"from": "claude_qa",        "to": "reset_for_coder"},
      {"from": "reset_for_coder",  "to": "claude_coder"}
    ]
  },
  "max_retries": 0,
  "db_path": "apex_coder.db",
  "sessions_file": "sessions.json"
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_apex_coder.py -v`
Expected: 12 passed

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python3 -m pytest tests/test_apex_coder.py test_colony_coder.py test_e2e_colony_coder.py test_e2e_debate.py test_cli.py tests/test_debug_reporter.py tests/test_subgraph_init_node.py -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add blueprints/functional_graphs/apex_coder/entity.json tests/test_apex_coder.py
git commit -m "feat(apex_coder): rewrite entity.json to 4-node TDD pipeline"
```

---

## Task 6: Update debug runner + smoke test

**Files:**
- Modify: `run_apex_coder_debug.py`
- Modify: `run_benchmark_apex.py`

- [ ] **Step 1: Update run_apex_coder_debug.py**

Replace the import of `apex_coder.state` and update the init_state to include `ApexCoderState` fields:

```python
#!/usr/bin/env python3
"""ApexCoder debug runner — TDD pipeline (QA writes tests → Coder passes them)."""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)

# Register state schema
import blueprints.functional_graphs.apex_coder.state  # noqa: F401

from framework.agent_loader import EntityLoader
from framework.debug import set_debug
from framework.debug_reporter import DebugConsoleReporter
from langchain_core.messages import HumanMessage

SNAKE_TASK = (
    "用 Python 写一个双蛇对战游戏（Snake Battle）。\n"
    "\n"
    "## 核心要求\n"
    "1. 使用 curses 库实现终端 UI\n"
    "2. 两条蛇同时出现在棋盘上，全部由 AI 控制（无人类玩家），玩家只是观战者\n"
    "3. 屏幕上同时存在多个食物，蛇吃到食物后身体变长\n"
    "4. 蛇撞墙、撞自己、或者撞对方身体则死亡\n"
    "5. 最后存活的蛇获胜；如果都活着则比长度\n"
    "\n"
    "## AI 设计\n"
    "你需要自己设计两个不同策略的 AI（AI-Alpha 和 AI-Beta），让它们各控制一条蛇。\n"
    "AI 的目标：尽量吃食物让自己变长，同时尽量消灭对方。\n"
    "两个 AI 必须使用不同的策略，让对战有趣。\n"
    "\n"
    "## UI 要求\n"
    "- 顶部状态栏显示双方信息和当前帧数\n"
    "- 游戏区域有边框\n"
    "- 两条蛇用不同颜色区分\n"
    "- 游戏结束显示获胜者\n"
    "- 按 Q 退出\n"
    "- 帧率默认 ~10 FPS\n"
    "\n"
    "## 技术要求\n"
    "- 单文件实现，保存到 /tmp/snake_battle_apex/snake_battle.py\n"
    "- 代码结构清晰，两个 AI 分别是独立的类\n"
    "- 可直接 python3 snake_battle.py 运行\n"
    "- 必须能在标准 24x80 终端下正常运行\n"
    "\n"
    "## 工作目录: /tmp/snake_battle_apex\n"
)


async def main():
    set_debug(True)

    loader = EntityLoader(Path("blueprints/functional_graphs/apex_coder"))
    graph = await loader.build_graph(checkpointer=None)

    reporter = DebugConsoleReporter("apex_coder")

    print("=" * 70)
    print("  ApexCoder TDD — Snake Battle")
    print("=" * 70)
    print(flush=True)

    init_state = {"messages": [HumanMessage(content=SNAKE_TASK)]}

    async for namespace, event in graph.astream(
        init_state, stream_mode="updates", subgraphs=True
    ):
        reporter.on_event(namespace, event)

    reporter.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Update run_benchmark_apex.py**

Add `import blueprints.functional_graphs.apex_coder.state` to the benchmark runner (replace the old plan.md logic — splitter now handles file path input natively):

Replace the `main()` function body. The key change: pass task text directly as message (no plan.md file needed — splitter handles both formats):

```python
async def main():
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(TASKS.keys())}>")
        sys.exit(1)

    task_name = sys.argv[1]
    task_desc, working_dir, _ = TASKS[task_name]

    set_debug(True)

    # Register state schema
    import blueprints.functional_graphs.apex_coder.state  # noqa: F401

    loader = EntityLoader(Path("blueprints/functional_graphs/apex_coder"))
    graph = await loader.build_graph(checkpointer=None)
    reporter = DebugConsoleReporter(f"apex_{task_name}")

    print("=" * 70)
    print(f"  ApexCoder TDD — {task_name}")
    print("=" * 70, flush=True)

    # Append working_directory hint to task
    task_with_wd = f"{task_desc}\n## 工作目录: {working_dir}\n"
    init_state = {"messages": [HumanMessage(content=task_with_wd)]}

    async for ns, event in graph.astream(init_state, stream_mode="updates", subgraphs=True):
        reporter.on_event(ns, event)
    reporter.print_summary()
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('run_apex_coder_debug.py').read()); ast.parse(open('run_benchmark_apex.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add run_apex_coder_debug.py run_benchmark_apex.py
git commit -m "feat(apex_coder): update debug and benchmark runners for TDD pipeline"
```
