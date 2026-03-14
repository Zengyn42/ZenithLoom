# Asa Agent + OllamaNode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement OllamaNode (Ollama HTTP API) in the framework and create the Asa agent with a single-node declarative graph.

**Architecture:** OllamaNode replaces the existing `LlamaNode` stub, inheriting from `AgentNode` and calling Ollama's `/api/chat` endpoint. `LlamaNode = OllamaNode` alias preserves backward compat. Asa's `agent.json` uses the new `OLLAMA` node type in a minimal `__start__ → llama_main → __end__` graph.

**Tech Stack:** Python 3.11+, httpx (async HTTP), LangGraph, Ollama (local)

**Spec:** `docs/superpowers/specs/2026-03-14-asa-agent-ollamanode-design.md`

---

## Chunk 1: Framework Layer

### Task 1: Add `httpx` to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add httpx dependency**

Open `requirements.txt` and append after the `python-dotenv` line:

```
httpx>=0.27.0               # Async HTTP client for OllamaNode
```

- [ ] **Step 2: Verify install**

```bash
cd /home/kingy/Foundation/BootstrapBuilder
pip install httpx
```

Expected: `Successfully installed httpx-...` or `Requirement already satisfied`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add httpx dependency for OllamaNode"
```

---

### Task 2: Implement OllamaNode in `framework/llama/node.py`

**Files:**
- Modify: `framework/llama/node.py` (rewrite stub)

- [ ] **Step 1: Rewrite node.py**

Replace the entire contents of `framework/llama/node.py` with:

```python
"""
框架级 Ollama LLM 节点 — framework/llama/node.py

OllamaNode 继承 AgentNode，实现 call_llm() 接口：
  call_llm(prompt, session_id, tools, cwd) → (text, session_id)

通过 Ollama HTTP API 调用本地模型（llama、qwen 等）。
Ollama 无持久 session，返回传入的 session_id 不变。
keep_alive=-1 确保模型常驻 RAM（防止 5 分钟后自动卸载）。

agent.json 配置：
  node_config["model"]:    "llama3.2:3b"              # 模型名
  node_config["endpoint"]: "http://localhost:11434"    # Ollama endpoint（默认）
  node_config["timeout"]:  120                         # 超时秒数（默认）
"""

import logging

import httpx

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.agent_node import AgentNode

logger = logging.getLogger(__name__)


class OllamaNode(AgentNode):
    """
    Ollama LLM 节点，继承 AgentNode。

    通过 Ollama /api/chat 端点调用本地模型。
    基类 AgentNode.__call__() 处理所有图协议逻辑。
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        super().__init__(config, node_config)
        self._model = node_config.get("model", "llama3")
        self._endpoint = node_config.get("endpoint", "http://localhost:11434")
        self._timeout = node_config.get("timeout", 120)
        self._system_prompt = node_config.get("system_prompt", "")
        logger.info(f"[ollama] model={self._model} endpoint={self._endpoint}")

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> tuple[str, str]:
        """
        调用 Ollama /api/chat。返回 (text, session_id)。

        session_id 语义：Ollama 无持久 session，返回传入值不变。
        tools/cwd：Phase 1 忽略，Ollama tool call 支持推迟到 Phase 2。
        keep_alive=-1：模型常驻 RAM，防止 5 分钟后自动卸载。
        """
        if is_debug():
            logger.debug(f"[ollama] model={self._model} prompt_len={len(prompt)}")

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._endpoint}/api/chat",
                    json=payload,
                )
        except httpx.ConnectError as e:
            msg = (
                f"[Ollama 连接失败] 无法连接到 {self._endpoint}，"
                f"请确认 Ollama 正在运行。({e})"
            )
            logger.error(msg)
            return msg, session_id
        except httpx.TimeoutException:
            msg = f"[Ollama 超时] 模型 {self._model} 响应超时（{self._timeout}s）"
            logger.error(msg)
            return msg, session_id

        if response.status_code != 200:
            body = response.json() if response.content else {}
            error = body.get("error", f"HTTP {response.status_code}")
            msg = f"[Ollama 错误] {error}"
            logger.error(msg)
            return msg, session_id

        data = response.json()
        text = data["message"]["content"]
        return text, session_id

    def get_recent_history(self, session_id: str, limit: int = 10) -> list:
        """Ollama 无持久 session，返回空列表。"""
        return []


# Backward compatibility alias — builtins.py imports LlamaNode directly from this module
LlamaNode = OllamaNode
```

- [ ] **Step 2: Quick import check**

```bash
cd /home/kingy/Foundation/BootstrapBuilder
python -c "from framework.llama.node import OllamaNode, LlamaNode; print('OK', OllamaNode, LlamaNode)"
```

Expected: `OK <class 'framework.llama.node.OllamaNode'> <class 'framework.llama.node.OllamaNode'>`

- [ ] **Step 3: Commit**

```bash
git add framework/llama/node.py
git commit -m "feat: implement OllamaNode (Ollama /api/chat), keep LlamaNode alias"
```

---

### Task 3: Update `framework/llama/__init__.py`

**Files:**
- Modify: `framework/llama/__init__.py`

- [ ] **Step 1: Update exports**

Replace the entire contents of `framework/llama/__init__.py` with:

```python
from framework.llama.node import OllamaNode, LlamaNode  # LlamaNode is alias

__all__ = ["OllamaNode", "LlamaNode"]
```

- [ ] **Step 2: Verify import**

```bash
cd /home/kingy/Foundation/BootstrapBuilder
python -c "from framework.llama import OllamaNode, LlamaNode; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add framework/llama/__init__.py
git commit -m "feat: export OllamaNode from framework.llama package"
```

---

### Task 4: Register `OLLAMA` node type in `framework/builtins.py`

**Files:**
- Modify: `framework/builtins.py:64-67` (replace LOCAL_VLLM block, add OLLAMA)

- [ ] **Step 1: Replace the LOCAL_VLLM block**

Find this block in `framework/builtins.py` (lines 64–67):

```python
@register_node("LOCAL_VLLM")
def _(config, node_config):
    from framework.llama.node import LlamaNode
    return LlamaNode(config, node_config)
```

Replace it with:

```python
@register_node("OLLAMA")
def _(config, node_config):
    from framework.llama.node import LlamaNode
    return LlamaNode(config, node_config)


@register_node("LOCAL_VLLM")
def _(config, node_config):
    from framework.llama.node import LlamaNode
    return LlamaNode(config, node_config)
```

Also update the module docstring at the top of builtins.py — change the `LOCAL_VLLM` line to:

```
  OLLAMA           — OllamaNode(AgentNode)，Ollama HTTP API（别名 LOCAL_VLLM）
  LOCAL_VLLM       — OllamaNode(AgentNode)，同 OLLAMA（向后兼容别名）
```

- [ ] **Step 2: Verify both types resolve**

Note: `builtins.py` must be explicitly imported to trigger `@register_node()` calls — the registry is populated at import time.

```bash
cd /home/kingy/Foundation/BootstrapBuilder
python -c "
import framework.builtins  # triggers all @register_node() registrations
from framework.registry import get_node_factory
from framework.config import AgentConfig

cfg = AgentConfig(name='test', max_retries=1)
nc = {'model': 'llama3', 'id': 'test_node'}

for t in ['OLLAMA', 'LOCAL_VLLM']:
    factory = get_node_factory(t)
    node = factory(cfg, nc)
    print(f'{t}: {type(node).__name__}')
"
```

Expected:
```
OLLAMA: OllamaNode
LOCAL_VLLM: OllamaNode
```

- [ ] **Step 3: Commit**

```bash
git add framework/builtins.py
git commit -m "feat: register OLLAMA node type, keep LOCAL_VLLM as backward compat alias"
```

---

## Chunk 2: Asa Agent

### Task 5: Rewrite `agents/asa/agent.json`

**Files:**
- Modify: `agents/asa/agent.json` (full rewrite to declarative format)

- [ ] **Step 1: Rewrite agent.json**

Replace the entire contents of `agents/asa/agent.json` with:

```json
{
  "name": "asa",
  "llm": "llama",
  "channel_history_limit": 20,
  "graph": {
    "nodes": [
      {
        "id": "llama_main",
        "type": "OLLAMA",
        "model": "llama3.2:3b",
        "endpoint": "http://localhost:11434",
        "first_turn_suffix": "Asa:",
        "user_msg_prefix": "",
        "tombstone_enabled": false,
        "tool_rules": []
      }
    ],
    "edges": [
      {"from": "__start__", "to": "llama_main"},
      {"from": "llama_main", "to": "__end__"}
    ]
  },
  "max_retries": 2,
  "db_path": "asa.db",
  "sessions_file": "sessions.json",
  "persona_files": ["SOUL.md"]
}
```

Note: `discord_token` / `discord_allowed_users` are omitted — CLI-only in Phase 1.

- [ ] **Step 2: Verify AgentLoader can parse it**

```bash
cd /home/kingy/Foundation/BootstrapBuilder
python -c "
from framework.agent_loader import AgentLoader
loader = AgentLoader('agents/asa')
cfg = loader.load_config()
print('config OK:', cfg.name)
graph = loader.build_graph()
print('graph OK:', type(graph).__name__)
"
```

Expected:
```
config OK: asa
graph OK: CompiledStateGraph
```

- [ ] **Step 3: Commit**

```bash
git add agents/asa/agent.json
git commit -m "feat: rewrite asa agent.json to declarative OLLAMA graph"
```

---

### Task 6: Expand `agents/asa/SOUL.md`

**Files:**
- Modify: `agents/asa/SOUL.md`

- [ ] **Step 1: Rewrite SOUL.md**

Note: The bilingual format below is an intentional enhancement over the minimal English-only version in the spec — approved as better UX for a bilingual agent.

Replace the entire contents of `agents/asa/SOUL.md` with:

```markdown
## Asa — 身份与灵魂 / Identity & Soul

你是 Asa，无垠智穹（Boundless Intellect Dome）的第二个 Agent。

You are Asa, the second Agent of the Boundless Intellect Dome (无垠智穹).

你运行在本地 Llama 模型上（通过 Ollama），专注于：
- 快速、轻量的响应（CPU 推理，随时可用）
- 隐私敏感和离线场景
- 系统监控与健康检查（Phase 2 功能）

You run on a local Llama model via Ollama, specialized for:
- Fast, lightweight responses (CPU inference, always available)
- Privacy-sensitive and offline scenarios
- System monitoring and health checks (Phase 2)

### 性格 / Personality

- 简洁直接 — 你是小模型，不要浪费 token
- 务实可靠 — 专注于把事情做好
- 双语：中文和英文均可

- Concise and direct — you're a small model, don't waste tokens
- Practical and reliable — focus on getting things done
- Bilingual: Chinese (中文) and English

### 关系 / Relationships

你与 Hani 共同服务于同一位老板。
Hani 是你的同伴，负责更复杂的任务（Claude + Gemini 双模型）。
你专注于速度和隐私，Hani 专注于深度推理。

You and Hani serve the same boss (老板).
Hani is your peer, handling more complex tasks (Claude + Gemini dual-model).
You focus on speed and privacy; Hani focuses on deep reasoning.
```

- [ ] **Step 2: Verify persona loads**

```bash
cd /home/kingy/Foundation/BootstrapBuilder
python -c "
from framework.agent_loader import AgentLoader
loader = AgentLoader('agents/asa')
prompt = loader.load_system_prompt()
print('persona length:', len(prompt), 'chars')
print(prompt[:200])
"
```

Expected: prints persona length > 100 and shows first 200 chars of SOUL.md content.

- [ ] **Step 3: Commit**

```bash
git add agents/asa/SOUL.md
git commit -m "feat: expand Asa SOUL.md with full bilingual persona"
```

---

## Chunk 3: Smoke Test

### Task 7: Manual smoke test

Prerequisites: Ollama is running with `llama3.2:3b` pulled.

- [ ] **Step 1: Verify Ollama is up**

```bash
curl http://localhost:11434/api/tags
```

Expected: JSON response listing available models. If not running: `ollama serve` in another terminal.

- [ ] **Step 2: Pull model if needed**

```bash
ollama pull llama3.2:3b
```

Expected: Model downloaded or already present.

- [ ] **Step 3: Run Asa CLI**

```bash
cd /home/kingy/Foundation/BootstrapBuilder
python main.py --agent asa cli
```

Expected: CLI starts with Asa prompt.

- [ ] **Step 4: Send a message**

Type: `你好，你是谁？`

Expected: Asa responds in Chinese, identifying itself. Response comes from `llama3.2:3b`.

- [ ] **Step 5: Test error handling (Ollama down)**

In another terminal, stop Ollama: `pkill ollama` or `systemctl stop ollama`

Back in Asa CLI, send another message.

Expected: Friendly error message `[Ollama 连接失败] ...` — no Python traceback, no crash.

- [ ] **Step 6: Restart Ollama and verify recovery**

```bash
ollama serve
```

Send another message in Asa CLI.

Expected: Asa responds normally again.

- [ ] **Step 7: Final commit (if any tweaks were made)**

```bash
git add -p  # review any changes made during testing
git commit -m "fix: address issues found during Asa smoke test"
```
