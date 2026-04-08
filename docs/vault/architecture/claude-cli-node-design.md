# ClaudeCLINode Design

> Date: 2026-04-08
> Status: Approved

## Summary

新增 `ClaudeCLINode`，通过 `asyncio.create_subprocess_exec("claude", "-p", ...)` 直接调用 Claude CLI subprocess，作为 `ClaudeSDKNode`（通过 `claude_agent_sdk`）的替代方案。功能对齐 SDK 版本，包括 streaming 和 session resume。

## Motivation

现有 `ClaudeSDKNode` 通过 `claude_agent_sdk` 的 `query()` 函数调用，SDK 底层封装了 CLI 子进程。新增 `ClaudeCLINode` 直接调用 CLI，绕过 SDK 层，提供：
- 更直接的 CLI 控制（flag 级别参数透传）
- 不依赖 `claude_agent_sdk` Python 包
- 与 `GeminiCLINode` 对称的架构

## Architecture

`ClaudeCLINode` 与 `ClaudeSDKNode` 并列，共存于 `framework/nodes/llm/claude.py`：

```
claude.py
  ├── ClaudeSDKNode  (CLAUDE_SDK) — 通过 claude_agent_sdk
  └── ClaudeCLINode  (CLAUDE_CLI) — 通过 subprocess
```

### 继承关系

```
LlmNode (base)
  ├── ClaudeSDKNode   → call_llm() via claude_agent_sdk.query()
  ├── ClaudeCLINode   → call_llm() via asyncio.create_subprocess_exec("claude")
  ├── GeminiCLINode   → call_llm() via subprocess "gemini"
  ├── GeminiCodeAssistNode → call_llm() via HTTP API
  └── OllamaNode      → call_llm() via HTTP API
```

`ClaudeCLINode` **复用基类 `LlmNode.__call__()`**，只实现 `call_llm()`。

## CLI Invocation

```bash
claude -p \
  --output-format stream-json \
  --verbose \
  --include-partial-messages \
  --model <model> \
  --permission-mode <mode> \
  --system-prompt <prompt> \
  --allowedTools <tools...> \
  --disallowedTools <tools...> \
  --resume <session_id> \
  --add-dir <dirs...> \
  --setting-sources <sources> \
  --settings <json> \
  --mcp-config <json>
```

- Prompt 通过 **stdin** 传入（避免 yargs 参数解析问题，同 GeminiCLINode）
- 环境变量设置 `CLAUDE_AGENT_SDK=1` 抑制 hook 声音

## Stream-JSON Parsing

`--output-format stream-json --verbose --include-partial-messages` 输出为逐行 JSON：

| `type` | 处理方式 |
|---|---|
| `"stream_event"` | 解析 `event.content_block_delta`：`text_delta` → `cb(text, False)`，`thinking_delta` → `cb(thinking, True)` |
| `"result"` | 提取 `result`、`session_id`、`is_error`、`usage`；调用 `update_token_stats()` |
| `"system"` | 跳过（hook events、init info） |
| `"assistant"` | 跳过（partial assembled messages） |
| `"rate_limit_event"` | 跳过 |

## Session Resume

- 新 session：不传 `--resume`
- 续接：`--resume <session_id>`
- Resume 失败（returncode != 0）→ 新 session 重试 → 再失败返回错误文本 + 空 session_id
- 与 ClaudeSDKNode 的重试策略完全一致

## Permission Mode

CLI 原生支持 `--permission-mode`，直传 `self._permission_mode`。
`disallowed_tools` 通过 `--disallowedTools` 传入（基类 `_get_disallowed_tools()` 计算）。

## Timeout

动态超时，借鉴 GeminiCLINode：
- 基线：120s
- 上限：600s
- 缩放：`prompt_len / 200` 每字符增加时间
- 公式：`min(600, max(120, prompt_len // 200))`

## SDK Options to CLI Flags Mapping

| ClaudeSDKNode (SDK option) | ClaudeCLINode (CLI flag) |
|---|---|
| `system_prompt` | `--system-prompt` |
| `permission_mode` | `--permission-mode` |
| `model` | `--model` |
| `allowed_tools` | `--allowedTools` |
| `disallowed_tools` | `--disallowedTools` |
| `resume` | `--resume` |
| `add_dirs` | `--add-dir` |
| `setting_sources` | `--setting-sources` |
| `settings` (JSON) | `--settings` |
| `mcp_servers` | `--mcp-config` (JSON string) |
| `env` | subprocess env vars |
| `cwd` | subprocess cwd |
| `thinking` | 无直接 CLI 对应（依赖模型默认行为） |
| `max_buffer_size` | 不适用（自行管理 stdout） |
| `include_partial_messages` | `--include-partial-messages` |

## Registry Changes

| builtins.py 注册 | 指向 |
|---|---|
| `CLAUDE_CLI` | `ClaudeCLINode`（新） |
| `CLAUDE_SDK` | `ClaudeSDKNode`（不变） |

## Entity.json Migration

所有现有 entity.json 中的 `"type": "CLAUDE_CLI"` 改为 `"type": "CLAUDE_SDK"`，确保现有行为不变：

- `blueprints/role_agents/technical_architect/entity.json` (1 处)
- `blueprints/functional_graphs/debate_claude_first/entity.json` (3 处)
- `blueprints/functional_graphs/debate_gemini_first/entity.json` (2 处)
- `blueprints/functional_graphs/apex_coder/entity.json` (1 处)
- `blueprints/functional_graphs/tool_discovery/entity.json` (3 处)

## Not Doing

- **不覆写 `__call__`** — 复用基类 LlmNode，避免 GeminiCLINode 的重复逻辑
- **不使用 `--bare` 标志** — 保持与普通 CLI 行为一致，让 hooks/CLAUDE.md 正常工作
- **不做模型降级链** — ClaudeSDKNode 没有，保持一致；降级是 Gemini 特有需求
- **不需要 `get_recent_history` / `list_sessions`** — 这些是 SDK 专有方法，CLI 节点不需要
