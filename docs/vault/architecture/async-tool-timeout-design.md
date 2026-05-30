# External Tool Async Timeout Design

> Last updated: 2026-03-24
> Source: Three rounds of Claude-Gemini Design Review (debate_design)
> Status: **Approved, pending implementation**

---

## I. Problem Statement

The current LangGraph main loop blocks synchronously when executing external tools (Claude CLI, Gemini CLI, shell commands and other subprocess calls). When tool execution time is unpredictable, the entire graph hangs and users receive no response.

**Limitations of the existing timeout mechanism**: each node has a hard timeout (30s~300s); timeout directly kills the subprocess — all work is lost.

---

## II. Final Solution

**One-line summary: launch with tee + after soft_timeout do NOT kill — instead register Heartbeat monitoring + write result to Task Vault + ring bell to notify user to manually resume.**

### Core Data Flow

```
ExternalToolNode starts subprocess (tee to buffer + BoundedFile)
        │
        ├─ subprocess completes within soft_timeout → normal ToolMessage return
        │
        └─ soft_timeout reached →
              ① return ToolMessage(content="[PENDING]", metadata={task_id})
              ② AsyncTaskManager registers monitoring task with Heartbeat
              ③ Heartbeat non-blocking detects subprocess completion → writes result to Task Vault
              ④ SSE push "task_completed" → BaseInterface background thread receives it
              ⑤ Ring bell \a to alert user
              ⑥ On user's next input, automatically fetch result from Vault, overwrite PENDING message
              ⑦ Initiate new graph.invoke(), passing in corrected complete history
```

---

## III. V1 Scope

| Do | Don't Do (V2) |
|---|---|
| Background pure subprocess tools (shell, Gemini CLI) | Background Claude SDK (just add progress hints) |
| Single background task limit | Multi-task concurrency |
| Bell + user manually triggers result injection | Auto resume |
| PID Registry + atexit cleanup | process group approach |
| BoundedFileWriter 50MB truncation protection | Dynamic truncation threshold adjustment |

---

## IV. Key Design Decisions

| # | Decision | Choice | Core Reason |
|---|----------|--------|-------------|
| 1 | IO strategy | tee at launch, not mid-stream switch | Mid-stream pipe redirection risks data loss and broken pipe; tee is simple and reliable |
| 2 | Graph suspend mechanism | `[PENDING]` message overwrite + new invocation | No dependency on specific LangGraph version; compatible with synchronous CLI model; easy to debug |
| 3 | Result re-injection trigger | Bell + user manually triggers (not auto resume) | BaseInterface CLI is synchronous blocking model; event-driven refactor is too costly; V1 low-risk compromise |
| 4 | Claude SDK handling | V1 no background, just progress hints | Nested tool use background autonomous execution has security concerns, complexity uncontrollable |
| 5 | Concurrent background tasks | V1 single task limit | Avoids out-of-order injection and state consistency issues; concurrent long tasks are extremely rare in practice |
| 6 | Timeout management | `call_later` callback, not Min-Heap | Background task count ≤3, simple callback sufficient; Min-Heap is over-engineering |
| 7 | Process cleanup | PID Registry + atexit per-process cleanup | Avoids macOS compatibility issues with `os.setsid`; cross-platform safe |
| 8 | Result archiving | Task Vault persistence (diskcache/jsonl) | Results not lost after user rollback; supports `/task result <id>` post-query |
| 9 | Disk protection | BoundedFileWriter (50MB truncation) | Prevent infinite-loop output from filling disk; truncation does not kill process, kill is handled by hard_timeout |
| 10 | soft_timeout values | Differentiated by node type, Blueprint configurable | Shell 30s / Gemini CLI 60s / Claude SDK 90s (hint only) |

---

## V. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Orphan process leak (subprocess remaining after main process crash) | High | PID Registry + `atexit` hook; Heartbeat scans remaining PID files and cleans up on startup |
| Checkpoint discontinuity (PENDING overwrite breaks append-only semantics) | Medium | Overwrite using same message ID rather than appending; router node explicitly handles PENDING state |
| User rollback causes result injection context inconsistency | Medium | Verify task_id corresponds to conversation_turn before injection; if mismatch, archive without injecting |
| Disk full | Medium | BoundedFileWriter 50MB hard limit + truncation marker |
| Heartbeat itself restarts and loses monitoring state | Medium | Task registration info persisted to Task Vault; reconciliation scan on startup |
| Asyncio cross-thread communication deadlock | Medium | Strictly use `run_coroutine_threadsafe` + Heartbeat exposes `get_loop()` |
| soft_timeout false positives | Low | Differentiated configuration by node type; Blueprint supports runtime override |

---

## VI. File Structure and Interface

### Modification Scope (6 files)

```
# New files
framework/async_task_manager.py      # background task lifecycle management
framework/bounded_file_writer.py     # file writer with truncation protection

# Modifications (core)
framework/nodes/external_tool_node.py     # base class adds soft_timeout + tee logic
framework/heartbeat.py                    # register TASK_MONITOR + completion detection
framework/base_interface.py               # SSE listening + bell + PENDING consumption
mcp_servers/heartbeat/server.py           # new task_completed SSE event type

# Minimal modifications (config/registration only)
framework/builtins.py                     # register TASK_MONITOR node type
```

### Key Interfaces

```python
# === AsyncTaskManager (new) ===
class AsyncTaskManager:
    register_task(task_id, pid, output_path, hard_timeout) -> None
    query_task(task_id) -> TaskStatus  # RUNNING | COMPLETED | FAILED | TIMEOUT
    get_result(task_id) -> str | None
    cancel_task(task_id) -> bool
    cleanup_all() -> None              # called by atexit


# === BoundedFileWriter (new) ===
class BoundedFileWriter:
    __init__(path, max_bytes=50_000_000)  # 50MB
    write(data: bytes) -> int             # silently discards over limit, writes truncation marker
    close() -> None
    path -> str


# === ExternalToolNode new hooks (modified) ===
class ExternalToolNode:
    soft_timeout: int       # subclass can override, or read from Blueprint
    hard_timeout: int

    on_soft_timeout(proc, task_id, output_path) -> ToolMessage
        # default: register Heartbeat task, return PENDING message
        # Claude subclass overrides: just print progress hint, don't interrupt

    _tee_subprocess(cmd) -> (proc, buffer, BoundedFileWriter)
        # start subprocess + tee thread


# === BaseInterface new additions (modified) ===
class BaseInterface:
    _completed_tasks_queue: queue.Queue     # thread-safe queue
    _sse_listener_thread: Thread            # background Heartbeat SSE listener

    _on_task_completed(task_id) -> None     # enqueue + ring bell
    _consume_pending_tasks(state) -> state  # called before invoke, overwrites PENDING messages


# === Heartbeat additions (modified) ===
# heartbeat.py
get_loop() -> asyncio.AbstractEventLoop     # thread-safe loop accessor

# server.py — new SSE event
# event: task_completed
# data: { "task_id": "xxx", "status": "completed|failed|timeout" }
```

---

## VII. Implementation Order

1. `bounded_file_writer.py` + `async_task_manager.py` (infrastructure, can be tested independently)
2. `external_tool_node.py` refactor (core logic: tee + soft_timeout branching)
3. Heartbeat side (`heartbeat.py` + `server.py`, TASK_MONITOR registration and detection)
4. `base_interface.py` (result consumption + PENDING overwrite + bell notification)

---

## VIII. V2 Outlook (out of V1 scope)

- BaseInterface CLI introduces `prompt_toolkit`, implements event-driven auto resume
- LangGraph `interrupt/Command(resume=...)` native suspend/resume
- Claude SDK background: separate consumer + safe/unsafe tool whitelist
- Multi-task concurrency: introduce sequence number + ordered result queue
- Discord Interface auto resume (already has async event loop, low refactor cost)
