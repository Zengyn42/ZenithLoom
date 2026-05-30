# Plan: CLI Ctrl+C Two-Stage Interrupt + Soft-Stop Keyword Interception

## Context

The CLI is currently a single-threaded blocking model: stdin is occupied while `invoke_agent` is running, so the user cannot type anything to interrupt it.
Ctrl+C always exits the program immediately regardless of state — poor user experience.

Goals:
- Ctrl+C while agent is running → cancel current task, return to prompt
- Ctrl+C while idle → normal exit (existing behavior unchanged)
- Soft-stop keywords (stop/wait) during CLI input → intercept, do not send to agent, print hint

The Discord side `_channel_tasks` pattern has been validated as feasible; CLI side follows the same approach.

---

## Files Involved

| File | Changes |
|------|---------|
| `interfaces/cli.py` | Main changes: task tracking, two-stage Ctrl+C, soft-stop keywords |
| `framework/base_interface.py` | No changes needed (`invoke_agent` is already async, can be cancelled) |

---

## Implementation

### 1. `_CliInterface.__init__` New Field

```python
self._current_agent_task: asyncio.Task | None = None
```

---

### 2. Main Loop Refactor (`run()` method, lines 112-155)

#### 2a. Soft-Stop Keyword Interception (after `!` command check, before `invoke_agent`)

```python
_SOFT_STOP_WORDS = {"stop", "wait"}
if user_input.lower() in _SOFT_STOP_WORDS:
    print("No task is currently running.")
    continue
```

#### 2b. Wrap `invoke_agent` with `asyncio.create_task()`

Replace the original:
```python
response = await self.invoke_agent(user_input)
```

With:
```python
agent_task = asyncio.create_task(self.invoke_agent(user_input))
self._current_agent_task = agent_task
try:
    response = await agent_task
    if not self._streaming or self._last_stream_chunk_count == 0:
        print(response, end="", flush=True)
except (asyncio.CancelledError, KeyboardInterrupt):
    if not agent_task.done():
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
    print("\nStopped.\n")
    continue          # return to input prompt, do not exit
except Exception as e:
    print(f"\n[Error] {e}", file=sys.stderr)
finally:
    self._current_agent_task = None
```

#### 2c. Ctrl+C While Idle (input() stage) Preserves Exit Behavior

```python
except (KeyboardInterrupt, EOFError):
    print(f"\n\n{agent_name} standing by. Goodbye.")
    await loader.stop_heartbeat()
    break
```
`_current_agent_task` is always `None` here (task cannot be running to reach this point), so no extra check is needed.

---

## Complete `run()` Structure After Changes (pseudocode)

```
while True:
  try:
    user_input = await run_in_executor(input)   # Ctrl+C here → exit
  except KeyboardInterrupt/EOFError → exit

  if not user_input: continue
  if quit keyword → exit

  if user_input starts with "!":
    handle_command(); continue

  if user_input in SOFT_STOP_WORDS:
    print("No task is currently running."); continue

  # Normal conversation
  task = create_task(invoke_agent(user_input))
  _current_agent_task = task
  try:
    response = await task                        # Ctrl+C here → cancel, continue
  except CancelledError/KeyboardInterrupt → print "Stopped", continue
  finally: _current_agent_task = None

  log_snapshot()
```

---

## Validation Method

1. **Normal conversation**: send a message, wait for complete reply, confirm no regression
2. **Ctrl+C while running**: send a message (triggers long reply), immediately press Ctrl+C
   - Expected: prints "Stopped.", returns to `> ` prompt, program does not exit
3. **Ctrl+C while idle**: press Ctrl+C at `> ` prompt directly
   - Expected: prints goodbye message, program exits
4. **Soft-stop keywords**: type `stop`, `STOP`, `Wait`, `wait`
   - Expected: prints "No task is currently running.", returns to prompt, does not send to agent
5. **Snapshot logging**: confirm `log_snapshot()` is still called after normal conversation completion
6. **Run tests**: `python -m pytest test_commands.py -v`, confirm no test regressions
