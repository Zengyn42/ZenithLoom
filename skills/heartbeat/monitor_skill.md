# Background Task Monitor Skill

When an external tool you call (such as `render_slides` or `render_docs`) exceeds 120 seconds,
the tool returns a `[PENDING]` message containing the subprocess PID and related information.

**You must immediately use the `heartbeat_register_monitor` tool to register monitoring.**

---

## Flow

1. You route to an external tool (e.g. `render_slides`)
2. Tool has not returned after 120s → you receive a `[PENDING]` message containing:
   - `task_id`: unique task identifier
   - `pid`: subprocess PID
   - `output_path`: output file path
   - `hard_timeout`: maximum allowed run time (seconds)
3. **You immediately call `heartbeat_register_monitor`** with the above parameters
4. Tell the user the task has been moved to background monitoring and they will be notified when complete
5. The monitoring system checks the PID every 60 seconds and automatically notifies the user when done

## Call Example

When you receive a message like this:

```
[PENDING] Command exceeded 120s, subprocess is still running in the background.
task_id: tool_abc123
pid: 12345
output_path: /path/to/output
hard_timeout: 600
```

You should immediately call:

```
heartbeat_register_monitor(task_id="tool_abc123", pid=12345, output_path="/path/to/output", hard_timeout=600)
```

Then reply to the user: "The task is running in the background. I'll notify you automatically when it completes."

## Other Available Tools

- `heartbeat_my_monitors()` — View all background tasks you are currently monitoring
