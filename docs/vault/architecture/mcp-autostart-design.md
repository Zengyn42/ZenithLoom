# MCP Auto-Start Mechanism — Design Decision Record

> Created: 2026-04-01
> Status: Design phase, pending implementation
> Impact scope: agent_loader.py, all entity.json files, heartbeat, agent_mail, and all future MCPs

---

## Background

Current problems:
- Heartbeat MCP startup logic is hard-coded in `agent_loader.py`'s `start_heartbeat()` method
- agent_mail MCP needs similar auto-start capability
- Every time a new MCP is added, it requires separate handling in agent_loader

**Decision: Unify all MCP auto-start logic; blueprints/entity.json declare dependencies; framework handles automatically.**

---

## Core Design

### entity.json New `mcps` Field

```json
{
  "name": "administrative_officer",
  "blueprint": "...",
  "mcps": [
    {
      "name": "heartbeat",
      "module": "mcp_servers.heartbeat",
      "transport": "sse",
      "host": "127.0.0.1",
      "port": 8100,
      "pid_file": "data/heartbeat/mcp.pid"
    },
    {
      "name": "agent_mail",
      "module": "mcp_servers.agent_mail",
      "transport": "sse",
      "host": "127.0.0.1",
      "port": 8200,
      "pid_file": "data/agent_mail/mail.pid"
    }
  ]
}
```

Each MCP declares: `module` (Python module), `transport`, `host:port`, `pid_file` (for liveness detection).

---

## Startup Logic (Generic, Applies to All MCPs)

```
agent starts
    ↓
iterate entity.json mcps list
    ↓
for each MCP:
    _is_running(pid_file)?
        ├── no → acquire file lock → double-check → launch (detach) → release lock → wait for ready
        └── yes → skip launch
    ↓
connect (SSE) → register tools to TOOL_REGISTRY
```

### File Lock Prevents Race Condition (Fixes heartbeat race condition)

```python
lock_path = pid_file.with_suffix(".launch.lock")
with open(lock_path, "w") as lf:
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not _is_running(pid_file):        # double-check
            _launch(module, host, port, pid_file)
    except BlockingIOError:
        pass  # someone else is launching, just wait for connection retry
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
```

---

## agent_loader.py Refactor

### Current (Hard-coded)

```python
async def start_heartbeat(self): ...       # heartbeat-specific
async def start_mcp_servers(self): ...     # obsidian etc. specific
async def _connect_heartbeat_proxy_only(): # EXTERNAL_TOOL specific
```

### After Refactor (Generic)

```python
async def start_mcps(self):
    """Iterate entity.json mcps field, ensure each is running and connected."""
    for mcp_conf in self.entity.get("mcps", []):
        proxy = await MCPLauncher.ensure_and_connect(mcp_conf)
        if proxy:
            self._mcp_proxies[mcp_conf["name"]] = proxy
            # register tools to corresponding LLM tool registry
            self._register_mcp_tools(mcp_conf["name"], proxy)
```

`start_heartbeat()` is retained but internally routes through `start_mcps()` unified path; eventually deprecated.

---

## MCPLauncher (New Module)

Location: `framework/mcp_launcher.py`

Responsibilities:
- `ensure_and_connect(mcp_conf)` — check + auto-start + connect, return proxy
- `_is_running(pid_file)` — check process liveness
- `_launch(module, host, port, pid_file)` — detach launch + write PID file
- `_wait_ready(url, timeout=10)` — poll until SSE endpoint is ready

---

## Per-entity.json Update Plan

| Agent | Current MCP config | After migration mcps field |
|-------|-------------------|--------------------------|
| administrative_officer (administrative_officer) | `"heartbeat": [...]` | heartbeat + agent_mail |
| knowledge_curator (knowledge_curator) | none (but incorrectly connected heartbeat) | agent_mail |
| technical_architect (technical_architect) | none | agent_mail |

---

## Relationship to Existing Designs

- **Heartbeat race condition fix**: unified file lock, resolves the race condition at its root
- **knowledge_curator zombie fix**: knowledge_curator's mcps field does not include heartbeat, eliminating erroneous connections
- **agent_mail startup**: same status as heartbeat; first agent to start brings up the mail server

---

## Pending Implementation

- [ ] `framework/mcp_launcher.py` — generic MCPLauncher
- [ ] `agent_loader.py` — `start_mcps()` replaces `start_heartbeat()`
- [ ] `EdenGateway/agents/administrative_officer/identity.json` — migrate heartbeat config to mcps field, add agent_mail
- [ ] `EdenGateway/agents/knowledge_curator/identity.json` — add agent_mail (no heartbeat)
- [ ] `EdenGateway/agents/technical_architect/identity.json` — add agent_mail
- [ ] `mcp_servers/agent_mail/server.py` — complete SIGUSR1 notification logic
