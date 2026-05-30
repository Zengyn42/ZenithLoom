# Agent Mail — Design Decision Record

> Created: 2026-04-01
> Status: Design phase, pending implementation

---

## Background

Current problem: when the knowledge_curator's EXTERNAL_TOOL execution exceeds 120s, monitoring of the task needs to be delegated to the administrative_officer (the sole heartbeat owner in the system).
Core requirement: agents need a reliable asynchronous messaging mechanism.

---

## Design Principles

1. **administrative_officer is the sole heartbeat owner**: other agents do not start a heartbeat MCP server, and do not connect to the heartbeat proxy at startup
2. **Asynchronous decoupling**: sender fires and forgets; does not wait for the receiver to be online
3. **Persistence first**: inbox survives agent restarts
4. **MCP interface**: consistent with the existing MCP architecture; tools are invoked the same way

---

## Core Design: "Mailbox" Model

Each agent has an independent inbox. Communication uses three verbs:

```
send_mail(to, subject, body)   → writes to recipient's inbox
fetch_inbox(agent_name)        → reads own unprocessed mail
ack_mail(mail_id)              → marks as processed
```

### Message Structure

```json
{
  "mail_id": "uuid",
  "from_agent": "knowledge_curator",
  "to_agent": "administrative_officer",
  "subject": "monitor_delegate",
  "body": {
    "task_id": "tool_abc123",
    "pid": 9876,
    "output_path": "/path/to/output",
    "hard_timeout": 600,
    "notify_channel": "1488233657742393358"
  },
  "created_at": "2026-04-01T22:00:00Z",
  "acked_at": null
}
```

---

## Architecture Options

### Selection Process

| Option | Conclusion |
|--------|-----------|
| Discord channel as message bus | ❌ Critical path should not depend on external network |
| SQLite + inotify | ⚠️ Feasible but asyncio integration has pitfalls |
| Agent Bus MCP Server (new) | ⚠️ Architecture-consistent but administrative_officer lacks tool-calling ability |
| Unix Domain Socket | ✅ Low latency but no persistence |
| **Agent Mail MCP Server** | ✅ Final choice |
| Supervisor LLM (Grok proposal) | ❌ Transport layer should not use LLM; hallucination risk; single point of failure |

### Reference Projects

- [mcp_agent_mail](https://github.com/Dicklesworthstone/mcp_agent_mail) (1.9k stars): aligned design direction but too heavyweight (file reservations, git tracking, Beads integration, Python 3.14)
- Decision: **reference its interface design, build a lightweight version** `mcp_servers/agent_mail/`

---

## Implementation Plan

### Storage

Each agent's own SQLite db (e.g. `administrative_officer.db`) contains a `mailbox` table, or a unified `shared.db` is used.
Preference: **shared.db**, path `data/agent_mail/mail.db`, single source of truth, cross-agent queries are easy.

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS mailbox (
    mail_id     TEXT PRIMARY KEY,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,       -- JSON string
    created_at  TEXT NOT NULL,
    acked_at    TEXT DEFAULT NULL
);

CREATE INDEX idx_inbox ON mailbox (to_agent, acked_at);
```

### MCP Tool Interface

```python
@mcp.tool()
async def send_mail(to: str, subject: str, body: dict) -> dict

@mcp.tool()
async def fetch_inbox(agent_name: str, unread_only: bool = True) -> list[dict]

@mcp.tool()
async def ack_mail(mail_id: str) -> dict

@mcp.tool()
async def list_agents() -> list[dict]   # agent discovery
```

### Read/Write Separation: No Forced MCP Connection at Startup

**Core principle: reads go directly to SQL, writes go through MCP.**

```
Read path (receiving mail): agent process's background task reads mail.db directly
                  → no MCP persistent connection, no additional process dependency
                  → if mail server is down, inbox is still readable

Write path (sending mail): lazy-load, connect to mail MCP server only when first send is needed
                  → disconnect after sending, no persistent connection
```

| Agent | Connect to mail MCP at startup? | Reason |
|-------|--------------------------------|--------|
| administrative_officer | ❌ reads SQLite directly (background task) | monitor owner, continuously polls inbox |
| knowledge_curator | ❌ lazy-load when PENDING | only sends one mail on timeout delegation |
| technical_architect | ❌ lazy-load on demand | sends only when coordination is needed |

**The mail MCP server is only the write entry point** — not all agents need to connect to it at startup.

### Trigger Mechanism (How Agents Detect New Mail)

Each agent process starts an asyncio background task that directly queries `mail.db` every **1 second**:

```sql
SELECT * FROM mailbox WHERE to_agent = ? AND acked_at IS NULL ORDER BY created_at ASC
```

New mail received → triggers `_on_mail_received(mail)` callback → handled by the framework layer
(e.g.: administrative_officer receives `monitor_delegate` → calls `heartbeat_register_monitor`).

### Agent Discovery (see next section)

---

## Agent Discovery Mechanism

### Static Discovery: Directory Scan

The `EdenGateway/agents/` directory is itself a registry. Blueprint definitions are in the VoidDraft repo; instance runtime data lives in EdenGateway:

```
EdenGateway/agents/
├── administrative_officer/identity.json    → name: "administrative_officer"  (administrative_officer instance)
├── technical_architect/identity.json   → name: "technical_architect" (technical_architect instance)
└── knowledge_curator/identity.json    → name: "knowledge_curator"  (knowledge_curator instance)
```

The `list_agents()` tool scans this directory and returns all known agents.
Advantage: offline agents can still be discovered; no runtime registration required.

### Dynamic State: Heartbeat Registration

When an agent process starts, it calls `register_agent(name, pid)` on the mail server,
and calls `unregister_agent(name)` on shutdown.
The mail server maintains `online_since` / `last_seen` fields.

Combined: **statically know who exists, dynamically know who is online.**

---

## Integration with Existing Architecture

### Fix the knowledge_curator Startup Heartbeat Binding Issue

`agent_loader.py` lines 344-348 need to be modified:
```python
# Current (incorrect): connect heartbeat if EXTERNAL_TOOL nodes exist
if self._has_external_tool_nodes():
    return await self._connect_heartbeat_proxy_only()

# After fix: EXTERNAL_TOOL present but mail_delegate configured → use agent mail path
# knowledge_curator does not connect heartbeat at startup; delegates via mail when PENDING
```

### ExternalToolNode._on_timeout() New Delegation Logic

```python
# When PENDING occurs, send mail to administrative_officer
await send_mail(
    to="administrative_officer",
    subject="monitor_delegate",
    body={"task_id": task_id, "pid": proc.pid, ...}
)
```

### administrative_officer Inbox Handling

The administrative_officer's mailbox watcher receives `monitor_delegate` → directly calls `heartbeat_register_monitor` → after registration, sends mail back to the notify_channel's corresponding agent.

---

## Pending Decisions

- [x] ~~Must agents connect to mail MCP at startup?~~ → **Yes, through the unified MCP auto-start mechanism**
- [x] ~~Lazy-load vs startup connection?~~ → **Startup connection, declared in entity.json mcps field**
- [x] ~~Register PID via MCP or directly SQL?~~ → **Register via MCP tool on connection; connection = registration**
- [ ] shared.db vs individual dbs? → preference for shared.db (`data/agent_mail/mail.db`)
- [ ] Does `list_agents()` need "online status"?

## Related Documents

- [MCP Auto-Start Mechanism](./mcp-autostart-design.md) — the agent_mail startup logic is part of the framework-level general design
