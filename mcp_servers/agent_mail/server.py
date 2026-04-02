"""
Agent Mail MCP Server — mcp_servers/agent_mail/server.py

轻量 agent 间"邮件收件箱"通信机制。
收发分离原则：
  写路径（发邮件）：通过此 MCP server 的工具
  读路径（收邮件）：agent 进程直接读 SQLite，不经过 MCP

启动方式：
  python -m mcp_servers.agent_mail                          # stdio
  python -m mcp_servers.agent_mail --transport sse          # SSE（多客户端）
  python -m mcp_servers.agent_mail --transport sse --port 8200
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

# 确保项目根在 sys.path 中（支持 python -m 启动）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("agent_mail_mcp")

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

_DB_DIR: Path = _PROJECT_ROOT / "data" / "agent_mail"
_DB_PATH: Path = _DB_DIR / "mail.db"
_AGENTS_DIR: Path = Path.home() / "Foundation" / "EdenGateway" / "agents"

# 允许通过环境变量覆盖 agents 目录
_AGENTS_DIR = Path(os.environ.get("EDEN_AGENTS_DIR", str(_AGENTS_DIR)))

# ---------------------------------------------------------------------------
# DB 初始化
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mailbox (
    mail_id     TEXT PRIMARY KEY,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    acked_at    TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_inbox ON mailbox (to_agent, acked_at);

CREATE TABLE IF NOT EXISTS agents (
    name        TEXT PRIMARY KEY,
    pid         INTEGER,
    online_since TEXT,
    last_seen   TEXT
);
"""


async def _init_db() -> None:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executescript(_SCHEMA)
        await db.commit()
    logger.info(f"DB initialised at {_DB_PATH}")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

def _sigusr1_handler(signum, frame):
    """SIGUSR1 handler for the MCP server process.

    The server sends SIGUSR1 to agent processes upon mail delivery.  If a
    stale PID entry in the agents table coincidentally refers to the server
    process itself, or if the OS recycles a PID, the server would be killed
    by the default SIGUSR1 action (terminate).  Installing a no-op handler
    prevents accidental self-termination.
    """
    logger.debug("[agent_mail_mcp] SIGUSR1 received (ignored by server process)")


# Install the handler at module load time so it is active regardless of
# transport mode (stdio or SSE).
signal.signal(signal.SIGUSR1, _sigusr1_handler)


@asynccontextmanager
async def lifespan(server):
    await _init_db()
    logger.info("Agent Mail MCP Server started")
    yield
    logger.info("Agent Mail MCP Server shutting down")


# ---------------------------------------------------------------------------
# MCP Server 定义
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="agent_mail",
    instructions=(
        "Agent Mail: async inter-agent messaging via SQLite mailbox. "
        "Use send_mail to deliver a message, fetch_inbox to list received mail, "
        "ack_mail to mark as read, list_agents to discover known agents, "
        "register_agent / unregister_agent to track online status."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------

@mcp.tool()
async def send_mail(from_agent: str, to: str, subject: str, body: str) -> dict:
    """
    发送一封邮件给目标 agent。

    from_agent: 发件人 agent 名称
    to:         收件人 agent 名称
    subject:    邮件主题（如 "monitor_delegate"）
    body:       邮件正文，JSON string（调用方负责序列化）
    返回: {"mail_id": "...", "status": "sent"}
    """
    mail_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(
            "INSERT INTO mailbox (mail_id, from_agent, to_agent, subject, body, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (mail_id, from_agent, to, subject, body, created_at),
        )
        await db.commit()

        # 写入后通知目标 agent：查 agents 表拿 PID，发 SIGUSR1
        row = await db.execute("SELECT pid FROM agents WHERE name=?", (to,))
        r = await row.fetchone()
        if r and r[0]:
            try:
                os.kill(r[0], signal.SIGUSR1)
                logger.debug(f"send_mail: SIGUSR1 → {to} pid={r[0]}")
            except ProcessLookupError:
                pass  # agent 已离线，邮件留在收件箱等它上线读

    logger.info(f"send_mail: {from_agent} → {to} [{subject}] mail_id={mail_id}")
    return {"mail_id": mail_id, "status": "sent"}


@mcp.tool()
async def fetch_inbox(agent_name: str, unread_only: bool = True) -> list[dict]:
    """
    查询指定 agent 的收件箱。

    agent_name: 收件人 agent 名称
    unread_only: 若为 True（默认），只返回未读（acked_at IS NULL）的邮件
    返回: 邮件列表，每项包含 mail_id / from_agent / subject / body / created_at / acked_at
    """
    if unread_only:
        sql = (
            "SELECT mail_id, from_agent, to_agent, subject, body, created_at, acked_at "
            "FROM mailbox WHERE to_agent = ? AND acked_at IS NULL ORDER BY created_at ASC"
        )
        params = (agent_name,)
    else:
        sql = (
            "SELECT mail_id, from_agent, to_agent, subject, body, created_at, acked_at "
            "FROM mailbox WHERE to_agent = ? ORDER BY created_at ASC"
        )
        params = (agent_name,)

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

    return [dict(row) for row in rows]


@mcp.tool()
async def ack_mail(mail_id: str) -> dict:
    """
    标记邮件为已读（ack）。

    mail_id: 邮件 ID（来自 send_mail 返回值或 fetch_inbox 列表）
    返回: {"mail_id": "...", "status": "acked"} 或 {"status": "not_found"}
    """
    acked_at = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        cursor = await db.execute(
            "UPDATE mailbox SET acked_at = ? WHERE mail_id = ? AND acked_at IS NULL",
            (acked_at, mail_id),
        )
        await db.commit()
        affected = cursor.rowcount

    if affected:
        logger.info(f"ack_mail: mail_id={mail_id}")
        return {"mail_id": mail_id, "status": "acked"}
    return {"mail_id": mail_id, "status": "not_found"}


@mcp.tool()
async def list_agents() -> list[dict]:
    """
    发现所有已知 agent：静态扫描 EdenGateway/agents/ 目录 + 动态查询在线状态表。

    返回: [{"name": "asa", "online": true, "pid": 399, "last_seen": "..."}, ...]
    """
    # 静态发现：扫描目录
    known: set[str] = set()
    if _AGENTS_DIR.is_dir():
        for entry in _AGENTS_DIR.iterdir():
            if entry.is_dir():
                known.add(entry.name)

    # 动态在线状态：从 agents 表读
    online_map: dict[str, dict] = {}
    try:
        async with aiosqlite.connect(_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT name, pid, online_since, last_seen FROM agents"
            ) as cursor:
                rows = await cursor.fetchall()
        for row in rows:
            r = dict(row)
            online_map[r["name"]] = r
            known.add(r["name"])
    except Exception as exc:
        logger.warning(f"list_agents: DB query failed: {exc}")

    result = []
    for name in sorted(known):
        rec = online_map.get(name, {})
        result.append({
            "name": name,
            "online": name in online_map,
            "pid": rec.get("pid"),
            "online_since": rec.get("online_since"),
            "last_seen": rec.get("last_seen"),
        })
    return result


@mcp.tool()
async def register_agent(name: str, pid: int) -> dict:
    """
    agent 进程启动时注册在线状态。

    name: agent 名称
    pid:  agent 进程 PID
    返回: {"name": "...", "status": "registered"}
    """
    now = datetime.now(timezone.utc).isoformat()

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(
            "INSERT INTO agents (name, pid, online_since, last_seen) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET pid=excluded.pid, "
            "online_since=excluded.online_since, last_seen=excluded.last_seen",
            (name, pid, now, now),
        )
        await db.commit()

    logger.info(f"register_agent: {name} pid={pid}")
    return {"name": name, "status": "registered"}


@mcp.tool()
async def unregister_agent(name: str) -> dict:
    """
    agent 进程关闭时注销在线状态。

    name: agent 名称
    返回: {"name": "...", "status": "unregistered"} 或 {"status": "not_found"}
    """
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        cursor = await db.execute("DELETE FROM agents WHERE name = ?", (name,))
        await db.commit()
        affected = cursor.rowcount

    if affected:
        logger.info(f"unregister_agent: {name}")
        return {"name": name, "status": "unregistered"}
    return {"name": name, "status": "not_found"}


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agent Mail MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode (default: stdio)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8200, help="SSE port (default: 8200)")
    args = parser.parse_args()

    logger.info(f"Starting Agent Mail MCP Server (transport={args.transport})")

    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
