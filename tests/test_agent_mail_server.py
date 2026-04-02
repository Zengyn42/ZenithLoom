"""
Tests for mcp_servers/agent_mail/server.py — core tool logic.

No real MCP server is started. The tool functions are called directly with
a temporary SQLite database substituted for the module-level _DB_PATH.
"""

import os
import signal
from pathlib import Path
from unittest.mock import patch, MagicMock

import aiosqlite
import pytest

# We import the module so we can monkey-patch _DB_PATH before each test.
import mcp_servers.agent_mail.server as _srv


# ---------------------------------------------------------------------------
# Fixture: isolated in-memory SQLite database per test
# ---------------------------------------------------------------------------

@pytest.fixture
async def db_path(tmp_path):
    """Create a fresh SQLite DB for each test and patch _DB_PATH."""
    path = tmp_path / "test_mail.db"
    # Initialise the schema using the same SQL as the real server
    async with aiosqlite.connect(path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executescript(_srv._SCHEMA)
        await db.commit()

    with patch.object(_srv, "_DB_PATH", path):
        yield path


# ---------------------------------------------------------------------------
# send_mail
# ---------------------------------------------------------------------------

class TestSendMail:
    async def test_send_mail_writes_row_and_returns_mail_id(self, db_path):
        result = await _srv.send_mail("alice", "bob", "hello", '{"msg": "hi"}')
        assert result["status"] == "sent"
        assert len(result["mail_id"]) == 32  # uuid4 hex

        # Verify DB row
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM mailbox WHERE mail_id=?",
                                  (result["mail_id"],)) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row["from_agent"] == "alice"
        assert row["to_agent"] == "bob"
        assert row["acked_at"] is None

    async def test_send_mail_unknown_recipient_still_writes(self, db_path):
        """Recipient not in agents table — mail is stored anyway."""
        result = await _srv.send_mail("alice", "nonexistent_agent", "ping", "body")
        assert result["status"] == "sent"

    async def test_send_mail_sends_sigusr1_when_pid_known(self, db_path):
        """If recipient has a PID in agents table, SIGUSR1 should be sent."""
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO agents (name, pid, online_since, last_seen) VALUES (?, ?, ?, ?)",
                ("bob", 12345, "2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
            )
            await db.commit()

        with patch("os.kill") as mock_kill:
            await _srv.send_mail("alice", "bob", "subj", "body")

        mock_kill.assert_called_once_with(12345, signal.SIGUSR1)

    async def test_send_mail_skips_sigusr1_when_pid_not_in_table(self, db_path):
        """No agents row → no SIGUSR1 sent."""
        with patch("os.kill") as mock_kill:
            await _srv.send_mail("alice", "ghost", "subj", "body")
        mock_kill.assert_not_called()

    async def test_send_mail_ignores_process_lookup_error(self, db_path):
        """If process is dead, ProcessLookupError should be swallowed."""
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO agents (name, pid, online_since, last_seen) VALUES (?, ?, ?, ?)",
                ("bob", 999999999, "2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
            )
            await db.commit()

        with patch("os.kill", side_effect=ProcessLookupError):
            # Should not raise
            result = await _srv.send_mail("alice", "bob", "subj", "body")

        assert result["status"] == "sent"


# ---------------------------------------------------------------------------
# fetch_inbox
# ---------------------------------------------------------------------------

class TestFetchInbox:
    async def _seed_mails(self, db_path, to_agent: str, count: int = 2,
                          ack_first: bool = False):
        """Insert `count` mails for to_agent; optionally ack the first one."""
        import uuid
        from datetime import datetime, timezone

        mail_ids = []
        async with aiosqlite.connect(db_path) as db:
            for i in range(count):
                mid = uuid.uuid4().hex
                mail_ids.append(mid)
                acked = datetime.now(timezone.utc).isoformat() if (ack_first and i == 0) else None
                await db.execute(
                    "INSERT INTO mailbox (mail_id, from_agent, to_agent, subject, body, "
                    "created_at, acked_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (mid, "sender", to_agent, f"subj{i}", "body",
                     datetime.now(timezone.utc).isoformat(), acked),
                )
            await db.commit()
        return mail_ids

    async def test_fetch_inbox_unread_only_returns_unacked(self, db_path):
        await self._seed_mails(db_path, "bob", count=3, ack_first=True)
        rows = await _srv.fetch_inbox("bob", unread_only=True)
        assert len(rows) == 2
        for r in rows:
            assert r["acked_at"] is None

    async def test_fetch_inbox_all_returns_all_mails(self, db_path):
        await self._seed_mails(db_path, "bob", count=3, ack_first=True)
        rows = await _srv.fetch_inbox("bob", unread_only=False)
        assert len(rows) == 3

    async def test_fetch_inbox_empty_returns_empty_list(self, db_path):
        rows = await _srv.fetch_inbox("nobody", unread_only=True)
        assert rows == []

    async def test_fetch_inbox_only_returns_own_mails(self, db_path):
        await self._seed_mails(db_path, "alice", count=2)
        await self._seed_mails(db_path, "bob", count=1)
        rows = await _srv.fetch_inbox("alice", unread_only=False)
        assert len(rows) == 2
        for r in rows:
            assert r["to_agent"] == "alice"


# ---------------------------------------------------------------------------
# ack_mail
# ---------------------------------------------------------------------------

class TestAckMail:
    async def _insert_mail(self, db_path, mail_id: str, acked: bool = False):
        from datetime import datetime, timezone
        acked_at = datetime.now(timezone.utc).isoformat() if acked else None
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO mailbox (mail_id, from_agent, to_agent, subject, body, "
                "created_at, acked_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (mail_id, "s", "r", "subj", "body",
                 datetime.now(timezone.utc).isoformat(), acked_at),
            )
            await db.commit()

    async def test_ack_existing_mail_returns_acked(self, db_path):
        await self._insert_mail(db_path, "aaa111")
        result = await _srv.ack_mail("aaa111")
        assert result["status"] == "acked"
        assert result["mail_id"] == "aaa111"

    async def test_ack_mail_sets_acked_at_in_db(self, db_path):
        await self._insert_mail(db_path, "bbb222")
        await _srv.ack_mail("bbb222")
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT acked_at FROM mailbox WHERE mail_id=?",
                                  ("bbb222",)) as cur:
                row = await cur.fetchone()
        assert row[0] is not None

    async def test_ack_nonexistent_mail_returns_not_found(self, db_path):
        result = await _srv.ack_mail("does_not_exist")
        assert result["status"] == "not_found"

    async def test_ack_already_acked_mail_returns_not_found(self, db_path):
        """Second ack on same mail_id should return not_found (acked_at IS NULL guard)."""
        await self._insert_mail(db_path, "ccc333")
        await _srv.ack_mail("ccc333")
        result = await _srv.ack_mail("ccc333")
        assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------

class TestListAgents:
    async def test_list_agents_returns_dir_based_agents_offline(self, db_path, tmp_path):
        """Agents discovered from directory without DB records are offline."""
        agents_dir = tmp_path / "agents"
        (agents_dir / "alice").mkdir(parents=True)
        (agents_dir / "bob").mkdir(parents=True)

        with patch.object(_srv, "_AGENTS_DIR", agents_dir):
            result = await _srv.list_agents()

        names = {r["name"] for r in result}
        assert "alice" in names
        assert "bob" in names
        for r in result:
            assert r["online"] is False

    async def test_list_agents_online_when_in_db(self, db_path, tmp_path):
        """Agent in agents table → online=True."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir(parents=True)

        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO agents (name, pid, online_since, last_seen) VALUES (?, ?, ?, ?)",
                ("carol", 9999, "2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
            )
            await db.commit()

        with patch.object(_srv, "_AGENTS_DIR", agents_dir):
            result = await _srv.list_agents()

        carol = next(r for r in result if r["name"] == "carol")
        assert carol["online"] is True
        assert carol["pid"] == 9999

    async def test_list_agents_no_dir_no_agents_returns_empty(self, db_path, tmp_path):
        non_existent = tmp_path / "no_such_dir"
        with patch.object(_srv, "_AGENTS_DIR", non_existent):
            result = await _srv.list_agents()
        assert result == []

    async def test_list_agents_sorted_by_name(self, db_path, tmp_path):
        agents_dir = tmp_path / "agents"
        (agents_dir / "zoe").mkdir(parents=True)
        (agents_dir / "alice").mkdir(parents=True)
        (agents_dir / "mike").mkdir(parents=True)

        with patch.object(_srv, "_AGENTS_DIR", agents_dir):
            result = await _srv.list_agents()

        names = [r["name"] for r in result]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# register_agent / unregister_agent
# ---------------------------------------------------------------------------

class TestRegisterAgent:
    async def test_register_writes_to_db(self, db_path):
        result = await _srv.register_agent("dave", 4567)
        assert result["status"] == "registered"
        assert result["name"] == "dave"

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT pid FROM agents WHERE name=?", ("dave",)) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == 4567

    async def test_register_upserts_existing_agent(self, db_path):
        await _srv.register_agent("dave", 100)
        await _srv.register_agent("dave", 200)

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT pid FROM agents WHERE name=?", ("dave",)) as cur:
                row = await cur.fetchone()
        assert row[0] == 200


class TestUnregisterAgent:
    async def test_unregister_known_agent_returns_unregistered(self, db_path):
        await _srv.register_agent("eve", 5678)
        result = await _srv.unregister_agent("eve")
        assert result["status"] == "unregistered"
        assert result["name"] == "eve"

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT * FROM agents WHERE name=?", ("eve",)) as cur:
                row = await cur.fetchone()
        assert row is None

    async def test_unregister_unknown_agent_returns_not_found(self, db_path):
        result = await _srv.unregister_agent("nobody")
        assert result["status"] == "not_found"
