"""
框架级回滚日志 — framework/rollback_log.py

维护 agent SQLite 数据库中的 rollback_log 表。
每次 GitSnapshotNode 成功后，GraphController 调用 log_turn() 写入一条记录，
记录当时的 (commit_hash, node_sessions, project_root)。

!rollback N 通过 get_nth_ago(thread_id, N) 查询第 N 条（1=最近）历史快照，
获取 commit_hash 和 node_sessions 用于三层回退：
  1. git reset --hard <commit_hash>
  2. LangGraph aupdate_state → node_sessions 恢复为旧 UUID
  3. .DO_NOT_REPEAT.md tombstone 注入防止重犯
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RollbackLog:
    """rollback_log 表的 CRUD 封装，使用 agent.db（与 LangGraph checkpointer 同库）。"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rollback_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id     TEXT NOT NULL,
                commit_hash   TEXT NOT NULL,
                node_sessions TEXT NOT NULL,
                project_root  TEXT NOT NULL DEFAULT '',
                created_at    TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rl_thread ON rollback_log(thread_id)"
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_turn(
        self,
        thread_id: str,
        commit_hash: str,
        node_sessions: dict,
        project_root: str = "",
    ) -> int:
        """写入一条快照记录，返回 row id。"""
        ns_json = json.dumps(node_sessions, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        cur = conn.execute(
            "INSERT INTO rollback_log "
            "(thread_id, commit_hash, node_sessions, project_root, created_at) "
            "VALUES (?,?,?,?,?)",
            (thread_id, commit_hash, ns_json, project_root or "", now),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        logger.debug(
            f"[rollback_log] logged id={row_id} "
            f"thread={thread_id[:8]} commit={commit_hash[:8]}"
        )
        return row_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(self, thread_id: str, limit: int = 10) -> list[dict]:
        """
        返回最近 limit 条快照记录（最新在前）。
        每条 dict: {id, commit_hash, node_sessions, project_root, created_at}
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, commit_hash, node_sessions, project_root, created_at "
            "FROM rollback_log "
            "WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
            (thread_id, limit),
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            result.append(
                {
                    "id": r[0],
                    "commit_hash": r[1],
                    "node_sessions": json.loads(r[2]),
                    "project_root": r[3],
                    "created_at": r[4],
                }
            )
        return result

    def get_nth_ago(self, thread_id: str, n: int) -> dict | None:
        """
        返回第 n 条（n=1=最近一次，n=2=倒数第二次...）。
        没有足够记录时返回 None。
        """
        history = self.get_history(thread_id, limit=n)
        if len(history) < n:
            return None
        return history[n - 1]
