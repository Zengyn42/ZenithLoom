"""
框架级 Named Session 管理器

sessions.json 存储 name → session_id 映射，方便 !session 命令管理。
"""

import json
import logging
import os
import sqlite3

logger = logging.getLogger(__name__)


class SessionManager:
    """管理 sessions.json 和 LangGraph SQLite checkpoint。"""

    def __init__(self, sessions_file: str, db_path: str):
        self.sessions_file = sessions_file
        self.db_path = db_path
        self._sessions: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, encoding="utf-8") as f:
                    self._sessions = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._sessions = {}

    def _save(self) -> None:
        with open(self.sessions_file, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, indent=2, ensure_ascii=False)

    def get(self, name: str) -> str | None:
        """获取 named session 对应的 thread_id。"""
        return self._sessions.get(name)

    def set(self, name: str, thread_id: str) -> None:
        """创建或更新 named session。"""
        self._sessions[name] = thread_id
        self._save()

    def delete(self, name: str) -> bool:
        """删除 named session。"""
        if name in self._sessions:
            del self._sessions[name]
            self._save()
            return True
        return False

    def list_all(self) -> dict[str, str]:
        """列出所有 named sessions。"""
        return dict(self._sessions)

    def session_stats(self, thread_id: str) -> dict:
        """返回指定 thread_id 的 checkpoint 统计。"""
        stats = {"thread_id": thread_id, "message_count": 0, "db_size_kb": 0}
        db = os.path.abspath(self.db_path)
        if not os.path.exists(db):
            return stats
        stats["db_size_kb"] = round(os.path.getsize(db) / 1024, 1)
        try:
            conn = sqlite3.connect(db)
            rows = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            conn.close()
            stats["message_count"] = rows[0] if rows else 0
        except Exception:
            pass
        return stats

    def compact(self, thread_id: str, keep_last: int = 20) -> int:
        """只保留最近 keep_last 条 checkpoint，返回删除行数。"""
        db = os.path.abspath(self.db_path)
        if not os.path.exists(db):
            return 0
        try:
            conn = sqlite3.connect(db, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            deleted = conn.execute(
                """
                DELETE FROM checkpoints
                WHERE thread_id = ?
                  AND checkpoint_id NOT IN (
                      SELECT checkpoint_id FROM checkpoints
                      WHERE thread_id = ?
                      ORDER BY checkpoint_id DESC
                      LIMIT ?
                  )
                """,
                (thread_id, thread_id, keep_last),
            ).rowcount
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()
            conn.close()
            logger.info(
                f"[session] compact thread={thread_id!r} "
                f"deleted={deleted} kept={keep_last}"
            )
            return deleted
        except Exception as e:
            logger.error(f"[session] compact 失败: {e}")
            return 0

    def reset(self, thread_id: str) -> int:
        """删除指定 thread_id 的全部 checkpoint，返回删除行数。"""
        db = os.path.abspath(self.db_path)
        if not os.path.exists(db):
            return 0
        try:
            conn = sqlite3.connect(db, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            deleted = conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).rowcount
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()
            conn.close()
            logger.info(
                f"[session] reset thread={thread_id!r} deleted={deleted}"
            )
            return deleted
        except Exception as e:
            logger.error(f"[session] reset 失败: {e}")
            return 0
