"""
框架级 Named Session 管理器

sessions.json 存储 name → SessionEnvelope 映射。
SessionEnvelope 包含：
  - thread_id   : LangGraph checkpointer 用的 thread_id
  - node_sessions: 各节点 session UUID（claude_main, gemini_main, ...）
  - created_at / updated_at

向后兼容：若 sessions.json 中值为 plain string（旧格式），
_load() 自动包装为 SessionEnvelope。
"""

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SessionEnvelope
# ---------------------------------------------------------------------------

@dataclass
class SessionEnvelope:
    """一个命名 session 的完整信封：LangGraph thread_id + 所有节点 UUID + workspace。"""

    thread_id: str
    created_at: str
    updated_at: str
    node_sessions: dict = field(default_factory=dict)  # {"claude_main": uuid, ...}
    workspace: str = ""  # per-session 工作目录（注入到 BaseAgentState["workspace"]）

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "node_sessions": self.node_sessions,
            "workspace": self.workspace,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEnvelope":
        return cls(
            thread_id=d["thread_id"],
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            node_sessions=d.get("node_sessions", {}),
            workspace=d.get("workspace", ""),
        )

    @classmethod
    def new(cls, thread_id: str | None = None, workspace: str = "") -> "SessionEnvelope":
        now = datetime.now(timezone.utc).isoformat()
        tid = thread_id or f"session_{uuid4().hex[:8]}"
        return cls(thread_id=tid, created_at=now, updated_at=now, workspace=workspace)


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """管理 sessions.json（命名 session → SessionEnvelope）和 LangGraph SQLite checkpoint。"""

    def __init__(self, sessions_file: str, db_path: str):
        self.sessions_file = sessions_file
        self.db_path = db_path
        self._sessions: dict[str, SessionEnvelope] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.sessions_file):
            return
        try:
            with open(self.sessions_file, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        for name, value in raw.items():
            if isinstance(value, str):
                # 旧格式：plain string → 自动迁移
                self._sessions[name] = SessionEnvelope.new(value)
                logger.info(f"[session] migrated legacy session {name!r} → envelope")
            elif isinstance(value, dict):
                self._sessions[name] = SessionEnvelope.from_dict(value)

    def _save(self) -> None:
        data = {name: env.to_dict() for name, env in self._sessions.items()}
        with open(self.sessions_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Session CRUD (backward-compatible API)
    # ------------------------------------------------------------------

    def get(self, name: str) -> str | None:
        """获取 named session 对应的 thread_id（签名不变）。"""
        env = self._sessions.get(name)
        return env.thread_id if env else None

    def get_envelope(self, name: str) -> SessionEnvelope | None:
        """获取完整 SessionEnvelope。"""
        return self._sessions.get(name)

    def set(self, name: str, thread_id: str) -> None:
        """创建或更新 named session（向后兼容，只设 thread_id）。"""
        existing = self._sessions.get(name)
        if existing:
            existing.thread_id = thread_id
            existing.updated_at = datetime.now(timezone.utc).isoformat()
        else:
            self._sessions[name] = SessionEnvelope.new(thread_id)
        self._save()

    def create_session(self, name: str, workspace: str = "") -> SessionEnvelope:
        """创建新命名 session，生成随机 thread_id。若 name 已存在抛 ValueError。"""
        if name in self._sessions:
            raise ValueError(
                f"Session {name!r} 已存在。用 !switch {name} 切换，或用其他名称。"
            )
        env = SessionEnvelope.new(workspace=workspace)
        self._sessions[name] = env
        self._save()
        logger.info(f"[session] created {name!r} → thread_id={env.thread_id} workspace={workspace!r}")
        return env

    def delete(self, name: str) -> bool:
        """删除 named session。"""
        if name in self._sessions:
            del self._sessions[name]
            self._save()
            return True
        return False

    def list_all(self) -> dict[str, SessionEnvelope]:
        """列出所有命名 sessions。"""
        return dict(self._sessions)

    def update_node_session(self, name: str, node_key: str, session_id: str) -> None:
        """更新指定 session 下某个节点的 UUID。"""
        env = self._sessions.get(name)
        if env is None:
            raise KeyError(f"Session {name!r} 不存在")
        env.node_sessions[node_key] = session_id
        env.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()

    def find_name_by_thread_id(self, thread_id: str) -> str | None:
        """反向查找：thread_id → session name。"""
        for name, env in self._sessions.items():
            if env.thread_id == thread_id:
                return name
        return None

    def list_by_prefix(self, prefix: str) -> dict[str, SessionEnvelope]:
        """列出 name 以 prefix 开头的所有 sessions。"""
        return {n: e for n, e in self._sessions.items() if n.startswith(prefix)}

    def delete_by_prefix(self, prefix: str) -> int:
        """删除 name 以 prefix 开头的所有 sessions 及其 checkpoint 行。返回删除的 session 数。"""
        to_delete = [n for n in self._sessions if n.startswith(prefix)]
        if not to_delete:
            return 0
        for name in to_delete:
            env = self._sessions.pop(name)
            self.reset(env.thread_id)
        self._save()
        logger.info(f"[session] delete_by_prefix {prefix!r} → {len(to_delete)} sessions")
        return len(to_delete)

    # ------------------------------------------------------------------
    # LangGraph checkpoint management
    # ------------------------------------------------------------------

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
