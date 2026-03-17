"""
Gemini CLI 兼容 Session 存储 — framework/gemini/session.py

与 Gemini CLI (google-gemini/gemini-cli) 的 ConversationRecord 格式完全兼容。
存储路径：~/.gemini/tmp/{project_id}/chats/session-{YYYY-MM-DD-HH-MM}-{uuid[:8]}.json

互操作：
  - 本模块写入的 session 文件可被 `gemini --list-sessions` 列出
  - 可通过 `gemini --resume {sessionId}` 在 Gemini CLI 中继续对话
  - 反之，Gemini CLI 创建的 session 也可被本模块加载

格式参考：
  packages/core/src/services/chatRecordingService.ts
  packages/core/src/types/session.ts（ConversationRecord、MessageRecord）
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from framework.debug import is_debug

logger = logging.getLogger(__name__)
_GEMINI_DIR = Path.home() / ".gemini" / "tmp"


# ---------------------------------------------------------------------------
# Dataclasses — 对应 Gemini CLI TypeScript 类型
# ---------------------------------------------------------------------------

@dataclass
class TokensSummary:
    """对应 TokensSummary interface。"""
    input: int
    output: int
    cached: int
    total: int
    thoughts: int | None = None
    tool: int | None = None


@dataclass
class ToolCallRecord:
    """对应 ToolCallRecord interface。"""
    id: str
    name: str
    args: dict
    status: str   # 'pending' | 'success' | 'error'
    timestamp: str
    result: list | None = None
    displayName: str | None = None
    description: str | None = None


@dataclass
class MessageRecord:
    """
    对应 MessageRecord type（BaseMessageRecord & ConversationRecordExtra）。

    type='user'  → 用户消息
    type='gemini'→ 模型回复（含可选的 toolCalls / thoughts / tokens / model）
    type='info' / 'error' / 'warning' → 系统消息，不参与 API history
    """
    id: str           # UUID
    timestamp: str    # ISO 8601
    type: str         # 'user' | 'gemini' | 'info' | 'error' | 'warning'
    content: list     # PartListUnion: [{text: "..."}]
    toolCalls: list | None = None
    thoughts: list | None = None
    tokens: TokensSummary | None = None
    model: str | None = None


@dataclass
class ConversationRecord:
    """
    对应 ConversationRecord interface — session 文件的根对象。

    sessionId 是该 Gemini session 的唯一 UUID，存入
    BaseAgentState.node_sessions["gemini_main"]。
    """
    sessionId: str
    projectHash: str    # project slug, e.g. "bootstrapbuilder"
    startTime: str      # ISO 8601
    lastUpdated: str    # ISO 8601
    messages: list      # list[MessageRecord]（序列化时用 dict）
    summary: str | None = None
    kind: str | None = None   # 'main' | 'subagent'


# ---------------------------------------------------------------------------
# Project ID
# ---------------------------------------------------------------------------

def get_project_id(workspace: str) -> str:
    """
    从工作目录生成 project slug，与 Gemini CLI ProjectRegistry 逻辑一致：
      - 取目录名（basename）
      - 转小写
      - 非字母数字字符替换为 '-'
      - 合并连续 '-'，去除首尾 '-'

    e.g. "/home/kingy/Foundation/BootstrapBuilder" → "bootstrapbuilder"
    """
    if not workspace:
        workspace = os.getcwd()
    basename = Path(workspace).name
    slug = basename.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "default"


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def _chats_dir(project_id: str) -> Path:
    return _GEMINI_DIR / project_id / "chats"


def _session_filename(record: ConversationRecord) -> str:
    """
    session-{YYYY-MM-DD-HH-MM}-{uuid[:8]}.json
    与 Gemini CLI chatRecordingService.ts 命名逻辑一致。
    """
    dt = datetime.fromisoformat(record.startTime.replace("Z", "+00:00"))
    ts = dt.strftime("%Y-%m-%d-%H-%M")
    return f"session-{ts}-{record.sessionId[:8]}.json"


def _find_session_file(session_id: str, project_id: str) -> Path | None:
    """按 sessionId 前缀扫描 chats/ 目录，找到对应文件。"""
    chats = _chats_dir(project_id)
    if not chats.exists():
        return None
    prefix = session_id[:8]
    for f in chats.glob(f"session-*-{prefix}.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("sessionId") == session_id:
                return f
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def new_session(project_id: str, model: str = "") -> ConversationRecord:
    """创建新 ConversationRecord，生成 UUID sessionId。"""
    now = _now_iso()
    record = ConversationRecord(
        sessionId=str(uuid4()),
        projectHash=project_id,
        startTime=now,
        lastUpdated=now,
        messages=[],
        kind="main",
    )
    logger.debug(f"[gemini.session] new session {record.sessionId[:8]}")
    return record


def save_session(record: ConversationRecord, project_id: str) -> Path:
    """
    将 ConversationRecord 序列化并写入磁盘。
    若已有同 sessionId 文件则覆盖，否则按命名规则新建。
    目录不存在时自动创建。
    """
    chats = _chats_dir(project_id)
    chats.mkdir(parents=True, exist_ok=True)

    existing = _find_session_file(record.sessionId, project_id)
    path = existing or (chats / _session_filename(record))

    data = _record_to_dict(record)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug(f"[gemini.session] saved {path.name} ({len(record.messages)} messages)")
    return path


def load_session(session_id: str, project_id: str) -> ConversationRecord | None:
    """
    按 sessionId UUID 从 ~/.gemini/tmp/{project_id}/chats/ 加载 session。
    返回 None 若文件不存在或解析失败。
    """
    path = _find_session_file(session_id, project_id)
    if path is None:
        if is_debug():
            logger.debug(f"[gemini.session] session {session_id[:8]} not found on disk")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        record = _dict_to_record(data)
        logger.debug(f"[gemini.session] loaded {session_id[:8]} ({len(record.messages)} messages)")
        return record
    except Exception as e:
        logger.warning(f"[gemini.session] load failed ({session_id[:8]}): {e}")
        return None


def delete_session(session_id: str, project_id: str = "") -> bool:
    """
    按 sessionId 删除磁盘上的 session 文件。

    若 project_id 为空，则扫描所有 project 目录。
    返回 True 若成功删除，False 若文件不存在。
    """
    if project_id:
        path = _find_session_file(session_id, project_id)
        if path and path.exists():
            path.unlink()
            logger.info(f"[gemini.session] deleted {path.name}")
            return True
        return False

    # project_id 未知时，扫描所有 project 目录
    if not _GEMINI_DIR.exists():
        return False
    for proj_dir in _GEMINI_DIR.iterdir():
        if proj_dir.is_dir():
            path = _find_session_file(session_id, proj_dir.name)
            if path and path.exists():
                path.unlink()
                logger.info(f"[gemini.session] deleted {path.name}")
                return True
    return False


def list_sessions(project_id: str) -> list[ConversationRecord]:
    """列出该 project 下所有 session，按 lastUpdated 降序。"""
    chats = _chats_dir(project_id)
    if not chats.exists():
        return []
    records = []
    for f in chats.glob("session-*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            records.append(_dict_to_record(data))
        except Exception:
            continue
    records.sort(key=lambda r: r.lastUpdated, reverse=True)
    return records


# ---------------------------------------------------------------------------
# History conversion
# ---------------------------------------------------------------------------

def append_history(
    record: ConversationRecord,
    history: list[dict],
    model: str = "",
) -> None:
    """
    将 _CodeAssistClient._history（API contents 格式）增量追加到 record.messages。

    只追加尚未在 messages 中出现的条目（按内容去重，避免重复写入）。
    更新 record.lastUpdated。

    history 格式：[{"role": "user"/"model", "parts": [{"text": "..."}]}, ...]
    """
    existing_count = sum(
        1 for m in record.messages if m.get("type") in ("user", "gemini")
    ) if record.messages and isinstance(record.messages[0], dict) else sum(
        1 for m in record.messages if getattr(m, "type", None) in ("user", "gemini")
    )

    new_entries = history[existing_count:]
    now = _now_iso()

    for entry in new_entries:
        role = entry.get("role", "")
        parts = entry.get("parts", [])
        if role == "user":
            msg = MessageRecord(
                id=str(uuid4()),
                timestamp=now,
                type="user",
                content=parts,
            )
        elif role == "model":
            msg = MessageRecord(
                id=str(uuid4()),
                timestamp=now,
                type="gemini",
                content=parts,
                model=model or None,
            )
        else:
            continue
        record.messages.append(msg)

    record.lastUpdated = now
    if is_debug() and new_entries:
        logger.debug(f"[gemini.session] appended {len(new_entries)} entries")


def to_api_history(record: ConversationRecord) -> list[dict]:
    """
    将 ConversationRecord.messages 转为 API contents 格式的 _history。

    只提取 type in ('user', 'gemini')，忽略 info/error/warning。
    role 映射：'user' → 'user'，'gemini' → 'model'
    """
    history = []
    for msg in record.messages:
        if isinstance(msg, dict):
            mtype = msg.get("type", "")
            content = msg.get("content", [])
        else:
            mtype = msg.type
            content = msg.content

        if mtype == "user":
            history.append({"role": "user", "parts": content})
        elif mtype == "gemini":
            history.append({"role": "model", "parts": content})
    return history


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_to_dict(record: ConversationRecord) -> dict:
    """ConversationRecord → JSON-serializable dict。"""
    def _msg_to_dict(msg) -> dict:
        if isinstance(msg, dict):
            return msg
        d = {
            "id": msg.id,
            "timestamp": msg.timestamp,
            "type": msg.type,
            "content": msg.content,
        }
        if msg.toolCalls is not None:
            d["toolCalls"] = msg.toolCalls
        if msg.thoughts is not None:
            d["thoughts"] = msg.thoughts
        if msg.tokens is not None:
            d["tokens"] = asdict(msg.tokens) if isinstance(msg.tokens, TokensSummary) else msg.tokens
        if msg.model is not None:
            d["model"] = msg.model
        return d

    return {
        "sessionId": record.sessionId,
        "projectHash": record.projectHash,
        "startTime": record.startTime,
        "lastUpdated": record.lastUpdated,
        "messages": [_msg_to_dict(m) for m in record.messages],
        **({"summary": record.summary} if record.summary is not None else {}),
        **({"kind": record.kind} if record.kind is not None else {}),
    }


def _dict_to_record(data: dict) -> ConversationRecord:
    """JSON dict → ConversationRecord（messages 保持 dict 格式，to_api_history 兼容）。"""
    return ConversationRecord(
        sessionId=data["sessionId"],
        projectHash=data.get("projectHash", ""),
        startTime=data.get("startTime", ""),
        lastUpdated=data.get("lastUpdated", ""),
        messages=data.get("messages", []),
        summary=data.get("summary"),
        kind=data.get("kind"),
    )
