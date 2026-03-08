"""
框架级 Git 节点 — GitSnapshotNode / GitRollbackNode

可插入任何 agent 图。依赖 git_ops.py 纯函数。
"""

import datetime
import logging
import os

from framework.nodes.git_ops import ensure_repo, rollback, snapshot

logger = logging.getLogger(__name__)

_TOMBSTONE_FILE = ".DO_NOT_REPEAT.md"


class GitSnapshotNode:
    """claude_node 前自动执行 git snapshot，记录稳定 commit hash。"""

    def __call__(self, state: dict) -> dict:
        root = state.get("project_root") or ""
        if not root or not os.path.isdir(root):
            return {}
        ensure_repo(root)
        h = snapshot(root, "Auto-snapshot before agent task")
        if h:
            logger.info(f"[git_snapshot] {h[:8]} @ {root}")
        return {"last_stable_commit": h or ""}


class GitRollbackNode:
    """验证失败时 git reset --hard 回到 last_stable_commit。"""

    def __call__(self, state: dict) -> dict:
        root = state.get("project_root") or ""
        commit = state.get("last_stable_commit", "")
        reason = state.get("rollback_reason", "")

        if root and commit:
            # 回滚前写耻辱柱
            bad_output = ""
            msgs = state.get("messages")
            if msgs:
                bad_output = msgs[-1].content if hasattr(msgs[-1], "content") else ""
            _write_tombstone(root, reason, bad_output)

            ok = rollback(root, commit)
            if not ok:
                logger.error(
                    f"[git_rollback] 回退失败，commit={commit[:8]!r}"
                )
        else:
            logger.warning(
                "[git_rollback] project_root 或 commit hash 为空，跳过"
            )

        return {
            "retry_count": state.get("retry_count", 0) + 1,
        }


def _write_tombstone(project_root: str, reason: str, bad_output: str) -> None:
    """把失败案例追加到 .DO_NOT_REPEAT.md（不受 Git 控制）。"""
    if not project_root or not os.path.isdir(project_root):
        return
    tombstone_path = os.path.join(project_root, _TOMBSTONE_FILE)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snippet = bad_output[:500].strip() if bad_output else "(空输出)"
    entry = (
        f"\n---\n"
        f"## [{ts}] 失败案例（已回滚）\n"
        f"**失败原因：** {reason}\n\n"
        f"**问题输出片段（前500字）：**\n```\n{snippet}\n```\n"
    )
    with open(tombstone_path, "a", encoding="utf-8") as f:
        f.write(entry)
    logger.info(f"[tombstone] 已写入耻辱柱: {tombstone_path}")


def read_tombstone(project_root: str) -> str:
    """读取耻辱柱内容（最近 2000 字符），注入 prompt。"""
    if not project_root:
        return ""
    tombstone_path = os.path.join(project_root, _TOMBSTONE_FILE)
    if not os.path.exists(tombstone_path):
        return ""
    with open(tombstone_path, encoding="utf-8") as f:
        content = f.read().strip()
    return content[-2000:] if content else ""
