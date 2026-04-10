"""
无垠智穹 Hani - Git 状态回退引擎

把 Git 包装成 LangGraph 的底层时间机器：
  - ensure_repo : 确保 project_root 是 git repo（自动 init + .gitignore）
  - snapshot    : git add -A && git commit，返回 commit hash
  - rollback    : git reset --hard <hash>
  - get_current_hash : git rev-parse HEAD

所有操作在 project_root（项目目录）执行，不影响 ZenithLoom 自身的 git 状态。
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# 默认 .gitignore：排除大文件、模型权重、图片视频、缓存
_DEFAULT_GITIGNORE = """\
# 模型权重 / 大文件
*.ckpt
*.safetensors
*.bin
*.pt
*.pth
*.gguf
*.onnx

# 图片 / 视频（生成内容不进 git）
*.png
*.jpg
*.jpeg
*.gif
*.bmp
*.webp
*.mp4
*.mov
*.avi
*.mkv

# Python 缓存
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Node
node_modules/
dist/
build/

# 环境变量
.env
.env.*

# 系统文件
.DS_Store
Thumbs.db

# 跨时空耻辱柱（不受 Git 管控，时光机不得抹除）
.DO_NOT_REPEAT.md
"""


def _run(cmd: list[str], cwd: str) -> tuple[int, str, str]:
    """执行 git 命令，返回 (returncode, stdout, stderr)。"""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def ensure_repo(cwd: str) -> bool:
    """
    确保 cwd 是一个 git 仓库。
    - 若不是，执行 git init + git config user（避免 commit 报错）
    - 若 .gitignore 不存在，写入默认模板
    返回是否成功。
    """
    if not os.path.isdir(cwd):
        logger.warning(f"[git_ops] ensure_repo: 目录不存在 {cwd!r}")
        return False

    git_dir = os.path.join(cwd, ".git")
    if not os.path.isdir(git_dir):
        rc, _, err = _run(["git", "init"], cwd)
        if rc != 0:
            logger.error(f"[git_ops] git init 失败: {err}")
            return False
        # 设置最小 git 配置，避免 commit 要求 user.email
        _run(["git", "config", "user.email", "hani@wuyin.ai"], cwd)
        _run(["git", "config", "user.name", "Hani"], cwd)
        logger.info(f"[git_ops] git init 完成: {cwd}")

    gitignore = os.path.join(cwd, ".gitignore")
    if not os.path.exists(gitignore):
        with open(gitignore, "w", encoding="utf-8") as f:
            f.write(_DEFAULT_GITIGNORE)
        logger.info(f"[git_ops] 写入默认 .gitignore")

    return True


def snapshot(cwd: str, message: str = "Auto-snapshot") -> str | None:
    """
    对 cwd 执行 git add -A && git commit。
    - 若无变更（nothing to commit），返回当前 HEAD hash（不是 None）
    - 返回 commit hash（40位），失败返回 None
    """
    if not os.path.isdir(os.path.join(cwd, ".git")):
        logger.warning(f"[git_ops] snapshot: {cwd!r} 不是 git repo，跳过")
        return None

    _run(["git", "add", "-A"], cwd)
    rc, out, err = _run(["git", "commit", "-m", message], cwd)

    if rc != 0:
        if "nothing to commit" in out or "nothing to commit" in err:
            # 无变更，返回当前 HEAD
            return get_current_hash(cwd)
        logger.error(f"[git_ops] git commit 失败: {err}")
        return None

    return get_current_hash(cwd)


def rollback(cwd: str, commit_hash: str) -> bool:
    """
    执行 git reset --hard <commit_hash>。
    返回是否成功。
    """
    if not commit_hash:
        logger.warning("[git_ops] rollback: commit_hash 为空，跳过")
        return False

    rc, _, err = _run(["git", "reset", "--hard", commit_hash], cwd)
    if rc != 0:
        logger.error(f"[git_ops] git reset --hard {commit_hash[:8]} 失败: {err}")
        return False

    logger.info(f"[git_ops] 已回退到 {commit_hash[:8]} @ {cwd}")
    return True


def get_current_hash(cwd: str) -> str | None:
    """
    返回当前 HEAD 的完整 commit hash（40位）。
    若不是 git repo 或无任何 commit，返回 None。
    """
    rc, out, _ = _run(["git", "rev-parse", "HEAD"], cwd)
    if rc != 0 or not out:
        return None
    return out
