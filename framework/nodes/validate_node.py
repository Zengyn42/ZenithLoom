"""
框架级验证节点 — ValidateNode

检查 claude_node 输出质量，验证失败时触发 git rollback。
"""

import logging
import os
import py_compile

from framework.config import AgentConfig

logger = logging.getLogger(__name__)


class ValidateNode:
    def __init__(self, config: AgentConfig):
        self.max_retries = config.max_retries

    def __call__(self, state: dict) -> dict:
        retry = state.get("retry_count", 0)
        if retry >= self.max_retries:
            logger.warning(f"[validate] retry_count={retry} 已达上限，放行")
            return {"rollback_reason": ""}

        msgs = state.get("messages")
        last_output = msgs[-1].content if msgs and hasattr(msgs[-1], "content") else ""
        root = state.get("project_root") or ""
        reason = _check_failure(last_output, root)
        if reason:
            logger.warning(f"[validate] 验证失败: {reason}")
        return {"rollback_reason": reason}


def _check_failure(last_output: str, project_root: str) -> str:
    """
    规则验证 claude_node 输出。
    返回失败原因（非空 = 失败），空字符串 = 通过。

    仅在 project_root 存在（代码任务模式）时做严格检查。
    普通对话模式只检查完全空输出。
    """
    if not last_output or not last_output.strip():
        return "输出为空"
    if last_output.lstrip().startswith("[错误]"):
        return f"输出包含错误前缀: {last_output[:80]}"
    if "CLI 超时" in last_output or "已强制终止" in last_output:
        return f"CLI 超时: {last_output[:80]}"

    # 以下检查仅在代码任务模式下生效（有 project_root 时）
    if not project_root or not os.path.isdir(project_root):
        return ""

    # 检查 .py 文件语法错误
    for dirpath, _, filenames in os.walk(project_root):
        if any(
            skip in dirpath
            for skip in (".git", "node_modules", "__pycache__")
        ):
            continue
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                py_compile.compile(fpath, doraise=True)
            except py_compile.PyCompileError as e:
                return f"Python 语法错误: {e}"

    return ""
