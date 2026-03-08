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
    """
    if not last_output or len(last_output.strip()) < 10:
        return "输出为空或过短"
    if last_output.lstrip().startswith("[错误]"):
        return f"输出包含错误前缀: {last_output[:80]}"
    if "CLI 超时" in last_output or "已强制终止" in last_output:
        return f"CLI 超时: {last_output[:80]}"

    # 检查 .py 文件语法错误
    if project_root and os.path.isdir(project_root):
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
