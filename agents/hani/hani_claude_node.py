"""
Hani 专属 Claude 节点

组合 ClaudeNode + Hani 路由逻辑（consult_gemini 信号、@Gemini 触发、
rollback 警告注入、耻辱柱、项目上下文读取）。
"""

import json
import logging
import os
import re

from langchain_core.messages import AIMessage

from agents.hani.config import load_hani_system_prompt, select_tools
from framework.nodes.claude_node import ClaudeNode
from framework.nodes.gemini_node import GeminiNode
from framework.nodes.git_nodes import read_tombstone
from framework.state import BaseAgentState

logger = logging.getLogger(__name__)

# consult_gemini 信号检测
_CONSULT_SIGNAL_RE = re.compile(r'\{"action"\s*:\s*"consult_gemini".*?\}', re.DOTALL)
_GEMINI_MENTION_RE = re.compile(r"@[Gg]emini\s*", re.IGNORECASE)
_IN_PROGRESS_RE = re.compile(r"^## In Progress", re.MULTILINE)

# 文件截取限制
PLAN_MAX_CHARS = 3000
TASKS_MAX_CHARS = 2000


class HaniClaudeNode:
    """Hani 的主控节点，包装 ClaudeNode 并添加 Hani 专属逻辑。"""

    def __init__(
        self,
        claude_node: ClaudeNode,
        gemini_node: GeminiNode,
    ):
        self.claude = claude_node
        self.gemini = gemini_node
        self._system_prompt = load_hani_system_prompt()

    async def __call__(self, state: BaseAgentState) -> dict:
        latest_input = state["messages"][-1].content

        # @Gemini 触发检测
        gemini_override_topic = None
        if _GEMINI_MENTION_RE.search(latest_input):
            gemini_override_topic = (
                _GEMINI_MENTION_RE.sub("", latest_input).strip() or latest_input
            )
            logger.info(
                f"[hani] @Gemini 触发: topic={gemini_override_topic!r}"
            )

        gemini_ctx = state.get("gemini_context", "")
        project_root = state.get("project_root", "") or None
        session_id = state.get("claude_session_id", "")

        # 动态注入内容
        rollback_reason = state.get("rollback_reason", "")
        last_commit = state.get("last_stable_commit", "")

        rollback_warning = ""
        if rollback_reason:
            commit_short = last_commit[:8] if last_commit else "上一个稳定版本"
            rollback_warning = (
                f"⚠️ 【系统警告·时光倒流】\n"
                f"你上一次的操作触发了验证失败（原因：{rollback_reason}）。\n"
                f"系统已执行 git reset --hard，文件已恢复到 {commit_short}。\n"
                f"请换一种思路，不要重复同样的错误。\n"
            )

        tombstone_raw = read_tombstone(project_root or "")
        tombstone_section = ""
        if tombstone_raw:
            tombstone_section = (
                f"⛔ 【跨时空耻辱柱·绝对禁止重蹈】\n{tombstone_raw}\n"
                f"绝对不要重复以上任何模式。\n"
            )

        gemini_section = ""
        if gemini_ctx and not gemini_ctx.startswith("__PENDING__"):
            gemini_section = (
                f"[Gemini 首席架构师建议（经3轮对抗验证）]\n{gemini_ctx}\n"
                f"[建议结束]\n"
                f"⚠️ 严禁再次输出 consult_gemini JSON，否则造成死循环。\n"
            )

        project_section = _build_project_section(state)

        # @Gemini 强制组装指令
        if gemini_override_topic:
            user_msg = (
                f"老板: {latest_input}\n\n"
                f"【系统指令·强制咨询】老板明确要求咨询 Gemini 首席架构师，"
                f"关于：{gemini_override_topic}\n"
                f"你的任务：结合当前项目上下文，把这个问题组装成专业提问，"
                f"**第一行且只有第一行**输出以下 JSON，不要任何前缀或解释：\n"
                f'{{"action": "consult_gemini", "topic": "<提炼后的问题>", '
                f'"context": "<相关项目状态>"}}'
            )
        else:
            user_msg = f"老板: {latest_input}"

        dynamic_injections = "".join(
            filter(
                None,
                [rollback_warning, tombstone_section, gemini_section, project_section],
            )
        )

        if not session_id:
            # 首轮：完整 prompt（SDK 会用 system_prompt 初始化 session）
            prompt = f"{dynamic_injections}\n\n{user_msg}\n\nHani:"
        else:
            # 后续轮：只发动态注入 + 消息，历史由 SDK session 管理
            parts = [p for p in [dynamic_injections, user_msg] if p]
            prompt = "\n\n".join(parts)

        dynamic_tools = select_tools(latest_input, self.claude.config.tools)

        raw_output, new_session_id = await self.claude.call_claude(
            prompt,
            session_id=session_id,
            tools=dynamic_tools,
            cwd=project_root,
        )

        debug = os.getenv("DEBUG", "").lower() in ("1", "true")
        if debug:
            logger.debug(f"[hani] raw_output={raw_output[:300]!r}")

        # consult_gemini 信号检测（只检查第一行）
        first_line = raw_output.lstrip().split("\n")[0].strip()
        if first_line.startswith("{") and "consult_gemini" in first_line:
            signal = _extract_json(first_line)
            if signal and signal.get("action") == "consult_gemini":
                topic = signal.get("topic", "")
                context = signal.get("context", "")
                return {
                    "messages": [AIMessage(content=raw_output)],
                    "gemini_context": f"__PENDING__{topic}|{context}",
                    "claude_session_id": new_session_id or session_id,
                }

        return {
            "messages": [AIMessage(content=raw_output)],
            "gemini_context": "",
            "consult_count": 0,
            "rollback_reason": "",
            "retry_count": 0,
            "claude_session_id": new_session_id or session_id,
        }


def _extract_json(text: str) -> dict | None:
    """简单 JSON 提取（不再调用 LLM 自纠正，依赖 SDK 输出质量）。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _read_project_file(
    path: str, is_tasks: bool = False, max_chars: int = 3000
) -> str:
    """实时读取磁盘文件。tasks 从 In Progress 锚点/尾部截取，plan 从头截取。"""
    if not path:
        return ""
    full_path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if not os.path.exists(full_path):
        return ""

    with open(full_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    if is_tasks:
        match = _IN_PROGRESS_RE.search(content)
        if match:
            content = content[match.start() :]
        if len(content) > max_chars:
            content = "...(older tasks truncated)\n" + content[-max_chars:]
    else:
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(rest truncated)"

    return content.strip()


def _build_project_section(state: BaseAgentState) -> str:
    """根据 project_meta 实时读取文件，构建项目上下文段落。"""
    meta = state.get("project_meta") or {}
    root = state.get("project_root") or ""
    sections = []

    plan_file = meta.get("plan", "")
    if plan_file:
        path = os.path.join(root, plan_file) if root else plan_file
        content = _read_project_file(path, is_tasks=False, max_chars=PLAN_MAX_CHARS)
        if content:
            sections.append(f"[当前项目架构 — {plan_file}]\n{content}")

    tasks_file = meta.get("tasks", "")
    if tasks_file:
        path = os.path.join(root, tasks_file) if root else tasks_file
        content = _read_project_file(path, is_tasks=True, max_chars=TASKS_MAX_CHARS)
        if content:
            sections.append(
                f"[当前任务列表 — {tasks_file}（尾部/In Progress 截取）]\n{content}"
            )

    return "\n\n".join(sections)
