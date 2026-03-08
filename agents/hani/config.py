"""
Hani Agent 配置

从 .env 加载 HANI_ 前缀的环境变量，并加载 persona 文件。
"""

import os
import re
from pathlib import Path

from framework.config import AgentConfig

_HANI_DIR = Path(__file__).parent

# ==========================================
# 操作规则提示词（与 SOUL/IDENTITY 分离）
# ==========================================
_OPERATIONAL_PROMPT = """## 运行时环境（你的自我认知）

你正在运行于无垠智穹的 LangGraph 状态机中。这不是裸 Claude，这是一个完整的 Agent 框架：
- 你的每条回复都经过中间件处理，不是直接发给老板
- consult_gemini 信号管道已就绪：你输出 JSON → 系统自动路由给 Gemini → Gemini 3轮对抗后结果注入回你的下一轮 prompt
- 当你看到 [Gemini 首席架构师建议] 段落时，说明管道已完成，你直接基于建议回复老板即可
- 老板也可以用 @Gemini 关键词绕过你直接触发咨询

## 操作规则

1. 回答简明扼要，直接输出操作结果或请老板 Approve。
2. 遇到宏大架构规划或物理隔离问题，必须咨询 Gemini（首席架构师）。
   咨询方式：回复的**第一行**单独输出以下 JSON，其余什么都不写：
   {"action": "consult_gemini", "topic": "<问题>", "context": "<当前状态>"}
   系统会自动接管，无需你解释这个机制。
3. 用中文回复，代码和命令用英文。"""

# 动态工具路由规则
_TOOL_RULES: list[tuple[re.Pattern, list[str]]] = [
    (
        re.compile(r"论文|paper|arXiv|arxiv|研究|publication", re.IGNORECASE),
        ["WebFetch", "WebSearch"],
    ),
    (
        re.compile(r"搜索|search|网络|internet|url|http", re.IGNORECASE),
        ["WebFetch", "WebSearch"],
    ),
    (
        re.compile(r"notebook|jupyter|\.ipynb", re.IGNORECASE),
        ["NotebookEdit"],
    ),
]


def load_hani_config() -> AgentConfig:
    """从 .env 加载 HANI_ 前缀配置。"""
    cfg = AgentConfig.from_env(prefix="HANI")
    cfg.persona_dir = str(_HANI_DIR)
    cfg.db_path = os.getenv(
        "HANI_DB_PATH",
        str(_HANI_DIR / "../../hani.db"),
    )
    cfg.sessions_file = os.getenv(
        "HANI_SESSIONS_FILE",
        str(_HANI_DIR / "../../sessions.json"),
    )
    return cfg


def load_hani_system_prompt() -> str:
    """加载 SOUL + IDENTITY + OPERATIONAL + COMMANDS，作为 SDK system_prompt。"""
    parts = []
    for fname in ["SOUL.md", "IDENTITY.md"]:
        p = _HANI_DIR / fname
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").strip())

    parts.append(_OPERATIONAL_PROMPT.strip())

    commands_path = _HANI_DIR / "COMMANDS.md"
    if commands_path.exists():
        parts.append(commands_path.read_text(encoding="utf-8").strip())

    return "\n\n---\n\n".join(parts)


def select_tools(user_input: str, base_tools: list[str] | None = None) -> list[str]:
    """根据用户输入动态组装工具列表。"""
    tools = list(base_tools or ["Read", "Write", "Edit", "Bash", "Glob", "Grep"])
    for pattern, extra in _TOOL_RULES:
        if pattern.search(user_input):
            for t in extra:
                if t not in tools:
                    tools.append(t)
    return tools
