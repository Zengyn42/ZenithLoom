"""
框架级通用配置 — AgentConfig dataclass

所有 Agent 共用此配置结构。通过 from_env(prefix="HANI") 加载环境变量。
"""

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """所有 Agent 的通用配置，从 .env 加载。"""

    name: str = "agent"
    workspace: str = ""  # 默认项目目录（可被 !setproject 覆盖）
    tools: list[str] = field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    )
    permission_mode: str = "bypassPermissions"
    max_retries: int = 2  # git rollback 最大重试
    max_gemini_consults: int = 3  # 每轮最多咨询 Gemini 次数
    db_path: str = "./agent.db"
    sessions_file: str = "./sessions.json"
    persona_dir: str = "./agents/hani/"  # SOUL.md / IDENTITY.md / COMMANDS.md 目录
    session_thread_id: str = "default_session"
    timeout: int = 120  # Claude SDK 超时（秒）
    claude_model: str | None = None  # None = 用 Claude Code CLI 默认模型

    @classmethod
    def from_env(cls, prefix: str = "HANI") -> "AgentConfig":
        return cls(
            name=prefix.lower(),
            workspace=os.getenv(f"{prefix}_WORKSPACE", ""),
            tools=[
                t.strip()
                for t in os.getenv(
                    f"{prefix}_TOOLS", "Read,Write,Edit,Bash,Glob,Grep"
                ).split(",")
                if t.strip()
            ],
            permission_mode=os.getenv(
                f"{prefix}_PERMISSION_MODE", "bypassPermissions"
            ),
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "2")),
            max_gemini_consults=int(os.getenv(f"{prefix}_MAX_GEMINI_CONSULTS", "3")),
            db_path=os.getenv(f"{prefix}_DB_PATH", f"./{prefix.lower()}.db"),
            sessions_file=os.getenv(f"{prefix}_SESSIONS_FILE", "./sessions.json"),
            session_thread_id=os.getenv(
                f"{prefix}_THREAD_ID", f"{prefix.lower()}_session"
            ),
            timeout=int(os.getenv(f"{prefix}_TIMEOUT", "120")),
            claude_model=os.getenv(f"{prefix}_CLAUDE_MODEL") or None,
        )
