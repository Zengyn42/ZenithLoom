"""
框架级通用配置 — AgentConfig dataclass

所有 Agent 共用此配置结构。支持两种加载方式：
  AgentConfig.from_env(prefix)      — 从环境变量加载（旧方式，向后兼容）
  AgentConfig.from_json(path, prefix) — 从 agent.json 加载，env var 优先覆盖（推荐）

Agent 专属配置（persona_files、tool_rules 等）存放在各自的 agent.json 中，
由 agent 层自行读取，不属于 AgentConfig。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """所有 Agent 的通用配置。"""

    name: str = "agent"
    workspace: str = ""                   # 默认项目目录（可被 !setproject 覆盖）
    tools: list[str] = field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    )
    permission_mode: str = "bypassPermissions"
    max_retries: int = 2                  # git rollback 最大重试次数
    max_gemini_consults: int = 1          # 每轮最多咨询 Gemini 次数
    db_path: str = "./agent.db"
    sessions_file: str = "./sessions.json"
    claude_model: str | None = None       # None = 用 Claude Code CLI 默认模型
    gemini_model: str = "gemini-2.5-flash"
    setting_sources: list[str] | None = None  # None = SDK默认(不加载); ["user"] = 继承已安装skill; ["user","project"] = 同时读工作目录skill
    settings_override: dict | None = None     # 传给 --settings 的 JSON 对象（可覆盖 enabledPlugins 等）
    discord_token: str = ""               # Discord Bot Token（建议用 DISCORD_BOT_TOKEN 环境变量覆盖）
    discord_allowed_users: list[str] = field(default_factory=list)  # 授权用户 Discord ID 列表

    @classmethod
    def from_env(cls, prefix: str = "HANI") -> "AgentConfig":
        """从环境变量加载（向后兼容）。"""
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
            permission_mode=os.getenv(f"{prefix}_PERMISSION_MODE", "bypassPermissions"),
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "2")),
            max_gemini_consults=int(os.getenv(f"{prefix}_MAX_GEMINI_CONSULTS", "1")),
            db_path=os.getenv(f"{prefix}_DB_PATH", f"./{prefix.lower()}.db"),
            sessions_file=os.getenv(f"{prefix}_SESSIONS_FILE", "./sessions.json"),
            claude_model=os.getenv(f"{prefix}_CLAUDE_MODEL") or None,
        )

    @classmethod
    def from_json(cls, path, env_prefix: str | None = None) -> "AgentConfig":
        """
        从 agent.json 加载，环境变量优先覆盖 JSON 值。

        JSON 中只包含 AgentConfig 标准字段；Agent 专属字段（persona_files、
        tool_rules 等）由 agent 层自行从同一 JSON 文件中读取。
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        def _get_str(key: str, env_key: str, default: str) -> str:
            if env_prefix:
                val = os.getenv(f"{env_prefix}_{env_key}")
                if val:
                    return val
            return str(data.get(key, default))

        def _get_int(key: str, env_key: str, default: int) -> int:
            if env_prefix:
                val = os.getenv(f"{env_prefix}_{env_key}")
                if val:
                    return int(val)
            return int(data.get(key, default))

        # tools: env var 用逗号分隔字符串，JSON 用数组
        tools_env = os.getenv(f"{env_prefix}_TOOLS") if env_prefix else None
        if tools_env:
            tools = [t.strip() for t in tools_env.split(",") if t.strip()]
        else:
            tools = data.get("tools", ["Read", "Write", "Edit", "Bash", "Glob", "Grep"])

        # setting_sources: JSON 数组或 env var 逗号分隔
        sources_env = os.getenv(f"{env_prefix}_SETTING_SOURCES") if env_prefix else None
        if sources_env is not None:
            setting_sources = [s.strip() for s in sources_env.split(",") if s.strip()] or None
        else:
            raw = data.get("setting_sources")
            setting_sources = raw if isinstance(raw, list) and raw else None

        # settings_override: 直接从 JSON 读取（dict），不支持 env var 覆盖
        settings_override = data.get("settings_override") or None
        if not isinstance(settings_override, dict):
            settings_override = None

        # discord_token: env var DISCORD_BOT_TOKEN 优先（不加 prefix），其次 agent.json
        discord_token = (
            os.getenv("DISCORD_BOT_TOKEN")
            or (os.getenv(f"{env_prefix}_DISCORD_TOKEN") if env_prefix else None)
            or data.get("discord_token", "")
        )

        # discord_allowed_users: JSON 数组，env var 逗号分隔覆盖
        users_env = os.getenv("DISCORD_ALLOWED_USERS") or (
            os.getenv(f"{env_prefix}_DISCORD_ALLOWED_USERS") if env_prefix else None
        )
        if users_env is not None:
            discord_allowed_users = [u.strip() for u in users_env.split(",") if u.strip()]
        else:
            raw_users = data.get("discord_allowed_users", [])
            discord_allowed_users = [str(u) for u in raw_users] if isinstance(raw_users, list) else []

        return cls(
            name=_get_str("name", "NAME", "agent"),
            workspace=_get_str("workspace", "WORKSPACE", ""),
            tools=tools,
            permission_mode=_get_str("permission_mode", "PERMISSION_MODE", "bypassPermissions"),
            max_retries=_get_int("max_retries", "MAX_RETRIES", 2),
            max_gemini_consults=_get_int("max_gemini_consults", "MAX_GEMINI_CONSULTS", 1),
            db_path=_get_str("db_path", "DB_PATH", "./agent.db"),
            sessions_file=_get_str("sessions_file", "SESSIONS_FILE", "./sessions.json"),
            claude_model=_get_str("claude_model", "CLAUDE_MODEL", "") or None,
            gemini_model=_get_str("gemini_model", "GEMINI_MODEL", "gemini-2.5-flash"),
            setting_sources=setting_sources,
            settings_override=settings_override,
            discord_token=discord_token,
            discord_allowed_users=discord_allowed_users,
        )
