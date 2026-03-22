"""
框架级通用配置 — AgentConfig dataclass

所有 Agent 共用此配置结构。从 agent.json 加载：
  AgentConfig.from_json(path, prefix) — 从 agent.json 加载（推荐）

策略：
  - API 密钥（ANTHROPIC_API_KEY、GOOGLE_API_KEY）只放 .env
  - Discord 密钥（DISCORD_BOT_TOKEN、DISCORD_ALLOWED_USERS）放 .env 或 shell
  - 所有行为配置（model、tools、permission_mode 等）只放 agent.json
  - agent 专属字段（persona_files、tool_rules 等）由 agent 层自行从 agent.json 读取

已移除字段（迁移说明）：
  workspace          → entity.json 定义默认值，session 可覆盖
  max_gemini_consults→ edge max_retry（agent.json graph edges）
  claude_model       → node_config["model"]（agent.json nodes）
  gemini_model       → node_config["model"]（agent.json nodes）
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """所有 Agent 的通用配置。"""

    name: str = "agent"
    tools: list[str] = field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    )
    permission_mode: str = "bypassPermissions"
    max_retries: int = 2                  # git rollback 最大重试次数
    db_path: str = "agent.db"
    sessions_file: str = "./sessions.json"
    setting_sources: list[str] | None = None  # None = SDK默认; ["user"] 继承已安装skill
    settings_override: dict | None = None     # 传给 --settings 的 JSON 对象
    discord_token: str = ""               # DISCORD_BOT_TOKEN env var 优先
    discord_allowed_users: list[str] = field(default_factory=list)
    gchat_space: str = ""                 # GChat space name, e.g. "spaces/AAAA..."
    gchat_gcp_project: str = ""           # GCP project for Workspace Events API
    gchat_event_types: str = "google.workspace.chat.message.v1.created"
    workspace: str = ""                   # entity 级默认工作目录（session workspace 可覆盖）

    @classmethod
    def from_json(cls, path, env_prefix: str | None = None) -> "AgentConfig":
        """
        从 agent.json 加载。Discord 密钥允许 env var 覆盖，其余只读 JSON。
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        def _get_str(key: str, default: str) -> str:
            return str(data.get(key, default))

        def _get_int(key: str, default: int) -> int:
            return int(data.get(key, default))

        # tools: JSON 数组
        tools = data.get("tools", ["Read", "Write", "Edit", "Bash", "Glob", "Grep"])

        # setting_sources: JSON 数组
        raw = data.get("setting_sources")
        setting_sources = raw if isinstance(raw, list) and raw else None

        # settings_override: JSON dict
        settings_override = data.get("settings_override") or None
        if not isinstance(settings_override, dict):
            settings_override = None

        # discord_token: DISCORD_BOT_TOKEN env var 优先（不带 prefix），其次 agent.json
        discord_token = (
            os.getenv("DISCORD_BOT_TOKEN")
            or (os.getenv(f"{env_prefix}_DISCORD_TOKEN") if env_prefix else None)
            or data.get("discord_token", "")
        )

        # discord_allowed_users: DISCORD_ALLOWED_USERS env var 逗号分隔覆盖
        users_env = os.getenv("DISCORD_ALLOWED_USERS") or (
            os.getenv(f"{env_prefix}_DISCORD_ALLOWED_USERS") if env_prefix else None
        )
        if users_env is not None:
            discord_allowed_users = [u.strip() for u in users_env.split(",") if u.strip()]
        else:
            raw_users = data.get("discord_allowed_users", [])
            discord_allowed_users = [str(u) for u in raw_users] if isinstance(raw_users, list) else []

        return cls(
            name=_get_str("name", "agent"),
            tools=tools,
            permission_mode=_get_str("permission_mode", "bypassPermissions"),
            max_retries=_get_int("max_retries", 2),
            db_path=_get_str("db_path", "agent.db"),
            sessions_file=_get_str("sessions_file", "./sessions.json"),
            setting_sources=setting_sources,
            settings_override=settings_override,
            discord_token=discord_token,
            discord_allowed_users=discord_allowed_users,
            gchat_space=_get_str("gchat_space", ""),
            gchat_gcp_project=_get_str("gchat_gcp_project", ""),
            gchat_event_types=_get_str(
                "gchat_event_types",
                "google.workspace.chat.message.v1.created",
            ),
        )

    @classmethod
    def from_blueprint_and_instance(
        cls,
        blueprint_path: Path,
        instance_path: "Path | None",
        env_prefix: str = "",
    ) -> "AgentConfig":
        """
        从 blueprint agent.json 加载基础配置，然后用 instance entity.json 覆盖
        实例专属字段（name、discord_token、discord_allowed_users）。

        db_path 规则：
          - 若 agent.json 中显式设置了 db_path，直接使用该值。
          - 否则，若 entity.json 提供了 name，则 db_path = f"{name}.db"。
          - 否则，退回到默认值 "agent.db"。
        """
        # 读取 blueprint（复用 from_json 逻辑）
        cfg = cls.from_json(blueprint_path, env_prefix=env_prefix)

        # 判断 agent.json 是否显式设置了 db_path（与默认值不同则视为显式设置）
        blueprint_data = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
        db_path_explicit = "db_path" in blueprint_data

        if instance_path is not None and Path(instance_path).is_file():
            try:
                inst = json.loads(Path(instance_path).read_text(encoding="utf-8"))
            except Exception:
                inst = {}

            # 合并 name
            name = inst.get("name", "")
            if name:
                cfg.name = name
                # 若 agent.json 未显式设置 db_path，用实例名推导
                if not db_path_explicit:
                    cfg.db_path = f"{name}.db"

            # 合并 discord_token — 支持嵌套 {discord: {token:}} 和旧式扁平字段
            discord_cfg = inst.get("discord", {})
            inst_token = discord_cfg.get("token", "") or inst.get("discord_token", "")
            if inst_token and not cfg.discord_token:
                cfg.discord_token = inst_token

            # 合并 workspace — entity 级默认工作目录
            inst_workspace = inst.get("workspace", "")
            if inst_workspace:
                cfg.workspace = inst_workspace

            # 合并 discord_allowed_users — 同上支持嵌套 discord.allowed_users
            if not cfg.discord_allowed_users:
                raw_users = (
                    discord_cfg.get("allowed_users", [])
                    or inst.get("discord_allowed_users", [])
                )
                if isinstance(raw_users, list):
                    cfg.discord_allowed_users = [str(u) for u in raw_users]

        return cfg
