"""向后兼容 — schema 已迁移到 framework/schema/ 包。

旧代码 import framework.state 时自动转发到新位置，同时触发 schema 自注册。
"""

from framework.schema.base import BaseAgentState  # noqa: F401
from framework.schema.debate import DebateState  # noqa: F401
