"""
框架级 state schema 包 — framework/schema/

所有 LangGraph state schema 的定义和自注册。
import framework.schema 会触发所有内置 schema 的注册。

内置 schema：
  base_schema              → BaseAgentState（默认，主图及大部分图使用）
  debate_schema            → DebateState（辩论子图专用）
  tool_discovery_schema    → ToolDiscoveryState（工具发现子图专用）

业务图 schema 在各自目录的 state.py 中自注册（如 colony_coder/state.py → colony_coder_schema）。
"""

# import 子模块触发自注册
from framework.schema.base import BaseAgentState  # noqa: F401
from framework.schema.debate import DebateState  # noqa: F401
from framework.schema.tool_discovery import ToolDiscoveryState  # noqa: F401
