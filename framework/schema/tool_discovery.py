"""
tool_discovery_schema — 工具发现子图专用 state schema

ToolDiscoveryState: 继承 BaseAgentState，覆盖 messages 使用 add_messages reducer，
增加工具搜索流水线专属字段。
query_expand → search_aggregate → candidate_filter → sandbox_eval → report_gen
"""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from framework.schema.base import BaseAgentState


class ToolDiscoveryState(BaseAgentState):
    """工具发现子图 state：继承 BaseAgentState，覆盖 messages 保留全部历史。"""
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Discovery 专属字段（JSON 序列化字符串）──────────────────
    user_query: str              # 原始自然语言需求
    search_intent: str           # JSON: {keywords, github_queries, web_queries}
    raw_candidates: str          # JSON: [{repo, stars, license, description, ...}]
    filtered_candidates: str     # JSON: Top-K [{..., relevance_score, rationale}]
    evaluation_results: str      # JSON: [{install_ok, test_pass_rate, ...}]
    discovery_config: str        # JSON: {depth: 5, timeout: 300}
    discovery_errors: str        # 累积错误日志


# Auto-register on import
from framework.registry import register_schema

register_schema("tool_discovery_schema", ToolDiscoveryState)
