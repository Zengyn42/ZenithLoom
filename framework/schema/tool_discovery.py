"""
tool_discovery_schema — 工具发现子图专用 state schema

ToolDiscoveryState: 工具搜索流水线的状态容器。
query_expand → search_aggregate → candidate_filter → sandbox_eval → report_gen
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from framework.schema.reducers import _merge_dict


class ToolDiscoveryState(TypedDict):
    """工具发现子图 state：保留全部消息历史，供各节点读取上游输出。"""
    messages: Annotated[list[BaseMessage], add_messages]

    # 框架必需字段（SubgraphRefNode 注入）
    routing_target: str
    routing_context: str
    workspace: str
    project_root: str
    node_sessions: Annotated[dict, _merge_dict]

    # ── Discovery 专属字段（JSON 序列化字符串）──────────────────
    user_query: str              # 原始自然语言需求
    search_intent: str           # JSON: {keywords, github_queries, web_queries}
    raw_candidates: str          # JSON: [{repo, stars, license, description, ...}]
    filtered_candidates: str     # JSON: Top-K [{..., relevance_score, rationale}]
    evaluation_results: str      # JSON: [{install_ok, test_pass_rate, ...}]
    discovery_report: str        # 最终评估报告（Markdown）
    discovery_config: str        # JSON: {depth: 5, timeout: 300}
    discovery_errors: str        # 累积错误日志


# Auto-register on import
from framework.registry import register_schema

register_schema("tool_discovery_schema", ToolDiscoveryState)
