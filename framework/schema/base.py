"""
base_schema — 框架默认 state schema

BaseAgentState: 对话历史由 Claude SDK session 管理，LangGraph state 只保留最近 2 条消息。
主图（Hani）、ApexCoder 等不需要自定义字段的图默认使用此 schema。
agent.json 中不声明 state_schema 时自动使用。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage

from framework.schema.reducers import _merge_dict


def _keep_last_2(existing: list, new) -> list:
    """只保留最近 2 条消息。对话历史交给 SDK session 管理。"""
    combined = existing + (new if isinstance(new, list) else [new])
    return combined[-2:]


class BaseAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], _keep_last_2]
    routing_target: str   # 路由目标节点 ID（Claude 写入，如 "debate_brainstorm"；空 = 无路由请求）
    routing_context: str  # 路由上下文（问题/背景，目标节点读取；替代旧 gemini_context）
    workspace: str        # 当前工作目录（per-session，GraphController 注入）
    project_root: str     # 运行时覆盖目录（!setproject 设置）
    project_meta: dict    # {"plan": "path", "tasks": "path"}
    consult_count: int    # 当轮已路由咨询次数
    last_stable_commit: str  # git 快照 hash
    retry_count: int      # 当轮回退重试次数
    rollback_reason: str  # 触发回退的原因（非空 = 需要回退）
    node_sessions: Annotated[dict, _merge_dict]  # {"claude_main": uuid, ...} — 所有节点 session UUID; merge reducer prevents parallel node writes from clobbering each other
    knowledge_vault: str    # 知识库根路径（Obsidian vault 或任意 .md 目录）；agent 用 Read/Glob/Grep 按需读取
    project_docs: str       # 当前子项目 /docs/ 路径（技术文档，随 repo 走）
    debate_conclusion: str  # 辩论子图最终结论（最后发言节点的输出，由 AgentRefNode 写入）
    apex_conclusion: str    # ApexCoder 子图执行结论（由 AgentRefNode 写入，claude_main 读取）
    knowledge_result: str   # knowledge_shelf 子图结论（由 SubgraphRefNode 写入，gemini_main 读取）
    discovery_report: str   # tool_discovery 子图结论（由 SubgraphRefNode 写入，claude_main 读取）
    subgraph_call_counts: dict  # {"debate_brainstorm": 2, ...} — 按子图 ID 计数，用于 max_retry 限速
    connector: str              # 接口类型标识（"cli" / "discord"），由 BaseInterface 注入，LlmNode 用于动态调整 user_msg_prefix


# Auto-register on import
from framework.registry import register_schema

register_schema("base_schema", BaseAgentState)
