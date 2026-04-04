import operator
"""
base_schema — 框架默认 state schema

BaseAgentState: 对话历史由 Claude SDK session 管理，LangGraph state 只保留最近 2 条消息。
主图（Hani）、ApexCoder 等不需要自定义字段的图默认使用此 schema。
entity.json 中不声明 state_schema 时自动使用。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage

from framework.schema.reducers import _merge_dict


def _keep_last_2(existing: list, new) -> list:
    """只保留最近 2 条非空消息。对话历史交给 SDK session 管理。
    空消息（EXTERNAL_TOOL 等无输出节点）不参与竞争，保留已有有效消息。
    """
    new_list = new if isinstance(new, list) else [new]
    non_empty_new = [m for m in new_list if getattr(m, "content", "").strip()]
    if not non_empty_new:
        return existing[-2:] if existing else []
    combined = existing + non_empty_new
    return combined[-2:]


class BaseAgentState(TypedDict):
    resilience_log: Annotated[list[dict], operator.add]
    messages: Annotated[list[BaseMessage], _keep_last_2]
    routing_target: str   # 路由目标节点 ID（Claude 写入，如 "debate_brainstorm"；空 = 无路由请求）
    routing_context: str  # 路由上下文（问题/背景，目标节点读取；替代旧 gemini_context）
    workspace: str        # 当前工作目录（per-session，GraphController 注入）
    project_root: str     # 运行时覆盖目录（!setproject 设置）
    project_meta: dict    # {"plan": "path", "tasks": "path"}
    last_stable_commit: str  # git 快照 hash
    retry_count: int      # 当轮回退重试次数
    rollback_reason: str  # 触发回退的原因（非空 = 需要回退）
    node_sessions: Annotated[dict, _merge_dict]  # {"claude_main": uuid, ...} — 所有节点 session UUID; merge reducer prevents parallel node writes from clobbering each other
    knowledge_vault: str    # 知识库根路径（Obsidian vault 或任意 .md 目录）；agent 用 Read/Glob/Grep 按需读取
    project_docs: str       # 当前子项目 /docs/ 路径（技术文档，随 repo 走）
    debate_conclusion: str  # 辩论子图最终结论（子图写回）
    apex_conclusion: str    # ApexCoder 子图执行结论
    knowledge_result: str   # knowledge_shelf 子图结论
    discovery_report: str   # tool_discovery 子图结论
    refined_plan: str       # 经辩论/评审后的精炼计划（colony_coder_planner 等写入）
    connector: str              # 接口类型标识（"cli" / "discord"），由 BaseInterface 注入，LlmNode 用于动态调整 user_msg_prefix


class SubgraphInputState(TypedDict):
    """子图 input schema：控制父图 state 流入子图的字段。

    用于 StateGraph(BaseAgentState, input=SubgraphInputState)。

    messages 字段故意缺失：LangGraph 原生阻断父图消息进入子图。

    node_sessions 使用与 BaseAgentState 完全相同的 reducer（_merge_dict）：
      - LangGraph 1.0.10 不允许 input_schema 与 state_schema 对同字段使用【不同】reducer，
        但允许使用【相同】reducer → _merge_dict 可以安全声明。
      - 加入后，session_mode=inherit 的注入能穿透 input schema 到达子图节点；
        session_mode=persistent 的 checkpoint 恢复不再被 input schema 清零；
        session_mode=fresh_per_call/isolated 的 wrapper 注入 {} → merge({}, ...) = {} 依然 fresh。
    """
    resilience_log: Annotated[list[dict], operator.add]
    routing_context: str
    routing_target: str
    workspace: str
    project_root: str
    project_meta: dict
    last_stable_commit: str
    retry_count: int
    rollback_reason: str
    node_sessions: Annotated[dict, _merge_dict]  # 与 BaseAgentState 相同 reducer，允许 inherit/persistent 正常工作
    knowledge_vault: str
    project_docs: str
    debate_conclusion: str
    apex_conclusion: str
    knowledge_result: str
    discovery_report: str
    refined_plan: str
    connector: str
    # messages 字段故意缺失 → LangGraph 不从父图透传 messages


# Auto-register on import
from framework.registry import register_schema

register_schema("base_schema", BaseAgentState)
register_schema("subgraph_input_schema", SubgraphInputState)
