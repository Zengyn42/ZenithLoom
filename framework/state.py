"""
框架级基础状态 — BaseAgentState / DebateState

BaseAgentState: 对话历史由 Claude SDK session 管理，LangGraph state 只保留最近 2 条消息。
DebateState:    辩论子图专用，保留全部消息历史（每轮发言都可见）。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


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
    claude_session_id: str  # SDK resume 用的 session UUID（向后兼容，镜像 node_sessions["claude_main"]）
    node_sessions: dict     # {"claude_main": uuid, ...} — 所有节点 session UUID
    knowledge_vault: str    # 知识库根路径（Obsidian vault 或任意 .md 目录）；agent 用 Read/Glob/Grep 按需读取
    project_docs: str       # 当前子项目 /docs/ 路径（技术文档，随 repo 走）
    debate_conclusion: str  # 辩论子图最终结论（最后发言节点的输出，由 AgentRefNode 写入）
    apex_conclusion: str    # ApexCoder 子图执行结论（由 AgentRefNode 写入，claude_main 读取）
    subgraph_call_counts: dict  # {"debate_brainstorm": 2, ...} — 按子图 ID 计数，用于 max_retry 限速


class DebateState(TypedDict):
    """辩论子图专用 state：保留全部消息历史，供每个发言节点看到完整辩论记录。"""
    messages: Annotated[list[BaseMessage], add_messages]
    knowledge_vault: str   # 知识库路径（从父图透传）
    project_docs: str      # 子项目文档路径（从父图透传）
    # 以下字段供 GeminiNode / ClaudeNode __call__ 正常读写（不会真正用于辩论逻辑）
    node_sessions: dict      # 节点 session UUID 映射
    routing_target: str      # 路由信号（辩论内不使用，但节点 __call__ 会清零写回）
    routing_context: str     # 路由上下文（首轮 Gemini 读取辩题）
    consult_count: int       # 咨询计数
    workspace: str           # 工作目录（ClaudeNode cwd）
    project_root: str        # 项目根目录
    rollback_reason: str     # 回退原因（辩论内始终为空）
    retry_count: int         # 重试计数
