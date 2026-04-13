import operator
"""
base_schema — 框架默认 state schema

BaseAgentState: 对话历史使用 add_messages reducer 累积消息。子图内部消息由 _subgraph_exit 节点清理，不会污染父图。
主图（Hani）、ApexCoder 等不需要自定义字段的图默认使用此 schema。
entity.json 中不声明 state_schema 时自动使用。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from framework.schema.reducers import _merge_dict


class BaseAgentState(TypedDict):
    resilience_log: Annotated[list[dict], operator.add]
    messages: Annotated[list[BaseMessage], add_messages]
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
    subgraph_topic: str         # 子图主题锚点（SubgraphMapperNode 入口写入、出口清空，LLM 节点只读注入）
    previous_node_output: str   # 前一节点输出（每个 LLM 节点执行后写入，下一节点注入 prompt；SubgraphMapperNode 入口清空）
    debate_conclusion: str  # 辩论子图最终结论（子图写回）
    apex_conclusion: str    # ApexCoder 子图执行结论
    knowledge_result: str   # knowledge_shelf 子图结论
    discovery_report: str   # tool_discovery 子图结论
    refined_plan: str       # 经辩论/评审后的精炼计划（colony_coder_planner 等写入）
    connector: str              # 接口类型标识（"cli" / "discord"），由 BaseInterface 注入，LlmNode 用于动态调整 user_msg_prefix


class SubgraphInputState(TypedDict):
    """子图 input schema：控制父图 state 流入子图的字段。

    用于 StateGraph(BaseAgentState, input=SubgraphInputState)。

    设计原则：只包含父图真正需要透传给子图的上下文。
    子图的输出字段、每次调用的临时字段、以及 messages 统统不在 input schema，
    LangGraph 会原生阻断它们从父图流入子图。

    故意从 input schema 移除的字段（防止跨次调用的状态污染）：

      messages:
        子图有自己的对话语境，父图历史不应流入。

      debate_conclusion / apex_conclusion / knowledge_result / discovery_report:
        子图的【输出】字段。LlmNode._build_gemini_section() 会把这些注入
        Claude 节点 prompt（供 claude_main 读取子图结论）。若从父图流入子图，
        第二次调用同一子图时，内部 Claude 节点会看到上一次的结论被注入到
        自己的 prompt 里，导致辩论跑偏 / 产出被污染。

      previous_node_output / subgraph_topic:
        子图内部节点间通信的临时字段，由 SubgraphMapperNode 或 LLM 节点
        自己维护。父图的残留值不应流入，每次子图调用都应从空值开始。

      refined_plan:
        colony_coder_planner 的输出，同理不应反向流入其他子图。

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
    connector: str
    # 以下字段故意缺失 → LangGraph 阻断它们从父图流入子图：
    #   messages, debate_conclusion, apex_conclusion, knowledge_result,
    #   discovery_report, refined_plan, previous_node_output, subgraph_topic


# Auto-register on import
from framework.registry import register_schema

register_schema("base_schema", BaseAgentState)
register_schema("subgraph_input_schema", SubgraphInputState)
