"""
测试：Claude 与 Gemini 通过 LangGraph 节点互相讨论，来回 3 个回合。

话题：「AI Agent 应该用状态机还是纯 LLM 驱动？」
流程：Claude 先发言 → Gemini 回应 → 重复 3 轮 → END

运行：python3 test_debate.py
"""

import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agent.cli_wrapper import call_claude, call_gemini

# ==========================================
# 状态定义
# ==========================================
TOTAL_ROUNDS = 3

class DebateState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    round: int       # 当前回合（1-based）
    speaker: str     # "claude" 或 "gemini"


TOPIC = "AI Agent 应该用状态机（如 LangGraph）还是纯 LLM 端到端驱动？请给出你的核心论点，50字以内。"

# ==========================================
# 节点：Claude 发言
# ==========================================
def claude_speaks(state: DebateState) -> dict:
    history = _format_history(state["messages"])
    round_n = state["round"]

    if round_n == 1:
        prompt = f"""你是 Claude，正在和 Gemini 讨论一个技术问题。
话题：{TOPIC}
请给出你的开场论点（50字以内，中文）。"""
    else:
        prompt = f"""你是 Claude，正在和 Gemini 进行第 {round_n} 轮辩论。
话题：{TOPIC}

之前的讨论：
{history}

请针对 Gemini 的上一条回复做出回应（50字以内，中文）。"""

    response = call_claude(prompt)
    print(f"\n[第{round_n}轮 · Claude] {response}")
    return {
        "messages": [AIMessage(content=f"[Claude] {response}")],
        "round": round_n,
        "speaker": "gemini",
    }


# ==========================================
# 节点：Gemini 回应
# ==========================================
def gemini_speaks(state: DebateState) -> dict:
    history = _format_history(state["messages"])
    round_n = state["round"]

    prompt = f"""You are Gemini, debating with Claude about: {TOPIC}

Discussion so far:
{history}

Give your response to Claude's last point (within 50 Chinese characters, reply in Chinese)."""

    response = call_gemini(prompt)
    print(f"[第{round_n}轮 · Gemini] {response}")
    return {
        "messages": [AIMessage(content=f"[Gemini] {response}")],
        "round": round_n + 1,  # 完成一轮，推进回合数
        "speaker": "claude",
    }


# ==========================================
# 路由：是否继续下一轮
# ==========================================
def route(state: DebateState) -> str:
    if state["speaker"] == "gemini":
        return "gemini"
    if state["round"] > TOTAL_ROUNDS:
        return "end"
    return "claude"


# ==========================================
# 辅助：格式化历史消息
# ==========================================
def _format_history(messages: list[BaseMessage]) -> str:
    return "\n".join(m.content for m in messages) if messages else "（无）"


# ==========================================
# 构建图
# ==========================================
def build_debate_graph():
    g = StateGraph(DebateState)
    g.add_node("claude", claude_speaks)
    g.add_node("gemini", gemini_speaks)
    g.set_entry_point("claude")
    g.add_conditional_edges(
        "claude",
        lambda s: "gemini",
        {"gemini": "gemini"},    # 明确声明边，让图可视化识别
    )
    g.add_conditional_edges(
        "gemini",
        route,                    # Gemini 说完：继续 or 结束
        {"claude": "claude", "end": END},
    )
    return g


# ==========================================
# 主流程
# ==========================================
if __name__ == "__main__":
    print(f"🎙️  辩题：{TOPIC}")
    print(f"   共 {TOTAL_ROUNDS} 轮，Claude 先发言\n{'─'*60}")

    conn = sqlite3.connect(":memory:", check_same_thread=False)  # 仅测试用，内存 DB
    conn.execute("PRAGMA journal_mode=WAL")

    graph = build_debate_graph().compile(checkpointer=SqliteSaver(conn))

    config = {"configurable": {"thread_id": "debate_test_01"}}
    initial_state: DebateState = {
        "messages": [],
        "round": 1,
        "speaker": "claude",
    }

    final = graph.invoke(initial_state, config=config)

    print(f"\n{'─'*60}")
    print(f"✅ 辩论结束，共 {len(final['messages'])} 条发言")
