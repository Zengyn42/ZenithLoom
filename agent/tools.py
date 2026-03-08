"""
无垠智穹 0号管家 - 工具库（CLI 版）

注意：由于使用 CLI subprocess 而非 SDK，不使用 @tool 装饰器。
工具函数直接被 core.py 的节点逻辑调用，或可单独测试。
"""

from agent.cli_wrapper import call_gemini


def consult_ceo_gemini(topic: str, current_context: str) -> str:
    """
    唤醒 Gemini 首席架构师，获取战略建议。
    由 gemini_node 在 LangGraph 图中调用，也可单独测试。
    """
    prompt = f"""你是无垠智穹的首席架构师（Gemini）。Hani遇到了问题需要战略建议。
请根据"5090显存绝对清场"和"物理隔离"铁律，给出极致架构建议。
问题：{topic}
当前上下文：{current_context}
直接输出建议或代码草案。"""

    return call_gemini(prompt)
