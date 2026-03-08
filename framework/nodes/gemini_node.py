"""
框架级 Gemini 顾问节点 — GeminiNode

Phase 1: subprocess 占位（保持现有行为）
Phase 2: 替换为 ADK-based（InMemorySessionService）
"""

import io
import logging
import os
import re
import subprocess
import threading
from subprocess import DEVNULL, PIPE

from framework.config import AgentConfig
from framework.nodes.claude_node import ClaudeNode

logger = logging.getLogger(__name__)

# ANSI 清洗器
_ANSI_ESCAPE = re.compile(
    r"""
    \x1B
    (?:
        [@-Z\\-_]
        | \[[0-?]*[ -/]*[@-~]
        | \][^\x07]*\x07
    )
    """,
    re.VERBOSE,
)

_INTERACTIVE_KEYWORDS = re.compile(
    r"\[y/N\]|\[Y/n\]|\(y/n\)|password:|Press Enter|Agree\?|proceed\?|"
    r"Do you want to|Continue\?|Are you sure|Enter passphrase",
    re.IGNORECASE,
)


def _clean(raw: str) -> str:
    return _ANSI_ESCAPE.sub("", raw).strip()


def _call_gemini_subprocess(prompt: str, timeout: int = 60) -> str:
    """Gemini CLI subprocess 调用（Phase 1 占位）。"""
    cmd = ["gemini", "-p", prompt]
    yolo = os.getenv("GEMINI_YOLO", "false").lower() == "true"
    if yolo:
        cmd += ["--approval-mode", "yolo"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=DEVNULL,
        )
        return _clean(result.stdout)
    except FileNotFoundError:
        return "[Gemini CLI 不可用。]"
    except subprocess.TimeoutExpired:
        return f"[Gemini CLI 超时（{timeout}s）。]"


class GeminiNode:
    """
    Gemini 顾问节点（3轮对抗咨询）。

    Phase 1: 每次调用通过 subprocess 传完整上下文。
    Phase 2: 替换为 ADK-based，Gemini 自管 session。
    """

    def __init__(self, config: AgentConfig, claude_node: ClaudeNode):
        self.config = config
        self.claude_node = claude_node

    async def consult(
        self,
        topic: str,
        context: str,
        session_id: str = "",
    ) -> str:
        """
        3轮对抗咨询：
          Round 1: Gemini 首次回答
          Round 2: Claude 挑刺 → Gemini 修订
          Round 3: Claude 深度挑刺 → Gemini 最终建议
        """
        gemini_system = (
            "你是无垠智穹的首席架构师（Gemini）。\n"
            "请根据问题给出极致架构建议。直接输出建议，不需要客套。"
        )

        # Round 1
        logger.info("[gemini] Round 1/3: Gemini 首次回答")
        g1 = _call_gemini_subprocess(
            f"{gemini_system}\n\n问题：{topic}\n当前上下文：{context}",
            timeout=60,
        )

        # Round 2: Claude 挑刺
        logger.info("[gemini] Round 2/3: Claude 挑刺")
        critique_prompt = (
            f"以下是 Gemini 架构师对「{topic}」的建议，请找出其中的逻辑漏洞、"
            f"遗漏的边界情况或过于理想化的假设（简明扼要，3点以内）：\n\n{g1}"
        )
        critique, _ = await self.claude_node.call_claude(critique_prompt)

        logger.info("[gemini] Round 2/3: Gemini 修订")
        g2 = _call_gemini_subprocess(
            f"{gemini_system}\n\n原始问题：{topic}\n"
            f"你的第一轮建议：\n{g1}\n\n"
            f"Hani的质疑：\n{critique}\n\n"
            "请针对上述质疑修订你的建议：",
            timeout=60,
        )

        # Round 3: Claude 深度挑刺
        logger.info("[gemini] Round 3/3: Claude 深度挑刺")
        nitpick_prompt = (
            f"对以下修订后的架构建议进行最后一轮深度审查。\n"
            f"重点关注：实施复杂度、潜在的单点故障、与现有环境的兼容性。\n\n{g2}"
        )
        nitpick, _ = await self.claude_node.call_claude(nitpick_prompt)

        logger.info("[gemini] Round 3/3: Gemini 最终建议")
        g_final = _call_gemini_subprocess(
            f"{gemini_system}\n\n原始问题：{topic}\n"
            f"经过两轮修订后的建议：\n{g2}\n\n"
            f"Hani 的最终审查意见：\n{nitpick}\n\n"
            "请给出你的最终建议（这将直接被采纳执行）：",
            timeout=60,
        )

        logger.info("[gemini] 3轮咨询完成")
        return g_final
