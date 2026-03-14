"""
框架级 API 封装 — ClaudeAPI / GeminiAPI

面向 LangGraph 节点设计的高层接口，隐藏 session / history 管理细节。

ClaudeAPI:
  - 自动追踪 session_id，支持连续对话
  - fresh=True 开启新 session
  - reset() 清空内存中的 session 引用（不删除 Claude 磁盘 session）

GeminiAPI:
  两种模式，通过 memory 参数控制：
  - memory=False（默认）：每次调用前清空历史，完全无状态
  - memory=True：历史跨调用保留，实现连续多轮对话

  方法：
    chat(message, *, memory=True, system="")  → 单条消息，直接追加 history
    consult(topic, context, *, memory=False)  → 3 轮对抗咨询（Claude 挑刺）
    reset()                                   → 强制清空 history

使用示例：
    from framework.api import ClaudeAPI, GeminiAPI
    from framework.agent_loader import AgentLoader
    from pathlib import Path

    cfg = AgentLoader(Path("agents/hani")).load_config()
    claude = ClaudeAPI(cfg)
    gemini = GeminiAPI(cfg, claude_api=claude)

    # 无状态 consult（每次独立）
    advice = await gemini.consult("微服务 vs 单体", "小团队 Python 项目")

    # 有记忆 chat（连续对话）
    r1 = await gemini.chat("你好，我们讨论下数据库选型")
    r2 = await gemini.chat("那 PostgreSQL 和 MySQL 呢？")

    # Claude 连续对话
    r1 = await claude.chat("请帮我设计一个简单的 API")
    r2 = await claude.chat("加上错误处理")   # 自动续接同一 session
"""

import logging

from framework.config import AgentConfig
from framework.claude.node import ClaudeSDKNode
from framework.gemini.node import (
    GeminiQuotaError,
    _CodeAssistClient,
    _GEMINI_SYSTEM,
)

logger = logging.getLogger(__name__)


class ClaudeAPI:
    """
    Claude 连续对话 API。

    自动追踪 session_id，同一实例的连续 chat() 调用共享 Claude session。
    LangGraph 节点可直接持有此实例，无需手动管理 session_id。
    """

    def __init__(self, config: AgentConfig, system_prompt: str = ""):
        self._node = ClaudeSDKNode(config, system_prompt)
        self._session_id: str = ""

    @property
    def session_id(self) -> str:
        return self._session_id

    async def chat(
        self,
        prompt: str,
        *,
        fresh: bool = False,
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> str:
        """
        发送一条消息，返回 Claude 文本回复。

        fresh=True：强制开启新 session，不续接旧会话。
        """
        sid = "" if fresh else self._session_id
        text, new_sid = await self._node.call_claude(
            prompt, session_id=sid, tools=tools, cwd=cwd
        )
        self._session_id = new_sid or self._session_id
        return text

    def reset(self):
        """清空内存中的 session 引用（不删除 Claude 磁盘 session）。"""
        old = self._session_id
        self._session_id = ""
        if old:
            logger.info(f"[ClaudeAPI] session reset (was {old[:8]})")

    @property
    def node(self) -> ClaudeSDKNode:
        """暴露底层 ClaudeSDKNode，供需要 call_claude() 完整签名的场景使用。"""
        return self._node


class GeminiAPI:
    """
    Gemini 双模式 API。

    chat() 和 consult() 共用同一个 _CodeAssistClient（同一段 history）。
    memory 参数决定调用前是否清空 history：

      memory=False：调用前 reset()，每次无状态。
      memory=True ：不 reset()，历史累积。

    GeminiQuotaError（403/429）不捕获，直接穿透给调用方（LangGraph 图）。
    """

    def __init__(self, config: AgentConfig, claude_api: ClaudeAPI, model: str = "gemini-2.5-pro"):
        self._client = _CodeAssistClient(model)
        self._claude = claude_api

    async def chat(
        self,
        message: str,
        *,
        memory: bool = True,
        system: str = "",
    ) -> str:
        """
        单条消息直发 Gemini，返回文本回复。

        memory=True（默认）：追加到历史，支持连续多轮对话。
        memory=False：发送前清空历史，每次独立。
        """
        if not memory:
            self._client.reset()
        return await self._client.chat(message, system=system or _GEMINI_SYSTEM)

    async def consult(
        self,
        topic: str,
        context: str,
        *,
        memory: bool = False,
    ) -> str:
        """
        3 轮对抗咨询（Gemini 首次 → Claude 挑刺 → Gemini 修订，共 3 轮）。

        memory=False（默认）：consult 前清空 history，每次独立。
        memory=True：在同一 GeminiAPI 实例内保留上下文，适合渐进式咨询。

        GeminiQuotaError（403/429）直接穿透，不捕获。
        其他异常包装为字符串返回。
        """
        if not memory:
            self._client.reset()

        try:
            # Round 1: Gemini 首次回答
            logger.info("[GeminiAPI] consult Round 1/3: Gemini 首次回答")
            g1 = await self._client.chat(
                f"问题：{topic}\n当前上下文：{context}",
                system=_GEMINI_SYSTEM,
            )

            # Round 2: Claude 挑刺 → Gemini 修订
            logger.info("[GeminiAPI] consult Round 2/3: Claude 挑刺")
            critique_prompt = (
                f"以下是 Gemini 架构师对「{topic}」的建议，请找出其中的逻辑漏洞、"
                f"遗漏的边界情况或过于理想化的假设（简明扼要，3点以内）：\n\n{g1}"
            )
            critique = await self._claude.chat(critique_prompt)

            logger.info("[GeminiAPI] consult Round 2/3: Gemini 修订")
            g2 = await self._client.chat(
                f"Hani 的质疑（请针对以下质疑修订你的建议）：\n{critique}"
            )

            # Round 3: Claude 深度挑刺 → Gemini 最终建议
            logger.info("[GeminiAPI] consult Round 3/3: Claude 深度挑刺")
            nitpick_prompt = (
                f"对以下修订后的架构建议进行最后一轮深度审查。\n"
                f"重点关注：实施复杂度、潜在的单点故障、与现有环境的兼容性。\n\n{g2}"
            )
            nitpick = await self._claude.chat(nitpick_prompt)

            logger.info("[GeminiAPI] consult Round 3/3: Gemini 最终建议")
            g_final = await self._client.chat(
                f"Hani 的最终审查意见：\n{nitpick}\n\n"
                "请给出你的最终建议（这将直接被采纳执行）："
            )

            logger.info("[GeminiAPI] 3轮咨询完成")
            return g_final

        except GeminiQuotaError:
            raise  # 穿透给 LangGraph 图
        except Exception as e:
            logger.error(f"[GeminiAPI] consult 失败: {e}")
            return f"[GeminiAPI 咨询失败: {e}]"

    def reset(self):
        """强制清空 Gemini 对话历史。"""
        self._client.reset()
        logger.info("[GeminiAPI] history reset")
