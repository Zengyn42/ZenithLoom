"""
E2E 测试：Claude ↔ Gemini 3轮对抗咨询，验证模型名称。

直接调用 GeminiNode.consult()（内含3轮 ClaudeNode 交互），
绕过 LangGraph 图的 session resume（避免嵌套 Claude Code 环境限制）。

运行：
    python3 test_e2e_debate.py

预期日志（INFO 级别）：
    [claude_node] model=claude-sonnet-4-6 sid=new
    [gemini_client] model=gemini-3-pro-preview jitter=X.Xs
    [gemini] Round 1/3: Gemini 首次回答
    [gemini] Round 2/3: Claude 挑刺
    [gemini] Round 2/3: Gemini 修订
    [gemini] Round 3/3: Claude 深度挑刺
    [gemini] Round 3/3: Gemini 最终建议
    [gemini] 3轮咨询完成
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("test_e2e_debate")

TOPIC = "微服务与单体架构在 AI Agent 编排场景下的选型"
CONTEXT = "当前项目为 Python LangGraph 多 Agent 框架，团队规模 1-3 人。"


async def run():
    from agents.hani.config import load_hani_config
    from framework.nodes.claude_node import ClaudeNode
    from framework.nodes.gemini_node import GeminiNode

    cfg = load_hani_config()
    gemini_model = os.getenv("HANI_GEMINI_MODEL", "gemini-2.5-flash")

    logger.info("=== E2E Debate 测试开始 ===")
    logger.info(f"Claude 模型: {cfg.claude_model or 'CLI default'}")
    logger.info(f"Gemini 模型: {gemini_model}")
    logger.info(f"话题: {TOPIC}")
    logger.info("=" * 50)

    claude = ClaudeNode(cfg)
    gemini = GeminiNode(cfg, claude)

    result = await gemini.consult(
        topic=TOPIC,
        context=CONTEXT,
        session_id="",
    )

    logger.info("=" * 50)
    logger.info("=== 测试完成 ===")
    logger.info(f"Gemini 最终建议长度: {len(result)} 字符")
    logger.info(f"Gemini 最终建议预览:\n{result[:400]}")

    # 断言
    assert result and len(result) > 20, f"Gemini 输出为空或过短: {result!r}"
    assert "[Gemini 咨询失败" not in result, f"Gemini 调用失败: {result[:200]}"
    assert "[Gemini 配额错误" not in result, f"Gemini 配额错误: {result[:200]}"

    print(f"\n✅ E2E Debate 测试通过")
    print(f"   Claude 模型: {cfg.claude_model or 'CLI default'}")
    print(f"   Gemini 模型: {gemini_model}")
    print(f"   Gemini 最终建议: {result[:100]}...")


if __name__ == "__main__":
    asyncio.run(run())
