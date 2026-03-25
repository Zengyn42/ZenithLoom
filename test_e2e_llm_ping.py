"""
E2E 测试：单节点 Claude / Gemini ping 验证

覆盖：
  1. 单 ClaudeNode 收到 "ping" 能正常回复
  2. 单 GeminiNode 收到 "ping" 能正常回复

运行：
    python3 test_e2e_llm_ping.py
"""

import asyncio
import logging
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("test_e2e_llm_ping")

PING_PROMPT = "ping"


def _make_config(extra: dict | None = None):
    from framework.config import AgentConfig
    return AgentConfig(
        tools=[],
        permission_mode="bypassPermissions",
        setting_sources=None,
        settings_override={"enabledPlugins": []},
        **(extra or {}),
    )


import pytest

@pytest.mark.skip(reason="requires live Claude service")
async def test_claude_ping():
    """单 ClaudeNode: 发送 ping，验证返回非空文本。"""
    from framework.nodes.llm.claude import ClaudeNode

    config = _make_config()
    node = ClaudeNode(
        config=config,
        node_config={"id": "claude_ping", "model": None},
        system_prompt="You are a helpful assistant. Reply concisely.",
    )

    logger.info("[claude_ping] sending 'ping'...")
    text, session_id = await node.call_llm(PING_PROMPT)

    assert text, "ClaudeNode 返回空文本"
    assert session_id, "ClaudeNode 未返回 session_id"
    logger.info(f"[claude_ping] ✅ reply={text!r:.80} sid={session_id[:8]}")


@pytest.mark.skip(reason="requires live Gemini service")
async def test_gemini_ping():
    """单 GeminiNode: 发送 ping，验证返回非空文本。"""
    from framework.nodes.llm.gemini import GeminiNode

    config = _make_config()
    node = GeminiNode(
        config=config,
        node_config={
            "id": "gemini_ping",
            "model": "gemini-2.5-flash",
            "system_prompt": "You are a helpful assistant. Reply concisely.",
        },
    )

    logger.info("[gemini_ping] sending 'ping'...")
    text, session_id = await node.call_llm(PING_PROMPT)

    assert text, "GeminiNode 返回空文本"
    assert session_id, "GeminiNode 未返回 session_id"
    logger.info(f"[gemini_ping] ✅ reply={text!r:.80} sid={session_id[:8]}")


async def run():
    logger.info("=== E2E LLM Ping 测试开始 ===")

    await test_claude_ping()
    await test_gemini_ping()

    logger.info("=" * 50)
    print("\n✅ 全部测试通过")
    print("   ClaudeNode ping: 正常回复")
    print("   GeminiNode ping: 正常回复")


if __name__ == "__main__":
    asyncio.run(run())
