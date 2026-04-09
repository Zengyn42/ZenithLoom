"""
最小 LLM 节点冒烟测试 — 直接调用 Claude Agent SDK，不经过 LangGraph。

用法：
  python3 test_llm_ping.py
"""

import asyncio
import sys
import time

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    query as sdk_query,
)
from claude_agent_sdk.types import StreamEvent


async def test_claude_sdk_direct():
    """直接调用 sdk_query，测试 Claude CLI 是否能正常响应。"""
    import sys as _sys
    print("[1] Claude Agent SDK 直接调用测试", flush=True)
    _sys.stdout.flush()
    print(f"    prompt: 'say hi'")

    import json, os
    os.environ["CLAUDE_AGENT_SDK"] = "1"
    os.environ.pop("CLAUDECODE", None)
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        permission_mode="bypassPermissions",
        allowed_tools=["Read"],
        setting_sources=None,
        settings=json.dumps({"enabledPlugins": []}),
        cli_path="/home/kingy/.local/bin/claude",
        stderr=lambda line: print(f"    [stderr] {line.rstrip()}"),
    )

    result_text = ""
    session_id = ""
    start = time.time()

    try:
        async for msg in sdk_query(prompt="say hi", options=options):
            if isinstance(msg, StreamEvent):
                ev = msg.event
                etype = ev.get("type")
                if etype == "content_block_delta":
                    delta = ev.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        print(f"    stream: {text}", end="", flush=True)
            elif isinstance(msg, ResultMessage):
                session_id = msg.session_id or ""
                if msg.result:
                    result_text = msg.result.strip()
                if msg.is_error:
                    print(f"\n    ERROR in ResultMessage: {msg.result}")

        elapsed = time.time() - start
        print(f"\n    result: {result_text[:200]}")
        print(f"    session_id: {session_id[:8] if session_id else 'none'}")
        print(f"    elapsed: {elapsed:.1f}s")
        print("    STATUS: OK" if result_text else "    STATUS: EMPTY RESPONSE")
        return bool(result_text)

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n    EXCEPTION after {elapsed:.1f}s: {type(e).__name__}: {e}")
        return False


async def main():
    ok = await test_claude_sdk_direct()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
