#!/usr/bin/env python3
"""
Colony Coder 全真实运行（debug 模式）

Planner: Claude SDK × 4 节点（独立 session）
Executor: Ollama qwen3.5:27b × 3 节点 + EXTERNAL_TOOL + DETERMINISTIC
Integrator: Ollama qwen3.5:27b + Claude rescue + DETERMINISTIC

所有 debug logs 存到 logs/YYYY-MM-DD/colony_coder/...（按图层级目录）
"""

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)

# ── 注册 colony_coder_schema state schema ──
import blueprints.functional_graphs.colony_coder.state  # noqa: F401

from framework.agent_loader import EntityLoader
from framework.debug import set_debug, push_graph_scope, pop_graph_scope
from langchain_core.messages import HumanMessage


async def main():
    set_debug(True)
    print("🔧 Debug mode ON", flush=True)
    print("📁 Logs → logs/YYYY-MM-DD/colony_coder/...\n", flush=True)

    loader = EntityLoader(Path("blueprints/functional_graphs/colony_coder"))
    graph = await loader.build_graph(checkpointer=None)

    task = (
        "用 Python 写一个终端贪吃蛇游戏（Snake Game）。\n"
        "要求：\n"
        "1. 使用 curses 库实现终端 UI\n"
        "2. 蛇可以上下左右移动，吃到食物变长\n"
        "3. 撞墙或撞自己 Game Over，显示分数\n"
        "4. 单文件实现，保存到 /tmp/colony_game/snake.py\n"
        "5. 代码要有注释，可直接 python3 snake.py 运行"
    )

    init_state = {
        "messages": [HumanMessage(content=task)],
        "tasks": [],
        "execution_order": [],
        "refined_plan": "",
        "working_directory": "/tmp/colony_game",
        "completed_tasks": [],
        "final_files": [],
        "success": False,
        "abort_reason": "",
        "node_sessions": {},
        "ollama_sessions": {},
        "current_task_index": 0,
        "current_task_id": "",
        "retry_count": 0,
        "transient_retry_count": 0,
        "error_history": [],
        "cross_task_issues": [],
        "validation_output": None,
        "rescue_scope": "",
        "rescue_rationale": "",
        "affected_task_ids": [],
    }

    Path("/tmp/colony_game").mkdir(parents=True, exist_ok=True)

    push_graph_scope("colony_coder")
    try:
        print("🚀 Starting colony_coder (全真实: Claude + Ollama qwen3.5:27b)\n")
        print("=" * 70, flush=True)

        last_state = {}
        async for event in graph.astream(init_state, stream_mode="updates"):
            for node_id, update in event.items():
                if node_id in ("__start__", "__end__"):
                    continue
                last_state.update(update)

                msgs = update.get("messages", [])
                for msg in msgs:
                    content = getattr(msg, "content", "")
                    if not content:
                        continue
                    lines = content.split("\n")
                    print(f"\n┌─ [{node_id}]", flush=True)
                    for line in lines[:50]:
                        print(f"│ {line}", flush=True)
                    if len(lines) > 50:
                        print(f"│ ... ({len(content)} chars total)", flush=True)
                    print(f"└─", flush=True)

                rt = update.get("routing_target")
                if rt:
                    print(f"  ⤳ routing_target = {rt!r}", flush=True)

                # Token guard 触发时显示醒目提示
                abort = update.get("abort_reason", "")
                if abort and "Token 安全阀" in abort:
                    print(f"\n🛑 TOKEN GUARD TRIGGERED: {abort}", flush=True)

        print("\n" + "=" * 70, flush=True)
        success = last_state.get("success", False)
        files = last_state.get("final_files", [])
        abort = last_state.get("abort_reason", "")
        print(f"\n{'✅' if success else '❌'} Done! success={success}", flush=True)
        if files:
            print(f"   files: {files}", flush=True)
        if abort:
            print(f"   abort_reason: {abort}", flush=True)

    finally:
        pop_graph_scope()

    # ── 展示 log 文件列表 ──
    from datetime import datetime
    date = datetime.now().strftime("%Y-%m-%d")
    log_base = Path("logs") / date / "colony_coder"

    print(f"\n📁 Debug logs: {log_base}/", flush=True)
    if log_base.exists():
        for p in sorted(log_base.rglob("*.md")):
            rel = p.relative_to(log_base)
            size = p.stat().st_size
            print(f"   {rel} ({size:,} bytes)", flush=True)
    else:
        print("   (no logs generated)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
