"""
无垠智穹 0号管家 - 本地 CLI 接口 (v2 — framework 版)
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage

from agents.hani.config import load_hani_config
from agents.hani.graph import get_engine, get_config

TMUX_SESSION_NAME = "bootstrap_boss"


def run_cli():
    """本地对话循环入口（同步包装）。"""
    asyncio.run(_run_cli_async())


async def _run_cli_async():
    """异步对话循环：流式打印 token，历史存 SQLite。"""
    config = load_hani_config()
    engine = await get_engine()
    graph_config = get_config()

    print(f"\nHani 已启动 (session: {config.session_thread_id})")
    print("   输入 'q' 或 Ctrl+C 退出\n")

    loop = asyncio.get_event_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(
                None, lambda: input("老板指令 > ").strip()
            )
        except (KeyboardInterrupt, EOFError):
            print("\n\n管家待命中，再见老板。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("管家待命中，再见老板。")
            break

        print("\n[CTO 思考中...]\n", end="", flush=True)

        try:
            async for chunk, metadata in engine.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=graph_config,
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)

        print("\n")


def run_tmux():
    """Tmux 模式。"""
    try:
        import libtmux
    except ImportError:
        print("请先安装 libtmux：pip install libtmux")
        sys.exit(1)

    import os
    main_py = os.path.join(os.path.dirname(__file__), "..", "main.py")
    main_py = os.path.abspath(main_py)

    server = libtmux.Server()

    existing = None
    for s in server.sessions:
        if s.name == TMUX_SESSION_NAME:
            existing = s
            break

    if existing:
        print(f"[Tmux] session '{TMUX_SESSION_NAME}' 已存在，正在附加...")
        server.attach_session(target_session=TMUX_SESSION_NAME)
    else:
        print(f"[Tmux] 创建新 session '{TMUX_SESSION_NAME}'...")
        session = server.new_session(session_name=TMUX_SESSION_NAME, detach=True)
        session.active_window.active_pane.send_keys(f"python {main_py} cli")
        server.attach_session(target_session=TMUX_SESSION_NAME)
