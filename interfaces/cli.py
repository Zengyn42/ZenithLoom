"""
无垠智穹 0号管家 - 本地 CLI 接口
支持两种模式：
  - cli:  直接在当前终端运行（VSCode 集成终端 / 普通 shell）
  - tmux: 自动创建/附加 tmux session "bootstrap_boss"
"""

import sys
from langchain_core.messages import HumanMessage

from agent.core import get_engine, get_config, SESSION_THREAD_ID

TMUX_SESSION_NAME = "bootstrap_boss"


# ==========================================
# 核心对话循环
# ==========================================
def run_cli():
    """本地对话循环：流式打印 token，历史存 SQLite。"""
    engine = get_engine()
    config = get_config()

    print(f"\n🤖 Hani 已启动 (session: {SESSION_THREAD_ID})")
    print("   输入 'q' 或 Ctrl+C 退出\n")

    while True:
        try:
            user_input = input("老板指令 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n管家待命中，再见老板。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("管家待命中，再见老板。")
            break

        print("\n[CTO 思考中...]\n", end="", flush=True)

        # stream_mode="messages" → 逐 token 输出
        try:
            for chunk, metadata in engine.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # 只打印 AI 的文字 token，跳过 tool call chunk
                if hasattr(chunk, "content") and isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)

        print("\n")  # 每轮结束换行


# ==========================================
# Tmux 模式：自动创建/附加 named session
# ==========================================
def run_tmux():
    """
    检测 tmux session 'bootstrap_boss'：
      - 存在 → 直接 attach
      - 不存在 → 新建，在其中启动 CLI，然后 attach
    历史记忆在 SQLite，不在 tmux buffer，detach/reattach 不丢上下文。
    """
    try:
        import libtmux
    except ImportError:
        print("请先安装 libtmux：pip install libtmux")
        sys.exit(1)

    # main.py 的绝对路径，确保在任意 cwd 下都能找到
    import os
    main_py = os.path.join(os.path.dirname(__file__), "..", "main.py")
    main_py = os.path.abspath(main_py)

    server = libtmux.Server()

    # 查找已有 session
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
