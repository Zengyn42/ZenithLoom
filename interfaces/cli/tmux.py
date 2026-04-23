"""
Discord connector — tmux session launcher.
"""

import os
import sys

TMUX_SESSION_NAME = "bootstrap_boss"


def run_tmux(loader=None):
    """Tmux 模式。"""
    try:
        import libtmux
    except ImportError:
        print("请先安装 libtmux：pip install libtmux")
        sys.exit(1)

    main_py = os.path.join(os.path.dirname(__file__), "..", "..", "main.py")
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
        agent_name = loader.name if loader else "hani"
        print(f"[Tmux] 创建新 session '{TMUX_SESSION_NAME}'...")
        session = server.new_session(session_name=TMUX_SESSION_NAME, detach=True)
        session.active_window.active_pane.send_keys(
            f"python {main_py} --agent {agent_name} cli"
        )
        server.attach_session(target_session=TMUX_SESSION_NAME)
