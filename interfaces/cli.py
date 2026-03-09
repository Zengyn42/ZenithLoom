"""
无垠智穹 — 本地 CLI 接口（Agent 无关）

内联命令（以 ! 开头）：
  !new <名称>     创建并切换到新命名 session
  !switch <名称>  切换到已有命名 session
  !sessions       列出所有命名 session（当前用 ◀ 标注）
  !session        显示当前 session 名称和 thread_id
  q / quit / exit 退出

用法：
  由 main.py 注入 AgentLoader 后调用 run_cli(loader)。
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage

from framework.graph import get_config, switch_session, new_session

TMUX_SESSION_NAME = "bootstrap_boss"


def run_cli(loader=None):
    """本地对话循环入口（同步包装）。"""
    asyncio.run(_run_cli_async(loader))


async def _run_cli_async(loader):
    session_mgr = loader.session_mgr
    engine = await loader.get_engine()

    if loader.json.get("heartbeat"):
        from framework.heartbeat import heartbeat_loop, run_heartbeat_once
        await run_heartbeat_once()
        asyncio.create_task(heartbeat_loop())

    cfg = get_config()
    thread_id = cfg["configurable"]["thread_id"]
    name = session_mgr.find_name_by_thread_id(thread_id) or "默认"
    agent_name = loader.name
    print(f"\n{agent_name} 已启动 (session: {name} | thread: {thread_id})")
    print("   !new <名> / !switch <名> / !sessions / !session  管理 session")
    print("   输入 'q' 或 Ctrl+C 退出\n")

    loop = asyncio.get_event_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(
                None, lambda: input("> ").strip()
            )
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{agent_name} 待命中，再见。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print(f"{agent_name} 待命中，再见。")
            break

        # --- 内联 session 命令 ---
        if user_input.startswith("!"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "!new":
                if not arg:
                    print("用法：!new <session名称>")
                    continue
                try:
                    tid = await new_session(arg, session_mgr)
                    loader.invalidate_engine()
                    engine = await loader.get_engine()
                    print(f"✅ 新 session '{arg}' 已创建并激活 (thread: {tid})")
                except ValueError as e:
                    print(f"❌ {e}")
                except Exception as e:
                    print(f"创建失败: {e}")
                continue

            elif cmd == "!switch":
                if not arg:
                    print("用法：!switch <session名称>")
                    continue
                try:
                    tid = await switch_session(arg, session_mgr)
                    loader.invalidate_engine()
                    engine = await loader.get_engine()
                    print(f"✅ 已切换到 session '{arg}' (thread: {tid})")
                except ValueError as e:
                    print(f"❌ {e}")
                except Exception as e:
                    print(f"切换失败: {e}")
                continue

            elif cmd == "!sessions":
                all_sessions = session_mgr.list_all()
                if not all_sessions:
                    print("还没有任何命名 session。用 !new <名称> 创建第一个。")
                    continue
                cur_tid = get_config()["configurable"]["thread_id"]
                for sname, env in all_sessions.items():
                    marker = " ◀" if env.thread_id == cur_tid else ""
                    print(f"  {sname} → {env.thread_id}{marker}")
                continue

            elif cmd == "!session":
                cur_cfg = get_config()
                cur_tid = cur_cfg["configurable"]["thread_id"]
                cur_name = session_mgr.find_name_by_thread_id(cur_tid) or "（默认）"
                print(f"当前 session: {cur_name} | thread_id: {cur_tid}")
                continue

            else:
                print(f"未知命令：{cmd}  （试试 !new / !switch / !sessions / !session）")
                continue

        # --- 正常对话 ---
        print(f"\n[{agent_name} 思考中...]\n", end="", flush=True)

        try:
            async for chunk, metadata in engine.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=get_config(),
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)

        print("\n")


def run_tmux(loader=None):
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
        agent_name = loader.name if loader else "hani"
        print(f"[Tmux] 创建新 session '{TMUX_SESSION_NAME}'...")
        session = server.new_session(session_name=TMUX_SESSION_NAME, detach=True)
        session.active_window.active_pane.send_keys(
            f"python {main_py} --agent {agent_name} cli"
        )
        server.attach_session(target_session=TMUX_SESSION_NAME)
