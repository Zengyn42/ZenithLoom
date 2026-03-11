"""
无垠智穹 — 本地 CLI 接口（Agent 无关）

内联命令（以 ! 开头）：
  !new <名称> [工作目录]  创建并切换到新命名 session（可选 workspace 路径）
  !switch <名称>  切换到已有命名 session
  !sessions       列出所有命名 session（当前用 ◀ 标注）
  !session        显示当前 session 名称和 thread_id
  !resources      查看所有资源锁状态（GPU/CPU）
  !tokens         查看 token 消耗统计（!tokens reset 重置）
  !topology       显示当前 agent 的图拓扑结构
  !debug          查看 debug 模式状态
  q / quit / exit 退出

用法：
  由 main.py 注入 AgentLoader 后调用 run_cli(loader)。
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage

try:
    import readline  # noqa: F401 — 启用方向键历史导航
except ImportError:
    pass

TMUX_SESSION_NAME = "bootstrap_boss"


def format_topology(agent_json: dict) -> str:
    """从 agent.json 生成可读的拓扑文本。"""
    name = agent_json.get("name", "agent")
    graph = agent_json.get("graph", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    lines = [f"=== {name} 拓扑图 ===", ""]

    # 节点列表
    lines.append(f"节点 ({len(nodes)}):")
    for n in nodes:
        nid = n.get("id", "?")
        ntype = n.get("type", "?")
        extra = ""
        if ntype == "AGENT_REF":
            extra = f" → {n.get('agent_dir', '?')}"
        elif n.get("model"):
            extra = f" [{n['model']}]"
        lines.append(f"  ● {nid:<22} [{ntype}]{extra}")

    lines.append("")

    # 边列表
    lines.append(f"边 ({len(edges)}):")
    for e in edges:
        src = e.get("from", "?")
        dst = e.get("to", "?")
        etype = e.get("type", "")
        max_retry = e.get("max_retry")
        if etype:
            retry_hint = f", max_retry={max_retry}" if max_retry is not None else ""
            lines.append(f"  {src} →[{etype}{retry_hint}]→ {dst}")
        else:
            lines.append(f"  {src} → {dst}")

    return "\n".join(lines)


def run_cli(loader=None):
    """本地对话循环入口（同步包装）。"""
    asyncio.run(_run_cli_async(loader))


async def _run_cli_async(loader):
    controller = await loader.get_controller()
    session_mgr = controller.session_mgr
    engine = controller._graph

    if loader.json.get("heartbeat"):
        from framework.heartbeat import heartbeat_loop, run_heartbeat_once
        await run_heartbeat_once()
        asyncio.create_task(heartbeat_loop())

    thread_id = controller.active_thread_id
    name = session_mgr.find_name_by_thread_id(thread_id) or "默认"
    agent_name = loader.name
    print(f"\n{agent_name} 已启动 (session: {name} | thread: {thread_id})")
    print("   !new / !switch / !sessions / !session / !resources / !tokens / !topology / !debug / !clear")
    print("   !snapshots — 查看历史快照  !rollback N — 回退到第 N 条快照")
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
                    print("用法：!new <session名称> [工作目录]")
                    continue
                # 分割 name 和可选 workspace（以空格分隔）
                new_parts = arg.split(maxsplit=1)
                new_name = new_parts[0]
                new_workspace = new_parts[1].strip() if len(new_parts) > 1 else ""
                try:
                    await controller.new_session(new_name, workspace=new_workspace)
                    engine = controller._graph  # 引用不变，thread_id 已更新
                    ws_hint = f" workspace={new_workspace!r}" if new_workspace else ""
                    print(f"✅ 新 session '{new_name}' 已创建并激活 (thread: {controller.active_thread_id}{ws_hint})")
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
                    await controller.switch_session(arg)
                    print(f"✅ 已切换到 session '{arg}' (thread: {controller.active_thread_id})")
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
                cur_tid = controller.active_thread_id
                for sname, env in all_sessions.items():
                    marker = " ◀" if env.thread_id == cur_tid else ""
                    print(f"  {sname} → {env.thread_id}{marker}")
                continue

            elif cmd == "!session":
                cur_tid = controller.active_thread_id
                cur_name = session_mgr.find_name_by_thread_id(cur_tid) or "（默认）"
                print(f"当前 session: {cur_name} | thread_id: {cur_tid}")
                continue

            elif cmd == "!resources":
                from framework.resource_lock import format_resource_status
                print(format_resource_status())
                continue

            elif cmd == "!topology":
                print(format_topology(loader.json))
                continue

            elif cmd == "!tokens":
                from framework.token_tracker import get_token_stats, reset_token_stats
                if arg == "reset":
                    reset_token_stats()
                    print("Token 计数已重置。")
                else:
                    s = get_token_stats()
                    inp = s["input_tokens"]
                    out = s["output_tokens"]
                    cr = s["cache_read_input_tokens"]
                    cc = s["cache_creation_input_tokens"]
                    calls = s["calls"]
                    cost_usd = (inp * 3 + out * 15 + cr * 0.3 + cc * 3.75) / 1_000_000
                    saved_usd = cr * (3 - 0.3) / 1_000_000
                    print(f"调用次数      : {calls}")
                    print(f"Input tokens  : {inp:,}")
                    print(f"Output tokens : {out:,}")
                    print(f"Cache read    : {cr:,}  (省了 ${saved_usd:.4f})")
                    print(f"Cache create  : {cc:,}")
                    print(f"估算费用      : ~${cost_usd:.4f} USD")
                continue

            elif cmd == "!debug":
                from framework.debug import is_debug
                print(f"Debug mode: {'ON' if is_debug() else 'OFF'}")
                continue

            elif cmd == "!clear":
                # 重置当前 session：生成新 thread_id，不碰 SQLite（避免与 LangGraph 连接冲突）
                cur_tid = controller.active_thread_id
                cur_name = session_mgr.find_name_by_thread_id(cur_tid) or "default"
                old_env = session_mgr.get_envelope(cur_name)
                workspace = old_env.workspace if old_env else ""
                session_mgr.delete(cur_name)
                new_env = session_mgr.create_session(cur_name, workspace=workspace)
                controller._active_thread_id = new_env.thread_id
                print(f"Session '{cur_name}' 已重置。(new thread: {new_env.thread_id[:8]})")
                continue

            elif cmd == "!snapshots":
                history = controller.rollback_log.get_history(
                    controller.active_thread_id, limit=10
                )
                if not history:
                    print("当前 session 还没有任何 git 快照记录。")
                    print("（需要 project_root 指向一个 git repo，每轮对话会自动快照）")
                else:
                    print(f"最近 {len(history)} 条快照（最新在前）：")
                    for i, rec in enumerate(history, 1):
                        ts = rec["created_at"][:19].replace("T", " ")
                        root = rec["project_root"] or "(无 project_root)"
                        print(f"  [{i}] {rec['commit_hash'][:8]}  {ts}  {root}")
                    print("用法：!rollback <序号>  （1=最近一次）")
                continue

            elif cmd == "!rollback":
                if not arg:
                    print("用法：!rollback <序号>  （1=最近，2=倒数第二...用 !snapshots 查看列表）")
                    continue
                try:
                    n = int(arg)
                    if n < 1:
                        raise ValueError
                except ValueError:
                    print(f"❌ 序号必须是正整数，例如：!rollback 3")
                    continue

                # 先展示目标快照
                record = controller.rollback_log.get_nth_ago(controller.active_thread_id, n)
                if not record:
                    print(f"❌ 没有找到第 {n} 条快照（用 !snapshots 查看当前记录数）")
                    continue
                ts = record["created_at"][:19].replace("T", " ")
                print(f"目标快照：{record['commit_hash'][:8]}  {ts}  {record['project_root'] or '(无 git repo)'}")
                print("请输入本次回退原因（回车跳过，将写入 .DO_NOT_REPEAT.md）：", end="", flush=True)
                try:
                    reason = await loop.run_in_executor(None, lambda: input().strip())
                except (KeyboardInterrupt, EOFError):
                    print("\n已取消")
                    continue

                result = await controller.rollback_to_turn(n, reason=reason)
                if result["ok"]:
                    ns_keys = list(result.get("node_sessions", {}).keys())
                    print(f"✅ {result['msg']}")
                    print(f"   恢复的节点 UUID：{ns_keys}")
                    if reason:
                        print(f"   已写入 .DO_NOT_REPEAT.md")
                else:
                    print(f"❌ {result['msg']}")
                continue

            else:
                print(f"未知命令：{cmd}  （试试 !new / !switch / !sessions / !session / !resources / !tokens / !topology / !clear / !snapshots / !rollback）")
                continue

        # --- 正常对话（流式输出）---
        print(f"\n[{agent_name} 思考中...]\n", end="", flush=True)

        try:
            async for chunk, metadata in engine.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=controller.get_config(),
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)

        print("\n")

        # 每轮结束后记录 git 快照到 rollback_log（有 project_root + git repo 时有效）
        try:
            await controller.log_snapshot()
        except Exception:
            pass


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
