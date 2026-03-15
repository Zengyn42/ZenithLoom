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
  !snapshots      查看历史快照
  !rollback N     回退到第 N 条快照
  !stream         切换流式输出 ON/OFF
  q / quit / exit 退出

用法：
  由 main.py 注入 AgentLoader 后调用 run_cli(loader)。
"""

import asyncio
import sys

from framework.base_interface import BaseInterface

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
    iface = _CliInterface(loader)
    await iface.setup()
    await iface.run()


class _CliInterface(BaseInterface):
    """CLI 专用接口，继承 BaseInterface 公共命令，保留流式输出和 CLI 专属命令。"""

    # ANSI: gray (90) + italic (3); reset (0)
    _THINK_OPEN  = "\x1b[90m\x1b[3m"
    _THINK_CLOSE = "\x1b[0m"

    def __init__(self, loader):
        super().__init__(loader)
        self._in_thinking: bool = False

    def _on_stream_reset(self) -> None:
        if self._in_thinking:
            print(f"\n{self._THINK_OPEN}[/thinking]{self._THINK_CLOSE}\n", end="", flush=True)
        self._in_thinking = False

    def _on_stream_chunk(self, text: str, is_thinking: bool = False) -> None:
        self._last_stream_chunk_count += 1
        if is_thinking:
            if not self._in_thinking:
                self._in_thinking = True
                print(f"{self._THINK_OPEN}[thinking]\n", end="", flush=True)
            if text:
                print(text, end="", flush=True)
        else:
            if self._in_thinking:
                self._in_thinking = False
                print(f"\n{self._THINK_OPEN}[/thinking]{self._THINK_CLOSE}\n\n", end="", flush=True)
            if text:
                print(text, end="", flush=True)

    async def run(self):
        controller = self._controller
        session_mgr = self._session_mgr
        engine = controller._graph
        loader = self._loader

        hb_graph, hb_cfg = await loader.build_heartbeat_graph()
        if hb_graph is not None:
            from framework.heartbeat import heartbeat_loop, run_heartbeat_once
            asyncio.create_task(run_heartbeat_once(hb_graph, hb_cfg))
            asyncio.create_task(heartbeat_loop(hb_graph, hb_cfg))

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

            if user_input.startswith("!"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                # CLI-specific commands
                if cmd == "!topology":
                    print(format_topology(loader.json))
                    continue

                if cmd == "!debug":
                    from framework.debug import is_debug
                    print(f"Debug mode: {'ON' if is_debug() else 'OFF'}")
                    continue

                if cmd == "!snapshots":
                    await self._handle_snapshots(controller)
                    continue

                if cmd == "!rollback":
                    await self._handle_rollback(controller, arg, loop)
                    continue

                # Common commands via BaseInterface
                reply = await self.handle_command(cmd, arg)
                if reply is not None:
                    print(reply)
                else:
                    print(
                        f"未知命令：{cmd}  "
                        "（试试 !new / !switch / !sessions / !session / !resources / "
                        "!tokens / !topology / !clear / !snapshots / !rollback）"
                    )
                continue

            # --- 正常对话（流式 / 非流式）---
            print(f"\n\x1b[90m[{agent_name}]\x1b[0m ", end="", flush=True)
            try:
                response = await self.invoke_agent(user_input)
                if not self._streaming or self._last_stream_chunk_count == 0:
                    print(response, end="", flush=True)
            except Exception as e:
                print(f"\n[错误] {e}", file=sys.stderr)

            print("\n")

            try:
                await controller.log_snapshot()
            except Exception:
                pass

    async def _handle_snapshots(self, controller):
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

    async def _handle_rollback(self, controller, arg: str, loop):
        if not arg:
            print("用法：!rollback <序号>  （1=最近，2=倒数第二...用 !snapshots 查看列表）")
            return
        try:
            n = int(arg)
            if n < 1:
                raise ValueError
        except ValueError:
            print("❌ 序号必须是正整数，例如：!rollback 3")
            return

        record = controller.rollback_log.get_nth_ago(controller.active_thread_id, n)
        if not record:
            print(f"❌ 没有找到第 {n} 条快照（用 !snapshots 查看当前记录数）")
            return
        ts = record["created_at"][:19].replace("T", " ")
        print(f"目标快照：{record['commit_hash'][:8]}  {ts}  {record['project_root'] or '(无 git repo)'}")
        print("请输入本次回退原因（回车跳过，将写入 .DO_NOT_REPEAT.md）：", end="", flush=True)
        try:
            reason = await loop.run_in_executor(None, lambda: input().strip())
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return

        result = await controller.rollback_to_turn(n, reason=reason)
        if result["ok"]:
            ns_keys = list(result.get("node_sessions", {}).keys())
            print(f"✅ {result['msg']}")
            print(f"   恢复的节点 UUID：{ns_keys}")
            if reason:
                print("   已写入 .DO_NOT_REPEAT.md")
        else:
            print(f"❌ {result['msg']}")


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
