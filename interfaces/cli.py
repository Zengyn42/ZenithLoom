"""
无垠智穹 — 本地 CLI 接口（Agent 无关）

内联命令（以 ! 开头）：
  通用命令由 BaseInterface.handle_command() 处理（见 !help）。
  CLI 专属：
    !snapshots      查看历史快照
    !rollback N     回退到第 N 条快照（交互式输入原因）

用法：
  由 main.py 注入 AgentLoader 后调用 run_cli(loader)。
"""

import asyncio
import sys

from framework.base_interface import BaseInterface
from framework.command_registry import Connector

try:
    import readline  # noqa: F401 — 启用方向键历史导航
except ImportError:
    pass

TMUX_SESSION_NAME = "bootstrap_boss"


def run_cli(loader=None):
    """本地对话循环入口（同步包装）。"""
    asyncio.run(_run_cli_async(loader))


async def _run_cli_async(loader):
    iface = _CliInterface(loader)
    await iface.setup()
    await iface.run()


class _CliInterface(BaseInterface):
    """CLI 专用接口，继承 BaseInterface 公共命令，保留流式输出和 CLI 专属交互。"""

    _connector = Connector.CLI

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

    async def _prompt_rollback_reason(self, record: dict) -> str:
        """CLI 专属：显示快照信息并交互式读取回退原因。"""
        ts   = record["created_at"][:19].replace("T", " ")
        root = record["project_root"] or "(无 git repo)"
        print(f"目标快照：{record['commit_hash'][:8]}  {ts}  {root}")
        print("请输入本次回退原因（回车跳过，将写入 .DO_NOT_REPEAT.md）：", end="", flush=True)
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, lambda: input().strip())
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return ""

    async def run(self):
        controller = self._controller
        session_mgr = self._session_mgr
        loader = self._loader

        hb_graph, hb_cfg = await loader.build_heartbeat_graph()
        if hb_graph is not None:
            from framework.heartbeat import heartbeat_loop, run_heartbeat_once
            asyncio.create_task(run_heartbeat_once(hb_graph, hb_cfg))
            asyncio.create_task(heartbeat_loop(hb_graph, hb_cfg))

        thread_id  = controller.active_thread_id
        name       = session_mgr.find_name_by_thread_id(thread_id) or "默认"
        agent_name = loader.name
        print(f"\n{agent_name} 已启动 (session: {name} | thread: {thread_id})")
        print("   输入 !help 查看所有命令，输入 'q' 或 Ctrl+C 退出\n")

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

                reply = await self.handle_command(cmd, arg)
                if reply is not None:
                    print(reply)
                else:
                    print(f"未知命令：{cmd}  （输入 !help 查看可用命令）")
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
