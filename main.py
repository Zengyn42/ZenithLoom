"""
无垠智穹 — 统一入口

用法：
  python main.py [--agent <name>] [--debug] <mode>

  mode:
    cli      本地 CLI（VSCode 终端 / 普通 shell）
    tmux     在 tmux session 'bootstrap_boss' 中运行
    discord  启动 Discord Bot（远程）
    gchat    启动 GChat Bot（通过 gws events +subscribe 监听消息）

  --agent:
    hani     （默认）使用 Claude 的 Hani
    asa      使用 Llama 的 Asa

  --debug:
    启用详细调试日志

示例：
  python main.py cli
  python main.py --debug cli
  python main.py --agent asa cli
  python main.py --agent hani discord

环境变量：
  AGENT=hani   等价于 --agent hani
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def _parse_args():
    """解析 --agent <name>、--debug 和 mode 参数。"""
    args = sys.argv[1:]
    agent_name = None
    mode = None
    debug = False

    i = 0
    while i < len(args):
        if args[i] == "--agent" and i + 1 < len(args):
            agent_name = args[i + 1]
            i += 2
        elif args[i] == "--debug":
            debug = True
            i += 1
        elif not args[i].startswith("--"):
            mode = args[i].lower()
            i += 1
        else:
            i += 1

    return agent_name, mode, debug


def main():
    import os

    agent_name, mode, debug = _parse_args()

    if debug:
        from framework.debug import set_debug
        set_debug(True)

    # 优先 --agent，其次 AGENT 环境变量，默认 hani
    agent_name = agent_name or os.getenv("AGENT", "hani")

    if not mode:
        print(__doc__)
        sys.exit(1)

    # 加载 AgentLoader
    agents_dir = Path(__file__).parent / "agents"
    agent_dir = agents_dir / agent_name
    if not agent_dir.is_dir():
        print(f"❌ Agent 目录不存在：{agent_dir}")
        print(f"   可用 agents: {[d.name for d in agents_dir.iterdir() if d.is_dir() and (d / 'agent.json').exists()]}")
        sys.exit(1)

    from framework.agent_loader import AgentLoader
    loader = AgentLoader(agent_dir)

    if mode == "cli":
        from interfaces.cli import run_cli
        run_cli(loader)

    elif mode == "tmux":
        from interfaces.cli import run_tmux
        run_tmux(loader)

    elif mode == "discord":
        from interfaces.discord_bot import run_discord
        run_discord(loader)

    elif mode == "gchat":
        import asyncio
        from interfaces.gchat_bot import run_gchat
        asyncio.run(run_gchat(loader))

    else:
        print(f"未知模式: {mode}")
        print("可用模式: cli | tmux | discord | gchat")
        sys.exit(1)


if __name__ == "__main__":
    main()
