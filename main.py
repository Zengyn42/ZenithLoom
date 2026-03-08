"""
无垠智穹 0号管家 - 统一入口

用法：
  python main.py cli      # 本地 CLI（VSCode 终端 / 普通 shell）
  python main.py tmux     # 在 tmux session 'bootstrap_boss' 中运行
  python main.py discord  # 启动 Discord Bot（远程）
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "cli":
        from interfaces.cli import run_cli
        run_cli()

    elif mode == "tmux":
        from interfaces.cli import run_tmux
        run_tmux()

    elif mode == "discord":
        from interfaces.discord_bot import run_discord
        run_discord()

    else:
        print(f"未知模式: {mode}")
        print("可用模式: cli | tmux | discord")
        sys.exit(1)


if __name__ == "__main__":
    main()
