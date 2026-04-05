"""
无垠智穹 — 实体唤醒入口

读取 identity.json，直接运行对应的 connector。

用法：
  python awaken.py --entity <path> [--connector <type>] [--debug] [--debug-output <file>]

  <path>: 实体目录路径，含 identity.json
          e.g. ~/Foundation/EdenGateway/agents/hani

  --connector:     覆盖 identity.json 中的 connector（cli | discord | tmux | gchat）
  --debug:         启用详细调试日志
  --debug-output:  将所有 LLM 节点完整输出追加写入指定文件（与 --debug 独立）
                   e.g. --debug-output /tmp/debate_debug.txt

示例：
  python awaken.py --entity ~/Foundation/EdenGateway/agents/hani
  python awaken.py --entity ~/Foundation/EdenGateway/agents/hani --connector cli

identity.json 格式：
  {
    "name": "hani",
    "blueprint": "/path/to/BootstrapBuilder/blueprints/role_agents/technical_architect",
    "connector": "discord",
    "discord": {
      "token": "...",
      "allowed_users": ["..."]
    }
  }

环境变量：
  ENTITY=<path>               等价于 --entity <path>
  BB_DEBUG_OUTPUT_FILE=<path>  等价于 --debug-output <path>
"""

import json
import os
import sys
from pathlib import Path


def _parse_args():
    """解析 --entity、--connector、--debug 和 --debug-output 参数。"""
    args = sys.argv[1:]
    entity_path = None
    connector = None
    debug = False
    debug_output_file = None

    i = 0
    while i < len(args):
        if args[i] == "--entity" and i + 1 < len(args):
            entity_path = args[i + 1]
            i += 2
        elif args[i] == "--connector" and i + 1 < len(args):
            connector = args[i + 1]
            i += 2
        elif args[i] == "--debug":
            debug = True
            i += 1
        elif args[i] == "--debug-output" and i + 1 < len(args):
            debug_output_file = args[i + 1]
            i += 2
        else:
            i += 1

    return entity_path, connector, debug, debug_output_file


def main():
    from dotenv import load_dotenv
    load_dotenv()

    entity_path, connector_override, debug, debug_output_file = _parse_args()
    entity_path = entity_path or os.getenv("ENTITY")
    debug_output_file = debug_output_file or os.getenv("BB_DEBUG_OUTPUT_FILE")

    if not entity_path:
        print(__doc__)
        sys.exit(1)

    # ── 读取 identity.json ────────────────────────────────────────────────────
    entity_dir = Path(entity_path).expanduser().resolve()
    entity_file = entity_dir / "identity.json"

    if not entity_file.exists():
        print(f"❌ identity.json not found: {entity_file}")
        sys.exit(1)

    entity = json.loads(entity_file.read_text(encoding="utf-8"))

    blueprint_path = entity.get("blueprint")
    connector = connector_override or entity.get("connector", "cli")

    if not blueprint_path:
        print(f"❌ identity.json missing 'blueprint' field: {entity_file}")
        sys.exit(1)

    # ── 解析 blueprint + framework ──────────────────────────────────────────
    blueprint_dir = Path(blueprint_path).expanduser().resolve()
    if not blueprint_dir.is_dir():
        print(f"❌ Blueprint not found: {blueprint_dir}")
        sys.exit(1)

    # framework root = blueprints/<sub>/<role> の 3 levels up
    framework_dir = blueprint_dir.parent.parent.parent
    framework_str = str(framework_dir)
    if framework_str not in sys.path:
        sys.path.insert(0, framework_str)

    # 切换 CWD 到 framework root，使 entity.json 里的相对路径（如 agent_dir）正确 resolve
    os.chdir(framework_dir)

    if debug:
        from framework.debug import set_debug
        set_debug(True)

    if debug_output_file:
        from framework.debug import set_debug_output_file
        set_debug_output_file(debug_output_file)
        print(f"📝 debug output file: {debug_output_file}")

    # ── EntityLoader ────────────────────────────────────────────────────────
    from framework.agent_loader import EntityLoader
    loader = EntityLoader(blueprint_dir, data_dir=entity_dir)

    # ── 启动 connector ───────────────────────────────────────────────────────
    if connector == "discord":
        from interfaces.discord_bot import run_discord
        run_discord(loader)

    elif connector == "tmux":
        from interfaces.cli import run_tmux
        run_tmux(loader)

    elif connector == "gchat":
        import asyncio
        from interfaces.gchat_bot import run_gchat
        asyncio.run(run_gchat(loader))

    else:  # cli (default)
        from interfaces.cli import run_cli
        run_cli(loader)


if __name__ == "__main__":
    main()
