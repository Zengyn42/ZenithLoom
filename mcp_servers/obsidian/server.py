"""
Obsidian Vault MCP Server — mcp_servers/obsidian/server.py

直接操作 Obsidian Vault 文件系统的 MCP 服务。不需要 Obsidian GUI 运行。
供无垠智穹全体 Agent 调用。

工具清单：
  读取：obsidian_read_note, obsidian_list_files
  写入：obsidian_write_note, obsidian_patch_note
  管理：obsidian_move_note, obsidian_delete_note,
        obsidian_get_frontmatter, obsidian_update_frontmatter,
        obsidian_manage_tags
  搜索：obsidian_search_files, obsidian_get_links

启动方式：
  python -m mcp_servers.obsidian.server                          # stdio
  python -m mcp_servers.obsidian.server --transport sse          # SSE（多客户端）
  python -m mcp_servers.obsidian.server --vault /path/to/vault   # 指定 vault 路径

安全：
  L1: 路径沙箱（所有操作限制在 vault 目录内）
  L2: 敏感目录黑名单（.obsidian/, .git/, .trash/）
  L3: 删除保护（默认移至 .trash/）
"""

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 确保项目根在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server import FastMCP

from mcp_servers.obsidian.core.vault import Vault

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("obsidian_mcp")

# 默认 vault 路径
_DEFAULT_VAULT = os.environ.get(
    "VAULT_BASE_DIR",
    "/home/kingy/Foundation/Vault",
)

# 全局 vault 实例
_vault: Vault | None = None


@asynccontextmanager
async def lifespan(server):
    """MCP Server 生命周期管理。"""
    logger.info(f"Obsidian Vault MCP 启动 — vault: {_vault.base_dir}")
    logger.info(f"已注册 tools: {len(server._tool_manager._tools)}")
    yield
    logger.info("Obsidian Vault MCP 关闭")


# MCP Server 定义
mcp = FastMCP(
    name="obsidian-vault",
    instructions=(
        "Obsidian Vault 操作工具。"
        "提供笔记的读写、搜索、链接查询、标签管理、frontmatter 操作等功能。"
        "所有写操作支持 CAS 乐观锁（基于 SHA-256 content hash），防止并发冲突。"
        "文件操作限制在 vault 目录内，敏感目录（.obsidian/ .git/）受保护。"
    ),
    lifespan=lifespan,
)


def _register_all_tools(vault: Vault):
    """注册所有 tool 模块。"""
    from mcp_servers.obsidian.tools import read, write, manage, search

    read.register(mcp, vault)
    write.register(mcp, vault)
    manage.register(mcp, vault)
    search.register(mcp, vault)

    logger.info("所有 tools 已注册")


def main():
    global _vault

    parser = argparse.ArgumentParser(description="Obsidian Vault MCP Server")
    parser.add_argument(
        "--vault", default=_DEFAULT_VAULT,
        help=f"Vault 目录路径 (default: {_DEFAULT_VAULT})",
    )
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="传输模式 (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE host")
    parser.add_argument("--port", type=int, default=8765, help="SSE port")
    args = parser.parse_args()

    # 初始化 vault
    vault_path = Path(args.vault).resolve()
    if not vault_path.is_dir():
        logger.error(f"Vault 目录不存在: {vault_path}")
        sys.exit(1)

    _vault = Vault(vault_path)
    logger.info(f"Vault: {_vault.base_dir}")

    # 注册 tools
    _register_all_tools(_vault)

    # 启动
    if args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        logger.info(f"SSE endpoint: http://{args.host}:{args.port}/sse")

    logger.info(f"启动 Obsidian Vault MCP Server (transport={args.transport})")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
