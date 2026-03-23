"""
命令注册表 — 所有 ! 命令的唯一定义来源。

每条记录标注支持哪些 Connector，供以下场景使用：
  - !help 按当前 connector 动态生成帮助文本
  - BaseInterface.handle_command() 统一处理所有通用命令
  - 未来可做权限验证（命令是否适用于当前 connector）

添加新命令：
  1. 在此文件 _r() 调用处注册（只需一行）
  2. 在 BaseInterface.handle_command() 添加处理逻辑
  3. 若某 connector 需要特殊处理（如 Discord per-channel 逻辑），
     在对应 interface 里以 @bot.command / 显式 if 优先拦截即可
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Connector 枚举
# ---------------------------------------------------------------------------

class Connector(str, Enum):
    CLI     = "cli"
    DISCORD = "discord"


ALL = frozenset({Connector.CLI, Connector.DISCORD})


# ---------------------------------------------------------------------------
# CommandDef
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandDef:
    name:       str        # "!topology"
    description: str       # "显示 agent 图拓扑"
    connectors: frozenset  # 支持的 connectors
    usage:      str = ""   # 参数说明，如 "<名称> [工作目录]"


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

REGISTRY: dict[str, CommandDef] = {}


def _r(name: str, desc: str, conn: frozenset, usage: str = "") -> None:
    REGISTRY[name] = CommandDef(name, desc, conn, usage)


_DISCORD = frozenset({Connector.DISCORD})

# ── 通用命令（CLI + Discord）─────────────────────────────────────────────
_r("!help",       "显示帮助信息",                           ALL)
_r("!topology",   "显示 agent 图拓扑结构",                  ALL)
_r("!debug",      "查看 debug 模式状态",                    ALL)
_r("!stream",     "切换流式输出 ON/OFF",                    ALL)
_r("!tokens",     "查看 token 消耗统计",                    ALL,  "[reset]")
_r("!resources",  "查看资源锁状态",                         ALL)
_r("!new",        "创建并切换到新命名 session",              ALL,  "<名称> [工作目录]")
_r("!switch",     "切换到已有命名 session",                 ALL,  "<名称>")
_r("!sessions",   "列出所有命名 session",                   ALL)
_r("!session",    "显示当前 session 信息",                  ALL)
_r("!clear",      "重置当前 session（保留名称）",            ALL)
_r("!memory",     "查看当前 session 的 checkpoint 统计",    ALL)
_r("!compact",    "压缩当前 session，保留最近 N 条",         ALL,  "[N=20]")
_r("!reset",      "清空当前 session 全部记忆",              ALL,  "confirm")
_r("!setproject", "设置当前 session 的工作目录",            ALL,  "<路径>")
_r("!project",    "查看当前 session 的工作目录",            ALL)
_r("!snapshots",  "查看历史 git 快照",                      ALL)
_r("!rollback",   "回退到第 N 条 git 快照",                 ALL,  "<N> [原因]")
_r("!heartbeat",  "查看 heartbeat 状态",                    ALL,  "[list|status <id>|run <id>|alerts]")
_r("!discover",   "搜索和评估开源工具",                     ALL,  "<需求描述>")

# ── Discord 专属命令 ──────────────────────────────────────────────────────
_r("!stop",       "停止当前频道正在运行的任务",              _DISCORD)
_r("!channels",   "列出所有频道的 session 数据",            _DISCORD)
_r("!whoami",     "显示你的 Discord 用户 ID",               _DISCORD)
