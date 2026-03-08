"""
无垠智穹 0号管家 - Discord 远程接口 (P0 v3)

改进：
- 原生 typing indicator（channel.typing() keepalive 循环，参考 openclaw 模式）
- Debug mode：DEBUG=1 时打印完整 prompt/response 到终端
- !sessions 命令：查看 SQLite 里的会话列表
- Fence-Aware Chunking
- 错误时 edit 占位消息，不静默失败
- 身份认证：DISCORD_ALLOWED_USERS 白名单，仅授权用户可交互
"""

import asyncio
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import discord
from discord.ext import commands
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.core import (
    get_engine, get_config, SESSION_THREAD_ID, DB_PATH,
    session_stats, session_compact, session_reset,
)
from agent.cli_wrapper import get_token_stats, reset_token_stats

load_dotenv()

# ==========================================
# Debug Mode
# ==========================================
DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true")

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("discord_bot")

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_MAX_CHARS = 1900
_executor = ThreadPoolExecutor(max_workers=2)

# ==========================================
# 身份认证白名单
# ==========================================
def _load_allowed_users() -> set[int]:
    """
    从 DISCORD_ALLOWED_USERS 读取授权用户 ID（逗号分隔）。
    留空 = 关闭认证（任何人均可使用，仅限私有服务器）。
    示例：DISCORD_ALLOWED_USERS=123456789,987654321
    """
    raw = os.getenv("DISCORD_ALLOWED_USERS", "").strip()
    if not raw:
        return set()
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
        else:
            logger.warning(f"[Auth] 无效的用户 ID，已跳过: {part!r}")
    return ids

_ALLOWED_USERS: set[int] = _load_allowed_users()


def _is_authorized(user: discord.User | discord.Member) -> bool:
    """白名单为空时放行所有人；否则仅放行白名单内 ID。"""
    if not _ALLOWED_USERS:
        return True
    return user.id in _ALLOWED_USERS


# ==========================================
# Fence-Aware Chunking
# ==========================================
def split_fence_aware(text: str, max_chars: int = DISCORD_MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text
    in_fence = False
    fence_lang = ""

    while len(remaining) > max_chars:
        candidate = remaining[:max_chars]
        current_in_fence = in_fence
        current_lang = fence_lang

        for line in candidate.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                if current_in_fence:
                    current_in_fence = False
                    current_lang = ""
                else:
                    current_in_fence = True
                    current_lang = stripped[3:].strip()

        if current_in_fence:
            chunk = candidate + "\n```"
            remaining = f"```{current_lang}\n" + remaining[max_chars:]
        else:
            chunk = candidate
            remaining = remaining[max_chars:]

        in_fence = current_in_fence
        fence_lang = current_lang
        chunks.append(chunk)

    if remaining:
        chunks.append(remaining)

    return chunks


# ==========================================
# 当前项目目录（!setproject 命令设置，持久到进程重启）
# ==========================================
_current_project_root: str = ""


# ==========================================
# 同步 agent 调用（在线程池中运行）
# ==========================================
def _invoke_agent_sync(user_input: str) -> str:
    engine = get_engine()
    config = get_config()
    full_response = []

    if DEBUG:
        logger.debug(f"[agent] input={user_input!r}, project_root={_current_project_root!r}")

    init_state: dict = {"messages": [HumanMessage(content=user_input)]}
    if _current_project_root:
        init_state["project_root"] = _current_project_root

    for chunk, _ in engine.stream(
        init_state,
        config=config,
        stream_mode="messages",
    ):
        if hasattr(chunk, "content") and isinstance(chunk.content, str):
            full_response.append(chunk.content)

    result = "".join(full_response)
    if DEBUG:
        logger.debug(f"[agent] output={result[:500]!r}")
    return result


# ==========================================
# Discord Bot
# ==========================================
intents = discord.Intents.default()
intents.message_content = True  # 必须在 Developer Portal 开启 Message Content Intent

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    logger.info(f"[Discord] Hani 已上线: {bot.user} (session: {SESSION_THREAD_ID})")
    logger.info(f"[Discord] 已加入 {len(bot.guilds)} 个服务器")
    if _ALLOWED_USERS:
        logger.info(f"[Auth] 白名单已启用，授权用户 ID: {_ALLOWED_USERS}")
    else:
        logger.warning("[Auth] ⚠️  白名单为空，任何人均可使用（建议生产环境设置 DISCORD_ALLOWED_USERS）")
    if DEBUG:
        for g in bot.guilds:
            logger.debug(f"  - {g.name} ({g.id})")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # ── 身份认证 ──────────────────────────────────────────────────────────
    if not _is_authorized(message.author):
        if DEBUG:
            logger.debug(f"[Auth] 拒绝未授权用户: {message.author} ({message.author.id})")
        return  # 静默忽略，不暴露 bot 存在

    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    user_input = message.content.strip()
    if not user_input:
        return

    if DEBUG:
        logger.debug(f"[on_message] {message.author} ({message.author.id}): {user_input!r}")

    # ── Typing Indicator（openclaw 模式）─────────────────────────────
    # `async with channel.typing()` 让 discord.py 内部自动 keepalive（每 5 秒），
    # 用户看到 "Hani is typing..." 直到消息发出，无需占位消息。
    loop = asyncio.get_event_loop()
    full_text = None
    error_msg = None

    async with message.channel.typing():
        try:
            full_text = await loop.run_in_executor(
                _executor, _invoke_agent_sync, user_input
            )
        except Exception as e:
            logger.error(f"[agent] 调用出错: {e}", exc_info=DEBUG)
            error_msg = str(e)

    if error_msg:
        await message.channel.send(f"❌ Hani 出错了: {error_msg}")
        return

    if not full_text:
        await message.channel.send("（Hani 没有输出，请重试）")
        return

    chunks = split_fence_aware(full_text)
    for chunk in chunks:
        await message.channel.send(chunk)


# ==========================================
# 命令
# ==========================================
async def _check_auth(ctx) -> bool:
    """命令前置认证检查，失败时静默返回 False。"""
    if not _is_authorized(ctx.author):
        if DEBUG:
            logger.debug(f"[Auth] 命令被拒绝: {ctx.author} ({ctx.author.id})")
        return False
    return True


@bot.command(name="session")
async def show_session(ctx):
    """!session — 显示当前 session ID"""
    if not await _check_auth(ctx):
        return
    await ctx.send(f"📎 当前 session: `{SESSION_THREAD_ID}`")


@bot.command(name="sessions")
async def list_sessions(ctx):
    """!sessions — 列出 SQLite 里所有保存的 thread_id"""
    if not await _check_auth(ctx):
        return
    db = os.path.abspath(DB_PATH)
    if not os.path.exists(db):
        await ctx.send("📭 数据库不存在，还没有任何对话历史。")
        return
    try:
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        conn.close()
        if not rows:
            await ctx.send("📭 数据库里没有任何 session。")
            return
        lines = [f"- `{r[0]}`" for r in rows]
        await ctx.send("📋 **已保存的 sessions：**\n" + "\n".join(lines))
    except Exception as e:
        await ctx.send(f"❌ 查询失败: {e}")


@bot.command(name="debug")
async def toggle_debug(ctx):
    """!debug — 显示当前 debug 模式状态"""
    if not await _check_auth(ctx):
        return
    status = "ON 🔍" if DEBUG else "OFF（设置 DEBUG=1 启动以开启）"
    await ctx.send(f"🛠️ Debug mode: {status}")


@bot.command(name="whoami")
async def whoami(ctx):
    """!whoami — 显示你的 Discord 用户 ID（用于配置白名单）"""
    authorized = "✅ 已授权" if _is_authorized(ctx.author) else "❌ 未授权"
    await ctx.send(f"👤 你的 Discord ID: `{ctx.author.id}` ({authorized})")


@bot.command(name="memory")
async def show_memory(ctx):
    """!memory — 查看当前 session 的消息数量和 DB 大小"""
    if not await _check_auth(ctx):
        return
    stats = session_stats()
    await ctx.send(
        f"🧠 **Session 状态**\n"
        f"- thread: `{stats['thread_id']}`\n"
        f"- checkpoints: `{stats['message_count']}` 条\n"
        f"- DB 大小: `{stats['db_size_kb']} KB`"
    )


@bot.command(name="compact")
async def compact_session(ctx, keep: int = 20):
    """!compact [N] — 压缩 session，只保留最近 N 条（默认 20）"""
    if not await _check_auth(ctx):
        return
    deleted = session_compact(keep_last=keep)
    await ctx.send(
        f"🗜️ Compact 完成：删除了 `{deleted}` 条旧记录，保留最近 `{keep}` 条。"
    )


@bot.command(name="reset")
async def reset_session(ctx, confirm: str = ""):
    """!reset confirm — 清空当前 session 全部记忆（需要输入 confirm 确认）"""
    if not await _check_auth(ctx):
        return
    if confirm != "confirm":
        await ctx.send(
            "⚠️ 此操作将**清空 Hani 的全部记忆**，无法恢复。\n"
            "确认请输入：`!reset confirm`"
        )
        return
    deleted = session_reset()
    await ctx.send(f"🗑️ Session 已重置，清空了 `{deleted}` 条记录。Hani 从零开始。")


@bot.command(name="tokens")
async def show_tokens(ctx, reset: str = ""):
    """!tokens — 查看本次进程累计 token 消耗；!tokens reset 重置计数"""
    if not await _check_auth(ctx):
        return
    if reset == "reset":
        reset_token_stats()
        await ctx.send("🔄 Token 计数已重置。")
        return
    s = get_token_stats()
    inp = s["input_tokens"]
    out = s["output_tokens"]
    cr  = s["cache_read_input_tokens"]
    cc  = s["cache_creation_input_tokens"]
    calls = s["calls"]
    # Sonnet 4.5 价格：input $3/M, output $15/M, cache_read $0.3/M, cache_write $3.75/M
    cost_usd = (inp * 3 + out * 15 + cr * 0.3 + cc * 3.75) / 1_000_000
    # 如果全走 cache_read 能省多少：
    saved_usd = cr * (3 - 0.3) / 1_000_000
    await ctx.send(
        f"📊 **Token 统计（本次启动后）**\n"
        f"```\n"
        f"调用次数      : {calls}\n"
        f"Input tokens  : {inp:,}\n"
        f"Output tokens : {out:,}\n"
        f"Cache read    : {cr:,}  ← 省了 ${saved_usd:.4f}\n"
        f"Cache create  : {cc:,}  ← 首次写入成本\n"
        f"估算费用      : ~${cost_usd:.4f} USD\n"
        f"```"
    )


@bot.command(name="clear")
async def clear_session(ctx):
    """!clear — !reset confirm 的快捷别名，无需二次确认"""
    if not await _check_auth(ctx):
        return
    deleted = session_reset()
    await ctx.send(f"🗑️ Session 已清空（{deleted} 条）。Hani 从零开始。")


@bot.command(name="setproject")
async def set_project(ctx, *, path: str = ""):
    """!setproject <路径> — 设置 Hani 的工作目录（Git 时间机器在此目录生效）"""
    if not await _check_auth(ctx):
        return
    global _current_project_root
    if not path:
        current = _current_project_root or "（未设置）"
        await ctx.send(f"📂 当前项目目录：`{current}`")
        return
    path = os.path.expanduser(path.strip())
    if not os.path.isdir(path):
        await ctx.send(f"❌ 路径不存在：`{path}`")
        return
    _current_project_root = path
    await ctx.send(f"✅ 项目目录已设置为：`{path}`\nHani 的下一次操作将在此目录执行，Git 时间机器已就位。")


@bot.command(name="project")
async def show_project(ctx):
    """!project — 查看当前项目目录"""
    if not await _check_auth(ctx):
        return
    current = _current_project_root or "（未设置，Hani 在全局模式运行）"
    await ctx.send(f"📂 当前项目目录：`{current}`")


# ==========================================
# 入口
# ==========================================
def run_discord():
    if not DISCORD_TOKEN:
        print("❌ 请在 .env 中设置 DISCORD_BOT_TOKEN")
        return
    logger.info(f"[Discord] 启动中... DEBUG={'ON' if DEBUG else 'OFF'}")
    bot.run(DISCORD_TOKEN, log_handler=None)  # 用我们自己的 logging
