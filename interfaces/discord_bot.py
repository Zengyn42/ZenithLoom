"""
无垠智穹 — Discord 远程接口（Agent 无关）

由 main.py 注入 AgentLoader 后调用 run_discord(loader)。
所有 agent 专属名称从 loader.name 动态获取。
"""

import asyncio
import logging
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from framework.debug import is_debug
from framework.token_tracker import get_token_stats, reset_token_stats

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG if is_debug() else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("discord_bot")

DISCORD_MAX_CHARS = 1900

# 由 run_discord() 注入
_loader = None
_controller = None
_session_mgr = None


# ==========================================
# 身份认证白名单（从 loader.config 读取）
# ==========================================
def _get_allowed_users() -> set[int]:
    if _loader is None:
        return set()
    users = _loader.load_config().discord_allowed_users
    return {int(u) for u in users if u.isdigit()}


def _is_authorized(user: discord.User | discord.Member) -> bool:
    allowed = _get_allowed_users()
    if not allowed:
        return True
    return user.id in allowed


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
# 当前项目目录
# ==========================================
_current_project_root: str = ""


# ==========================================
# 文件发送解析（图片 / 视频）
# ==========================================
import re as _re

_SEND_FILE_RE = _re.compile(r"\[SEND_FILE:\s*([^\]]+)\]")


def _extract_attachments(text: str) -> tuple[str, list[str]]:
    """
    从 agent 输出中提取所有 [SEND_FILE: /path/to/file] 标记。
    返回 (清理后的文字, [文件路径列表])。
    Agent 约定：在回复中写 [SEND_FILE: /绝对路径] 表示要发送该文件。
    """
    paths = [m.group(1).strip() for m in _SEND_FILE_RE.finditer(text)]
    clean_text = _SEND_FILE_RE.sub("", text).strip()
    return clean_text, paths


# ==========================================
# 频道历史 — 写入文件（Agent 按需用 Read 工具读取）
# ==========================================
_HISTORY_FILENAME = ".discord_channel_history.txt"


async def _refresh_history_file(channel, limit: int, exclude_msg_id: int) -> str | None:
    """
    拉取最近 limit 条频道消息，写入 workspace/.discord_channel_history.txt。
    返回文件绝对路径，写入失败或无 workspace 时返回 None。
    Agent 仅在用户主动要求时通过 Read 工具读取该文件，不自动注入。
    """
    workspace = (_loader.json.get("workspace", "") if _loader else "") or _current_project_root
    if not workspace:
        return None

    lines = []
    async for msg in channel.history(limit=limit + 1):
        if msg.id == exclude_msg_id:
            continue
        if msg.content.startswith("!"):
            continue
        ts = msg.created_at.strftime("%H:%M")
        lines.append(f"[{ts}] {msg.author.display_name}: {msg.content.strip()}")
        if len(lines) >= limit:
            break
    lines.reverse()  # 由旧到新

    path = os.path.join(workspace, _HISTORY_FILENAME)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Discord 频道历史（最近 {limit} 条，截止本消息前）\n\n")
            f.write("\n".join(lines))
        return path
    except OSError as e:
        logger.warning(f"[discord] 历史文件写入失败: {e}")
        return None


# ==========================================
# Agent 调用
# ==========================================
async def _invoke_agent_async(user_input: str, message=None) -> str:
    engine = _controller._graph
    config = _controller.get_config()

    # 后台刷新历史文件（Agent 不会自动读取，只在被要求时用 Read 工具读取）
    history_limit = (_loader.json.get("channel_history_limit", 0) if _loader else 0)
    if history_limit and message is not None:
        history_path = await _refresh_history_file(
            message.channel, history_limit, exclude_msg_id=message.id
        )
        if history_path and is_debug():
            logger.debug(f"[discord] 历史文件已刷新: {history_path}")

    if is_debug():
        logger.debug(f"[agent] input_len={len(user_input)} project_root={_current_project_root!r}")

    init_state: dict = {"messages": [HumanMessage(content=user_input)]}
    if _current_project_root:
        init_state["project_root"] = _current_project_root

    full_response = []
    async for chunk, _ in engine.astream(
        init_state,
        config=config,
        stream_mode="messages",
    ):
        if hasattr(chunk, "content") and isinstance(chunk.content, str):
            full_response.append(chunk.content)

    result = "".join(full_response)
    if is_debug():
        logger.debug(f"[agent] output={result[:500]!r}")
    return result


# ==========================================
# Discord Bot
# ==========================================
intents = discord.Intents.default()
intents.message_content = True  # Content Intent（开发者后台已启用）

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


@bot.event
async def on_ready():
    agent_name = _loader.name if _loader else "Agent"
    logger.info(f"[Discord] {agent_name} 已上线: {bot.user}")
    allowed = _get_allowed_users()
    if allowed:
        logger.info(f"[Auth] 白名单已启用: {allowed}")
    else:
        logger.warning("[Auth] 白名单为空，任何人均可使用")

    if _loader and _loader.json.get("heartbeat"):
        from framework.heartbeat import heartbeat_loop, run_heartbeat_once
        logger.info(f"[Discord] heartbeat 已启动（agent={agent_name}）")
        await run_heartbeat_once()
        asyncio.create_task(heartbeat_loop())


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if not _is_authorized(message.author):
        return
    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    user_input = message.content.strip()
    if not user_input:
        return

    full_text = None
    error_msg = None

    async with message.channel.typing():
        try:
            full_text = await _invoke_agent_async(user_input, message=message)

        except Exception as e:
            logger.error(f"[agent] 调用出错: {e}", exc_info=is_debug())
            error_msg = str(e)

    agent_name = _loader.name if _loader else "Agent"
    if error_msg:
        await message.channel.send(f"{agent_name} 出错了: {error_msg}")
        return
    if not full_text:
        await message.channel.send(f"（{agent_name} 没有输出，请重试）")
        return

    # 提取文件附件标记，剩余部分正常发送
    clean_text, file_paths = _extract_attachments(full_text)

    if clean_text:
        for chunk in split_fence_aware(clean_text):
            await message.channel.send(chunk)

    for path in file_paths:
        if not os.path.isfile(path):
            await message.channel.send(f"⚠️ 文件不存在，无法发送：`{path}`")
            logger.warning(f"[discord] SEND_FILE 文件不存在: {path}")
            continue
        try:
            await message.channel.send(file=discord.File(path))
            logger.info(f"[discord] 已发送文件: {path}")
        except discord.HTTPException as e:
            await message.channel.send(f"⚠️ 文件发送失败：{e}")
            logger.error(f"[discord] 文件发送失败 {path}: {e}")


# ==========================================
# 命令前置认证
# ==========================================
async def _check_auth(ctx) -> bool:
    return _is_authorized(ctx.author)


# ==========================================
# 命令
# ==========================================
@bot.command(name="help")
async def show_help(ctx):
    if not await _check_auth(ctx):
        return
    await ctx.send(
        "**Agent 命令手册**\n"
        "```\n"
        "── Session 管理（框架通用）──────────────────────\n"
        "!new <名称>        创建并切换到新命名 session\n"
        "!switch <名称>     切换到已有命名 session\n"
        "!session           显示当前 session 名称和 thread_id\n"
        "!sessions          列出所有命名 sessions（当前用 ◀ 标注）\n"
        "!memory            查看 session 消息数和 DB 大小\n"
        "!compact [N]       压缩 session，保留最近 N 条（默认 20）\n"
        "!reset confirm     清空当前 session 全部记忆\n"
        "!clear             清空 session（无需确认）\n"
        "\n"
        "── 工具 ─────────────────────────────────────────\n"
        "!tokens            查看 token 消耗统计\n"
        "!tokens reset      重置 token 计数\n"
        "!setproject <路径>  设置工作目录\n"
        "!project           查看当前项目目录\n"
        "!debug             查看 debug 模式状态\n"
        "!whoami            显示你的 Discord ID\n"
        "!help              显示此帮助信息\n"
        "```\n"
        "**提示**：消息中包含 `@Gemini` 可强制触发 Gemini 架构咨询。"
    )


@bot.command(name="new")
async def new_session_cmd(ctx, *, name: str = ""):
    if not await _check_auth(ctx):
        return
    if not name:
        await ctx.send("用法：`!new <session名称>`")
        return
    try:
        await _controller.new_session(name)
        await ctx.send(f"✅ 新 session `{name}` 已创建并激活\nthread_id: `{_controller.active_thread_id}`")
    except ValueError as e:
        await ctx.send(f"❌ {e}")
    except Exception as e:
        await ctx.send(f"创建失败: {e}")


@bot.command(name="switch")
async def switch_session_cmd(ctx, *, name: str = ""):
    if not await _check_auth(ctx):
        return
    if not name:
        await ctx.send("用法：`!switch <session名称>`")
        return
    try:
        await _controller.switch_session(name)
        await ctx.send(f"✅ 已切换到 session `{name}`\nthread_id: `{_controller.active_thread_id}`")
    except ValueError as e:
        await ctx.send(f"❌ {e}")
    except Exception as e:
        await ctx.send(f"切换失败: {e}")


@bot.command(name="session")
async def show_session(ctx):
    if not await _check_auth(ctx):
        return
    thread_id = _controller.active_thread_id
    name = _session_mgr.find_name_by_thread_id(thread_id) or "（默认）"
    await ctx.send(f"当前 session: `{name}`\nthread_id: `{thread_id}`")


@bot.command(name="sessions")
async def list_sessions_cmd(ctx):
    if not await _check_auth(ctx):
        return
    all_sessions = _session_mgr.list_all()
    if not all_sessions:
        await ctx.send("还没有任何命名 session。用 `!new <名称>` 创建第一个。")
        return
    current_tid = _controller.active_thread_id
    lines = []
    for sname, env in all_sessions.items():
        marker = " ◀" if env.thread_id == current_tid else ""
        lines.append(f"- `{sname}` → `{env.thread_id}`{marker}")
    await ctx.send("**命名 Sessions：**\n" + "\n".join(lines))


@bot.command(name="debug")
async def toggle_debug(ctx):
    if not await _check_auth(ctx):
        return
    status = "ON" if is_debug() else "OFF（设置 DEBUG=1 启动以开启）"
    await ctx.send(f"Debug mode: {status}")


@bot.command(name="whoami")
async def whoami(ctx):
    authorized = "已授权" if _is_authorized(ctx.author) else "未授权"
    await ctx.send(f"你的 Discord ID: `{ctx.author.id}` ({authorized})")


@bot.command(name="memory")
async def show_memory(ctx):
    if not await _check_auth(ctx):
        return
    thread_id = _controller.active_thread_id
    stats = _session_mgr.session_stats(thread_id)
    name = _session_mgr.find_name_by_thread_id(thread_id) or "默认"
    await ctx.send(
        f"**Session 状态**（`{name}`）\n"
        f"- thread: `{stats['thread_id']}`\n"
        f"- checkpoints: `{stats['message_count']}` 条\n"
        f"- DB 大小: `{stats['db_size_kb']} KB`"
    )


@bot.command(name="compact")
async def compact_session(ctx, keep: int = 20):
    if not await _check_auth(ctx):
        return
    thread_id = _controller.active_thread_id
    deleted = _session_mgr.compact(thread_id, keep_last=keep)
    _loader.invalidate_engine()
    await ctx.send(f"Compact 完成：删除了 `{deleted}` 条旧记录，保留最近 `{keep}` 条。")


@bot.command(name="reset")
async def reset_session(ctx, confirm: str = ""):
    if not await _check_auth(ctx):
        return
    agent_name = _loader.name if _loader else "Agent"
    if confirm != "confirm":
        await ctx.send(
            f"此操作将**清空 {agent_name} 的全部记忆**，无法恢复。\n"
            "确认请输入：`!reset confirm`"
        )
        return
    thread_id = _controller.active_thread_id
    deleted = _session_mgr.reset(thread_id)
    _loader.invalidate_engine()
    await ctx.send(f"Session 已重置，清空了 `{deleted}` 条记录。{agent_name} 从零开始。")


@bot.command(name="tokens")
async def show_tokens(ctx, reset: str = ""):
    if not await _check_auth(ctx):
        return
    if reset == "reset":
        reset_token_stats()
        await ctx.send("Token 计数已重置。")
        return
    s = get_token_stats()
    inp = s["input_tokens"]
    out = s["output_tokens"]
    cr = s["cache_read_input_tokens"]
    cc = s["cache_creation_input_tokens"]
    calls = s["calls"]
    cost_usd = (inp * 3 + out * 15 + cr * 0.3 + cc * 3.75) / 1_000_000
    saved_usd = cr * (3 - 0.3) / 1_000_000
    await ctx.send(
        f"**Token 统计（本次启动后）**\n"
        f"```\n"
        f"调用次数      : {calls}\n"
        f"Input tokens  : {inp:,}\n"
        f"Output tokens : {out:,}\n"
        f"Cache read    : {cr:,}  (省了 ${saved_usd:.4f})\n"
        f"Cache create  : {cc:,}\n"
        f"估算费用      : ~${cost_usd:.4f} USD\n"
        f"```"
    )


@bot.command(name="clear")
async def clear_session(ctx):
    if not await _check_auth(ctx):
        return
    agent_name = _loader.name if _loader else "Agent"
    thread_id = _controller.active_thread_id
    deleted = _session_mgr.reset(thread_id)
    _loader.invalidate_engine()
    await ctx.send(f"Session 已清空（{deleted} 条）。{agent_name} 从零开始。")


@bot.command(name="resources")
async def show_resources(ctx):
    if not await _check_auth(ctx):
        return
    from framework.resource_lock import format_resource_status
    status = format_resource_status()
    await ctx.send(f"**资源锁状态**\n```\n{status}\n```")


@bot.command(name="setproject")
async def set_project(ctx, *, path: str = ""):
    if not await _check_auth(ctx):
        return
    global _current_project_root
    if not path:
        current = _current_project_root or "（未设置）"
        await ctx.send(f"当前项目目录：`{current}`")
        return
    path = os.path.expanduser(path.strip())
    if not os.path.isdir(path):
        await ctx.send(f"路径不存在：`{path}`")
        return
    _current_project_root = path
    await ctx.send(f"项目目录已设置为：`{path}`\nGit 时间机器已就位。")


@bot.command(name="project")
async def show_project(ctx):
    if not await _check_auth(ctx):
        return
    current = _current_project_root or "（未设置，全局模式）"
    await ctx.send(f"当前项目目录：`{current}`")


# ==========================================
# 入口
# ==========================================
def run_discord(loader=None):
    global _loader, _controller, _session_mgr
    _loader = loader
    _session_mgr = loader.session_mgr if loader else None

    # 初始化 GraphController（同步包装）
    if loader:
        import asyncio
        _controller = asyncio.get_event_loop().run_until_complete(loader.get_controller())

    token = loader.load_config().discord_token if loader else ""
    if not token:
        print("请在 agent.json 或 .env（DISCORD_BOT_TOKEN）中设置 Discord Bot Token")
        return
    logger.info(f"[Discord] 启动中... agent={loader.name if loader else '?'} DEBUG={'ON' if is_debug() else 'OFF'}")
    bot.run(token, log_handler=None)
