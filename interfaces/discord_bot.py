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

from framework.base_interface import BaseInterface
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

# 流式输出开关（!stream 切换）
_discord_streaming: bool = True

# 每个频道的并发锁（防止多条消息同时被处理）
_channel_locks: dict = {}


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
# Fence-Aware Chunking — delegated to BaseInterface
# ==========================================
def split_fence_aware(text: str, max_chars: int = DISCORD_MAX_CHARS) -> list[str]:
    return BaseInterface.split_fence_aware(text, max_chars)


# ==========================================
# 当前项目目录
# ==========================================
_current_project_root: str = ""


# ==========================================
# 文件发送解析（图片 / 视频）— delegated to BaseInterface
# ==========================================
def _extract_attachments(text: str) -> tuple[str, list[str]]:
    return BaseInterface.extract_attachments(text)


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


async def _invoke_agent_streaming(user_input: str, message) -> None:
    """
    流式模式：
    - 立即发送 typing 指示器，每 8s 续命（Discord 有效期 ~10s）
    - 每个频道加锁，避免并发打断
    - 思考块 → 独立斜体消息，实时更新，保留在频道（不追加最终编辑）
    - 文字 token → 独立草稿，从头追加（不截尾），完成后删除
    - 最终回复以全新消息发出（无 "(edited)"）
    """
    from framework.claude.node import set_stream_callback
    from framework.base_interface import BaseInterface

    channel_id = message.channel.id
    if channel_id not in _channel_locks:
        _channel_locks[channel_id] = asyncio.Lock()

    lock = _channel_locks[channel_id]
    if lock.locked():
        await message.channel.send("⏳ 正在处理上一条消息，请稍候…")
        return

    async with lock:
        engine = _controller._graph
        config = _controller.get_config()
        agent_name = _loader.name if _loader else "Agent"

        history_limit = (_loader.json.get("channel_history_limit", 0) if _loader else 0)
        if history_limit:
            await _refresh_history_file(message.channel, history_limit, exclude_msg_id=message.id)

        init_state: dict = {"messages": [HumanMessage(content=user_input)]}
        if _current_project_root:
            init_state["project_root"] = _current_project_root

        # ── Typing 指示器：立即发送一次，背景每 8s 续命 ────────────────────
        await message.channel.trigger_typing()

        async def _typing_loop():
            try:
                while True:
                    await asyncio.sleep(8)
                    await message.channel.trigger_typing()
            except asyncio.CancelledError:
                pass

        typing_task = asyncio.create_task(_typing_loop())

        # ── 事件队列：区分思考 / 文字 token ────────────────────────────────
        event_queue: asyncio.Queue = asyncio.Queue()
        _in_thinking = [False]

        def _discord_cb(text: str) -> None:
            if "\x1b[2m" in text:
                _in_thinking[0] = True
                return
            if "\x1b[0m" in text:
                _in_thinking[0] = False
                return
            if "\x1b" in text:
                return
            if not text:
                return
            event_queue.put_nowait(("thinking" if _in_thinking[0] else "text", text))

        # ── 编辑器任务 ──────────────────────────────────────────────────────
        thinking_msg = [None]
        thinking_buf: list[str] = []
        text_draft = [None]
        text_buf: list[str] = []

        async def _editor():
            loop = asyncio.get_event_loop()
            thinking_last = 0.0
            text_last = 0.0
            THINKING_THROTTLE = 1.5
            TEXT_THROTTLE = 0.5

            while True:
                sentinel = False
                while True:
                    try:
                        kind, chunk = event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if kind is None:
                        sentinel = True
                        break
                    if kind == "thinking":
                        thinking_buf.append(chunk)
                    else:
                        text_buf.append(chunk)

                now = loop.time()

                # 思考消息（斜体，独立持久，流式更新，不做最终二次编辑）
                if thinking_buf and (sentinel or now - thinking_last >= THINKING_THROTTLE):
                    thinking_last = now
                    preview = "".join(thinking_buf)
                    # 截头保尾，不超 1900 字符
                    display = f"*💭 {preview[-1890:]}*" if len(preview) > 1890 else f"*💭 {preview}*"
                    if thinking_msg[0] is None:
                        thinking_msg[0] = await message.channel.send(display)
                    else:
                        try:
                            await thinking_msg[0].edit(content=display)
                        except Exception:
                            pass

                # 文字草稿（从头追加，不截尾；完成后删除）
                if text_buf and (sentinel or now - text_last >= TEXT_THROTTLE):
                    text_last = now
                    preview = "".join(text_buf)
                    cursor = "" if sentinel else " ▌"
                    # 从头显示，只在超长时截掉开头（保持末尾可见）
                    if len(preview) + len(cursor) > 1900:
                        display = "…" + preview[-(1897 - len(cursor)):] + cursor
                    else:
                        display = preview + cursor
                    if text_draft[0] is None:
                        text_draft[0] = await message.channel.send("▌")
                    try:
                        await text_draft[0].edit(content=display)
                    except Exception:
                        pass

                if sentinel:
                    break
                await asyncio.sleep(0.1)

        editor_task = asyncio.create_task(_editor())
        set_stream_callback(_discord_cb)
        try:
            result_state = await engine.ainvoke(init_state, config=config)
        finally:
            set_stream_callback(None)
            event_queue.put_nowait((None, None))
            await editor_task
            typing_task.cancel()

        final_text = BaseInterface._extract_response(result_state)

        # 删除文字草稿（(edited) 随草稿消失）
        if text_draft[0]:
            try:
                await text_draft[0].delete()
            except Exception:
                pass

        # 思考消息保留原样（不追加最终编辑，避免 "(edited)" 出现在最后）

        if not final_text:
            await message.channel.send(f"（{agent_name} 没有输出，请重试）")
            return

        clean_text, file_paths = _extract_attachments(final_text)
        chunks = split_fence_aware(clean_text) if clean_text else []

        for chunk in chunks:
            await message.channel.send(chunk)

        for path in file_paths:
            if not os.path.isfile(path):
                await message.channel.send(f"⚠️ 文件不存在：`{path}`")
                continue
            try:
                await message.channel.send(file=discord.File(path))
            except discord.HTTPException as e:
                await message.channel.send(f"⚠️ 文件发送失败：{e}")


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

    agent_name = _loader.name if _loader else "Agent"
    full_text = None

    if _discord_streaming:
        # 流式模式：_invoke_agent_streaming 全权处理所有 Discord 消息
        try:
            await _invoke_agent_streaming(user_input, message)
        except Exception as e:
            logger.error(f"[agent] 流式调用出错: {e}", exc_info=is_debug())
            await message.channel.send(f"{agent_name} 出错了: {e}")
        return
    else:
        # 非流式模式：typing 占位 → 收集完整回复 → 发送
        async with message.channel.typing():
            try:
                full_text = await _invoke_agent_async(user_input, message=message)
            except Exception as e:
                logger.error(f"[agent] 调用出错: {e}", exc_info=is_debug())
                await message.channel.send(f"{agent_name} 出错了: {e}")
                return

    if not full_text:
        await message.channel.send(f"（{agent_name} 没有输出，请重试）")
        return

    # 提取文件附件标记，分块发送
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
        "!stream            切换流式输出 ON/OFF\n"
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
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!new", name)
    await ctx.send(reply or "❌ 创建失败")


@bot.command(name="switch")
async def switch_session_cmd(ctx, *, name: str = ""):
    if not await _check_auth(ctx):
        return
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!switch", name)
    await ctx.send(reply or "❌ 切换失败")


@bot.command(name="session")
async def show_session(ctx):
    if not await _check_auth(ctx):
        return
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!session", "")
    await ctx.send(reply)


@bot.command(name="sessions")
async def list_sessions_cmd(ctx):
    if not await _check_auth(ctx):
        return
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!sessions", "")
    await ctx.send(reply)


@bot.command(name="debug")
async def toggle_debug(ctx):
    if not await _check_auth(ctx):
        return
    status = "ON" if is_debug() else "OFF（设置 DEBUG=1 启动以开启）"
    await ctx.send(f"Debug mode: {status}")


@bot.command(name="stream")
async def toggle_stream(ctx):
    if not await _check_auth(ctx):
        return
    global _discord_streaming
    _discord_streaming = not _discord_streaming
    state = "ON" if _discord_streaming else "OFF"
    await ctx.send(f"Streaming: {state}")


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
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!tokens", reset)
    await ctx.send(f"**Token 统计（本次启动后）**\n```\n{reply}\n```")


@bot.command(name="clear")
async def clear_session(ctx):
    if not await _check_auth(ctx):
        return
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!clear", "")
    await ctx.send(reply)


@bot.command(name="resources")
async def show_resources(ctx):
    if not await _check_auth(ctx):
        return
    iface = _DiscordInterface(_loader)
    reply = await iface.handle_command("!resources", "")
    await ctx.send(f"**资源锁状态**\n```\n{reply}\n```")


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
# DiscordInterface — wraps BaseInterface for command delegation
# ==========================================
class _DiscordInterface(BaseInterface):
    """
    Thin wrapper so discord command handlers can delegate to BaseInterface.handle_command()
    without managing controller lifecycle themselves.  The module-level _controller /
    _session_mgr are already initialised by run_discord(); we just reference them.
    """

    def __init__(self, loader) -> None:
        super().__init__(loader)
        # Reuse the already-initialised globals
        self._controller = _controller
        self._session_mgr = _session_mgr


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
