"""
Discord connector — _DiscordInterface class and per-channel message consumer.
"""

import asyncio
import logging
import os

import discord
from interfaces.command_registry import Connector
from framework.debug import is_debug
from interfaces.base_interface import BaseInterface
from interfaces.discord import state as _state
from interfaces.discord.alerts import _register_pending_tasks_for_channel
from interfaces.discord.formatting import format_persona_response
from interfaces.discord.messaging import _refresh_history_file, send_to_channel

logger = logging.getLogger("discord_bot")


async def _channel_consumer(channel_id: int) -> None:
    """
    每个频道一个消费者协程：从队列中顺序取出消息并处理。
    队列空闲 60s 后自动退出，下次有消息时重新创建。
    !stop 取消当前 agent task → CancelledError 传到此处 → 清空剩余队列并退出。
    """
    queue = _state._channel_queues[channel_id]
    agent_name = _state._loader.name if _state._loader else "Agent"
    while True:
        try:
            user_input, message = await asyncio.wait_for(queue.get(), timeout=60.0)
        except asyncio.TimeoutError:
            _state._channel_consumers.pop(channel_id, None)
            return

        # ── 积压合并：如果队列里还有消息，全部取出合并为一条 ──────────
        merged_inputs = [user_input]
        extra_drained = 0
        while not queue.empty():
            try:
                extra_input, extra_msg = queue.get_nowait()
                merged_inputs.append(extra_input)
                message = extra_msg  # 用最新的 message 对象做回复锚点
                extra_drained += 1
            except asyncio.QueueEmpty:
                break
        if extra_drained:
            user_input = "\n\n".join(merged_inputs)
            logger.info(f"[discord] 合并了 {extra_drained + 1} 条积压消息 (channel={channel_id})")

        print(f"DEBUG: Consumer picking up message from channel {channel_id}: {user_input[:20]}...", flush=True)
        iface = _DiscordInterface(_state._loader, channel_id=channel_id)
        print("DEBUG: Calling iface.invoke...", flush=True)
        task = asyncio.create_task(iface.invoke(user_input, message))
        _state._channel_tasks[channel_id] = task
        try:
            await task
        except asyncio.CancelledError:
            drained = 0
            while not queue.empty():
                queue.get_nowait()
                drained += 1
            if drained:
                await message.channel.send(f"已停止，清除了 {drained} 条待处理消息。")
            _state._channel_consumers.pop(channel_id, None)
            return
        except Exception as e:
            logger.error(f"[agent] 出错: {e}", exc_info=is_debug())
            await message.channel.send(f"{agent_name} 出错了: {e}")
        finally:
            _state._channel_tasks.pop(channel_id, None)

        # 检测刚发送的回复中是否有 PENDING 标记，注册后台 poller
        _register_pending_tasks_for_channel(channel_id)
        queue.task_done()
        # 补偿额外 drain 的 task_done 计数
        for _ in range(extra_drained):
            queue.task_done()


class _DiscordInterface(BaseInterface):
    """
    Discord connector 的 BaseInterface 实现。

    - _connector = Connector.DISCORD：!help 自动过滤 Discord 命令
    - channel_id：per-channel session 解析，覆写 _resolve_thread_id /
      _resolve_session_name，确保所有通用命令操作正确的频道 session
    - invoke(user_input, message)：统一入口，处理历史刷新 + 流式/非流式 Discord UI
    """

    _connector = Connector.DISCORD

    def __init__(self, loader, channel_id: int | None = None) -> None:
        super().__init__(loader)
        self._controller  = _state._controller
        self._session_mgr = _state._session_mgr
        self._config      = loader.load_config()
        self._channel_id  = channel_id
        self._streaming   = _state._discord_streaming
        self._event_queue: asyncio.Queue = asyncio.Queue()

    # ── per-channel session 解析 ────────────────────────────────────

    def _resolve_thread_id(self) -> str:
        if self._channel_id is not None and _state._session_mgr is not None:
            name = _state._ensure_channel_session(self._channel_id)
            env  = _state._session_mgr.get_envelope(name)
            if env:
                return env.thread_id
        return super()._resolve_thread_id()

    def _resolve_session_name(self) -> str | None:
        if self._channel_id is not None:
            return _state._channel_active_session.get(
                self._channel_id,
                _state._channel_default_session(self._channel_id),
            )
        return super()._resolve_session_name()

    # ── per-channel command overrides ────────────────────────────────

    def _friendly_name(self, full_name: str) -> str:
        """Return user-visible session name (strip channel prefix)."""
        if self._channel_id is None:
            return full_name
        default_name = _state._channel_default_session(self._channel_id)
        if full_name == default_name:
            return "default"
        prefix = f"{_state._channel_prefix(self._channel_id)}-"
        if full_name.startswith(prefix):
            return full_name[len(prefix):]
        return full_name

    async def handle_command(self, cmd: str, arg: str) -> str | None:
        """
        Discord per-channel overrides for session/stream commands.
        Delegates everything else to BaseInterface.handle_command().
        """
        channel_id = self._channel_id

        if cmd == "!new":
            if not arg:
                return "用法：!new <session名称>"
            if channel_id is None:
                return await super().handle_command(cmd, arg)
            name = arg.split()[0]
            full_name = f"{_state._channel_prefix(channel_id)}-{name}"
            try:
                env = _state._session_mgr.create_session(full_name)
                _state._channel_active_session[channel_id] = full_name
                return f"已创建并切换到 session `{name}`\nthread: `{env.thread_id}`"
            except ValueError as e:
                return f"❌ {e}"

        if cmd == "!switch":
            if not arg:
                return "用法：!switch <session名称> 或 !switch default"
            if channel_id is None:
                return await super().handle_command(cmd, arg)
            name = arg.strip()
            if name == "default":
                full_name = _state._channel_default_session(channel_id)
            else:
                full_name = f"{_state._channel_prefix(channel_id)}-{name}"
            env = _state._session_mgr.get_envelope(full_name)
            if not env:
                return f"❌ Session `{name}` 不存在。用 `!sessions` 查看可用 sessions。"
            _state._channel_active_session[channel_id] = full_name
            return f"已切换到 session `{name}`\nthread: `{env.thread_id}`"

        if cmd == "!session":
            if channel_id is None:
                return await super().handle_command(cmd, arg)
            name = _state._channel_active_session.get(channel_id, _state._channel_default_session(channel_id))
            env = _state._session_mgr.get_envelope(name) if _state._session_mgr else None
            display = self._friendly_name(name)
            tid = env.thread_id if env else "?"
            ns = env.node_sessions if env else {}
            ns_display = ", ".join(f"{k}={v[:8]}" for k, v in ns.items() if v) if ns else "（无）"
            return (
                f"session: `{display}`\n"
                f"thread: `{tid}`\n"
                f"node_sessions: `{ns_display}`"
            )

        if cmd == "!sessions":
            if channel_id is None:
                return await super().handle_command(cmd, arg)
            prefix = _state._channel_prefix(channel_id)
            sessions = _state._session_mgr.list_by_prefix(prefix)
            if not sessions:
                return "当前频道没有 session。"
            active = _state._channel_active_session.get(channel_id, _state._channel_default_session(channel_id))
            lines = []
            for sname, env in sessions.items():
                marker = " ◀" if sname == active else ""
                display = self._friendly_name(sname)
                lines.append(f"  {display:<40} thread={env.thread_id}{marker}")
            return "**频道 Sessions**\n```\n" + "\n".join(lines) + "\n```"

        if cmd == "!stream":
            _state._discord_streaming = not _state._discord_streaming
            return f"Streaming: {'ON' if _state._discord_streaming else 'OFF'}"

        return await super().handle_command(cmd, arg)

    # ── 流式回调（invoke_agent 经由 BaseInterface 调用）─────────────

    def _on_stream_chunk(self, text: str, is_thinking: bool = False) -> None:
        if not text:
            return
        self._event_queue.put_nowait(("thinking" if is_thinking else "text", text))

    def _on_stream_reset(self) -> None:
        self._event_queue = asyncio.Queue()

    # ── 统一调用入口 ──────────────────────────────────────────────────

    async def invoke(self, user_input: str, message) -> None:
        """统一入口：处理历史刷新 + 流式/非流式 Discord UI，再调用 invoke_agent()。"""
        # 注入当前 channel 供 Discord tool server 使用
        from interfaces.discord.tool_server import set_current_channel
        set_current_channel(message.channel)
        _state._current_channel = message.channel

        history_limit = (_state._loader.json.get("channel_history_limit", 0) if _state._loader else 0)
        if history_limit and self._channel_id is not None:
            await _refresh_history_file(
                message.channel, history_limit, exclude_msg_id=message.id
            )
        if self._streaming:
            await self._invoke_streaming(user_input, message)
        else:
            await self._invoke_non_streaming(user_input, message)

    async def _invoke_non_streaming(self, user_input: str, message) -> None:
        from framework.nodes.llm.llm_node import set_channel_send_callback

        agent_name = _state._loader.name if _state._loader else "Agent"

        async def _subgraph_send(text: str) -> None:
            await send_to_channel(message.channel, text)

        set_channel_send_callback(_subgraph_send)
        try:
            async with message.channel.typing():
                result = await self.invoke_agent(user_input)
        finally:
            set_channel_send_callback(None)
        if result:
            clean_text, file_paths = BaseInterface.extract_attachments(result)
            clean_text = format_persona_response(clean_text)
            await send_to_channel(message.channel, clean_text)
            for path in file_paths:
                if not os.path.isfile(path):
                    await message.channel.send(f"⚠️ 文件不存在：`{path}`")
                    continue
                try:
                    await message.channel.send(file=discord.File(path))
                except discord.HTTPException as e:
                    await message.channel.send(f"⚠️ 文件发送失败：{e}")
        else:
            await message.channel.send(f"（{agent_name} 没有输出，请重试）")

    async def _invoke_streaming(self, user_input: str, message) -> None:
        """
        流式模式：
        - 立即发送 typing 指示器，每 8s 续命（Discord 有效期 ~10s）
        - 由调用方（_channel_consumer）保证串行，此函数不加锁
        - 思考块 → 独立斜体消息，实时更新，保留在频道（不追加最终编辑）
        - 文字 token → 独立草稿，从头追加（不截尾），完成后删除
        - 最终回复以全新消息发出（无 "(edited)"）
        """
        agent_name = _state._loader.name if _state._loader else "Agent"

        # ── Typing 指示器 ─────────────────────────────────────────────
        typing_ctx = message.channel.typing()
        await typing_ctx.__aenter__()

        # ── 编辑器任务状态 ────────────────────────────────────────────
        thinking_msg = [None]
        thinking_buf: list[str] = []
        text_draft   = [None]
        text_buf: list[str]     = []

        async def _editor():
            loop = asyncio.get_event_loop()
            thinking_last     = loop.time()
            text_last         = loop.time()
            THINKING_THROTTLE = 3.0
            TEXT_THROTTLE     = 5.0
            # 429 动态退避：遇到 rate limit 后逐步加大编辑间隔，恢复后递减
            _backoff_until    = 0.0      # 退避截止时间（loop.time）
            _backoff_level    = 0        # 当前退避级别（0=正常）
            _BACKOFF_STEPS    = [0, 2.0, 5.0, 10.0]  # 级别 → 额外等待秒数
            _BACKOFF_DECAY    = 15.0     # 无 429 持续 N 秒后降一级

            _last_429_time    = 0.0

            async def _safe_edit(msg_obj, content: str) -> bool:
                """编辑消息，捕获 429 并触发退避。返回 True=成功/跳过，False=429。"""
                nonlocal _backoff_until, _backoff_level, _last_429_time
                now = loop.time()
                if now < _backoff_until:
                    return True
                try:
                    await msg_obj.edit(content=content)
                    if _backoff_level > 0 and now - _last_429_time > _BACKOFF_DECAY:
                        _backoff_level = max(0, _backoff_level - 1)
                        if _backoff_level == 0:
                            logger.debug("[discord-editor] 退避已恢复到正常频率")
                    return True
                except discord.HTTPException as e:
                    if e.status == 429:
                        retry_after = getattr(e, "retry_after", None) or _BACKOFF_STEPS[min(_backoff_level + 1, len(_BACKOFF_STEPS) - 1)]
                        _backoff_level = min(_backoff_level + 1, len(_BACKOFF_STEPS) - 1)
                        _backoff_until = now + retry_after
                        _last_429_time = now
                        logger.warning(
                            f"[discord-editor] 429 rate limit, 退避 {retry_after:.1f}s "
                            f"(level={_backoff_level})"
                        )
                        return False
                    return True
                except Exception:
                    return True

            while True:
                sentinel = False
                while True:
                    try:
                        kind, chunk = self._event_queue.get_nowait()
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
                extra_wait = _BACKOFF_STEPS[_backoff_level] if _backoff_level > 0 else 0

                if thinking_buf and (sentinel or now - thinking_last >= THINKING_THROTTLE + extra_wait):
                    thinking_last = now
                    preview   = "".join(thinking_buf)
                    max_body  = 1860
                    body_text = f"…{preview[-max_body:]}" if len(preview) > max_body else preview
                    quoted    = "\n".join(f"> {line}" if line else ">" for line in body_text.splitlines())
                    display   = f"*💭*\n{quoted}"
                    if thinking_msg[0] is None:
                        thinking_msg[0] = await message.channel.send(display)
                    else:
                        await _safe_edit(thinking_msg[0], display)

                if text_buf and (sentinel or now - text_last >= TEXT_THROTTLE + extra_wait):
                    text_last = now
                    preview = "".join(text_buf)
                    cursor  = "" if sentinel else " ▌"
                    if len(preview) + len(cursor) > 1900:
                        display = "…" + preview[-(1897 - len(cursor)):] + cursor
                    else:
                        display = preview + cursor
                    if text_draft[0] is None:
                        text_draft[0] = await message.channel.send(display)
                        text_last = now
                    else:
                        await _safe_edit(text_draft[0], display)

                if sentinel:
                    break
                await asyncio.sleep(0.1)

        # 子图节点内容回调
        from framework.nodes.llm.llm_node import set_channel_send_callback
        _subgraph_draft_cleared = [False]

        async def _subgraph_send(text: str) -> None:
            if not _subgraph_draft_cleared[0] and text_draft[0]:
                try:
                    await text_draft[0].delete()
                except Exception:
                    pass
                text_draft[0] = None
                text_buf.clear()
                _subgraph_draft_cleared[0] = True
            await send_to_channel(message.channel, text)

        set_channel_send_callback(_subgraph_send)

        editor_task = asyncio.create_task(_editor())
        try:
            result = await self.invoke_agent(user_input)
        except asyncio.CancelledError:
            self._event_queue.put_nowait((None, None))
            await editor_task
            await typing_ctx.__aexit__(None, None, None)
            if text_draft[0]:
                try:
                    await text_draft[0].delete()
                except Exception:
                    pass
            raise
        finally:
            set_channel_send_callback(None)
            self._event_queue.put_nowait((None, None))
            await editor_task
            await typing_ctx.__aexit__(None, None, None)

        streamed_full = "".join(text_buf) if text_buf else ""
        if _subgraph_draft_cleared[0]:
            final_text = result or streamed_full or ""
        else:
            final_text = streamed_full or result or ""

        if not final_text:
            await message.channel.send(f"（{agent_name} 没有输出，请重试）")
            return

        clean_text, file_paths = BaseInterface.extract_attachments(final_text)
        clean_text = format_persona_response(clean_text)
        chunks = BaseInterface.split_fence_aware(clean_text, _state.DISCORD_MAX_CHARS) if clean_text else []

        if text_draft[0] and chunks:
            if _subgraph_draft_cleared[0]:
                try:
                    await text_draft[0].delete()
                except Exception:
                    pass
                text_draft[0] = None
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                if len(chunks) == 1:
                    try:
                        await text_draft[0].edit(content=chunks[0])
                    except Exception:
                        await message.channel.send(chunks[0])
                else:
                    try:
                        await text_draft[0].delete()
                    except Exception:
                        pass
                    text_draft[0] = None
                    for chunk in chunks:
                        await message.channel.send(chunk)
        elif text_draft[0]:
            try:
                await text_draft[0].delete()
            except Exception:
                pass
        else:
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
