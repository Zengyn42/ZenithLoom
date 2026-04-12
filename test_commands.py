"""
用户命令全覆盖测试

覆盖所有暴露给用户的命令：
  CLI  : !new, !switch, !sessions, !session
  Discord: !help, !new, !switch, !session, !sessions, !whoami,
           !debug, !memory, !compact, !reset, !tokens,
           !setproject, !project, !stop, !channels
  Per-channel: _ensure_channel_session, _get_channel_config,
               _get_channel_workspace, _cleanup_channel
  SessionManager: list_by_prefix, delete_by_prefix
  工具函数: split_fence_aware（消息分块），auth whitelist

不做实际 API 调用，使用 tempfile + unittest.mock。
运行：python3 test_commands.py
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_session_mgr(tmp_dir: str):
    from framework.session_mgr import SessionManager
    return SessionManager(
        sessions_file=os.path.join(tmp_dir, "sessions.json"),
        db_path=os.path.join(tmp_dir, "test.db"),
    )


def _mock_ctx(user_id: int = 12345, channel_id: int = 100) -> MagicMock:
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.id = user_id
    ctx.send = AsyncMock()
    ctx.channel = MagicMock()
    ctx.channel.id = channel_id
    ctx.guild = MagicMock()
    return ctx


def _mock_loader(sm, name: str = "hani") -> MagicMock:
    loader = MagicMock()
    loader.name = name
    loader.session_mgr = sm
    loader.get_engine = AsyncMock(return_value=MagicMock())
    loader.invalidate_engine = MagicMock()
    # Make load_config().discord_allowed_users return an empty list (allow all)
    config = MagicMock()
    config.discord_allowed_users = []
    config.workspace = ""
    loader.load_config = MagicMock(return_value=config)
    loader.json = {}
    return loader


def _setup_bot(sm, name: str = "hani"):
    """Set module-level globals in discord_bot for testing."""
    import interfaces.discord_bot as bot
    loader = _mock_loader(sm, name)
    bot._loader = loader
    bot._session_mgr = sm
    bot._controller = MagicMock()
    bot._channel_active_session.clear()
    bot._channel_tasks.clear()
    bot._channel_queues.clear()
    bot._channel_consumers.clear()
    # Set up AsyncMock methods on controller for handle_command tests
    bot._controller.checkpoint_stats = AsyncMock(return_value=0)
    bot._controller.compact_checkpoint = AsyncMock(return_value=5)
    bot._controller.compact_claude_session = AsyncMock(return_value="无 Claude session 可压缩")
    bot._controller.reset_checkpoint = AsyncMock(return_value=10)
    return bot, loader


def _set_active(name: str, thread_id: str):
    """Helper: set framework.graph._active without await."""
    import framework.graph as fg
    from dataclasses import dataclass

    @dataclass
    class _A:
        name: str
        thread_id: str

    fg._active = _A(name=name, thread_id=thread_id)


# ─────────────────────────────────────────────
# 1. SessionManager
# ─────────────────────────────────────────────

def test_session_manager_create():
    print("--- SessionManager.create_session ---")
    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        env = sm.create_session("alpha")
        assert env.thread_id
        assert sm.get("alpha") == env.thread_id
        print(f"   alpha → {env.thread_id[:8]}")

        # Duplicate raises
        try:
            sm.create_session("alpha")
            assert False, "should raise"
        except ValueError:
            pass

        print("✅ create_session OK\n")


def test_session_manager_list_find():
    print("--- SessionManager.list_all / find_name_by_thread_id ---")
    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        env_a = sm.create_session("alpha")
        sm.create_session("beta")
        assert set(sm.list_all()) == {"alpha", "beta"}
        assert sm.find_name_by_thread_id(env_a.thread_id) == "alpha"
        assert sm.find_name_by_thread_id("nonexistent") is None
        print("✅ list_all / find_name_by_thread_id OK\n")


def test_session_manager_persistence():
    print("--- SessionManager persistence ---")
    with tempfile.TemporaryDirectory() as tmp:
        sm1 = _make_session_mgr(tmp)
        env = sm1.create_session("persisted")
        tid = env.thread_id

        sm2 = _make_session_mgr(tmp)
        assert sm2.get("persisted") == tid
        print(f"   persisted: {tid[:8]}")
        print("✅ persistence OK\n")


def test_session_manager_legacy_migration():
    print("--- SessionManager legacy (plain string) migration ---")
    with tempfile.TemporaryDirectory() as tmp:
        sf = os.path.join(tmp, "sessions.json")
        with open(sf, "w") as f:
            json.dump({"old-proj": "legacy-thread-id-abc"}, f)

        sm = _make_session_mgr(tmp)
        assert sm.get("old-proj") == "legacy-thread-id-abc"
        print("✅ legacy migration OK\n")


def test_session_manager_prefix_ops():
    print("--- SessionManager.list_by_prefix / delete_by_prefix ---")
    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        sm.create_session("discord-100")
        sm.create_session("discord-100-dev")
        sm.create_session("discord-200")
        sm.create_session("cli-default")

        # list_by_prefix
        ch100 = sm.list_by_prefix("discord-100")
        assert set(ch100.keys()) == {"discord-100", "discord-100-dev"}

        all_discord = sm.list_by_prefix("discord-")
        assert len(all_discord) == 3

        cli = sm.list_by_prefix("cli-")
        assert set(cli.keys()) == {"cli-default"}

        empty = sm.list_by_prefix("nonexistent-")
        assert len(empty) == 0

        # delete_by_prefix
        deleted = sm.delete_by_prefix("discord-100")
        assert deleted == 2
        assert sm.get("discord-100") is None
        assert sm.get("discord-100-dev") is None
        assert sm.get("discord-200") is not None  # untouched
        assert sm.get("cli-default") is not None  # untouched

        # delete nonexistent prefix
        assert sm.delete_by_prefix("ghost-") == 0

        print("✅ list_by_prefix / delete_by_prefix OK\n")


# ─────────────────────────────────────────────
# 2. Per-channel helpers
# ─────────────────────────────────────────────

def test_per_channel_helpers():
    print("--- Per-channel helpers ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _setup_bot(sm)

        # _channel_prefix / _channel_default_session
        assert bot._channel_prefix(100) == "discord-100"
        assert bot._channel_default_session(100) == "discord-100"
        print("   prefix/default OK")

        # _ensure_channel_session — auto-creates
        name = bot._ensure_channel_session(100)
        assert name == "discord-100"
        assert sm.get("discord-100") is not None
        assert bot._channel_active_session[100] == "discord-100"
        print(f"   ensure created: {name}")

        # _ensure_channel_session — returns existing
        name2 = bot._ensure_channel_session(100)
        assert name2 == "discord-100"
        print("   ensure idempotent OK")

        # _get_channel_config
        config = bot._get_channel_config(100)
        env = sm.get_envelope("discord-100")
        assert config["configurable"]["thread_id"] == env.thread_id
        print(f"   config thread_id: {env.thread_id[:8]}")

        # _get_channel_workspace — empty by default
        ws = bot._get_channel_workspace(100)
        assert ws == ""
        print("   workspace empty by default OK")

        # _get_channel_workspace — after setting workspace
        env.workspace = "/test/path"
        sm._save()
        ws2 = bot._get_channel_workspace(100)
        assert ws2 == "/test/path"
        print("   workspace after set OK")

    print("✅ Per-channel helpers OK\n")


# ─────────────────────────────────────────────
# 4. Discord !new (per-channel)
# ─────────────────────────────────────────────

async def test_discord_new():
    print("--- Discord !new ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _setup_bot(sm)

        # Success — creates discord-100-proj-a
        ctx = _mock_ctx(channel_id=100)
        await bot.new_session_cmd(ctx, name="proj-a")
        msg = ctx.send.call_args[0][0]
        assert "proj-a" in msg
        assert sm.get("discord-100-proj-a") is not None
        assert bot._channel_active_session[100] == "discord-100-proj-a"
        print(f"   success: {msg[:70]}")

        # No name provided
        ctx2 = _mock_ctx(channel_id=100)
        await bot.new_session_cmd(ctx2, name="")
        msg2 = ctx2.send.call_args[0][0]
        assert "用法" in msg2
        print(f"   no name: {msg2}")

        # Duplicate
        ctx3 = _mock_ctx(channel_id=100)
        await bot.new_session_cmd(ctx3, name="proj-a")
        msg3 = ctx3.send.call_args[0][0]
        assert "❌" in msg3
        print(f"   duplicate: {msg3[:60]}")

        # Different channel can use same name
        ctx4 = _mock_ctx(channel_id=200)
        await bot.new_session_cmd(ctx4, name="proj-a")
        msg4 = ctx4.send.call_args[0][0]
        assert "proj-a" in msg4
        assert sm.get("discord-200-proj-a") is not None
        print(f"   same name diff channel: OK")

    print("✅ Discord !new OK\n")


# ─────────────────────────────────────────────
# 5. Discord !switch (per-channel)
# ─────────────────────────────────────────────

async def test_discord_switch():
    print("--- Discord !switch ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _setup_bot(sm)

        # Pre-create sessions
        sm.create_session("discord-100")
        sm.create_session("discord-100-dev")

        # Switch to named session
        ctx = _mock_ctx(channel_id=100)
        await bot.switch_session_cmd(ctx, name="dev")
        msg = ctx.send.call_args[0][0]
        assert "dev" in msg
        assert bot._channel_active_session[100] == "discord-100-dev"
        print(f"   switch to dev: {msg[:70]}")

        # Switch to default
        ctx2 = _mock_ctx(channel_id=100)
        await bot.switch_session_cmd(ctx2, name="default")
        msg2 = ctx2.send.call_args[0][0]
        assert bot._channel_active_session[100] == "discord-100"
        print(f"   switch to default: {msg2[:70]}")

        # No name
        ctx3 = _mock_ctx(channel_id=100)
        await bot.switch_session_cmd(ctx3, name="")
        msg3 = ctx3.send.call_args[0][0]
        assert "用法" in msg3
        print(f"   no name: {msg3}")

        # Nonexistent
        ctx4 = _mock_ctx(channel_id=100)
        await bot.switch_session_cmd(ctx4, name="ghost")
        msg4 = ctx4.send.call_args[0][0]
        assert "❌" in msg4
        print(f"   nonexistent: {msg4[:60]}")

    print("✅ Discord !switch OK\n")


# ─────────────────────────────────────────────
# 6. Discord !session / !sessions (per-channel)
# ─────────────────────────────────────────────

async def test_discord_session_info():
    print("--- Discord !session / !sessions ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _setup_bot(sm)

        # Ensure channel has sessions
        sm.create_session("discord-100")
        sm.create_session("discord-100-dev")
        bot._channel_active_session[100] = "discord-100"

        # !session
        ctx = _mock_ctx(channel_id=100)
        await bot.show_session(ctx)
        msg = ctx.send.call_args[0][0]
        assert "default" in msg  # discord-100 → display "default"
        print(f"   !session: {msg}")

        # Switch and check display name
        bot._channel_active_session[100] = "discord-100-dev"
        ctx1b = _mock_ctx(channel_id=100)
        await bot.show_session(ctx1b)
        msg1b = ctx1b.send.call_args[0][0]
        assert "dev" in msg1b
        print(f"   !session (dev): {msg1b}")

        # !sessions — shows only this channel's sessions, marks active with ◀
        bot._channel_active_session[100] = "discord-100-dev"
        ctx2 = _mock_ctx(channel_id=100)
        await bot.list_sessions_cmd(ctx2)
        msg2 = ctx2.send.call_args[0][0]
        assert "default" in msg2
        assert "dev" in msg2
        assert "◀" in msg2
        print(f"   !sessions:\n{msg2}")

        # !sessions — empty channel
        ctx3 = _mock_ctx(channel_id=999)
        await bot.list_sessions_cmd(ctx3)
        msg3 = ctx3.send.call_args[0][0]
        assert "没有" in msg3
        print(f"   !sessions (empty): {msg3}")

        # Other channel's sessions not shown
        sm.create_session("discord-200")
        ctx4 = _mock_ctx(channel_id=100)
        await bot.list_sessions_cmd(ctx4)
        msg4 = ctx4.send.call_args[0][0]
        assert "200" not in msg4
        print("   channel isolation OK")

    print("✅ Discord !session / !sessions OK\n")


# ─────────────────────────────────────────────
# 7. Discord !whoami / !debug
# ─────────────────────────────────────────────

async def test_discord_whoami_debug():
    print("--- Discord !whoami / !debug ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _, loader = _setup_bot(sm)

        ctx_ok = _mock_ctx(user_id=12345)
        await bot.whoami(ctx_ok)
        msg_ok = ctx_ok.send.call_args[0][0]
        assert "12345" in msg_ok
        print(f"   !whoami: {msg_ok}")

        iface_d = bot._DiscordInterface(loader)
        reply_d = await iface_d.handle_command("!debug", "")
        assert "Debug mode" in reply_d
        print(f"   !debug: {reply_d}")

    print("✅ Discord !whoami / !debug OK\n")


# ─────────────────────────────────────────────
# 8. Discord !tokens
# ─────────────────────────────────────────────

async def test_discord_tokens():
    print("--- Discord !tokens ---")
    import interfaces.discord_bot as bot
    import framework.token_tracker as tt
    from framework import token_display

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _, loader = _setup_bot(sm)

        # Inject known stats
        tt._token_stats.update({
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 100,
            "calls": 5,
        })

        iface = bot._DiscordInterface(loader)

        # 1) no-arg — cumulative stats + toggle state line
        reply = await iface.handle_command("!tokens", "")
        assert "1,000" in reply or "1000" in reply
        assert "500" in reply
        assert "内联显示" in reply
        print(f"   !tokens preview:\n{reply[:300]}")

        # 2) reset — preserved behavior
        reply2 = await iface.handle_command("!tokens", "reset")
        assert "重置" in reply2
        assert tt._token_stats["input_tokens"] == 0
        print(f"   !tokens reset: {reply2}")

        # 3) off — disable inline display
        reply3 = await iface.handle_command("!tokens", "off")
        assert "关闭" in reply3
        assert token_display.is_token_display_enabled() is False
        print(f"   !tokens off: {reply3}")

        # 4) status — report current state
        reply4 = await iface.handle_command("!tokens", "status")
        assert "关闭" in reply4
        print(f"   !tokens status (off): {reply4}")

        # 5) on — re-enable
        reply5 = await iface.handle_command("!tokens", "on")
        assert "开启" in reply5
        assert token_display.is_token_display_enabled() is True
        print(f"   !tokens on: {reply5}")

        # 6) unknown arg — usage hint
        reply6 = await iface.handle_command("!tokens", "gibberish")
        assert "用法" in reply6 or "on|off|status" in reply6
        print(f"   !tokens gibberish: {reply6}")

    # Restore default for test isolation
    token_display.set_token_display(True)
    print("✅ Discord !tokens OK\n")


# ─────────────────────────────────────────────
# 9. Discord !memory (per-channel)
# ─────────────────────────────────────────────

async def test_discord_memory():
    print("--- Discord !memory ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _, loader = _setup_bot(sm)

        # Ensure channel session exists
        sm.create_session("discord-100")
        bot._channel_active_session[100] = "discord-100"

        iface = bot._DiscordInterface(loader, channel_id=100)
        reply = await iface.handle_command("!memory", "")
        assert "discord-100" in reply  # session name
        assert "KB" in reply
        print(f"   !memory: {reply}")

    print("✅ Discord !memory OK\n")


# ─────────────────────────────────────────────
# 10. Discord !compact / !reset (per-channel)
# ─────────────────────────────────────────────

async def test_discord_compact_reset():
    print("--- Discord !compact / !reset ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _, loader = _setup_bot(sm)

        sm.create_session("discord-100")
        bot._channel_active_session[100] = "discord-100"

        iface = bot._DiscordInterface(loader, channel_id=100)

        # !compact (default keep=20) — now reports both checkpoint DB and Claude session
        reply = await iface.handle_command("!compact", "")
        assert "Compact" in reply
        assert "checkpoint DB" in reply
        assert "Claude session" in reply
        print(f"   !compact:\n{reply}")

        # !reset without confirm → shows warning
        reply2 = await iface.handle_command("!reset", "")
        assert "confirm" in reply2 or "确认" in reply2
        print(f"   !reset (no confirm): {reply2[:70]}")

        # !reset confirm → executes
        reply3 = await iface.handle_command("!reset", "confirm")
        assert "重置" in reply3 or "Session" in reply3
        print(f"   !reset confirm: {reply3[:70]}")

    print("✅ Discord !compact / !reset OK\n")


# ─────────────────────────────────────────────
# 11. Discord !setproject / !project (per-channel)
# ─────────────────────────────────────────────

async def test_discord_setproject_project():
    print("--- Discord !setproject / !project ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _, loader = _setup_bot(sm)

        # Ensure channel session
        sm.create_session("discord-100")
        bot._channel_active_session[100] = "discord-100"

        iface100 = bot._DiscordInterface(loader, channel_id=100)

        with tempfile.TemporaryDirectory() as real_dir:
            # Valid path
            reply = await iface100.handle_command("!setproject", real_dir)
            assert real_dir in reply
            print(f"   !setproject (valid): {reply}")

            # Workspace persisted in SessionEnvelope
            env = sm.get_envelope("discord-100")
            assert env.workspace == real_dir
            print(f"   workspace persisted: {env.workspace}")

            # !project shows it
            reply2 = await iface100.handle_command("!project", "")
            assert real_dir in reply2
            print(f"   !project: {reply2}")

            # Different channel unaffected
            sm.create_session("discord-200")
            bot._channel_active_session[200] = "discord-200"
            iface200 = bot._DiscordInterface(loader, channel_id=200)
            reply_other = await iface200.handle_command("!project", "")
            assert "未设置" in reply_other
            print(f"   other channel unaffected: {reply_other}")

        # Invalid path
        reply3 = await iface100.handle_command("!setproject", "/nonexistent/xyz/abc")
        assert "不存在" in reply3
        print(f"   !setproject (invalid): {reply3}")

        # No path → shows current
        reply4 = await iface100.handle_command("!setproject", "")
        assert "当前" in reply4
        print(f"   !setproject (no arg): {reply4}")

    print("✅ Discord !setproject / !project OK\n")


# ─────────────────────────────────────────────
# 12. Discord !stop
# ─────────────────────────────────────────────

async def test_discord_stop():
    print("--- Discord !stop ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _setup_bot(sm)

        # No running task
        ctx = _mock_ctx(channel_id=100)
        await bot.stop_task(ctx)
        msg = ctx.send.call_args[0][0]
        assert "没有" in msg
        print(f"   no task: {msg}")

        # With a running task
        async def _slow():
            await asyncio.sleep(999)

        task = asyncio.create_task(_slow())
        bot._channel_tasks[100] = task

        ctx2 = _mock_ctx(channel_id=100)
        await bot.stop_task(ctx2)
        msg2 = ctx2.send.call_args[0][0]
        assert "已停止" in msg2
        # Let event loop process the cancellation
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()
        print(f"   stopped: {msg2}")

        # Completed task → "no task"
        async def _fast():
            return "done"
        task2 = asyncio.create_task(_fast())
        await task2
        bot._channel_tasks[100] = task2

        ctx3 = _mock_ctx(channel_id=100)
        await bot.stop_task(ctx3)
        msg3 = ctx3.send.call_args[0][0]
        assert "没有" in msg3
        print(f"   completed task: {msg3}")

        bot._channel_tasks.clear()

    print("✅ Discord !stop OK\n")


# ─────────────────────────────────────────────
# 13. Discord !channels
# ─────────────────────────────────────────────

async def test_discord_channels():
    print("--- Discord !channels ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _setup_bot(sm)

        # Empty
        ctx = _mock_ctx(channel_id=100)
        await bot.list_channels_cmd(ctx)
        msg = ctx.send.call_args[0][0]
        assert "没有" in msg
        print(f"   empty: {msg}")

        # With sessions
        sm.create_session("discord-100")
        sm.create_session("discord-100-dev")
        sm.create_session("discord-200")
        bot._channel_active_session[100] = "discord-100-dev"
        bot._channel_active_session[200] = "discord-200"

        # Mock guild channels
        ch100 = MagicMock()
        ch100.name = "general"
        ch200 = MagicMock()
        ch200.name = "random"
        ctx2 = _mock_ctx(channel_id=100)
        ctx2.guild.get_channel = lambda cid: {100: ch100, 200: ch200}.get(cid)
        await bot.list_channels_cmd(ctx2)
        msg2 = ctx2.send.call_args[0][0]
        assert "#general" in msg2
        assert "#random" in msg2
        assert "2 sessions" in msg2  # channel 100 has 2
        assert "1 sessions" in msg2  # channel 200 has 1
        assert "◀" in msg2
        print(f"   with sessions:\n{msg2}")

        # Orphan channel (not in guild)
        sm.create_session("discord-999")
        ctx3 = _mock_ctx(channel_id=100)
        ctx3.guild.get_channel = lambda cid: {100: ch100, 200: ch200}.get(cid)
        await bot.list_channels_cmd(ctx3)
        msg3 = ctx3.send.call_args[0][0]
        assert "orphan" in msg3
        print("   orphan detection OK")

    print("✅ Discord !channels OK\n")


# ─────────────────────────────────────────────
# 14. _cleanup_channel
# ─────────────────────────────────────────────

async def test_cleanup_channel():
    print("--- _cleanup_channel ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _setup_bot(sm)

        sm.create_session("discord-100")
        sm.create_session("discord-100-dev")
        sm.create_session("discord-200")
        bot._channel_active_session[100] = "discord-100"
        bot._channel_active_session[200] = "discord-200"
        deleted = await bot._cleanup_channel(100)
        assert deleted == 2
        assert sm.get("discord-100") is None
        assert sm.get("discord-100-dev") is None
        assert sm.get("discord-200") is not None  # untouched
        assert 100 not in bot._channel_active_session
        assert 200 in bot._channel_active_session  # untouched
        print(f"   cleaned up {deleted} sessions, channel 200 untouched")

        # Cleanup non-existent channel
        deleted2 = await bot._cleanup_channel(999)
        assert deleted2 == 0
        print("   non-existent channel: 0 deleted")

    print("✅ _cleanup_channel OK\n")


# ─────────────────────────────────────────────
# 15. Discord !help
# ─────────────────────────────────────────────

async def test_discord_help():
    print("--- Discord !help ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _, loader = _setup_bot(sm)

        iface = bot._DiscordInterface(loader)
        reply = await iface.handle_command("!help", "")
        for keyword in ("!new", "!switch", "!sessions", "!session",
                        "!tokens", "!memory", "!compact", "!reset",
                        "!setproject", "!whoami", "!stop", "!channels"):
            assert keyword in reply, f"!help missing keyword: {keyword}"
        print(f"   !help length: {len(reply)} chars — all keywords present")

    print("✅ Discord !help OK\n")


# ─────────────────────────────────────────────
# 16. Auth whitelist
# ─────────────────────────────────────────────

def test_auth_whitelist():
    print("--- Authorization whitelist ---")
    import interfaces.discord_bot as bot

    # With allowed_users set via loader
    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _, loader = _setup_bot(sm)

        # Set whitelist via config
        loader.load_config.return_value.discord_allowed_users = ["111", "222"]
        u_ok = MagicMock(); u_ok.id = 111
        u_bad = MagicMock(); u_bad.id = 999
        assert bot._is_authorized(u_ok)
        assert not bot._is_authorized(u_bad)
        print("   whitelist enforced OK")

        # Empty whitelist → allow all
        loader.load_config.return_value.discord_allowed_users = []
        assert bot._is_authorized(u_bad)
        print("   empty whitelist → allow all OK")

        # No loader → allow all
        bot._loader = None
        assert bot._is_authorized(u_bad)
        print("   no loader → allow all OK")

    print("✅ Auth whitelist OK\n")


# ─────────────────────────────────────────────
# 17. split_fence_aware (Discord message chunking)
# ─────────────────────────────────────────────

def test_split_fence_aware():
    print("--- split_fence_aware ---")
    from interfaces.discord_bot import split_fence_aware

    # Short text → 1 chunk
    chunks = split_fence_aware("hello world", max_chars=1900)
    assert chunks == ["hello world"]

    # Long plain text → split
    long_text = "x" * 3000
    chunks = split_fence_aware(long_text, max_chars=1900)
    assert len(chunks) == 2
    assert all(len(c) <= 1950 for c in chunks)  # slight slack for fence closing

    # Code fence → each non-final chunk closes its fence
    code_text = "intro\n```python\n" + "line\n" * 500 + "```\noutro"
    chunks = split_fence_aware(code_text, max_chars=1900)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks[:-1]):
        # Any chunk that opened a fence should close it
        assert c.count("```") % 2 == 0 or c.endswith("```"), (
            f"Chunk {i} has unclosed fence: {c[-50:]!r}"
        )
    print(f"   code fence: {len(chunks)} chunks, all fences balanced")
    print("✅ split_fence_aware OK\n")


# ─────────────────────────────────────────────
# 18. Restart recovery (on_ready session restore logic)
# ─────────────────────────────────────────────

def test_restart_recovery():
    print("--- Restart recovery ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        _setup_bot(sm)

        # Simulate sessions from a previous run
        env1 = sm.create_session("discord-100")
        env2 = sm.create_session("discord-100-dev")
        env3 = sm.create_session("discord-200")
        # Make "dev" the most recently updated for channel 100
        env2.updated_at = "2099-01-01T00:00:00+00:00"
        sm._save()

        # Clear active sessions (simulating restart)
        bot._channel_active_session.clear()

        # Run the recovery logic (extracted from on_ready)
        best = {}
        for sname, env in sm.list_by_prefix("discord-").items():
            parts = sname.split("-", 2)
            if len(parts) < 2:
                continue
            try:
                cid = int(parts[1])
            except ValueError:
                continue
            if cid not in best or env.updated_at > best[cid][1]:
                best[cid] = (sname, env.updated_at)
        for cid, (sname, _) in best.items():
            bot._channel_active_session[cid] = sname

        # Verify
        assert bot._channel_active_session[100] == "discord-100-dev"  # most recent
        assert bot._channel_active_session[200] == "discord-200"
        print(f"   channel 100 → discord-100-dev (most recent)")
        print(f"   channel 200 → discord-200")

    print("✅ Restart recovery OK\n")


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Sync tests
    test_session_manager_create()
    test_session_manager_list_find()
    test_session_manager_persistence()
    test_session_manager_legacy_migration()
    test_session_manager_prefix_ops()
    test_per_channel_helpers()
    test_auth_whitelist()
    test_split_fence_aware()
    test_restart_recovery()

    # Async tests
    async def _run():
        await test_discord_new()
        await test_discord_switch()
        await test_discord_session_info()
        await test_discord_whoami_debug()
        await test_discord_tokens()
        await test_discord_memory()
        await test_discord_compact_reset()
        await test_discord_setproject_project()
        await test_discord_stop()
        await test_discord_channels()
        await test_cleanup_channel()
        await test_discord_help()

    asyncio.run(_run())
    print("🎉 全部命令测试通过")
