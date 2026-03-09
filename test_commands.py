"""
用户命令全覆盖测试

覆盖所有暴露给用户的命令：
  CLI  : !new, !switch, !sessions, !session
  Discord: !help, !new, !switch, !session, !sessions, !whoami,
           !debug, !memory, !compact, !reset, !clear, !tokens,
           !setproject, !project
  工具函数: split_fence_aware（消息分块），auth whitelist

不做实际 API 调用，使用 tempfile + unittest.mock。
运行：python3 test_commands.py
"""

import asyncio
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


def _mock_ctx(user_id: int = 12345) -> MagicMock:
    ctx = MagicMock()
    ctx.author = MagicMock()
    ctx.author.id = user_id
    ctx.send = AsyncMock()
    return ctx


def _mock_loader(sm, name: str = "hani") -> MagicMock:
    loader = MagicMock()
    loader.name = name
    loader.session_mgr = sm
    loader.get_engine = AsyncMock(return_value=MagicMock())
    loader.invalidate_engine = MagicMock()
    return loader


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
    import json
    with tempfile.TemporaryDirectory() as tmp:
        sf = os.path.join(tmp, "sessions.json")
        with open(sf, "w") as f:
            json.dump({"old-proj": "legacy-thread-id-abc"}, f)

        sm = _make_session_mgr(tmp)
        assert sm.get("old-proj") == "legacy-thread-id-abc"
        print("✅ legacy migration OK\n")


# ─────────────────────────────────────────────
# 2. framework.graph session functions
# ─────────────────────────────────────────────

async def test_graph_new_and_switch():
    print("--- framework.graph new_session / switch_session / get_config ---")
    import framework.graph as fg
    fg._active = None

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)

        # new_session creates and activates
        tid_a = await fg.new_session("alpha", sm)
        assert fg._active.name == "alpha"
        assert fg.get_config()["configurable"]["thread_id"] == tid_a
        print(f"   new alpha: {tid_a[:8]}")

        await fg.new_session("beta", sm)
        assert fg._active.name == "beta"

        # switch_session goes back
        switched = await fg.switch_session("alpha", sm)
        assert switched == tid_a
        assert fg._active.name == "alpha"
        print(f"   switched back to alpha: {switched[:8]}")

        # switch to nonexistent raises
        try:
            await fg.switch_session("nonexistent", sm)
            assert False, "should raise"
        except ValueError:
            pass

    fg._active = None
    print("✅ framework.graph session functions OK\n")


# ─────────────────────────────────────────────
# 3. Discord !new / !switch
# ─────────────────────────────────────────────

async def test_discord_new():
    print("--- Discord !new ---")
    import interfaces.discord_bot as bot
    import framework.graph as fg
    fg._active = None

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        bot._loader = _mock_loader(sm)
        bot._session_mgr = sm

        # Success
        ctx = _mock_ctx()
        await bot.new_session_cmd(ctx, name="proj-a")
        msg = ctx.send.call_args[0][0]
        assert "✅" in msg and "proj-a" in msg
        print(f"   success: {msg[:70]}")

        # No name provided
        ctx2 = _mock_ctx()
        await bot.new_session_cmd(ctx2, name="")
        msg2 = ctx2.send.call_args[0][0]
        assert "用法" in msg2
        print(f"   no name: {msg2}")

        # Duplicate
        ctx3 = _mock_ctx()
        await bot.new_session_cmd(ctx3, name="proj-a")
        msg3 = ctx3.send.call_args[0][0]
        assert "❌" in msg3
        print(f"   duplicate: {msg3[:60]}")

    fg._active = None
    print("✅ Discord !new OK\n")


async def test_discord_switch():
    print("--- Discord !switch ---")
    import interfaces.discord_bot as bot
    import framework.graph as fg
    fg._active = None

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        sm.create_session("my-proj")
        bot._loader = _mock_loader(sm)
        bot._session_mgr = sm

        # Successful switch
        ctx = _mock_ctx()
        await bot.switch_session_cmd(ctx, name="my-proj")
        msg = ctx.send.call_args[0][0]
        assert "✅" in msg and "my-proj" in msg
        print(f"   success: {msg[:70]}")

        # No name
        ctx2 = _mock_ctx()
        await bot.switch_session_cmd(ctx2, name="")
        msg2 = ctx2.send.call_args[0][0]
        assert "用法" in msg2
        print(f"   no name: {msg2}")

        # Nonexistent
        ctx3 = _mock_ctx()
        await bot.switch_session_cmd(ctx3, name="ghost")
        msg3 = ctx3.send.call_args[0][0]
        assert "❌" in msg3
        print(f"   nonexistent: {msg3[:60]}")

    fg._active = None
    print("✅ Discord !switch OK\n")


# ─────────────────────────────────────────────
# 4. Discord !session / !sessions
# ─────────────────────────────────────────────

async def test_discord_session_info():
    print("--- Discord !session / !sessions ---")
    import interfaces.discord_bot as bot
    import framework.graph as fg
    fg._active = None

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        env_a = sm.create_session("work")
        sm.create_session("archive")
        bot._loader = _mock_loader(sm)
        bot._session_mgr = sm
        _set_active("work", env_a.thread_id)

        # !session
        ctx = _mock_ctx()
        await bot.show_session(ctx)
        msg = ctx.send.call_args[0][0]
        assert "work" in msg
        print(f"   !session: {msg}")

        # !sessions — current marked with ◀
        ctx2 = _mock_ctx()
        await bot.list_sessions_cmd(ctx2)
        msg2 = ctx2.send.call_args[0][0]
        assert "work" in msg2 and "archive" in msg2
        assert "◀" in msg2
        print(f"   !sessions:\n{msg2}")

        # !sessions — empty
        with tempfile.TemporaryDirectory() as tmp2:
            sm_empty = _make_session_mgr(tmp2)
            bot._session_mgr = sm_empty
            ctx3 = _mock_ctx()
            await bot.list_sessions_cmd(ctx3)
            msg3 = ctx3.send.call_args[0][0]
            assert "!new" in msg3
            print(f"   !sessions (empty): {msg3}")

    fg._active = None
    print("✅ Discord !session / !sessions OK\n")


# ─────────────────────────────────────────────
# 5. Discord !whoami / !debug
# ─────────────────────────────────────────────

async def test_discord_whoami_debug():
    print("--- Discord !whoami / !debug ---")
    import interfaces.discord_bot as bot

    bot._ALLOWED_USERS = {12345}

    ctx_ok = _mock_ctx(user_id=12345)
    await bot.whoami(ctx_ok)
    msg_ok = ctx_ok.send.call_args[0][0]
    assert "12345" in msg_ok and "已授权" in msg_ok
    print(f"   !whoami (authorized): {msg_ok}")

    ctx_bad = _mock_ctx(user_id=99999)
    await bot.whoami(ctx_bad)
    msg_bad = ctx_bad.send.call_args[0][0]
    assert "99999" in msg_bad and "未授权" in msg_bad
    print(f"   !whoami (unauthorized): {msg_bad}")

    ctx_d = _mock_ctx()
    await bot.toggle_debug(ctx_d)
    msg_d = ctx_d.send.call_args[0][0]
    assert "Debug mode" in msg_d
    print(f"   !debug: {msg_d}")

    bot._ALLOWED_USERS = set()
    print("✅ Discord !whoami / !debug OK\n")


# ─────────────────────────────────────────────
# 6. Discord !tokens
# ─────────────────────────────────────────────

async def test_discord_tokens():
    print("--- Discord !tokens ---")
    import interfaces.discord_bot as bot
    import framework.token_tracker as tt

    # Inject known stats
    tt._token_stats.update({
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 100,
        "calls": 5,
    })

    ctx = _mock_ctx()
    await bot.show_tokens(ctx)
    msg = ctx.send.call_args[0][0]
    assert "1,000" in msg or "1000" in msg
    assert "500" in msg
    assert "调用次数" in msg
    print(f"   !tokens preview:\n{msg[:300]}")

    ctx2 = _mock_ctx()
    await bot.show_tokens(ctx2, reset="reset")
    msg2 = ctx2.send.call_args[0][0]
    assert "重置" in msg2
    assert tt._token_stats["input_tokens"] == 0
    print(f"   !tokens reset: {msg2}")

    print("✅ Discord !tokens OK\n")


# ─────────────────────────────────────────────
# 7. Discord !memory
# ─────────────────────────────────────────────

async def test_discord_memory():
    print("--- Discord !memory ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        env = sm.create_session("mywork")
        bot._session_mgr = sm
        _set_active("mywork", env.thread_id)

        ctx = _mock_ctx()
        await bot.show_memory(ctx)
        msg = ctx.send.call_args[0][0]
        assert "mywork" in msg
        assert "KB" in msg
        print(f"   !memory: {msg}")

    import framework.graph as fg
    fg._active = None
    print("✅ Discord !memory OK\n")


# ─────────────────────────────────────────────
# 8. Discord !compact / !reset / !clear
# ─────────────────────────────────────────────

async def test_discord_compact_reset_clear():
    print("--- Discord !compact / !reset / !clear ---")
    import interfaces.discord_bot as bot

    with tempfile.TemporaryDirectory() as tmp:
        sm = _make_session_mgr(tmp)
        env = sm.create_session("test-sess")
        bot._loader = _mock_loader(sm)
        bot._session_mgr = sm
        _set_active("test-sess", env.thread_id)

        # !compact (default keep=20)
        ctx = _mock_ctx()
        await bot.compact_session(ctx)
        msg = ctx.send.call_args[0][0]
        assert "Compact" in msg
        print(f"   !compact: {msg}")

        # !reset without confirm → shows warning
        ctx2 = _mock_ctx()
        await bot.reset_session(ctx2)
        msg2 = ctx2.send.call_args[0][0]
        assert "confirm" in msg2 or "确认" in msg2
        print(f"   !reset (no confirm): {msg2[:70]}")

        # !reset confirm → executes
        ctx3 = _mock_ctx()
        await bot.reset_session(ctx3, confirm="confirm")
        msg3 = ctx3.send.call_args[0][0]
        assert "重置" in msg3 or "Session" in msg3
        print(f"   !reset confirm: {msg3[:70]}")

        # !clear → immediate reset (no confirm needed)
        ctx4 = _mock_ctx()
        await bot.clear_session(ctx4)
        msg4 = ctx4.send.call_args[0][0]
        assert "清空" in msg4 or "Session" in msg4
        print(f"   !clear: {msg4[:70]}")

    import framework.graph as fg
    fg._active = None
    print("✅ Discord !compact / !reset / !clear OK\n")


# ─────────────────────────────────────────────
# 9. Discord !setproject / !project
# ─────────────────────────────────────────────

async def test_discord_setproject_project():
    print("--- Discord !setproject / !project ---")
    import interfaces.discord_bot as bot
    bot._ALLOWED_USERS = set()

    with tempfile.TemporaryDirectory() as real_dir:
        # Valid path
        ctx = _mock_ctx()
        await bot.set_project(ctx, path=real_dir)
        msg = ctx.send.call_args[0][0]
        assert real_dir in msg
        print(f"   !setproject (valid): {msg}")

        # !project shows it
        ctx2 = _mock_ctx()
        await bot.show_project(ctx2)
        msg2 = ctx2.send.call_args[0][0]
        assert real_dir in msg2
        print(f"   !project: {msg2}")

    # Invalid path
    ctx3 = _mock_ctx()
    await bot.set_project(ctx3, path="/nonexistent/xyz/abc")
    msg3 = ctx3.send.call_args[0][0]
    assert "不存在" in msg3
    print(f"   !setproject (invalid): {msg3}")

    # No path → shows current
    ctx4 = _mock_ctx()
    await bot.set_project(ctx4, path="")
    msg4 = ctx4.send.call_args[0][0]
    assert "当前" in msg4 or "未设置" in msg4 or "项目" in msg4
    print(f"   !setproject (no arg): {msg4}")

    print("✅ Discord !setproject / !project OK\n")


# ─────────────────────────────────────────────
# 10. Discord !help
# ─────────────────────────────────────────────

async def test_discord_help():
    print("--- Discord !help ---")
    import interfaces.discord_bot as bot
    bot._ALLOWED_USERS = set()

    ctx = _mock_ctx()
    await bot.show_help(ctx)
    msg = ctx.send.call_args[0][0]
    for keyword in ("!new", "!switch", "!sessions", "!session",
                    "!tokens", "!memory", "!compact", "!reset",
                    "!setproject", "!whoami"):
        assert keyword in msg, f"!help missing keyword: {keyword}"
    print(f"   !help length: {len(msg)} chars — all keywords present")
    print("✅ Discord !help OK\n")


# ─────────────────────────────────────────────
# 11. Auth whitelist
# ─────────────────────────────────────────────

def test_auth_whitelist():
    print("--- Authorization whitelist ---")
    import interfaces.discord_bot as bot

    bot._ALLOWED_USERS = {111, 222}
    u_ok = MagicMock(); u_ok.id = 111
    u_bad = MagicMock(); u_bad.id = 999
    assert bot._is_authorized(u_ok)
    assert not bot._is_authorized(u_bad)

    # Empty whitelist → allow all
    bot._ALLOWED_USERS = set()
    assert bot._is_authorized(u_bad)

    bot._ALLOWED_USERS = set()
    print("✅ Auth whitelist OK\n")


# ─────────────────────────────────────────────
# 12. split_fence_aware (Discord message chunking)
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
# Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Sync tests
    test_session_manager_create()
    test_session_manager_list_find()
    test_session_manager_persistence()
    test_session_manager_legacy_migration()
    test_auth_whitelist()
    test_split_fence_aware()

    # Async tests
    async def _run():
        await test_graph_new_and_switch()
        await test_discord_new()
        await test_discord_switch()
        await test_discord_session_info()
        await test_discord_whoami_debug()
        await test_discord_tokens()
        await test_discord_memory()
        await test_discord_compact_reset_clear()
        await test_discord_setproject_project()
        await test_discord_help()

    asyncio.run(_run())
    print("🎉 全部命令测试通过")
