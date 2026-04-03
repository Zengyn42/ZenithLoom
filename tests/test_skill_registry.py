"""
SkillRegistry 单元测试 — tests/test_skill_registry.py

覆盖：
  - 单例行为
  - inline register + load
  - load 不存在的 skill 名（警告跳过，不抛异常）
  - load 多个 skill 用 --- 分隔
  - list_available 返回可发现的名称
  - reload 后重新扫描（用 tmp_path fixture 模拟 skills/ 目录）
  - command 类型：mock subprocess，验证缓存和 TTL
"""

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── 工具函数：创建干净的 SkillRegistry 实例（绕过单例）──────────────────────

def _make_registry():
    """创建一个新的 SkillRegistry 实例，不影响全局单例。"""
    from framework.skill_registry import SkillRegistry
    inst = SkillRegistry.__new__(SkillRegistry)
    inst._lock = threading.Lock()
    inst._static_cache = {}
    inst._command_configs = {}
    inst._command_cache = {}
    inst._discovered_names = set()
    return inst


# ── 单例行为 ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_instance_returns_same_object(self):
        """两次调用 get_instance() 返回同一个对象。"""
        from framework.skill_registry import SkillRegistry
        a = SkillRegistry.get_instance()
        b = SkillRegistry.get_instance()
        assert a is b

    def test_get_instance_is_skillregistry(self):
        """返回值是 SkillRegistry 实例。"""
        from framework.skill_registry import SkillRegistry
        assert isinstance(SkillRegistry.get_instance(), SkillRegistry)

    def test_concurrent_get_instance(self):
        """并发调用 get_instance() 仍返回同一对象（线程安全单例）。"""
        from framework.skill_registry import SkillRegistry
        results = []

        def _get():
            results.append(SkillRegistry.get_instance())

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = results[0]
        assert all(r is first for r in results), "并发得到了不同实例"


# ── inline register + load ───────────────────────────────────────────────────

class TestInlineRegister:
    def test_register_then_load(self):
        """inline 注册后可以 load 到内容。"""
        reg = _make_registry()
        reg.register("my-skill", "## My Skill\nDo something.")
        result = reg.load(["my-skill"])
        assert "My Skill" in result

    def test_register_appears_in_list_available(self):
        """inline 注册的 skill 出现在 list_available() 中。"""
        reg = _make_registry()
        reg.register("inline-test", "content")
        assert "inline-test" in reg.list_available()

    def test_register_overwrite(self):
        """同名 inline 注册覆盖旧内容。"""
        reg = _make_registry()
        reg.register("foo", "old content")
        reg.register("foo", "new content")
        assert reg.load(["foo"]) == "new content"

    def test_load_strips_content(self):
        """load 结果是 strip 后的内容。"""
        reg = _make_registry()
        reg.register("padded", "  hello world  ")
        assert reg.load(["padded"]) == "hello world"


# ── load 不存在的 skill 名（警告跳过） ────────────────────────────────────────

class TestLoadMissing:
    def test_missing_skill_returns_empty_string(self):
        """不存在的 skill 返回空字符串，不抛异常。"""
        reg = _make_registry()
        result = reg.load(["nonexistent-skill-xyz"])
        assert result == ""

    def test_missing_skill_warns(self, caplog):
        """不存在的 skill 发出 WARNING 日志。"""
        import logging
        reg = _make_registry()
        with caplog.at_level(logging.WARNING, logger="framework.skill_registry"):
            reg.load(["no-such-skill"])
        assert any("no-such-skill" in msg for msg in caplog.messages)

    def test_mixed_existing_and_missing(self):
        """已有和不存在的 skill 混合时，只返回已有的内容。"""
        reg = _make_registry()
        reg.register("exists", "I exist")
        result = reg.load(["exists", "missing-xyz"])
        assert "I exist" in result
        assert "missing" not in result


# ── load 多个 skill 用 --- 分隔 ───────────────────────────────────────────────

class TestLoadMultiple:
    def test_two_skills_separated_by_divider(self):
        """两个 skill 用 \\n\\n---\\n\\n 分隔。"""
        reg = _make_registry()
        reg.register("alpha", "Alpha content")
        reg.register("beta", "Beta content")
        result = reg.load(["alpha", "beta"])
        assert "\n\n---\n\n" in result
        assert "Alpha content" in result
        assert "Beta content" in result

    def test_three_skills_two_dividers(self):
        """三个 skill 有两个分隔符。"""
        reg = _make_registry()
        reg.register("a", "A")
        reg.register("b", "B")
        reg.register("c", "C")
        result = reg.load(["a", "b", "c"])
        assert result.count("---") == 2

    def test_empty_skill_content_skipped(self):
        """内容为空的 skill 不加入拼接（避免多余分隔符）。"""
        reg = _make_registry()
        reg.register("real", "Real content")
        reg.register("empty", "   ")  # 全空白
        result = reg.load(["real", "empty"])
        # 只有一个 skill 有内容，不应出现分隔符
        assert "---" not in result
        assert "Real content" in result

    def test_single_skill_no_divider(self):
        """单个 skill 结果中不含分隔符。"""
        reg = _make_registry()
        reg.register("solo", "Solo content")
        result = reg.load(["solo"])
        assert "---" not in result


# ── list_available ───────────────────────────────────────────────────────────

class TestListAvailable:
    def test_empty_registry_returns_list(self):
        """空 registry 返回空列表（不报错）。"""
        reg = _make_registry()
        result = reg.list_available()
        assert isinstance(result, list)

    def test_inline_registered_appears(self):
        """inline 注册的 skill 出现在列表中。"""
        reg = _make_registry()
        reg.register("skill-a", "content-a")
        reg.register("skill-b", "content-b")
        available = reg.list_available()
        assert "skill-a" in available
        assert "skill-b" in available

    def test_returns_sorted_list(self):
        """返回的列表已排序。"""
        reg = _make_registry()
        reg.register("zzz", "c")
        reg.register("aaa", "a")
        reg.register("mmm", "b")
        result = reg.list_available()
        assert result == sorted(result)


# ── reload 后重新扫描（用 tmp_path 模拟 skills/ 目录） ────────────────────────

class TestReload:
    def test_reload_picks_up_new_static_file(self, tmp_path, monkeypatch):
        """reload 后能发现新增的 .md 文件。"""
        # 临时 skills/ 目录
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        from framework import skill_registry as sr_module
        # 将 _PROJECT_ROOT 指向 tmp_path
        monkeypatch.setattr(sr_module, "_PROJECT_ROOT", tmp_path)

        reg = _make_registry()
        reg._scan()  # 初始扫描（skills/ 为空）
        assert "new-skill" not in reg.list_available()

        # 新增文件
        (skills_dir / "new-skill.md").write_text("New skill content", encoding="utf-8")
        reg.reload()

        assert "new-skill" in reg.list_available()

    def test_reload_clears_static_cache(self, tmp_path, monkeypatch):
        """reload 后 static_cache 被清除，重新从文件读取。"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "my-skill.md").write_text("Original", encoding="utf-8")

        from framework import skill_registry as sr_module
        monkeypatch.setattr(sr_module, "_PROJECT_ROOT", tmp_path)

        reg = _make_registry()
        reg._scan()

        # 加载一次，写入缓存
        content_before = reg.load(["my-skill"])
        assert "Original" in content_before

        # 修改文件内容
        (skills_dir / "my-skill.md").write_text("Updated", encoding="utf-8")

        # reload 前缓存仍是旧内容
        assert "Original" in reg._static_cache.get("my-skill", "")

        reg.reload()
        content_after = reg.load(["my-skill"])
        assert "Updated" in content_after

    def test_subdir_skill_discovered(self, tmp_path, monkeypatch):
        """子目录形式 skills/foo/SKILL.md 也能被扫描到。"""
        skills_dir = tmp_path / "skills"
        foo_dir = skills_dir / "foo"
        foo_dir.mkdir(parents=True)
        (foo_dir / "SKILL.md").write_text("Foo skill via subdir", encoding="utf-8")

        from framework import skill_registry as sr_module
        monkeypatch.setattr(sr_module, "_PROJECT_ROOT", tmp_path)

        reg = _make_registry()
        reg._scan()

        assert "foo" in reg.list_available()
        content = reg.load(["foo"])
        assert "Foo skill via subdir" in content

    def test_flat_file_priority_over_subdir(self, tmp_path, monkeypatch):
        """平铺 .md 文件优先于子目录 SKILL.md（同名时以平铺文件内容为准）。"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # 平铺文件
        (skills_dir / "bar.md").write_text("Flat bar content", encoding="utf-8")
        # 子目录文件（同名）
        bar_dir = skills_dir / "bar"
        bar_dir.mkdir()
        (bar_dir / "SKILL.md").write_text("Subdir bar content", encoding="utf-8")

        from framework import skill_registry as sr_module
        monkeypatch.setattr(sr_module, "_PROJECT_ROOT", tmp_path)

        reg = _make_registry()
        reg._scan()
        content = reg.load(["bar"])
        # 平铺文件优先（_load_static 先检查 foo.md）
        assert "Flat bar content" in content


# ── command 类型：mock subprocess，验证缓存和 TTL ────────────────────────────

class TestCommandSkill:
    def _make_command_registry(self, command: str, ttl: int = 3600) -> object:
        """创建含 command skill 配置的 registry 实例。"""
        reg = _make_registry()
        reg._command_configs["my-cmd"] = {"type": "command", "command": command, "ttl": ttl}
        reg._discovered_names.add("my-cmd")
        return reg

    def test_command_output_returned(self):
        """command skill 运行命令并返回 stdout。"""
        reg = self._make_command_registry("echo hello")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="hello world\n", returncode=0, stderr="")
            result = reg.load(["my-cmd"])
        assert "hello world" in result

    def test_command_cached_within_ttl(self):
        """TTL 内第二次调用不重新运行命令（缓存命中）。"""
        reg = self._make_command_registry("echo cached", ttl=60)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="cached output\n", returncode=0, stderr="")
            # 第一次调用
            reg.load(["my-cmd"])
            # 第二次调用（TTL 未过期）
            reg.load(["my-cmd"])
        # subprocess.run 只被调用一次
        assert mock_run.call_count == 1

    def test_command_refreshed_after_ttl(self):
        """TTL 过期后重新运行命令。"""
        reg = self._make_command_registry("echo refreshed", ttl=1)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="output\n", returncode=0, stderr="")
            reg.load(["my-cmd"])

            # 手动将缓存时间设置到过去（过期）
            old_ts = time.monotonic() - 5  # 5 秒前，已超出 TTL=1
            reg._command_cache["my-cmd"] = ("output", old_ts)

            reg.load(["my-cmd"])

        # 应该调用了两次
        assert mock_run.call_count == 2

    def test_command_nonzero_exit_warns(self, caplog):
        """命令非零退出时发出警告。"""
        import logging
        reg = self._make_command_registry("false-cmd", ttl=3600)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=1, stderr="error msg")
            with caplog.at_level(logging.WARNING, logger="framework.skill_registry"):
                reg.load(["my-cmd"])
        assert any("my-cmd" in msg for msg in caplog.messages)

    def test_command_timeout_returns_missing(self, caplog):
        """命令超时时返回空（skill 视为不存在），发出警告。"""
        import logging
        import subprocess
        reg = self._make_command_registry("slow-cmd", ttl=3600)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("slow-cmd", 30)
            with caplog.at_level(logging.WARNING, logger="framework.skill_registry"):
                result = reg.load(["my-cmd"])
        assert result == ""

    def test_command_discovered_via_skill_json(self, tmp_path, monkeypatch):
        """skills/foo.skill.json 被扫描为 command 类型。"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        cfg = {"type": "command", "command": "foo-cli --help", "ttl": 7200}
        (skills_dir / "foo.skill.json").write_text(json.dumps(cfg), encoding="utf-8")

        from framework import skill_registry as sr_module
        monkeypatch.setattr(sr_module, "_PROJECT_ROOT", tmp_path)

        reg = _make_registry()
        reg._scan()

        assert "foo" in reg.list_available()
        assert "foo" in reg._command_configs
        assert reg._command_configs["foo"]["command"] == "foo-cli --help"

    def test_skill_json_command_priority_over_md(self, tmp_path, monkeypatch):
        """.skill.json 中 type=command 优先于同名 .md 文件。"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # 同名的 .md 文件
        (skills_dir / "tool.md").write_text("Static content", encoding="utf-8")
        # .skill.json 声明 command 类型
        cfg = {"type": "command", "command": "tool --help", "ttl": 100}
        (skills_dir / "tool.skill.json").write_text(json.dumps(cfg), encoding="utf-8")

        from framework import skill_registry as sr_module
        monkeypatch.setattr(sr_module, "_PROJECT_ROOT", tmp_path)

        reg = _make_registry()
        reg._scan()

        # 应该是 command 类型
        assert "tool" in reg._command_configs


# ── LlmNode 集成：_skill_names 解析 ─────────────────────────────────────────

def _make_llm_node(node_config: dict):
    """创建 LlmNode 具体子类实例（LlmNode 是抽象类，需要子类化）。"""
    from framework.config import AgentConfig
    from framework.nodes.llm.llm_node import LlmNode

    class _ConcreteNode(LlmNode):
        """测试用具体子类，实现 call_llm 抽象方法。"""
        async def call_llm(self, prompt, session_id="", tools=None, cwd=None, history=None):
            return ("", "")

    config = AgentConfig.__new__(AgentConfig)
    config.tools = []
    config.permission_mode = "default"

    return _ConcreteNode(config, node_config)


class TestLlmNodeSkillsField:
    def test_skill_names_parsed_from_node_config(self):
        """LlmNode.__init__ 正确解析 node_config['skills']。"""
        node = _make_llm_node({
            "id": "test_node",
            "type": "GEMINI_CLI",
            "skills": ["blender-index", "gws-gam"],
        })
        assert node._skill_names == ["blender-index", "gws-gam"]

    def test_skill_names_default_empty_list(self):
        """未声明 skills 时 _skill_names 默认为空列表。"""
        node = _make_llm_node({"id": "test_node", "type": "CLAUDE_CLI"})
        assert node._skill_names == []

    def test_load_skill_content_calls_skill_registry(self):
        """_load_skill_content() 在有 _skill_names 时调用 SkillRegistry.load()。"""
        from framework.skill_registry import SkillRegistry

        node = _make_llm_node({
            "id": "test_node",
            "type": "GEMINI_CLI",
            "skills": ["my-tool"],
        })

        # mock SkillRegistry.get_instance().load()
        mock_registry = MagicMock()
        mock_registry.load.return_value = "## My Tool\nUsage: my-tool [options]"

        with patch.object(SkillRegistry, "get_instance", return_value=mock_registry):
            result = node._load_skill_content()

        mock_registry.load.assert_called_once_with(["my-tool"])
        assert "My Tool" in result
