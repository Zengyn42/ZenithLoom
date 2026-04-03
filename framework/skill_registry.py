"""
SkillRegistry — framework/skill_registry.py

为 LlmNode 基类提供声明式工具技能注入机制。
支持三种 skill 类型：static（读 .md 文件）、command（运行命令缓存输出）、inline（代码注册，测试用）。

目录扫描规则：
  skills/foo.md        → skill 名 "foo"（平铺文件）
  skills/foo/SKILL.md  → skill 名 "foo"（子目录形式，兼容 CLI-Anything 输出）
  平铺文件优先；同名时 .skill.json 中 type=command 最优先。

线程安全：使用 threading.Lock 保护缓存，兼容 asyncio + threading 混合环境。
单例：通过 SkillRegistry.get_instance() 获取全局实例。
"""

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 项目根目录：framework/skill_registry.py 向上两级
_PROJECT_ROOT = Path(__file__).parent.parent


class SkillRegistry:
    """声明式 Skill 注册表，供 LlmNode 基类在初始化时注入工具技能描述。"""

    # 单例实例
    _instance: Optional["SkillRegistry"] = None
    # 保护单例创建的类级锁
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        # 实例级锁：保护所有缓存读写
        self._lock = threading.Lock()

        # static / inline skill 缓存：{ skill名: 内容字符串 }
        self._static_cache: dict[str, str] = {}

        # command skill 配置缓存：{ skill名: {"command": str, "ttl": int} }
        self._command_configs: dict[str, dict] = {}

        # command skill 输出缓存：{ skill名: (内容字符串, 缓存时间戳) }
        self._command_cache: dict[str, tuple[str, float]] = {}

        # 已知的所有 skill 名（扫描结果）
        self._discovered_names: set[str] = set()

        # 首次扫描
        self._scan()

    # ── 单例接口 ────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "SkillRegistry":
        """返回全局单例，线程安全。"""
        if cls._instance is None:
            with cls._class_lock:
                # 双重检查，防止并发创建
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 核心接口 ────────────────────────────────────────────────────────────

    def register(self, name: str, content: str) -> None:
        """inline 注册：直接将内容字符串注册为指定名称的 skill（主要用于测试）。

        inline 注册优先级最高，会覆盖同名 static/command skill。
        """
        with self._lock:
            self._static_cache[name] = content
            self._discovered_names.add(name)
            logger.debug(f"[SkillRegistry] inline registered: {name!r} ({len(content)} chars)")

    def load(self, names: list[str]) -> str:
        """加载并拼接多个 skill，用 '\\n\\n---\\n\\n' 分隔。

        不存在的 skill 名会发出警告并跳过，不抛异常。
        """
        parts = []
        for name in names:
            content = self._load_one(name)
            if content is None:
                logger.warning(f"[SkillRegistry] skill not found, skipped: {name!r}")
                continue
            if content.strip():
                parts.append(content.strip())
        return "\n\n---\n\n".join(parts)

    def list_available(self) -> list[str]:
        """返回所有可发现的 skill 名称（包含扫描到的和 inline 注册的）。"""
        with self._lock:
            return sorted(self._discovered_names)

    def reload(self) -> None:
        """重新扫描 skills/ 目录，清除 static 缓存（command TTL 独立，不受影响）。"""
        with self._lock:
            self._static_cache.clear()
            self._command_configs.clear()
            self._discovered_names.clear()
            # 注意：_command_cache 保留（TTL 独立管理）
        self._scan()
        logger.info("[SkillRegistry] reloaded")

    # ── 内部实现 ────────────────────────────────────────────────────────────

    def _scan(self) -> None:
        """扫描项目根目录下 skills/ 文件夹，识别所有可用 skill。"""
        skills_dir = _PROJECT_ROOT / "skills"
        if not skills_dir.exists():
            logger.debug("[SkillRegistry] skills/ 目录不存在，跳过扫描")
            return

        with self._lock:
            # 第一遍：扫描所有 .skill.json（command 类型）
            for json_file in skills_dir.glob("*.skill.json"):
                # 例：skills/blender.skill.json → skill 名 "blender"
                name = json_file.stem.replace(".skill", "")
                try:
                    cfg = json.loads(json_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning(f"[SkillRegistry] 解析 {json_file} 失败: {exc}")
                    continue
                if cfg.get("type") == "command":
                    self._command_configs[name] = cfg
                    self._discovered_names.add(name)
                    logger.debug(f"[SkillRegistry] discovered command skill: {name!r}")

            # 第二遍：平铺 .md 文件（skill 名已被 command 占用则跳过）
            for md_file in skills_dir.glob("*.md"):
                name = md_file.stem  # skills/foo.md → "foo"
                if name not in self._command_configs:
                    self._discovered_names.add(name)
                    logger.debug(f"[SkillRegistry] discovered static skill (flat): {name!r}")

            # 第三遍：子目录形式 skills/foo/SKILL.md
            for skill_md in skills_dir.glob("*/SKILL.md"):
                name = skill_md.parent.name  # skills/foo/SKILL.md → "foo"
                if name not in self._command_configs:
                    self._discovered_names.add(name)
                    logger.debug(f"[SkillRegistry] discovered static skill (subdir): {name!r}")

    def _load_one(self, name: str) -> Optional[str]:
        """加载单个 skill，返回内容字符串；不存在返回 None。

        优先级：
          1. inline（存在于 _static_cache 且不在 _command_configs 中）
          2. command（在 _command_configs 中）
          3. static（从 skills/ 扫描的 .md 文件）
        """
        with self._lock:
            # 1. inline 注册（_static_cache 中且不是 command 类型）
            if name in self._static_cache and name not in self._command_configs:
                return self._static_cache[name]

            # 2. command 类型
            if name in self._command_configs:
                return self._load_command(name)

        # 3. static 类型（从 skills/ 目录读取文件，不持锁避免 IO 阻塞）
        return self._load_static(name)

    def _load_command(self, name: str) -> Optional[str]:
        """运行命令获取 skill 内容，TTL 缓存。调用方需持 self._lock。"""
        cfg = self._command_configs[name]
        ttl = cfg.get("ttl", 3600)
        command = cfg.get("command", "")

        # 检查缓存是否有效
        if name in self._command_cache:
            cached_content, cached_at = self._command_cache[name]
            if time.monotonic() - cached_at < ttl:
                logger.debug(f"[SkillRegistry] command cache hit: {name!r}")
                return cached_content

        # 缓存失效，重新运行命令
        logger.info(f"[SkillRegistry] running command for skill {name!r}: {command!r}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            content = result.stdout.strip()
            if result.returncode != 0:
                logger.warning(
                    f"[SkillRegistry] command skill {name!r} exited {result.returncode}: "
                    f"{result.stderr[:200]}"
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"[SkillRegistry] command skill {name!r} timed out")
            content = ""
        except Exception as exc:
            logger.warning(f"[SkillRegistry] command skill {name!r} error: {exc}")
            content = ""

        # 存入缓存（即使为空，也缓存，避免频繁重试）
        self._command_cache[name] = (content, time.monotonic())
        return content if content else None

    def _load_static(self, name: str) -> Optional[str]:
        """从 skills/ 目录读取 .md 文件，不持锁。"""
        skills_dir = _PROJECT_ROOT / "skills"

        # 先检查是否已缓存
        with self._lock:
            if name in self._static_cache:
                return self._static_cache[name]

        # 按优先级尝试两种路径格式
        candidates = [
            skills_dir / f"{name}.md",         # 平铺格式：skills/foo.md
            skills_dir / name / "SKILL.md",    # 子目录格式：skills/foo/SKILL.md
        ]

        for path in candidates:
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    with self._lock:
                        self._static_cache[name] = content
                        self._discovered_names.add(name)
                    logger.info(f"[SkillRegistry] loaded static skill: {name!r} from {path}")
                    return content
                except Exception as exc:
                    logger.warning(f"[SkillRegistry] 读取 {path} 失败: {exc}")

        return None
