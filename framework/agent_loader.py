"""
框架级 Agent 加载器 — framework/agent_loader.py

AgentLoader(agent_dir) 从 agent 目录的 agent.json 加载所有配置，
提供完整的图构建和引擎管理能力。

任何新 Agent 只需：
  1. 创建 agents/<name>/ 目录
  2. 编写 agent.json（含 llm、persona_files、tool_rules 等）
  3. 放置 persona .md 文件
  4. 编写 6 行 graph.py

env_prefix 自动派生：agents/hani → "HANI"，agents/asa → "ASA"。

agent.json 支持的 LLM：
  "llm": "claude"  — Claude Code CLI SDK（默认）
  "llm": "llama"   — Llama（Ollama / vLLM 本地推理）
"""

import json
import logging
from pathlib import Path

from framework.config import AgentConfig
from framework.debug import is_debug

logger = logging.getLogger(__name__)


class AgentLoader:
    """
    从 agent 目录加载配置，构建 LangGraph 状态机，管理引擎单例。

    用法：
        loader = AgentLoader(Path("agents/hani"))
        engine = await loader.get_engine()
    """

    def __init__(self, agent_dir: Path):
        self._dir = Path(agent_dir).resolve()
        self._env_prefix = self._dir.name.upper()  # "hani" → "HANI"
        self._json: dict = json.loads(
            (self._dir / "agent.json").read_text(encoding="utf-8")
        )
        self._config: AgentConfig | None = None
        self._engine = None
        self._session_mgr = None

        if is_debug():
            logger.debug(
                f"[agent_loader] dir={self._dir} prefix={self._env_prefix}"
            )

    @property
    def json(self) -> dict:
        return self._json

    @property
    def name(self) -> str:
        return self._json.get("name", self._dir.name)

    def load_config(self) -> AgentConfig:
        """加载 AgentConfig，相对路径解析到 agent_dir（不是项目根）。"""
        if self._config is None:
            cfg = AgentConfig.from_json(
                self._dir / "agent.json", env_prefix=self._env_prefix
            )
            if not Path(cfg.db_path).is_absolute():
                cfg.db_path = str(self._dir / cfg.db_path)
            if not Path(cfg.sessions_file).is_absolute():
                cfg.sessions_file = str(self._dir / cfg.sessions_file)
            self._config = cfg
        return self._config

    def load_system_prompt(self) -> str:
        """按 persona_files 顺序拼接 system prompt。"""
        parts = []
        for fname in self._json.get("persona_files", []):
            p = self._dir / fname
            if p.exists():
                parts.append(p.read_text(encoding="utf-8").strip())
        return "\n\n---\n\n".join(parts)

    @property
    def session_mgr(self):
        """懒加载 SessionManager（使用 agent 目录内的路径）。"""
        if self._session_mgr is None:
            from framework.session_mgr import SessionManager
            cfg = self.load_config()
            self._session_mgr = SessionManager(cfg.sessions_file, cfg.db_path)
        return self._session_mgr

    def _make_llm_node(self, config: AgentConfig, system_prompt: str):
        """根据 agent.json["llm"] 创建对应的 LLM 节点。"""
        llm = self._json.get("llm", "claude").lower()
        if is_debug():
            logger.debug(f"[agent_loader] llm={llm!r}")
        if llm == "claude":
            from framework.claude.node import ClaudeNode
            return ClaudeNode(config, system_prompt)
        elif llm == "llama":
            from framework.llama.node import LlamaNode
            return LlamaNode(config, system_prompt)
        else:
            raise ValueError(f"未知 llm: {llm!r}，支持：'claude' | 'llama'")

    async def build_graph(self, checkpointer=None):
        """构建并返回编译好的 LangGraph 状态机。

        优先级：
          1. agent 目录下的 graph.py 存在且定义了 build_graph(loader, checkpointer) → 使用它
          2. 否则使用框架默认图（git_snapshot → agent → validate → ...）
        """
        # 检查 agent 是否有自定义图
        custom_graph_path = self._dir / "graph.py"
        if custom_graph_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"agents.{self._dir.name}.graph", custom_graph_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "build_graph"):
                logger.info(f"[agent_loader] using custom graph.py for {self.name!r}")
                return await mod.build_graph(self, checkpointer)

        # 框架默认图
        config = self.load_config()
        system_prompt = self.load_system_prompt()
        llm_node = self._make_llm_node(config, system_prompt)

        from framework.gemini.node import GeminiNode
        from framework.nodes.agent_node import AgentNode
        from framework.graph import build_agent_graph, GraphSpec

        gemini = GeminiNode(config, llm_node)
        agent_node = AgentNode(llm_node, gemini, node_config=self._json)

        # agent.json["graph"] 驱动拓扑；向后兼容旧 "vram_flush" 字段
        graph_dict = dict(self._json.get("graph") or {})
        if "use_vram_flush" not in graph_dict:
            graph_dict["use_vram_flush"] = bool(self._json.get("vram_flush", False))
        spec = GraphSpec.from_dict(graph_dict)

        logger.info(f"[agent_loader] building graph for {self.name!r}")
        if is_debug():
            logger.debug(f"[agent_loader] db={config.db_path!r}")

        return await build_agent_graph(
            config=config,
            agent_node=agent_node,
            gemini=gemini,
            checkpointer=checkpointer,
            spec=spec,
        )

    async def get_engine(self):
        """懒加载引擎单例。"""
        if self._engine is None:
            self._engine = await self.build_graph()
        return self._engine

    def invalidate_engine(self) -> None:
        """使引擎缓存失效（compact/reset 后调用）。"""
        self._engine = None
        logger.info(f"[agent_loader] engine invalidated for {self.name!r}")
