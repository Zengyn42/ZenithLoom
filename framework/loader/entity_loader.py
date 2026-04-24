"""
框架级实体加载器 — framework/loader/entity_loader.py

EntityLoader(blueprint_dir, data_dir) 从 blueprint 目录的 entity.json 加载角色定义，
从 data_dir 的 identity.json 加载实例专属配置，
提供完整的图构建和控制器管理能力。

图构建优先级：
  Priority 1: agent 目录下的 graph.py（定义 build_graph(loader, checkpointer)）
  Priority 2: entity.json["graph"] 含 "nodes" 和 "edges" → 声明式图
  Priority 3: entity.json["graph"] 含 GraphSpec 标志 → 框架默认图
"""

import json
import logging
from pathlib import Path

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.loader.graph_builder import _DEFAULT, _build_declarative, _extract_session_keys_from_json
from framework.loader.topology import _mermaid_render

logger = logging.getLogger(__name__)


class EntityLoader:
    """
    从 blueprint + entity 目录加载配置，构建 LangGraph 状态机，管理控制器单例。

    用法：
        loader = EntityLoader(Path("blueprints/role_agents/technical_architect"),
                              data_dir=Path("~/Foundation/EdenGateway/agents/hani"))
        controller = await loader.get_controller()
        response = await controller.run("用户输入")
    """

    def __init__(self, agent_dir: Path, data_dir: Path | None = None):
        self._dir = Path(agent_dir).resolve()
        self._data_dir = Path(data_dir).resolve() if data_dir else self._dir
        self._env_prefix = self._resolve_env_prefix()
        self._json: dict = json.loads(
            (self._dir / "entity.json").read_text(encoding="utf-8")
        )
        self._config: AgentConfig | None = None
        self._engine = None
        self._controller = None
        self._session_mgr = None
        self._heartbeat_proxy = None
        self._mcp_proxies: dict = {}

        if is_debug():
            logger.debug(
                f"[entity_loader] dir={self._dir} data_dir={self._data_dir} prefix={self._env_prefix}"
            )

    def _resolve_env_prefix(self) -> str:
        """Derive env prefix from identity.json["name"] if available, else dir name."""
        entity_path = self._data_dir / "identity.json"
        if entity_path.exists():
            try:
                inst = json.loads(entity_path.read_text(encoding="utf-8"))
                name = inst.get("name", "")
                if name:
                    return name.upper()
            except Exception:
                pass
        return self._dir.name.upper()

    @property
    def json(self) -> dict:
        return self._json

    @property
    def name(self) -> str:
        if self._config is not None and self._config.name:
            return self._config.name
        entity_path = self._data_dir / "identity.json"
        if entity_path.exists():
            try:
                inst = json.loads(entity_path.read_text(encoding="utf-8"))
                n = inst.get("name", "")
                if n:
                    return n
            except Exception:
                pass
        return self._json.get("name", self._dir.name)

    def load_config(self) -> AgentConfig:
        """加载 AgentConfig，相对路径解析到 data_dir。"""
        if self._config is None:
            blueprint_path = self._dir / "entity.json"
            entity_path = self._data_dir / "identity.json"
            cfg = AgentConfig.from_blueprint_and_instance(
                blueprint_path,
                entity_path if entity_path.exists() else None,
                env_prefix=self._env_prefix,
            )
            if not Path(cfg.db_path).is_absolute():
                cfg.db_path = str(self._data_dir / cfg.db_path)
            if not Path(cfg.sessions_file).is_absolute():
                cfg.sessions_file = str(self._data_dir / cfg.sessions_file)
            self._config = cfg
        return self._config

    def load_system_prompt(self) -> str:
        """按 persona_files 顺序拼接 system prompt，每段标注来源。

        查找顺序：data_dir（entity 目录）优先，其次 blueprint dir。
        entity 目录中额外存在的 SOUL.md / IDENTITY.md 自动追加到末尾。
        """
        parts = []
        seen: set[str] = set()

        for fname in self._json.get("persona_files", []):
            for search_dir, label in [
                (self._data_dir, self._data_dir.name),
                (self._dir, self._dir.name),
            ]:
                p = search_dir / fname
                if p.exists():
                    content = p.read_text(encoding="utf-8").strip()
                    header = f"<!-- [source: {label}/{fname}] -->"
                    parts.append(f"{header}\n{content}")
                    seen.add(fname)
                    break

        if self._data_dir != self._dir:
            for p in sorted(self._data_dir.glob("*.md")):
                if p.name not in seen:
                    content = p.read_text(encoding="utf-8").strip()
                    header = f"<!-- [source: {self._data_dir.name}/{p.name}] -->"
                    parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)

    @property
    def session_mgr(self):
        """懒加载 SessionManager。"""
        if self._session_mgr is None:
            from framework.session_mgr import SessionManager
            cfg = self.load_config()
            self._session_mgr = SessionManager(cfg.sessions_file, cfg.db_path)
        return self._session_mgr

    async def build_graph(
        self,
        checkpointer=_DEFAULT,
        extra_persona_text: str = "",
        is_subgraph: bool = False,
        force_unique_session_keys: bool = False,
        session_mode: str = "fresh_per_call",
        fresh_keep_fields: list[str] | None = None,
    ):
        """
        构建并返回编译好的 LangGraph 状态机。

        优先级：
          1. agent 目录下的 graph.py（定义 build_graph(loader, checkpointer)）
          2. entity.json["graph"] 含 "nodes" + "edges" → 声明式图
          3. 否则使用框架默认图（GraphSpec 标志驱动）
        """
        # Priority 1: custom graph.py
        custom_graph_path = self._dir / "graph.py"
        if custom_graph_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"agents.{self._dir.name}.graph", custom_graph_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "build_graph"):
                logger.info(f"[entity_loader] using custom graph.py for {self.name!r}")
                compiled = await mod.build_graph(self, checkpointer)

                if is_subgraph and session_mode in ("fresh_per_call", "isolated"):
                    from framework.nodes.subgraph_init_node import make_subgraph_init, make_subgraph_exit
                    from framework.schema.base import BaseAgentState, SubgraphInputState
                    from langgraph.graph import StateGraph, START, END

                    wrapper = StateGraph(BaseAgentState, input_schema=SubgraphInputState)
                    _init_fn = make_subgraph_init(session_mode, keep_fields=fresh_keep_fields)
                    _skeys = _extract_session_keys_from_json(self._json)
                    _exit_fn = make_subgraph_exit(session_mode=session_mode, subgraph_session_keys=_skeys)
                    wrapper.add_node("_subgraph_init", _init_fn)
                    wrapper.add_node("_inner", compiled)
                    wrapper.add_node("_subgraph_exit", _exit_fn)
                    wrapper.add_edge(START, "_subgraph_init")
                    wrapper.add_edge("_subgraph_init", "_inner")
                    wrapper.add_edge("_inner", "_subgraph_exit")
                    wrapper.add_edge("_subgraph_exit", END)
                    logger.debug(f"[entity_loader] wrapping custom graph.py with init/exit (mode={session_mode})")
                    return wrapper.compile(checkpointer=checkpointer)

                return compiled

        config = self.load_config()

        # Assemble instance-level extra_persona from identity.json persona_files
        _instance_persona_parts: list[str] = []
        _instance_dir = self._data_dir
        _blueprint_dir_path = Path(self._dir)

        _identity_json_path = _instance_dir / "identity.json"
        _instance_persona_files: list[str] = []
        if _identity_json_path.exists():
            try:
                _inst = json.loads(_identity_json_path.read_text(encoding="utf-8"))
                _instance_persona_files = _inst.get("persona_files", [])
                _instance_prompt = _inst.get("prompt", "")
            except Exception:
                _instance_persona_files = []
                _instance_prompt = ""
        else:
            _instance_prompt = ""

        if _instance_persona_files or _instance_prompt:
            from framework.loader.persona import _load_persona_text
            _inst_text = _load_persona_text(
                _instance_persona_files, _instance_dir,
                prompt=_instance_prompt, label=_instance_dir.name,
            )
            if _inst_text:
                _instance_persona_parts.append(_inst_text)

        if _instance_dir != _blueprint_dir_path:
            _seen = set(_instance_persona_files)
            for _p in sorted(_instance_dir.glob("*.md")):
                if _p.name not in _seen:
                    _content = _p.read_text(encoding="utf-8").strip()
                    _instance_persona_parts.append(
                        f"<!-- [source: {_instance_dir.name}/{_p.name}] -->\n{_content}"
                    )

        _all_extra = "\n\n---\n\n".join(p for p in [extra_persona_text] + _instance_persona_parts if p)

        graph_dict = dict(self._json.get("graph") or {})

        # Priority 2: declarative graph
        if "nodes" in graph_dict and "edges" in graph_dict:
            logger.info(f"[entity_loader] building declarative graph for {self.name!r}")
            return await _build_declarative(
                graph_dict, config, checkpointer,
                blueprint_dir=str(self._dir),
                is_subgraph=is_subgraph,
                force_unique_session_keys=force_unique_session_keys,
                extra_persona_text=_all_extra,
                session_mode=session_mode,
                fresh_keep_fields=fresh_keep_fields,
            )

        # Priority 3: GraphSpec default graph
        from framework.nodes.llm.claude import ClaudeSDKNode
        from framework.graph import build_agent_graph, GraphSpec

        if "use_vram_flush" not in graph_dict:
            graph_dict["use_vram_flush"] = bool(self._json.get("vram_flush", False))
        spec = GraphSpec.from_dict(graph_dict)

        _p3_system_prompt = self.load_system_prompt()
        claude_node_config = {**self._json}
        agent_node = ClaudeSDKNode(config, node_config=claude_node_config, system_prompt=_p3_system_prompt)

        logger.info(f"[entity_loader] building graph for {self.name!r}")
        if is_debug():
            logger.debug(f"[entity_loader] db={config.db_path!r}")

        compiled = await build_agent_graph(
            config=config,
            agent_node=agent_node,
            checkpointer=checkpointer,
            spec=spec,
        )
        compiled._llm_node_instances = {
            getattr(agent_node, "_session_key", "claude_main"): agent_node,
        }
        return compiled

    async def _build_engine(self):
        """懒加载编译图单例（内部使用）。"""
        if self._engine is None:
            self._engine = await self.build_graph()
        return self._engine

    async def get_controller(self):
        """懒加载 GraphController 单例（唯一推荐的外部入口）。"""
        if self._controller is None:
            from framework.graph_controller import GraphController
            graph = await self._build_engine()
            self._controller = GraphController(graph, self.session_mgr, self.load_config(), entity_name=self.name)
        return self._controller

    async def get_engine(self):
        """已废弃：请改用 get_controller()。保留供旧代码/测试向后兼容。"""
        return await self._build_engine()

    @property
    def heartbeat_proxy(self):
        """返回 HeartbeatMCPProxy 实例（可能为 None，需先调用 start_heartbeat）。"""
        return self._heartbeat_proxy

    # ------------------------------------------------------------------
    # MCP profile helpers
    # ------------------------------------------------------------------

    def _resolve_profile_path(self, entry: str) -> Path | None:
        """将单条 mcp entry 字符串解析为 profile.json 的绝对路径。

        支持三种格式：
          - "python:pkg_name"          → import pkg_name，取 __file__.parent.parent / profile.json
          - "relative/path/profile.json" → 相对于 ZenithLoom 项目根（_PROJECT_ROOT）
          - "/absolute/path/profile.json" → 直接使用
        """
        from framework.mcp_manager import _PROJECT_ROOT

        if entry.startswith("python:"):
            pkg_name = entry[7:]
            try:
                import importlib
                mod = importlib.import_module(pkg_name)
                pkg_root = Path(mod.__file__).parent.parent
                profile_path = pkg_root / "profile.json"
                if not profile_path.exists():
                    logger.error(f"[entity_loader] python:{pkg_name} profile.json not found at {profile_path}")
                    return None
                return profile_path
            except ImportError as exc:
                logger.error(f"[entity_loader] python:{pkg_name} import failed: {exc}")
                return None

        p = Path(entry)
        if p.is_absolute():
            return p
        return (_PROJECT_ROOT / p).resolve()

    def _load_profile(self, path: Path, source: str) -> dict | None:
        """读取 profile.json 并返回扁平化 spec dict（MCPManager 格式）。"""
        if not path.exists():
            logger.error(f"[entity_loader] mcp profile not found: {path} (from {source})")
            return None
        try:
            profile: dict = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[entity_loader] failed to read mcp profile {path}: {exc}")
            return None

        server = profile.get("server", {})
        spec: dict = {
            "name": profile.get("name", path.stem),
            "module": server.get("module", ""),
            "module_args": server.get("module_args", []),
            "url": server.get("url", ""),
            "shared": server.get("shared", True),
        }
        if profile.get("dependency"):
            spec["dependency"] = profile["dependency"]
        if profile.get("proxy"):
            spec["proxy"] = profile["proxy"]
        logger.debug(f"[entity_loader] loaded mcp profile {profile.get('name')!r} from {path}")
        return spec

    def _resolve_mcp_specs(self, mcp_entries: list) -> list[dict]:
        """将 mcp entry 列表解析为完整 spec dict 列表。

        每个 entry 可以是：
          - str "python:pkg"       → Python 包路径解析
          - str "relative/path"    → 相对 ZenithLoom root
          - str "/absolute/path"   → 绝对路径
          - dict                   → 旧格式（向后兼容，打 warning）
        """
        if not mcp_entries:
            return []

        specs: list[dict] = []
        for entry in mcp_entries:
            if isinstance(entry, str):
                profile_path = self._resolve_profile_path(entry)
                if profile_path is None:
                    continue
                spec = self._load_profile(profile_path, source=entry)
                if spec:
                    specs.append(spec)
            elif isinstance(entry, dict):
                name = entry.get("name", "")
                logger.warning(
                    f"[entity_loader] {self.name!r}: mcp entry {name!r} uses deprecated inline spec; "
                    f"migrate to profile.json"
                )
                specs.append(entry)
            else:
                logger.warning(f"[entity_loader] {self.name!r}: unknown mcp entry type {type(entry)}, skipped")
        return specs

    def _collect_mcp_entries(self) -> list:
        """合并 entity.json 和 identity.json 的 mcp 列表，按 name 去重（entity 优先）。"""
        entity_entries = self._json.get("mcp", [])

        # 读取 identity.json 的 mcp 追加列表
        identity_path = self._data_dir / "identity.json"
        identity_entries: list = []
        if identity_path.exists():
            try:
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
                identity_entries = identity.get("mcp", [])
            except Exception as exc:
                logger.warning(f"[entity_loader] failed to read identity mcp entries: {exc}")

        if not entity_entries and not identity_entries:
            return []

        # 先解析 entity（blueprint 层，角色必需）
        entity_specs = self._resolve_mcp_specs(entity_entries)
        seen_names: set[str] = {s["name"] for s in entity_specs}

        # 再追加 identity（实例层，部署追加），按 name 去重
        identity_specs = self._resolve_mcp_specs(identity_entries)
        for spec in identity_specs:
            name = spec.get("name", "")
            if name and name not in seen_names:
                entity_specs.append(spec)
                seen_names.add(name)
            elif name in seen_names:
                logger.debug(f"[entity_loader] identity mcp {name!r} already provided by blueprint, skipped")

        return entity_specs

    async def start_mcp_servers(self) -> list[str]:
        """启动 entity.json（blueprint）+ identity.json（instance）声明的 MCP Server。

        entity 层为角色必需 MCP（blueprint），identity 层为部署追加 MCP（instance），
        按 name 去重，entity 优先。
        """
        from framework.mcp_manager import MCPManager
        mgr = MCPManager.get_instance()
        specs = self._collect_mcp_entries()
        if not specs:
            return []
        started: list[str] = []
        acquired_specs: list[dict] = []
        for spec in specs:
            name = spec.get("name", "")
            if not name:
                logger.warning(f"[entity_loader] {self.name!r}: mcp entry missing 'name', skipped")
                continue
            ok = await mgr.acquire(spec, agent_name=self.name)
            if ok:
                started.append(name)
                acquired_specs.append(spec)
                logger.info(f"[entity_loader] {self.name!r}: mcp server {name!r} acquired")
                proxy_type = spec.get("proxy")
                if proxy_type:
                    await self._connect_proxy(name, spec, proxy_type)
            else:
                logger.error(f"[entity_loader] {self.name!r}: mcp server {name!r} failed to start")

        self._inject_gemini_mcp_configs(acquired_specs)
        return started

    def _inject_gemini_mcp_configs(self, acquired_specs: list[dict]) -> None:
        """如果 entity 包含 GEMINI_CLI 节点，将已启动的 MCP SSE URL 写入
        工作区的 .gemini/settings.json，使 Gemini CLI 能发现这些 MCP Server。

        写入格式：{"mcpServers": {"<name>": {"url": "<sse_url>"}}}
        仅追加/更新，不覆盖已有的其他 MCP 条目。
        写入位置：ZenithLoom 项目根（framework 上两级）及其父目录（Foundation workspace）。
        """
        if not acquired_specs:
            return

        nodes = self._json.get("graph", {}).get("nodes", [])
        has_gemini_cli = any(n.get("type") == "GEMINI_CLI" for n in nodes)
        if not has_gemini_cli:
            return

        # 确定要写入的 .gemini/settings.json 路径集合：
        # project_root = ZenithLoom/（MCPManager 所在的父目录）
        # workspace    = Foundation/（project_root 的父目录，即 agents 运行时的 cwd）
        from framework.mcp_manager import _PROJECT_ROOT
        candidate_dirs = [_PROJECT_ROOT, _PROJECT_ROOT.parent]

        for workspace in candidate_dirs:
            settings_path = workspace / ".gemini" / "settings.json"
            if not settings_path.parent.exists():
                continue  # 没有 .gemini/ 目录则跳过（避免意外创建）

            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
            except Exception as exc:
                logger.warning(f"[entity_loader] 读取 {settings_path} 失败: {exc}")
                continue

            mcp_servers: dict = settings.setdefault("mcpServers", {})
            changed = False
            for spec in acquired_specs:
                name = spec.get("name", "")
                url = spec.get("url", "")
                if not name or not url:
                    continue
                new_entry: dict = {"url": url}
                if mcp_servers.get(name) != new_entry:
                    mcp_servers[name] = new_entry
                    changed = True

            if changed:
                try:
                    settings_path.write_text(
                        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    logger.info(
                        f"[entity_loader] {self.name!r}: gemini mcp 配置已注入 → {settings_path}"
                    )
                except Exception as exc:
                    logger.warning(f"[entity_loader] 写入 {settings_path} 失败: {exc}")

    async def _connect_proxy(self, name: str, spec: dict, proxy_type: str) -> None:
        """Connect a proxy for framework tool registration (Gemini/Ollama nodes)."""
        proxy_class = _resolve_proxy_class(proxy_type)
        if proxy_class is None:
            logger.debug(f"[entity_loader] {self.name!r}: no proxy class for {proxy_type!r}, skip")
            return

        url = spec.get("url", "")
        try:
            proxy = proxy_class(url)
            await proxy.connect()
            self._mcp_proxies[name] = proxy
            logger.info(f"[entity_loader] {self.name!r}: proxy {name!r} connected")

            if proxy_type == "agent_mail" and hasattr(proxy, "register"):
                try:
                    await proxy.register(self.name)
                except Exception as e:
                    logger.warning(f"[entity_loader] {self.name!r}: agent_mail register failed: {e}")

            _register_mcp_tools(proxy_type, proxy)
        except Exception as e:
            logger.error(f"[entity_loader] {self.name!r}: proxy {name!r} connect failed: {e}")

    async def stop_mcp_servers(self) -> None:
        """释放本 agent 持有的所有 MCP Server 引用 + 断开 proxy 连接。"""
        for name, proxy in list(self._mcp_proxies.items()):
            if hasattr(proxy, "unregister"):
                try:
                    await proxy.unregister(self.name)
                except Exception as e:
                    logger.warning(f"[entity_loader] {self.name!r}: {name!r} unregister failed: {e}")
            try:
                await proxy.disconnect()
                logger.info(f"[entity_loader] {self.name!r}: proxy {name!r} disconnected")
            except Exception as e:
                logger.warning(f"[entity_loader] {self.name!r}: proxy {name!r} disconnect failed: {e}")
        self._mcp_proxies.clear()

        from framework.mcp_manager import MCPManager
        mgr = MCPManager.get_instance()
        specs = self._collect_mcp_entries()
        for spec in specs:
            name = spec.get("name", "")
            if name:
                await mgr.release(name, agent_name=self.name)
                logger.info(f"[entity_loader] {self.name!r}: mcp server {name!r} released")

    async def start_heartbeat(self):
        """连接 Heartbeat MCP Server 并装载 entity 指定的 blueprint。"""
        from framework.nodes.llm.heartbeat_tools import HeartbeatMCPProxy

        entity_path = self._data_dir / "identity.json"
        if not entity_path.exists():
            logger.debug(f"[entity_loader] {self.name!r}: no identity.json, skip heartbeat")
            return None

        try:
            entity = json.loads(entity_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[entity_loader] failed to read identity.json: {e}")
            return None

        hb_ref = entity.get("heartbeat")
        if not hb_ref:
            logger.debug(f"[entity_loader] {self.name!r}: no heartbeat configuration")
            return None

        if isinstance(hb_ref, str):
            entries = [{"blueprint": hb_ref}]
        elif isinstance(hb_ref, dict):
            entries = [hb_ref]
        elif isinstance(hb_ref, list):
            entries = []
            for item in hb_ref:
                if isinstance(item, str):
                    entries.append({"blueprint": item})
                elif isinstance(item, dict):
                    entries.append(item)
        else:
            logger.warning(f"[entity_loader] invalid heartbeat config: {hb_ref}")
            return None

        if not entries:
            return None

        server_url = entries[0].get("server_url", "http://127.0.0.1:8100/sse")

        proxy = HeartbeatMCPProxy(server_url)
        try:
            await proxy.connect()
        except Exception as e:
            logger.error(
                f"[entity_loader] {self.name!r}: failed to connect Heartbeat MCP Server "
                f"at {server_url}: {e}"
            )
            return None

        for entry in entries:
            bp_path = entry.get("blueprint", "")
            if not bp_path:
                continue
            overrides = entry.get("overrides")
            result = await proxy.load_blueprint(bp_path, overrides=overrides)
            logger.info(f"[entity_loader] {self.name!r} heartbeat: {result}")

        from framework.nodes.llm.heartbeat_tools import make_heartbeat_tools
        from framework.nodes.llm.tools import TOOL_REGISTRY, TOOL_SCHEMAS
        registry, schemas = make_heartbeat_tools(proxy)
        TOOL_REGISTRY.update(registry)
        TOOL_SCHEMAS.update(schemas)

        self._heartbeat_proxy = proxy
        from framework.nodes.llm.heartbeat_tools import set_active_proxy
        set_active_proxy(proxy)
        return proxy

    async def stop_heartbeat(self):
        """卸载本 agent 装载的所有 heartbeat blueprint 并断开连接。"""
        if self._heartbeat_proxy is not None:
            try:
                await self._heartbeat_proxy.cleanup()
            except Exception as e:
                logger.warning(f"[entity_loader] {self.name!r} heartbeat cleanup failed: {e}")
            self._heartbeat_proxy = None
            from framework.nodes.llm.heartbeat_tools import set_active_proxy
            set_active_proxy(None)

    def invalidate_engine(self) -> None:
        """使引擎和控制器缓存失效（compact/reset 后调用）。"""
        self._engine = None
        self._controller = None
        logger.info(f"[entity_loader] engine invalidated for {self.name!r}")

    def build_topology_mermaid(self) -> str:
        """从 entity.json 构建 Mermaid 拓扑图，展开 external subgraph 子图。"""
        graph_spec = self._json.get("graph", {})
        lines = ["flowchart LR"]
        _mermaid_render(graph_spec, lines, "  ", "")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP proxy factory helpers
# ---------------------------------------------------------------------------

def _resolve_proxy_class(mcp_name: str):
    """根据 MCP 名称返回对应的 proxy 类。"""
    if mcp_name == "heartbeat":
        from framework.nodes.llm.heartbeat_tools import HeartbeatMCPProxy
        return HeartbeatMCPProxy
    if mcp_name == "agent_mail":
        from framework.mcp_proxy_agent_mail import AgentMailProxy
        return AgentMailProxy
    return None


def _register_mcp_tools(mcp_name: str, proxy) -> None:
    """将 MCP proxy 的工具注册到框架 TOOL_REGISTRY / TOOL_SCHEMAS。"""
    from framework.nodes.llm.tools import TOOL_REGISTRY, TOOL_SCHEMAS

    if mcp_name == "heartbeat":
        from framework.nodes.llm.heartbeat_tools import make_heartbeat_tools
        registry, schemas = make_heartbeat_tools(proxy)
        TOOL_REGISTRY.update(registry)
        TOOL_SCHEMAS.update(schemas)
        logger.info(f"[entity_loader] registered heartbeat tools ({len(registry)} tools)")

    elif mcp_name == "agent_mail":
        try:
            from framework.mcp_proxy_agent_mail import make_agent_mail_tools
            registry, schemas = make_agent_mail_tools(proxy)
            TOOL_REGISTRY.update(registry)
            TOOL_SCHEMAS.update(schemas)
            logger.info(f"[entity_loader] registered agent_mail tools ({len(registry)} tools)")
        except ImportError:
            logger.debug(
                "[entity_loader] mcp_proxy_agent_mail not yet implemented, tools not registered"
            )
