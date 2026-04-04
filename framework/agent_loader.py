"""
框架级实体加载器 — framework/agent_loader.py

EntityLoader(blueprint_dir, data_dir) 从 blueprint 目录的 entity.json 加载角色定义，
从 data_dir 的 identity.json 加载实例专属配置，
提供完整的图构建和控制器管理能力。

任何新 Agent 只需：
  1. 创建 blueprints/role_agents/<name>/ 目录
  2. 编写 entity.json（含 llm、persona_files、tool_rules 等）
  3. 放置 persona .md 文件
  4. 可选：编写 graph.py（完全自定义图）

图构建优先级：
  Priority 1: agent 目录下的 graph.py（定义 build_graph(loader, checkpointer)）
  Priority 2: entity.json["graph"] 含 "nodes" 和 "edges" → 声明式图
  Priority 3: entity.json["graph"] 含 GraphSpec 标志 → 框架默认图

声明式图（Priority 2）支持完整节点/边/条件路由定义，
以及 SUBGRAPH 内联子图（递归构建，共享 checkpointer）。

图构建前验证（声明式图）：
  1. 节点 ID 全局唯一（含子图递归）
  2. 边引用的节点必须存在
  3. 从 __start__ 可达所有节点（BFS 连通性）
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from framework.config import AgentConfig
from framework.debug import is_debug, log_graph_flow, log_state_snapshot

logger = logging.getLogger(__name__)

# Sentinel: build_graph(checkpointer=_DEFAULT) → create SQLite checkpointer
#           build_graph(checkpointer=None)      → compile without any checkpointer
_DEFAULT = object()


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
        self._heartbeat_proxy = None  # HeartbeatMCPProxy | None
        self._mcp_proxies: dict = {}  # name → proxy（由 start_mcps() 填充）

        if is_debug():
            logger.debug(
                f"[agent_loader] dir={self._dir} data_dir={self._data_dir} prefix={self._env_prefix}"
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
        # If config already loaded, use entity name (most specific)
        if self._config is not None and self._config.name:
            return self._config.name
        # Try identity.json directly (instance-specific name)
        entity_path = self._data_dir / "identity.json"
        if entity_path.exists():
            try:
                import json as _json_mod
                inst = _json_mod.loads(entity_path.read_text(encoding="utf-8"))
                n = inst.get("name", "")
                if n:
                    return n
            except Exception:
                pass
        return self._json.get("name", self._dir.name)

    def load_config(self) -> AgentConfig:
        """加载 AgentConfig，相对路径解析到 data_dir（不是项目根）。"""
        if self._config is None:
            blueprint_path = self._dir / "entity.json"
            entity_path = self._data_dir / "identity.json"
            cfg = AgentConfig.from_blueprint_and_instance(
                blueprint_path,
                entity_path if entity_path.exists() else None,
                env_prefix=self._env_prefix,
            )
            # db_path 和 sessions_file 相对路径解析到 data_dir
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
            # data_dir 优先（entity 可覆盖 blueprint 文件）
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

        # entity 目录中额外的 .md 文件（SOUL.md、IDENTITY.md 等）自动追加
        if self._data_dir != self._dir:
            for p in sorted(self._data_dir.glob("*.md")):
                if p.name not in seen:
                    content = p.read_text(encoding="utf-8").strip()
                    header = f"<!-- [source: {self._data_dir.name}/{p.name}] -->"
                    parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)

    @property
    def session_mgr(self):
        """懒加载 SessionManager（使用 agent 目录内的路径）。"""
        if self._session_mgr is None:
            from framework.session_mgr import SessionManager
            cfg = self.load_config()
            self._session_mgr = SessionManager(cfg.sessions_file, cfg.db_path)
        return self._session_mgr

    async def build_graph(self, checkpointer=_DEFAULT, parent_persona: str = "", is_subgraph: bool = False, force_unique_session_keys: bool = False):
        """
        构建并返回编译好的 LangGraph 状态机。

        Args:
            checkpointer: LangGraph checkpointer（None=无，_DEFAULT=自动创建 SQLite）
            parent_persona: 父图透传的 persona（仅 external subgraph 使用）。
                           会追加到子图自身 system_prompt 之后。
            is_subgraph: True 时使用 SubgraphInputState 作为 input schema，
                         原生隔离父图 messages（由 external subgraph 路径传入）。

        优先级：
          1. agent 目录下的 graph.py（定义 build_graph(loader, checkpointer)）
          2. entity.json["graph"] 含 "nodes" + "edges" → 声明式图
          3. 否则使用框架默认图（GraphSpec 标志驱动）
        """
        # Priority 1: 自定义 graph.py
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

        config = self.load_config()
        system_prompt = self.load_system_prompt()
        # 合并父图透传的 persona（external subgraph 场景）
        if parent_persona:
            if system_prompt:
                system_prompt = system_prompt + "\n\n" + parent_persona
            else:
                system_prompt = parent_persona
            logger.info(
                f"[agent_loader] {self.name!r}: merged parent persona "
                f"({len(parent_persona)} chars) into system_prompt"
            )
        graph_dict = dict(self._json.get("graph") or {})

        # Priority 2: 声明式图
        if "nodes" in graph_dict and "edges" in graph_dict:
            logger.info(f"[agent_loader] building declarative graph for {self.name!r}")
            return await _build_declarative(
                graph_dict, config, system_prompt, checkpointer,
                blueprint_dir=str(self._dir),
                is_subgraph=is_subgraph,
                force_unique_session_keys=force_unique_session_keys,
            )

        # Priority 3: GraphSpec 默认图
        from framework.nodes.llm.claude import ClaudeSDKNode
        from framework.graph import build_agent_graph, GraphSpec

        # 向后兼容旧 "vram_flush" 字段
        if "use_vram_flush" not in graph_dict:
            graph_dict["use_vram_flush"] = bool(self._json.get("vram_flush", False))
        spec = GraphSpec.from_dict(graph_dict)

        # ClaudeSDKNode — model 从 entity.json 顶层 claude_model 字段读（向后兼容）
        claude_node_config = {**self._json}
        agent_node = ClaudeSDKNode(config, node_config=claude_node_config, system_prompt=system_prompt)

        logger.info(f"[agent_loader] building graph for {self.name!r}")
        if is_debug():
            logger.debug(f"[agent_loader] db={config.db_path!r}")

        return await build_agent_graph(
            config=config,
            agent_node=agent_node,
            checkpointer=checkpointer,
            spec=spec,
        )

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

    async def start_mcp_servers(self) -> list[str]:
        """
        按 entity.json 顶层 "mcp" 数组启动所需的 MCP Server 进程。

        读取格式：
          "mcp": [
            {
              "name": "obsidian-vault",
              "module": "mcp_servers.obsidian.server",
              "module_args": ["--transport", "sse", "--port", "8101",
                              "--vault", "/home/kingy/Foundation/Vault"],
              "url": "http://localhost:8101/sse",
              "shared": true
            }
          ]

        返回成功启动/已运行的 server 名称列表。
        """
        from framework.mcp_manager import MCPManager
        mgr = MCPManager.get_instance()
        specs = self._json.get("mcp", [])
        if not specs:
            return []

        started: list[str] = []
        for spec in specs:
            name = spec.get("name", "")
            if not name:
                logger.warning(f"[agent_loader] {self.name!r}: mcp entry missing 'name', skipped")
                continue
            ok = await mgr.acquire(spec, agent_name=self.name)
            if ok:
                started.append(name)
                logger.info(f"[agent_loader] {self.name!r}: mcp server {name!r} acquired")
            else:
                logger.error(f"[agent_loader] {self.name!r}: mcp server {name!r} failed to start")

        return started

    async def stop_mcp_servers(self) -> None:
        """
        释放本 agent 持有的所有 MCP Server 引用。
        引用归零时 MCPManager 负责停止进程。
        """
        from framework.mcp_manager import MCPManager
        mgr = MCPManager.get_instance()
        specs = self._json.get("mcp", [])
        for spec in specs:
            name = spec.get("name", "")
            if name:
                await mgr.release(name, agent_name=self.name)
                logger.info(f"[agent_loader] {self.name!r}: mcp server {name!r} released")

    async def start_heartbeat(self):
        """
        连接 Heartbeat MCP Server 并装载 entity 指定的 blueprint。

        生命周期：
          1. 读 entity.json 的 "heartbeat" 字段 → blueprint 路径 + MCP server URL
          2. connect() 自动检测 server 是否运行，未运行则启动（detach）
          3. load_blueprint() 装载 blueprint 并追踪名称
          4. 将 MCP 工具注册到框架 tool registry（供 Ollama 等使用）
          5. 返回 proxy（或 None）

        关闭时调用 stop_heartbeat() 卸载本 agent 装载的 blueprint。
        """
        from framework.nodes.llm.heartbeat_tools import HeartbeatMCPProxy

        entity_path = self._data_dir / "identity.json"
        if not entity_path.exists():
            logger.debug(f"[agent_loader] {self.name!r}: no identity.json, skip heartbeat")
            return None

        try:
            entity = json.loads(entity_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[agent_loader] failed to read identity.json: {e}")
            return None

        hb_ref = entity.get("heartbeat")
        if not hb_ref:
            logger.debug(f"[agent_loader] {self.name!r}: no heartbeat configuration")
            return None

        # 解析 heartbeat 配置 → 统一为 list[dict]
        # 支持三种格式：
        #   "path/to/blueprint.json"                       — 单 blueprint 字符串
        #   {"blueprint": "path", "server_url": "..."}     — 单 blueprint 字典
        #   ["path1.json", {"blueprint": "path2.json"}]    — 多 blueprint 数组
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
            logger.warning(f"[agent_loader] invalid heartbeat config: {hb_ref}")
            return None

        if not entries:
            return None

        # 取第一个 entry 的 server_url（所有 blueprint 共享同一个 MCP Server）
        server_url = entries[0].get("server_url", "http://127.0.0.1:8100/sse")

        # connect() 自动检测并启动 MCP Server
        proxy = HeartbeatMCPProxy(server_url)
        try:
            await proxy.connect()
        except Exception as e:
            logger.error(
                f"[agent_loader] {self.name!r}: failed to connect Heartbeat MCP Server "
                f"at {server_url}: {e}"
            )
            return None

        # 装载所有 blueprint（proxy 内部追踪名称，传递 entity 级覆写）
        for entry in entries:
            bp_path = entry.get("blueprint", "")
            if not bp_path:
                continue
            overrides = entry.get("overrides")
            result = await proxy.load_blueprint(bp_path, overrides=overrides)
            logger.info(f"[agent_loader] {self.name!r} heartbeat: {result}")

        # 注册工具到框架 tool registry（供 Ollama 使用）
        from framework.nodes.llm.heartbeat_tools import make_heartbeat_tools
        from framework.nodes.llm.tools import TOOL_REGISTRY, TOOL_SCHEMAS
        registry, schemas = make_heartbeat_tools(proxy)
        TOOL_REGISTRY.update(registry)
        TOOL_SCHEMAS.update(schemas)

        self._heartbeat_proxy = proxy
        # 注册全局引用，供 ExternalToolNode 复用持久 SSE 连接
        from framework.nodes.llm.heartbeat_tools import set_active_proxy
        set_active_proxy(proxy)
        return proxy

    async def stop_heartbeat(self):
        """
        卸载本 agent 装载的所有 heartbeat blueprint 并断开连接。
        MCP Server 在所有 blueprint 卸载后会自动退出。
        """
        if self._heartbeat_proxy is not None:
            try:
                await self._heartbeat_proxy.cleanup()
            except Exception as e:
                logger.warning(f"[agent_loader] {self.name!r} heartbeat cleanup failed: {e}")
            self._heartbeat_proxy = None
            from framework.nodes.llm.heartbeat_tools import set_active_proxy
            set_active_proxy(None)

    async def start_mcps(self):
        """
        遍历 identity.json 的 'mcps' 字段，对每个 MCP 调用
        MCPLauncher.ensure_and_connect()，将结果存入 self._mcp_proxies。

        连接成功后，若 proxy 提供 make_tools() 方法，则注册工具到框架 TOOL_REGISTRY。
        适用于 agent_mail 等通用 MCP，与 heartbeat 完全同等地位。
        """
        from framework.mcp_launcher import MCPLauncher

        entity_path = self._data_dir / "identity.json"
        if not entity_path.exists():
            logger.debug(f"[agent_loader] {self.name!r}: no identity.json, skip start_mcps")
            return

        try:
            entity = json.loads(entity_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[agent_loader] {self.name!r}: failed to read identity.json: {e}")
            return

        mcps_conf = entity.get("mcps", [])
        if not mcps_conf:
            logger.debug(f"[agent_loader] {self.name!r}: no 'mcps' field, skip start_mcps")
            return

        for mcp_conf in mcps_conf:
            mcp_name = mcp_conf.get("name", "unknown")
            proxy_class = _resolve_proxy_class(mcp_name)
            if proxy_class is None:
                logger.warning(
                    f"[agent_loader] {self.name!r}: no proxy class for MCP {mcp_name!r}, skip"
                )
                continue

            proxy = await MCPLauncher.ensure_and_connect(mcp_conf, proxy_class)
            if proxy is None:
                logger.error(
                    f"[agent_loader] {self.name!r}: failed to connect MCP {mcp_name!r}"
                )
                continue

            self._mcp_proxies[mcp_name] = proxy
            logger.info(f"[agent_loader] {self.name!r}: MCP {mcp_name!r} connected")

            # agent_mail: 连接成功后立即注册在线状态
            if mcp_name == "agent_mail" and hasattr(proxy, "register"):
                try:
                    await proxy.register(self.name)
                except Exception as e:
                    logger.warning(
                        f"[agent_loader] {self.name!r}: agent_mail register failed: {e}"
                    )

            # 注册工具到框架 tool registry（若 proxy 提供工厂方法）
            _register_mcp_tools(mcp_name, proxy)

    async def stop_mcps(self) -> None:
        """断开所有由 start_mcps() 建立的 MCP proxy 连接，并清空 _mcp_proxies。

        应在 agent 关闭时调用，与 stop_heartbeat() 同等地位。
        """
        for mcp_name, proxy in list(self._mcp_proxies.items()):
            # agent_mail: 注销在线状态再断连
            if mcp_name == "agent_mail" and hasattr(proxy, "unregister"):
                try:
                    await proxy.unregister(self.name)
                except Exception as e:
                    logger.warning(
                        f"[agent_loader] {self.name!r}: agent_mail unregister failed: {e}"
                    )
            try:
                await proxy.disconnect()
                logger.info(f"[agent_loader] {self.name!r}: MCP {mcp_name!r} disconnected")
            except Exception as e:
                logger.warning(
                    f"[agent_loader] {self.name!r}: MCP {mcp_name!r} disconnect failed: {e}"
                )
        self._mcp_proxies.clear()

    def invalidate_engine(self) -> None:
        """使引擎和控制器缓存失效（compact/reset 后调用）。"""
        self._engine = None
        self._controller = None
        logger.info(f"[agent_loader] engine invalidated for {self.name!r}")

    def build_topology_mermaid(self) -> str:
        """
        从 entity.json 构建 Mermaid 拓扑图，展开 external subgraph 子图。

        弥补 LangGraph get_graph(xray=True) 无法展开 wrapped subgraph callable 的缺陷。
        """
        graph_spec = self._json.get("graph", {})
        lines = ["flowchart LR"]
        _mermaid_render(graph_spec, lines, "  ", "")
        return "\n".join(lines)


# 向后兼容别名
AgentLoader = EntityLoader


# ---------------------------------------------------------------------------
# MCP proxy 工厂：根据 MCP name 返回对应的 proxy 类
# ---------------------------------------------------------------------------

def _resolve_proxy_class(mcp_name: str):
    """根据 MCP 名称返回对应的 proxy 类。

    当前支持：
      "heartbeat"  → HeartbeatMCPProxy
      "agent_mail" → AgentMailProxy

    未知名称返回 None。
    """
    if mcp_name == "heartbeat":
        from framework.nodes.llm.heartbeat_tools import HeartbeatMCPProxy
        return HeartbeatMCPProxy
    if mcp_name == "agent_mail":
        from framework.mcp_proxy_agent_mail import AgentMailProxy
        return AgentMailProxy
    return None


def _register_mcp_tools(mcp_name: str, proxy) -> None:
    """将 MCP proxy 的工具注册到框架 TOOL_REGISTRY / TOOL_SCHEMAS。

    目前支持：
      "heartbeat"  → make_heartbeat_tools(proxy)
      "agent_mail" → make_agent_mail_tools(proxy)

    其他 MCP 若无工具工厂则静默跳过。
    """
    from framework.nodes.llm.tools import TOOL_REGISTRY, TOOL_SCHEMAS

    if mcp_name == "heartbeat":
        from framework.nodes.llm.heartbeat_tools import make_heartbeat_tools
        registry, schemas = make_heartbeat_tools(proxy)
        TOOL_REGISTRY.update(registry)
        TOOL_SCHEMAS.update(schemas)
        logger.info(f"[agent_loader] registered heartbeat tools ({len(registry)} tools)")

    elif mcp_name == "agent_mail":
        try:
            from framework.mcp_proxy_agent_mail import make_agent_mail_tools
            registry, schemas = make_agent_mail_tools(proxy)
            TOOL_REGISTRY.update(registry)
            TOOL_SCHEMAS.update(schemas)
            logger.info(f"[agent_loader] registered agent_mail tools ({len(registry)} tools)")
        except ImportError:
            logger.debug(
                "[agent_loader] mcp_proxy_agent_mail not yet implemented, tools not registered"
            )


# ---------------------------------------------------------------------------
# 通用工具：条件边包装器（module-level，供测试直接导入）
# ---------------------------------------------------------------------------

def _maybe_limit(fn, max_retry):
    """
    向后兼容包装器。max_retry 参数已废弃，直接返回原函数。
    """
    return fn


# ---------------------------------------------------------------------------
# 声明式图构建（Priority 2）
# ---------------------------------------------------------------------------

def _collect_routing_hints(graph_spec: dict, base_dir: str = "") -> str:
    """
    遍历 graph_spec 中所有含 agent_dir 的子图节点，读取其 entity.json 的 routing_hint 字段，
    构建路由说明字符串，用于注入主节点 system_prompt。

    base_dir: blueprint 所在目录，用于解析相对 agent_dir 路径。
    """
    hints: list[str] = []
    for node_def in graph_spec.get("nodes", []):
        # 外部子图：有 agent_dir 且无 type（原 external subgraph）
        agent_dir = node_def.get("agent_dir", "")
        if not agent_dir or node_def.get("type"):
            continue
        node_id = node_def.get("id", "")
        # 相对路径优先用 base_dir 解析，再 fallback 到 cwd
        raw = Path(agent_dir)
        if not raw.is_absolute() and base_dir:
            agent_json_path = Path(base_dir) / raw / "entity.json"
        else:
            agent_json_path = raw / "entity.json"
        if not agent_json_path.exists():
            continue
        try:
            # 父节点本地声明的 routing_hint 优先（允许每个父图为同一子图定制提示）
            hint = node_def.get("routing_hint") or ""
            if not hint:
                sub_json = json.loads(agent_json_path.read_text(encoding="utf-8"))
                hint = sub_json.get("routing_hint", "")
            if hint:
                hints.append(f'  - "{node_id}": {hint} <!-- [auto-injected from {agent_dir}/entity.json:routing_hint] -->')
        except Exception:
            continue

    if not hints:
        return ""

    lines = [
        "",
        "<!-- [auto-generated section: routing hints collected from subgraph nodes] -->",
        "[可调用子图]",
        "遇到以下情况时，可将任务委托给对应子图。",
        "路由方式：在回复的第一行且仅第一行输出以下 JSON（不加任何前缀或解释）：",
        '{"route": "<节点id>", "context": "<清晰描述议题和相关背景>"}',
        "",
        "可用子图：",
    ] + hints + [
        "",
        "注意：路由是重操作（多轮 LLM 调用），仅在真正有价值时使用，日常任务直接回复。",
    ]
    return "\n".join(lines)


# ── 向后兼容：旧代码 import register_state_schema / _get_state_schemas ──
# 已迁移到 framework/registry.py（register_schema / get_all_schemas），
# 保留别名避免 import 报错。
from framework.registry import register_schema as _register_schema, get_all_schemas


def register_state_schema(name: str, cls):
    """向后兼容别名 — 请改用 framework.registry.register_schema()。"""
    _register_schema(name, cls)


def _get_state_schemas() -> dict:
    """向后兼容别名 — 请改用 framework.registry.get_all_schemas()。"""
    import framework.schema  # noqa: F401 — 确保内置 schema 已注册
    return get_all_schemas()



def _get_subgraph_session_keys(entity_json: dict) -> set:
    """Extract all unique session_key values from a subgraph's entity.json graph spec.

    Each LLM node declares a ``session_key`` (defaults to its ``id``).
    Used by ``session_mode: inherit`` to know which keys to inject.
    """
    keys: set[str] = set()
    for nd in entity_json.get("graph", {}).get("nodes", []):
        sk = nd.get("session_key", nd.get("id", ""))
        if sk:
            keys.add(sk)
    return keys


async def _build_declarative(
    graph_spec: dict,
    config: AgentConfig,
    system_prompt: str,
    checkpointer,
    blueprint_dir: str = "",
    is_subgraph: bool = False,
    force_unique_session_keys: bool = False,
) -> object:
    """
    从 entity.json["graph"]（含 nodes + edges）构建 LangGraph 状态机。

    执行三步验证后再构建图，任意失败直接抛 ValueError。

    is_subgraph=True：子图模式，使用 SubgraphInputState 作为 input schema，
                      让 LangGraph 原生阻止父图的 messages 进入子图。
    """
    import framework.builtins  # 确保内置类型已注册

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph

    from framework.registry import get_condition, get_node_factory, get_all_schemas
    from framework.schema.base import BaseAgentState, SubgraphInputState

    # ── Step 0: routing_hint 注入 system_prompt ──────────────────────────────
    if system_prompt:
        routing_section = _collect_routing_hints(graph_spec, base_dir=blueprint_dir)
        if routing_section:
            system_prompt = system_prompt + "\n\n" + routing_section

    # ── Step 1: 三步图验证 + 主节点唯一性 ───────────────────────────────────
    all_ids = _collect_all_ids(graph_spec)
    _check_edge_refs(graph_spec, all_ids)
    _check_reachable(graph_spec, all_ids)
    persona_targets: list[str] = []
    if system_prompt:
        persona_targets = _find_persona_targets(graph_spec)
        if persona_targets:
            logger.info(f"[agent_loader] persona injection targets: {persona_targets}")
        else:
            logger.warning("[agent_loader] system_prompt provided but no LLM node found for injection")

    # ── Step 2: 构建图节点 ────────────────────────────────────────────────
    import framework.schema  # noqa: F401 — 确保内置 schema 已注册
    all_schemas = get_all_schemas()
    state_schema = all_schemas.get(graph_spec.get("state_schema", "base_schema"), BaseAgentState)

    # is_subgraph=True 且使用 base_schema → 子图模式，使用 SubgraphInputState 隔离父图 messages。
    # 自定义 schema（如 colony_coder_schema）含有 base_schema 以外的字段，
    # SubgraphInputState 不覆盖这些字段，不能应用 input 隔离（否则自定义字段会丢失）。
    _schema_name = graph_spec.get("state_schema", "base_schema")
    if is_subgraph and _schema_name == "base_schema":
        builder = StateGraph(state_schema, input_schema=SubgraphInputState)
    else:
        builder = StateGraph(state_schema)

    for node_def in graph_spec.get("nodes", []):
        node_id = node_def["id"]
        node_type = node_def.get("type", "")

        if node_type == "SUBGRAPH":
            # 内联子图：递归构建（无 checkpointer，父图负责；is_subgraph=True 隔离 messages）
            inner = await _build_declarative(
                node_def["graph"], config, system_prompt, None,
                blueprint_dir=blueprint_dir,
                is_subgraph=True,
            )
            builder.add_node(node_id, _wrap_node_for_flow_log(node_id, inner))

        elif node_def.get("agent_dir") and not node_type:
            # External agent subgraph — 纯原生 LangGraph 子图接入。
            # agent_dir 字段存在且无 type 声明时自动识别为外部子图。
            # output_field 由子图末尾 LLM 节点的 node_config["output_field"] 处理。
            # session_mode: persistent(default) | fresh_per_call | inherit | isolated

            raw_dir = node_def.get("agent_dir", "")
            if not raw_dir:
                raise ValueError(
                    f"subgraph node '{node_id}' must declare 'agent_dir'"
                )
            raw_path = Path(raw_dir)
            if raw_path.is_absolute():
                inner_dir = raw_path
            elif raw_path.resolve().exists():
                # Relative to CWD (e.g. "blueprints/functional_graphs/...")
                inner_dir = raw_path
            else:
                # Relative to blueprint_dir (e.g. "../colony_coder_executor")
                if not blueprint_dir:
                    raise ValueError(
                        f"external subgraph '{node_id}': relative agent_dir "
                        f"'{raw_dir}' requires a known blueprint_dir"
                    )
                inner_dir = Path(blueprint_dir) / raw_dir
            inner_dir = inner_dir.resolve()
            if not inner_dir.exists():
                raise ValueError(
                    f"external subgraph '{node_id}': agent_dir not found: {inner_dir}"
                )

            session_mode = node_def.get("session_mode", "persistent")
            inner_loader = EntityLoader(inner_dir)
            _pass_persona = system_prompt or ""

            _force_unique = session_mode == "isolated"
            inner_graph = await inner_loader.build_graph(
                checkpointer=None, parent_persona=_pass_persona, is_subgraph=True,
                force_unique_session_keys=_force_unique,
            )

            if session_mode == "persistent":
                # Default: LangGraph manages subgraph checkpoint namespace;
                # node_sessions persist across calls.
                builder.add_node(node_id, inner_graph)

            elif session_mode == "fresh_per_call":
                # Clear node_sessions before each invocation so every call
                # starts fresh LLM sessions. Wrapping in an async function
                # prevents LangGraph from creating a checkpoint namespace
                # for the subgraph (inner_graph has checkpointer=None).
                _inner = inner_graph  # capture for closure
                _nid = node_id

                async def _fresh_wrapper(state: dict, config=None, *, _g=_inner, _n=_nid) -> dict:
                    patched = {**state, "node_sessions": {}}
                    logger.debug(f"[session_mode:fresh_per_call] {_n}: clearing node_sessions")
                    return await _g.ainvoke(patched)

                builder.add_node(node_id, _fresh_wrapper)

            elif session_mode == "inherit":
                # Inject parent node's session ID into all subgraph session keys.
                inherit_from = node_def.get("inherit_from", "claude_main")
                _inner = inner_graph
                _nid = node_id
                _sub_keys = _get_subgraph_session_keys(inner_loader.json)

                async def _inherit_wrapper(
                    state: dict, config=None,
                    *, _g=_inner, _n=_nid, _from=inherit_from, _keys=_sub_keys,
                ) -> dict:
                    parent_sid = (state.get("node_sessions") or {}).get(_from, "")
                    injected = {sk: parent_sid for sk in _keys}
                    patched = {**state, "node_sessions": injected}
                    logger.debug(
                        f"[session_mode:inherit] {_n}: inheriting session "
                        f"from {_from!r} → {len(_keys)} subgraph keys"
                    )
                    return await _g.ainvoke(patched)

                builder.add_node(node_id, _inherit_wrapper)

            elif session_mode == "isolated":
                # Each node gets its own unique session key (forced at build time)
                # AND node_sessions is cleared per call.
                _inner = inner_graph
                _nid = node_id

                async def _isolated_wrapper(state: dict, config=None, *, _g=_inner, _n=_nid) -> dict:
                    patched = {**state, "node_sessions": {}}
                    logger.debug(f"[session_mode:isolated] {_n}: clearing node_sessions (unique keys)")
                    return await _g.ainvoke(patched)

                builder.add_node(node_id, _isolated_wrapper)

            else:
                raise ValueError(
                    f"subgraph '{node_id}': unknown session_mode {session_mode!r} "
                    f"(valid: persistent, fresh_per_call, inherit, isolated)"
                )

        else:
            factory = get_node_factory(node_type)
            # persona 注入目标节点
            _base = (
                {**node_def, "system_prompt": system_prompt}
                if system_prompt and node_id in persona_targets
                else dict(node_def)
            )
            if node_type == "DETERMINISTIC" and blueprint_dir and "agent_dir" not in node_def:
                _base["agent_dir"] = blueprint_dir
            # force_unique_session_keys: override session_key to node_id
            # so each node uses its own independent session (isolated mode).
            if force_unique_session_keys and _base.get("session_key"):
                _base["session_key"] = node_id
            effective_def = _base
            node_instance = factory(config, effective_def)
            builder.add_node(node_id, _wrap_node_for_flow_log(node_id, node_instance))

    # ── Step 3: 添加边 ────────────────────────────────────────────────────
    # Separate routing_to edges (native routing) from named conditions per source.
    _routing_targets: dict[str, dict[str, str]] = defaultdict(dict)  # src → {route_key: dst_node}
    _named_conds: dict[str, list[tuple[str, object, str]]] = defaultdict(list)

    for edge in graph_spec.get("edges", []):
        src = edge["from"]
        dst = edge["to"]
        edge_type = edge.get("type")
        dst_node = END if dst == "__end__" else dst
        max_retry = edge.get("max_retry")  # 仅供命名条件边

        if not edge_type:
            # 无条件边（__start__ 用 add_edge(START,...) 支持多入口 fan-out）
            if src == "__start__":
                builder.add_edge(START, dst_node)
            else:
                builder.add_edge(src, dst_node)

        elif edge_type == "routing_to":
            # Native LangGraph routing: router returns state["routing_target"]
            # directly, LangGraph maps to same-named node.
            # Route key = original dst string (e.g. "debate_brainstorm" or "__end__")
            _routing_targets[src][dst] = dst_node

        else:
            # 命名条件边 — 从 registry 查找（on_error, no_routing, etc.）
            fn = _maybe_limit(get_condition(edge_type), max_retry)
            _named_conds[src].append((edge_type, fn, dst_node))

    # 动态补 entry→START、exit→END（如 __start__/__end__ 边未声明）
    _graph_entry = graph_spec.get("entry")
    _graph_exit  = graph_spec.get("exit")
    _has_start_edge = any(e["from"] == "__start__" for e in graph_spec.get("edges", []))
    _has_end_edge   = any(e["to"]   == "__end__"   for e in graph_spec.get("edges", []))

    if _graph_entry and not _has_start_edge:
        builder.add_edge(START, _graph_entry)
        logger.debug(f"[agent_loader] dynamic entry: START → {_graph_entry!r}")

    if _graph_exit and not _has_end_edge:
        builder.add_edge(_graph_exit, END)
        logger.debug(f"[agent_loader] dynamic exit: {_graph_exit!r} → END")

    # 注册条件边（按 src 分组，每个 src 只调用一次 add_conditional_edges）
    all_cond_sources = set(_routing_targets) | set(_named_conds)

    for src in all_cond_sources:
        rt_map = _routing_targets.get(src, {})
        nc_list = _named_conds.get(src, [])

        route_map: dict[str, str] = {}
        priority_fns: dict[str, object] = {}  # named conditions checked before routing
        no_routing_dst = None

        for ckey, fn, dst in nc_list:
            if ckey == "no_routing" and rt_map:
                # When routing_to edges exist, no_routing serves as fallback
                # for empty routing_target → don't add as a checked condition
                no_routing_dst = dst
            else:
                route_map[ckey] = dst
                priority_fns[ckey] = fn

        # Add routing_to targets to route_map (key = target name, maps to itself)
        for route_key, dst in rt_map.items():
            route_map[route_key] = dst

        # Add fallback for empty routing_target
        if rt_map:
            if no_routing_dst is not None:
                route_map[""] = no_routing_dst
            else:
                route_map[""] = END  # default: no routing → end

        has_routing = bool(rt_map)

        def _make_router(p_fns, rmap, has_rt, src_id):
            def _router(state):
                # Priority: named conditions checked first (on_error, etc.)
                for ckey, fn in p_fns.items():
                    if fn(state):
                        log_graph_flow("route", src_id, f"{ckey} → {rmap[ckey]}")
                        return ckey
                if has_rt:
                    # Native routing: return routing_target value directly
                    target = state.get("routing_target", "")
                    if target and target in rmap:
                        log_graph_flow("route", src_id, f"routing → {target}")
                        return target
                    # Fallback: empty routing_target → no_routing / END
                    fallback_dst = rmap.get("", END)
                    log_graph_flow("route", src_id, f"no_routing → {fallback_dst}")
                    return ""
                # No routing edges — use first available key as fallback
                fallback_key = next(iter(rmap), "")
                log_graph_flow("route", src_id, f"fallback → {rmap.get(fallback_key)}")
                return fallback_key
            return _router

        builder.add_conditional_edges(
            src,
            _make_router(priority_fns, route_map, has_routing, src),
            route_map,
        )

    # ── Step 4: Checkpointer ──────────────────────────────────────────────
    # _DEFAULT  → create default SQLite checkpointer (top-level graphs)
    # None      → no checkpointer (subgraphs embedded in a parent graph)
    if checkpointer is _DEFAULT:
        db_path = os.path.abspath(config.db_path)
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=10000")
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

    return builder.compile(checkpointer=checkpointer or None)


# ---------------------------------------------------------------------------
# Debug: 节点流转日志包裹器
# ---------------------------------------------------------------------------

def _wrap_node_for_flow_log(node_id: str, node_fn):
    """
    包裹节点 callable，在 debug 模式下记录 enter/exit 事件。

    非 debug 模式下仅增加一次 is_debug() 检查，开销可忽略。
    支持 sync 和 async callable，以及 CompiledStateGraph（子图）。
    """
    import asyncio
    import inspect

    # CompiledStateGraph（子图）：有 ainvoke 但不是普通 callable
    # LangGraph 会调用它的 ainvoke，不需要包裹
    if hasattr(node_fn, "ainvoke") and not callable(getattr(node_fn, "__call__", None)):
        return node_fn

    if inspect.iscoroutinefunction(node_fn) or (
        hasattr(node_fn, "__call__") and inspect.iscoroutinefunction(node_fn.__call__)
    ):
        async def _async_wrapper(state):
            if is_debug():
                # 记录入口
                msg_count = len(state.get("messages", []))
                rt = state.get("routing_target", "")
                detail = f"msgs={msg_count}"
                if rt:
                    detail += f" routing_target={rt!r}"
                log_graph_flow("enter", node_id, detail)

            result = await node_fn(state)

            if is_debug():
                # 记录出口
                if isinstance(result, dict):
                    keys = sorted(k for k in result.keys() if result[k])
                    rt_out = result.get("routing_target", "")
                    detail = f"keys=[{', '.join(keys)}]"
                    if rt_out:
                        detail += f" → routing_target={rt_out!r}"
                    # 记录 state snapshot
                    log_state_snapshot(node_id, result, full_state=state)
                else:
                    detail = f"type={type(result).__name__}"
                log_graph_flow("exit", node_id, detail)

            return result
        return _async_wrapper
    else:
        def _sync_wrapper(state):
            if is_debug():
                msg_count = len(state.get("messages", []))
                log_graph_flow("enter", node_id, f"msgs={msg_count}")

            result = node_fn(state)

            if is_debug():
                if isinstance(result, dict):
                    keys = sorted(k for k in result.keys() if result[k])
                    # 记录 state snapshot
                    log_state_snapshot(node_id, result, full_state=state)
                    log_graph_flow("exit", node_id, f"keys=[{', '.join(keys)}]")
                else:
                    log_graph_flow("exit", node_id, f"type={type(result).__name__}")

            return result
        return _sync_wrapper


# ---------------------------------------------------------------------------
# 图构建前验证
# ---------------------------------------------------------------------------

def _collect_all_ids(graph_spec: dict, seen: set | None = None) -> set:
    """
    递归收集所有节点 ID（包括 SUBGRAPH 内部）。
    发现重复 ID 立即抛 ValueError。
    """
    if seen is None:
        seen = set()
    for node_def in graph_spec.get("nodes", []):
        nid = node_def.get("id")
        if not nid:
            raise ValueError(f"Node missing 'id': {node_def}")
        if nid in seen:
            raise ValueError(f"Duplicate node ID: {nid!r}")
        seen.add(nid)
        if node_def.get("type") == "SUBGRAPH":
            _collect_all_ids(node_def.get("graph", {}), seen)
    return seen


def _check_edge_refs(graph_spec: dict, all_ids: set) -> None:
    """验证所有边引用的节点 ID 存在（__start__ 和 __end__ 为合法虚节点）。"""
    valid_ids = all_ids | {"__start__", "__end__"}
    for edge in graph_spec.get("edges", []):
        for key in ("from", "to"):
            ref = edge.get(key)
            if ref not in valid_ids:
                raise ValueError(
                    f"Edge references unknown node: {ref!r} "
                    f"(known: {sorted(valid_ids)})"
                )

    # 验证 entry/exit 字段引用的节点存在
    entry = graph_spec.get("entry")
    exit_node = graph_spec.get("exit")
    if entry and entry not in all_ids:
        raise ValueError(f"'entry' references unknown node: {entry!r} (known: {sorted(all_ids)})")
    if exit_node and exit_node not in all_ids:
        raise ValueError(f"'exit' references unknown node: {exit_node!r} (known: {sorted(all_ids)})")


def _check_reachable(graph_spec: dict, all_ids: set) -> None:
    """BFS 验证所有节点可达。支持 __start__ 边和 entry 字段两种入口声明。"""
    adjacency: dict[str, set] = defaultdict(set)
    for edge in graph_spec.get("edges", []):
        adjacency[edge["from"]].add(edge["to"])

    # 收集所有起始点
    start_nodes: set[str] = {"__start__"}
    entry = graph_spec.get("entry")
    if entry:
        start_nodes.add(entry)

    visited = set(start_nodes)
    queue = []
    for s in start_nodes:
        queue.extend(adjacency.get(s, []))

    while queue:
        node = queue.pop()
        if node not in visited:
            visited.add(node)
            queue.extend(adjacency.get(node, []))

    unreachable = all_ids - visited - {"__end__"}
    if unreachable:
        raise ValueError(f"Unreachable nodes from start: {unreachable}")


_LLM_NODE_TYPES = frozenset({
    "CLAUDE_CLI", "CLAUDE_SDK", "GEMINI_CLI", "GEMINI_API",
    "OLLAMA", "LOCAL_VLLM",
})


def _find_persona_targets(graph_spec: dict) -> list[str]:
    """找到 persona/system_prompt 应注入的 LLM 节点 ID 列表。

    规则：
      1. 优先选取所有 id 含 'main' 的 LLM 节点（可多个）。
      2. 若无 'main' 节点，遍历图拓扑，选取离 __end__ 最近的 LLM 节点。
      3. 若图中无任何 LLM 节点，返回空列表（不注入）。
    """
    nodes = graph_spec.get("nodes", [])
    edges = graph_spec.get("edges", [])

    # 所有 LLM 节点
    llm_ids = {
        n["id"] for n in nodes
        if n.get("type", "") in _LLM_NODE_TYPES
    }
    if not llm_ids:
        return []

    # 规则 1：含 'main' 的 LLM 节点
    main_ids = [nid for nid in llm_ids if "main" in nid]
    if main_ids:
        return main_ids

    # 规则 2：拓扑排序，取离 __end__ 最近的 LLM 节点
    # 构建反向邻接表（从 __end__ 反向 BFS）
    reverse_adj: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        src = edge["from"]
        dst = edge["to"]
        reverse_adj[dst].append(src)

    # 从 __end__ 反向 BFS，找到第一个 LLM 节点
    from collections import deque
    queue = deque(["__end__"])
    visited: set[str] = set()
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current in llm_ids:
            return [current]
        for pred in reverse_adj.get(current, []):
            if pred not in visited:
                queue.append(pred)

    # 兜底：无法从 __end__ 回溯到 LLM 节点，取任意一个
    return [next(iter(llm_ids))]


# ---------------------------------------------------------------------------
# 自定义 Mermaid 拓扑渲染（供 build_topology_mermaid 使用）
# ---------------------------------------------------------------------------

# 节点类型 → Mermaid 括号形状 (open, close)
_MERMAID_SHAPES: dict[str, tuple[str, str]] = {
    "VALIDATE":      ("{{", "}}"),     # 六边形：决策/校验
    "GIT_SNAPSHOT":  ("(", ")"),       # 圆角：存储操作
    "GIT_ROLLBACK":  ("(", ")"),       # 圆角：存储操作
    "EXTERNAL_TOOL": ('(["', '"])'),   # 子程序框：外部工具
}


def _mermaid_id(prefix: str, raw: str) -> str:
    """
    给节点原始 ID 加前缀，用于避免子图内 ID 冲突。
    __start__ / __end__ 在加前缀时去掉双下划线，变成 {prefix}start / {prefix}end。
    """
    if raw in ("__start__", "__end__"):
        stripped = raw.strip("_")          # "__start__" → "start"
        return (prefix + stripped) if prefix else raw
    return (prefix + raw) if prefix else raw


def _mermaid_render(spec: dict, lines: list, indent: str, prefix: str) -> None:
    """
    递归将 graph_spec 的节点和边渲染为 Mermaid flowchart 行。

    prefix  — 子图内节点 ID 前缀（消除全局冲突）。顶层传空字符串。
    indent  — 当前缩进字符串。
    """
    edges = spec.get("edges", [])
    all_refs = {e.get("from") for e in edges} | {e.get("to") for e in edges}

    # ── 节点 ─────────────────────────────────────────────────────────────
    for node_def in spec.get("nodes", []):
        raw   = node_def["id"]
        ntype = node_def.get("type", "")
        full  = _mermaid_id(prefix, raw)

        if node_def.get("agent_dir") and not ntype:
            # 外部子图：agent_dir 存在且无 type（原 external subgraph）
            _mermaid_agent_ref(node_def, lines, indent, full, raw)
            continue

        if ntype == "SUBGRAPH":
            lines.append(f'{indent}subgraph {full}["{raw}"]')
            lines.append(f'{indent}  direction LR')
            _mermaid_render(node_def.get("graph", {}), lines, indent + "  ", full + "_")
            lines.append(f'{indent}end')
            continue

        open_, close_ = _MERMAID_SHAPES.get(ntype, ('["', '"]'))
        label = f"{raw}\\n{ntype}" if ntype else raw
        lines.append(f'{indent}{full}{open_}{label}{close_}')

    # ── __start__ / __end__ ───────────────────────────────────────────────
    if "__start__" in all_refs:
        sid = _mermaid_id(prefix, "__start__")
        lbl = "start" if not prefix else " "
        lines.append(f'{indent}{sid}(({lbl}))')
    if "__end__" in all_refs:
        eid = _mermaid_id(prefix, "__end__")
        lbl = "end" if not prefix else " "
        lines.append(f'{indent}{eid}(({lbl}))')

    # ── 边 ────────────────────────────────────────────────────────────────
    for edge in edges:
        src   = _mermaid_id(prefix, edge["from"])
        dst   = _mermaid_id(prefix, edge["to"])
        etype = edge.get("type", "")
        arrow = f" -->|{etype}| " if etype else " --> "
        lines.append(f"{indent}{src}{arrow}{dst}")


def _mermaid_agent_ref(
    node_def: dict, lines: list, indent: str, full_id: str, raw: str
) -> None:
    """
    将 external subgraph 节点展开为 Mermaid subgraph，递归加载外部 entity.json。
    agent_dir 路径相对于进程 CWD。
    """
    agent_dir_str = node_def.get("agent_dir", "")
    if not agent_dir_str:
        lines.append(f'{indent}subgraph {full_id}["{raw} ⚠ no agent_dir"]')
        lines.append(f'{indent}  {full_id}_err["⚠ agent_dir missing"]')
        lines.append(f'{indent}end')
        return

    sub_json_path = Path(agent_dir_str) / "entity.json"
    if not sub_json_path.exists():
        lines.append(f'{indent}subgraph {full_id}["{raw}\\n⚠ not found"]')
        lines.append(f'{indent}  {full_id}_err["⚠ {agent_dir_str}"]')
        lines.append(f'{indent}end')
        return

    try:
        sub_json   = json.loads(sub_json_path.read_text(encoding="utf-8"))
        agent_name = sub_json.get("name", Path(agent_dir_str).name)
        sub_spec   = sub_json.get("graph", {})
        sub_prefix = full_id + "_"
        lines.append(f'{indent}subgraph {full_id}["{raw}\\n({agent_name})"]')
        lines.append(f'{indent}  direction LR')
        _mermaid_render(sub_spec, lines, indent + "  ", sub_prefix)
        lines.append(f'{indent}end')
    except Exception as exc:
        lines.append(f'{indent}subgraph {full_id}["{raw} ⚠ load error"]')
        lines.append(f'{indent}  {full_id}_err["⚠ {str(exc)[:60]}"]')
        lines.append(f'{indent}end')
