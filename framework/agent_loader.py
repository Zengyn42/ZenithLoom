"""
框架级实体加载器 — framework/agent_loader.py

EntityLoader(blueprint_dir, data_dir) 从 blueprint 目录的 agent.json 加载角色定义，
从 data_dir 的 entity.json 加载实例专属配置，
提供完整的图构建和控制器管理能力。

任何新 Agent 只需：
  1. 创建 blueprints/role_agents/<name>/ 目录
  2. 编写 agent.json（含 llm、persona_files、tool_rules 等）
  3. 放置 persona .md 文件
  4. 可选：编写 graph.py（完全自定义图）

图构建优先级：
  Priority 1: agent 目录下的 graph.py（定义 build_graph(loader, checkpointer)）
  Priority 2: agent.json["graph"] 含 "nodes" 和 "edges" → 声明式图
  Priority 3: agent.json["graph"] 含 GraphSpec 标志 → 框架默认图

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
            (self._dir / "agent.json").read_text(encoding="utf-8")
        )
        self._config: AgentConfig | None = None
        self._engine = None
        self._controller = None
        self._session_mgr = None

        if is_debug():
            logger.debug(
                f"[agent_loader] dir={self._dir} data_dir={self._data_dir} prefix={self._env_prefix}"
            )

    def _resolve_env_prefix(self) -> str:
        """Derive env prefix from entity.json["name"] if available, else dir name."""
        entity_path = self._data_dir / "entity.json"
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
        # Try entity.json directly (instance-specific name)
        entity_path = self._data_dir / "entity.json"
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
            blueprint_path = self._dir / "agent.json"
            entity_path = self._data_dir / "entity.json"
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

    async def build_graph(self, checkpointer=_DEFAULT):
        """
        构建并返回编译好的 LangGraph 状态机。

        优先级：
          1. agent 目录下的 graph.py（定义 build_graph(loader, checkpointer)）
          2. agent.json["graph"] 含 "nodes" + "edges" → 声明式图
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
        graph_dict = dict(self._json.get("graph") or {})

        # Priority 2: 声明式图
        if "nodes" in graph_dict and "edges" in graph_dict:
            logger.info(f"[agent_loader] building declarative graph for {self.name!r}")
            return await _build_declarative(
                graph_dict, config, system_prompt, checkpointer,
                blueprint_dir=str(self._dir),
            )

        # Priority 3: GraphSpec 默认图
        from framework.nodes.llm.claude import ClaudeSDKNode
        from framework.graph import build_agent_graph, GraphSpec

        # 向后兼容旧 "vram_flush" 字段
        if "use_vram_flush" not in graph_dict:
            graph_dict["use_vram_flush"] = bool(self._json.get("vram_flush", False))
        spec = GraphSpec.from_dict(graph_dict)

        # ClaudeSDKNode — model 从 agent.json 顶层 claude_model 字段读（向后兼容）
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

    def invalidate_engine(self) -> None:
        """使引擎和控制器缓存失效（compact/reset 后调用）。"""
        self._engine = None
        self._controller = None
        logger.info(f"[agent_loader] engine invalidated for {self.name!r}")

    def build_topology_mermaid(self) -> str:
        """
        从 agent.json 构建 Mermaid 拓扑图，正确展开 SUBGRAPH_REF / AGENT_REF 子图。

        弥补 LangGraph get_graph(xray=True) 无法展开 AgentRefNode 的缺陷：
        AgentRefNode 是普通 callable，xray 只展开 native CompiledStateGraph 节点。
        """
        graph_spec = self._json.get("graph", {})
        lines = ["flowchart LR"]
        _mermaid_render(graph_spec, lines, "  ", "")
        return "\n".join(lines)

    async def build_heartbeat_graph(self) -> tuple:
        """
        构建心跳调度图（无 checkpointer，每次 invocation 独立）。

        返回 (compiled_graph, hb_cfg)；若 heartbeat 未配置或无 graph 定义则返回 (None, {})。
        """
        hb_cfg = self._json.get("heartbeat")
        if not hb_cfg or not isinstance(hb_cfg, dict):
            return None, {}
        graph_spec = hb_cfg.get("graph")
        if not graph_spec or "nodes" not in graph_spec or "edges" not in graph_spec:
            return None, hb_cfg

        config = self.load_config()
        logger.info(f"[agent_loader] building heartbeat graph for {self.name!r}")
        # system_prompt="" → 跳过 persona 注入和 _check_single_main 验证
        # checkpointer=None → 心跳图本身无需持久化（各 AGENT_RUN 节点自有 DB）
        graph = await _build_declarative(
            graph_spec, config, system_prompt="", checkpointer=None,
            blueprint_dir=str(self._dir),
        )
        return graph, hb_cfg


# 向后兼容别名
AgentLoader = EntityLoader


# ---------------------------------------------------------------------------
# 通用工具：条件边限速器（module-level，供测试直接导入）
# ---------------------------------------------------------------------------

def _maybe_limit(fn, max_retry):
    """
    包装条件函数，当 state["consult_count"] >= max_retry 时强制返回 False。
    max_retry=None 时直接返回原函数（不限速）。
    """
    if max_retry is None:
        return fn
    def _limited(state):
        return fn(state) and state.get("consult_count", 0) < max_retry
    return _limited


# ---------------------------------------------------------------------------
# 声明式图构建（Priority 2）
# ---------------------------------------------------------------------------

def _collect_routing_hints(graph_spec: dict) -> str:
    """
    遍历 graph_spec 中所有 SUBGRAPH_REF / AGENT_REF 节点，读取其 agent.json 的 routing_hint 字段，
    构建路由说明字符串，用于注入主节点 system_prompt。
    """
    hints: list[str] = []
    for node_def in graph_spec.get("nodes", []):
        if node_def.get("type") not in ("SUBGRAPH_REF", "AGENT_REF"):
            continue
        node_id = node_def.get("id", "")
        agent_dir = node_def.get("agent_dir", "")
        if not agent_dir:
            continue
        agent_json_path = Path(agent_dir) / "agent.json"
        if not agent_json_path.exists():
            continue
        try:
            sub_json = json.loads(agent_json_path.read_text(encoding="utf-8"))
            hint = sub_json.get("routing_hint", "")
            if hint:
                hints.append(f'  - "{node_id}": {hint} <!-- [auto-injected from {agent_dir}/agent.json:routing_hint] -->')
        except Exception:
            continue

    if not hints:
        return ""

    lines = [
        "",
        "<!-- [auto-generated section: routing hints collected from AGENT_REF nodes] -->",
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


async def _build_declarative(
    graph_spec: dict,
    config: AgentConfig,
    system_prompt: str,
    checkpointer,
    blueprint_dir: str = "",
) -> object:
    """
    从 agent.json["graph"]（含 nodes + edges）构建 LangGraph 状态机。

    执行三步验证后再构建图，任意失败直接抛 ValueError。
    """
    import framework.builtins  # 确保内置类型已注册

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph

    from framework.registry import get_condition, get_node_factory, get_all_schemas
    from framework.schema.base import BaseAgentState

    # ── Step 0: routing_hint 注入 system_prompt ──────────────────────────────
    if system_prompt:
        routing_section = _collect_routing_hints(graph_spec)
        if routing_section:
            system_prompt = system_prompt + "\n\n" + routing_section

    # ── Step 1: 三步图验证 + 主节点唯一性 ───────────────────────────────────
    all_ids = _collect_all_ids(graph_spec)
    _check_edge_refs(graph_spec, all_ids)
    _check_reachable(graph_spec, all_ids)
    if system_prompt:  # 仅有 persona 需要注入时才要求恰好一个主节点
        _check_single_main(graph_spec)

    # ── Step 2: 构建图节点 ────────────────────────────────────────────────
    import framework.schema  # noqa: F401 — 确保内置 schema 已注册
    all_schemas = get_all_schemas()
    state_schema = all_schemas.get(graph_spec.get("state_schema", "base_schema"), BaseAgentState)
    builder = StateGraph(state_schema)

    for node_def in graph_spec.get("nodes", []):
        node_id = node_def["id"]
        node_type = node_def.get("type", "")

        if node_type == "SUBGRAPH":
            # 内联子图：递归构建（无 checkpointer，父图负责）
            inner = await _build_declarative(
                node_def["graph"], config, system_prompt, None,
                blueprint_dir=blueprint_dir,
            )
            builder.add_node(node_id, _wrap_node_for_flow_log(node_id, inner))
        else:
            factory = get_node_factory(node_type)
            # 仅主节点（ID 含 "main"）注入 system_prompt；其他节点忽略
            _base = (
                {**node_def, "system_prompt": system_prompt}
                if system_prompt and "main" in node_id
                else dict(node_def)
            )
            if node_type == "DETERMINISTIC" and blueprint_dir:
                _base["agent_dir"] = blueprint_dir
            effective_def = _base
            node_instance = factory(config, effective_def)
            builder.add_node(node_id, _wrap_node_for_flow_log(node_id, node_instance))

    # ── Step 3: 添加边 ────────────────────────────────────────────────────
    # conditional_edges[src] = list of (cond_key, fn, dst_node)
    conditional_edges: dict[str, list[tuple[str, object, str]]] = defaultdict(list)

    for edge in graph_spec.get("edges", []):
        src = edge["from"]
        dst = edge["to"]
        edge_type = edge.get("type")
        dst_node = END if dst == "__end__" else dst
        max_retry = edge.get("max_retry")  # 仅供非 routing_to 的条件边向后兼容

        if not edge_type:
            # 无条件边（__start__ 用 add_edge(START,...) 支持多入口 fan-out）
            if src == "__start__":
                builder.add_edge(START, dst_node)
            else:
                builder.add_edge(src, dst_node)

        elif edge_type == "routing_to":
            # 参数化路由边：仅匹配 routing_target，限速由 SubgraphRefNode 自行处理并回报
            target = dst_node  # 闭包捕获

            def _make_routing_fn(t):
                return lambda state: state.get("routing_target") == t

            fn = _make_routing_fn(target)
            cond_key = edge.get("id") or f"routing_to_{dst}"
            conditional_edges[src].append((cond_key, fn, dst_node))

        else:
            # 命名条件边 — 从 registry 查找
            fn = _maybe_limit(get_condition(edge_type), max_retry)
            conditional_edges[src].append((edge_type, fn, dst_node))

    # 注册条件边（按 src 分组，每个 src 只调用一次 add_conditional_edges）
    for src, cond_list in conditional_edges.items():
        route_map: dict[str, str] = {ckey: dnode for ckey, _, dnode in cond_list}
        fns: dict[str, object] = {ckey: fn for ckey, fn, _ in cond_list}

        def _make_router(fn_map, rmap, src_id):
            # Prefer no_routing key as fallback (maps to __end__); else first key
            _no_routing_key = next(
                (k for k, v in rmap.items() if k == "no_routing"), next(iter(rmap))
            )
            def _router(state):
                for ckey, fn in fn_map.items():
                    if fn(state):
                        dst = rmap.get(ckey, ckey)
                        log_graph_flow("route", src_id, f"{ckey} → {dst}")
                        return ckey
                dst = rmap.get(_no_routing_key, _no_routing_key)
                log_graph_flow("route", src_id, f"fallback({_no_routing_key}) → {dst}")
                return _no_routing_key
            return _router

        builder.add_conditional_edges(
            src,
            _make_router(fns, route_map, src),
            route_map,
        )

    # ── Step 4: Checkpointer ──────────────────────────────────────────────
    # _DEFAULT  → create default SQLite checkpointer (top-level graphs)
    # None      → no checkpointer (subgraphs embedded in a parent graph)
    if checkpointer is _DEFAULT:
        db_path = os.path.abspath(config.db_path)
        conn = await aiosqlite.connect(db_path)
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


def _check_reachable(graph_spec: dict, all_ids: set) -> None:
    """BFS 从 __start__ 出发，验证所有节点均可达。"""
    adjacency: dict[str, set] = defaultdict(set)
    for edge in graph_spec.get("edges", []):
        adjacency[edge["from"]].add(edge["to"])

    visited = {"__start__"}
    queue = list(adjacency["__start__"])
    while queue:
        node = queue.pop()
        if node not in visited:
            visited.add(node)
            queue.extend(adjacency.get(node, []))

    unreachable = all_ids - visited - {"__end__"}
    if unreachable:
        raise ValueError(f"Unreachable nodes from __start__: {unreachable}")


def _check_single_main(graph_spec: dict) -> None:
    """验证图中恰好存在一个 ID 含 'main' 的主节点（persona 注入点，不可多于一个）。"""
    main_ids = [
        node_def["id"]
        for node_def in graph_spec.get("nodes", [])
        if "main" in node_def.get("id", "")
    ]
    if len(main_ids) != 1:
        raise ValueError(
            f"Graph must have exactly 1 main agent node (id containing 'main'), "
            f"found {len(main_ids)}: {main_ids}"
        )


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

        if ntype in ("SUBGRAPH_REF", "AGENT_REF"):
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
    将 SUBGRAPH_REF / AGENT_REF 节点展开为 Mermaid subgraph，递归加载外部 agent.json。
    agent_dir 路径相对于进程 CWD（与 AgentRefNode 运行时行为一致）。
    """
    agent_dir_str = node_def.get("agent_dir", "")
    if not agent_dir_str:
        lines.append(f'{indent}subgraph {full_id}["{raw} ⚠ no agent_dir"]')
        lines.append(f'{indent}  {full_id}_err["⚠ agent_dir missing"]')
        lines.append(f'{indent}end')
        return

    sub_json_path = Path(agent_dir_str) / "agent.json"
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
