"""
框架级 Agent 加载器 — framework/agent_loader.py

AgentLoader(agent_dir) 从 agent 目录的 agent.json 加载所有配置，
提供完整的图构建和控制器管理能力。

任何新 Agent 只需：
  1. 创建 agents/<name>/ 目录
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
from framework.debug import is_debug

logger = logging.getLogger(__name__)

# Sentinel: build_graph(checkpointer=_DEFAULT) → create SQLite checkpointer
#           build_graph(checkpointer=None)      → compile without any checkpointer
_DEFAULT = object()


class AgentLoader:
    """
    从 agent 目录加载配置，构建 LangGraph 状态机，管理控制器单例。

    用法：
        loader = AgentLoader(Path("agents/hani"))
        controller = await loader.get_controller()
        response = await controller.run("用户输入")
    """

    def __init__(self, agent_dir: Path):
        self._dir = Path(agent_dir).resolve()
        self._env_prefix = self._dir.name.upper()  # "hani" → "HANI"
        self._json: dict = json.loads(
            (self._dir / "agent.json").read_text(encoding="utf-8")
        )
        self._config: AgentConfig | None = None
        self._engine = None
        self._controller = None
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
                graph_dict, config, system_prompt, checkpointer
            )

        # Priority 3: GraphSpec 默认图
        from framework.claude.node import ClaudeNode
        from framework.gemini.node import GeminiNode
        from framework.graph import build_agent_graph, GraphSpec

        # 向后兼容旧 "vram_flush" 字段
        if "use_vram_flush" not in graph_dict:
            graph_dict["use_vram_flush"] = bool(self._json.get("vram_flush", False))
        spec = GraphSpec.from_dict(graph_dict)

        # ClaudeNode — model 从 agent.json 顶层 claude_model 字段读（向后兼容）
        claude_node_config = {**self._json}
        agent_node = ClaudeNode(config, node_config=claude_node_config, system_prompt=system_prompt)

        # GeminiNode — model 从 agent.json 顶层 gemini_model 字段读（向后兼容）
        gemini_node_config = {"model": self._json.get("gemini_model", "gemini-2.5-flash")}
        gemini_node = GeminiNode(config, node_config=gemini_node_config)

        logger.info(f"[agent_loader] building graph for {self.name!r}")
        if is_debug():
            logger.debug(f"[agent_loader] db={config.db_path!r}")

        return await build_agent_graph(
            config=config,
            agent_node=agent_node,
            gemini=gemini_node,
            checkpointer=checkpointer,
            spec=spec,
        )

    async def get_engine(self):
        """懒加载引擎单例（向后兼容，推荐改用 get_controller()）。"""
        if self._engine is None:
            self._engine = await self.build_graph()
        return self._engine

    async def get_controller(self):
        """懒加载 GraphController 单例（推荐接口）。"""
        if self._controller is None:
            from framework.graph_controller import GraphController
            graph = await self.get_engine()
            self._controller = GraphController(graph, self.session_mgr, self.load_config())
        return self._controller

    def invalidate_engine(self) -> None:
        """使引擎和控制器缓存失效（compact/reset 后调用）。"""
        self._engine = None
        self._controller = None
        logger.info(f"[agent_loader] engine invalidated for {self.name!r}")


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

async def _build_declarative(
    graph_spec: dict,
    config: AgentConfig,
    system_prompt: str,
    checkpointer,
) -> object:
    """
    从 agent.json["graph"]（含 nodes + edges）构建 LangGraph 状态机。

    执行三步验证后再构建图，任意失败直接抛 ValueError。
    """
    import framework.builtins  # 确保内置类型已注册

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, StateGraph

    from framework.registry import get_condition, get_node_factory
    from framework.state import BaseAgentState, DebateState

    # ── Step 1: 三步图验证 + 主节点唯一性 ───────────────────────────────────
    all_ids = _collect_all_ids(graph_spec)
    _check_edge_refs(graph_spec, all_ids)
    _check_reachable(graph_spec, all_ids)
    if system_prompt:  # 仅有 persona 需要注入时才要求恰好一个主节点
        _check_single_main(graph_spec)

    # ── Step 2: 构建图节点 ────────────────────────────────────────────────
    _STATE_SCHEMAS = {"debate": DebateState, "base": BaseAgentState}
    state_schema = _STATE_SCHEMAS.get(graph_spec.get("state_schema", "base"), BaseAgentState)
    builder = StateGraph(state_schema)

    for node_def in graph_spec.get("nodes", []):
        node_id = node_def["id"]
        node_type = node_def.get("type", "")

        if node_type == "SUBGRAPH":
            # 内联子图：递归构建（无 checkpointer，父图负责）
            inner = await _build_declarative(
                node_def["graph"], config, system_prompt, None
            )
            builder.add_node(node_id, inner)
        else:
            factory = get_node_factory(node_type)
            # 仅主节点（ID 含 "main"）注入 system_prompt；其他节点忽略
            effective_def = (
                {**node_def, "system_prompt": system_prompt}
                if system_prompt and "main" in node_id
                else node_def
            )
            node_instance = factory(config, effective_def)
            builder.add_node(node_id, node_instance)

    # ── Step 3: 添加边 ────────────────────────────────────────────────────
    # conditional_edges[src] = list of (cond_key, fn, dst_node)
    conditional_edges: dict[str, list[tuple[str, object, str]]] = defaultdict(list)

    for edge in graph_spec.get("edges", []):
        src = edge["from"]
        dst = edge["to"]
        edge_type = edge.get("type")
        dst_node = END if dst == "__end__" else dst
        max_retry = edge.get("max_retry")

        if not edge_type:
            # 无条件边
            if src == "__start__":
                builder.set_entry_point(dst_node)
            else:
                builder.add_edge(src, dst_node)

        elif edge_type == "routing_to":
            # 参数化路由边：自动生成条件，检查 state["routing_target"] == dst
            target = dst_node  # 闭包捕获

            def _make_routing_fn(t):
                return lambda state: state.get("routing_target") == t

            fn = _maybe_limit(_make_routing_fn(target), max_retry)
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

        def _make_router(fn_map, rmap):
            def _router(state):
                for ckey, fn in fn_map.items():
                    if fn(state):
                        return ckey
                return next(iter(rmap))
            return _router

        builder.add_conditional_edges(
            src,
            _make_router(fns, route_map),
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
