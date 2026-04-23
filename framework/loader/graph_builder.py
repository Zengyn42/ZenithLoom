"""
Declarative LangGraph graph builder.

Handles Priority 2 graph construction: entity.json["graph"] with "nodes" + "edges".
Also owns the debug node-wrapping layer (_wrap_node_for_flow_log).
"""

import asyncio
import inspect
import logging
import os
from collections import defaultdict
from pathlib import Path

from framework.config import AgentConfig
from framework.debug import is_debug, log_graph_flow, log_state_snapshot
from framework.loader.graph_validator import _collect_all_ids, _check_edge_refs, _check_reachable
from framework.loader.persona import _collect_routing_hints, _load_persona_text

logger = logging.getLogger(__name__)

# Sentinel: _build_declarative(checkpointer=_DEFAULT) → create SQLite checkpointer
#           _build_declarative(checkpointer=None)      → compile without checkpointer
# Also imported by entity_loader so both sides share the same object identity.
_DEFAULT = object()

_LLM_NODE_TYPES = frozenset({
    "CLAUDE_CLI", "CLAUDE_SDK", "GEMINI_CLI", "GEMINI_API",
    "OLLAMA", "LOCAL_VLLM",
})


def _maybe_limit(fn, max_retry):
    """向后兼容包装器。max_retry 参数已废弃，直接返回原函数。"""
    return fn


def _extract_session_keys_from_json(entity_json: dict) -> list[str]:
    """Extract session_key values from all LLM nodes in entity.json graph spec."""
    graph = entity_json.get("graph", {})
    keys = []
    for node in graph.get("nodes", []):
        ntype = node.get("type", "")
        if ntype in _LLM_NODE_TYPES:
            keys.append(node.get("session_key", node["id"]))
    return keys


# ---------------------------------------------------------------------------
# Debug: node flow log wrapper
# ---------------------------------------------------------------------------

def _wrap_node_for_flow_log(node_id: str, node_fn):
    """
    包裹节点 callable，在 debug 模式下记录 enter/exit 事件。

    非 debug 模式下仅增加一次 is_debug() 检查，开销可忽略。
    支持 sync 和 async callable，以及 CompiledStateGraph（子图）。
    """
    # CompiledStateGraph: has ainvoke but __call__ is not a plain function.
    # LangGraph invokes it via ainvoke directly — no wrapping needed.
    if hasattr(node_fn, "ainvoke") and not callable(getattr(node_fn, "__call__", None)):
        return node_fn

    if inspect.iscoroutinefunction(node_fn) or (
        hasattr(node_fn, "__call__") and inspect.iscoroutinefunction(node_fn.__call__)
    ):
        async def _async_wrapper(state):
            if is_debug():
                msg_count = len(state.get("messages", []))
                rt = state.get("routing_target", "")
                detail = f"msgs={msg_count}"
                if rt:
                    detail += f" routing_target={rt!r}"
                log_graph_flow("enter", node_id, detail)

            result = await node_fn(state)

            if is_debug():
                if isinstance(result, dict):
                    keys = sorted(k for k in result.keys() if result[k])
                    rt_out = result.get("routing_target", "")
                    detail = f"keys=[{', '.join(keys)}]"
                    if rt_out:
                        detail += f" → routing_target={rt_out!r}"
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
                    log_state_snapshot(node_id, result, full_state=state)
                    log_graph_flow("exit", node_id, f"keys=[{', '.join(keys)}]")
                else:
                    log_graph_flow("exit", node_id, f"type={type(result).__name__}")

            return result
        return _sync_wrapper


# ---------------------------------------------------------------------------
# Declarative graph builder (Priority 2)
# ---------------------------------------------------------------------------

async def _build_declarative(
    graph_spec: dict,
    config: AgentConfig,
    checkpointer,
    blueprint_dir: str = "",
    is_subgraph: bool = False,
    force_unique_session_keys: bool = False,
    extra_persona_text: str = "",
    session_mode: str = "fresh_per_call",
    fresh_keep_fields: list[str] | None = None,
) -> object:
    """
    从 entity.json["graph"]（含 nodes + edges）构建 LangGraph 状态机。

    执行三步验证后再构建图，任意失败直接抛 ValueError。

    is_subgraph=True：子图模式，使用 SubgraphInputState 作为 input schema，
                      让 LangGraph 原生阻止父图的 messages 进入子图。
    """
    import framework.builtins  # noqa: F401 — 确保内置类型已注册

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, START, StateGraph

    from framework.registry import get_condition, get_node_factory, get_all_schemas
    from framework.schema.base import BaseAgentState, SubgraphInputState

    # ── Step 0: routing_hint collection ──────────────────────────────────────
    routing_section = _collect_routing_hints(graph_spec, base_dir=blueprint_dir)

    # ── Step 1: three-pass graph validation ──────────────────────────────────
    all_ids = _collect_all_ids(graph_spec)
    _check_edge_refs(graph_spec, all_ids)
    _check_reachable(graph_spec, all_ids)

    # ── Step 2: build nodes ───────────────────────────────────────────────────
    import framework.schema  # noqa: F401 — 确保内置 schema 已注册

    _schema_name = graph_spec.get("state_schema", "base_schema")
    if _schema_name != "base_schema" and blueprint_dir:
        _state_py = Path(blueprint_dir) / "state.py"
        if _state_py.exists() and _schema_name not in get_all_schemas():
            import importlib.util as _ilu
            import sys as _sys
            _mod_name = f"_schema_{_schema_name}"
            _spec = _ilu.spec_from_file_location(_mod_name, _state_py)
            _mod = _ilu.module_from_spec(_spec)
            # Register BEFORE exec_module so typing.get_type_hints() can resolve
            # forward references stored as strings by `from __future__ import annotations`.
            _sys.modules[_mod_name] = _mod
            _spec.loader.exec_module(_mod)
            logger.debug(f"[graph_builder] auto-imported state.py for schema {_schema_name!r}")

    all_schemas = get_all_schemas()
    state_schema = all_schemas.get(_schema_name, BaseAgentState)

    # Input isolation strategy:
    #   fresh_per_call: _subgraph_init manages messages, so parent messages must flow in.
    #   Other modes: SubgraphInputState blocks parent messages/output from entering subgraph.
    if is_subgraph and session_mode != "fresh_per_call":
        builder = StateGraph(state_schema, input_schema=SubgraphInputState)
        logger.debug(f"[graph_builder] SubgraphInputState applied (session_mode={session_mode})")
    else:
        builder = StateGraph(state_schema)

    _llm_node_instances: dict[str, object] = {}

    for node_def in graph_spec.get("nodes", []):
        node_id = node_def["id"]
        node_type = node_def.get("type", "")

        if node_type == "SUBGRAPH":
            inner = await _build_declarative(
                node_def["graph"], config, None,
                blueprint_dir=blueprint_dir,
                is_subgraph=True,
                extra_persona_text=extra_persona_text,
            )
            builder.add_node(node_id, _wrap_node_for_flow_log(node_id, inner))

        elif node_def.get("agent_dir") and not node_type:
            # External agent subgraph — lazy import avoids circular dependency.
            from framework.loader.entity_loader import EntityLoader

            raw_dir = node_def.get("agent_dir", "")
            if not raw_dir:
                raise ValueError(f"subgraph node '{node_id}' must declare 'agent_dir'")

            raw_path = Path(raw_dir)
            if raw_path.is_absolute():
                inner_dir = raw_path
            elif raw_path.resolve().exists():
                inner_dir = raw_path
            else:
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

            session_mode = node_def.get("session_mode", "fresh_per_call")
            inner_loader = EntityLoader(inner_dir)

            _child_extra_persona = ""
            _ep_def = node_def.get("extra_persona")
            if isinstance(_ep_def, dict):
                _bp_path = Path(blueprint_dir) if blueprint_dir else Path(".")
                _child_extra_persona = _load_persona_text(
                    _ep_def.get("persona_files", []),
                    _bp_path,
                    prompt=_ep_def.get("prompt", ""),
                    label=_bp_path.name,
                )
            elif isinstance(_ep_def, bool) and _ep_def:
                raise ValueError(
                    f"子图节点 '{node_id}': extra_persona:true 只能用于 LLM 节点，"
                    f"子图节点应使用 extra_persona: {{persona_files: [...], prompt: '...'}} 格式。"
                )

            _force_unique = session_mode == "isolated"
            _keep_fields = node_def.get("fresh_keep_fields")
            inner_graph = await inner_loader.build_graph(
                checkpointer=None, extra_persona_text=_child_extra_persona, is_subgraph=True,
                force_unique_session_keys=_force_unique,
                session_mode=session_mode,
                fresh_keep_fields=_keep_fields,
            )

            if session_mode in ("persistent", "fresh_per_call", "inherit", "isolated"):
                builder.add_node(node_id, inner_graph)
            else:
                raise ValueError(
                    f"subgraph '{node_id}': unknown session_mode {session_mode!r} "
                    f"(valid: persistent, fresh_per_call, inherit, isolated)"
                )

        else:
            factory = get_node_factory(node_type)
            _base = dict(node_def)

            if node_type in _LLM_NODE_TYPES:
                _node_extra = ""
                _ep_flag = node_def.get("extra_persona", False)

                if _ep_flag and not isinstance(_ep_flag, bool):
                    raise ValueError(
                        f"节点 '{node_id}': extra_persona 在 LLM 节点上必须是 bool，"
                        f"传入了 {type(_ep_flag).__name__}。"
                        f"（dict 格式仅用于子图节点声明中）"
                    )

                if _ep_flag and extra_persona_text:
                    _node_extra = extra_persona_text

                _node_pfiles = node_def.get("persona_files", [])
                _bp_dir = Path(blueprint_dir) if blueprint_dir else Path(".")
                _node_persona = _load_persona_text(_node_pfiles, _bp_dir, label=_bp_dir.name)

                _node_sys = node_def.get("system_prompt", "")
                if routing_section and _ep_flag:
                    _node_sys = (_node_sys + "\n\n" + routing_section).strip() if _node_sys else routing_section

                _assembled = "\n\n---\n\n".join(
                    p for p in [_node_extra, _node_persona, _node_sys] if p
                )
                if _assembled:
                    _base["system_prompt"] = _assembled

            if node_type == "DETERMINISTIC" and blueprint_dir and "agent_dir" not in node_def:
                _base["agent_dir"] = blueprint_dir
            if force_unique_session_keys and _base.get("session_key"):
                _base["session_key"] = node_id
            if is_subgraph:
                _base["_is_subgraph"] = True
            node_instance = factory(config, _base)
            _llm_node_instances[node_id] = node_instance
            builder.add_node(node_id, _wrap_node_for_flow_log(node_id, node_instance))

    # ── Pre-compute entry/exit metadata ──────────────────────────────────────
    _graph_entry = graph_spec.get("entry")
    _graph_exit  = graph_spec.get("exit")
    _has_start_edge = any(e["from"] == "__start__" for e in graph_spec.get("edges", []))
    _has_end_edge   = any(e["to"]   == "__end__"   for e in graph_spec.get("edges", []))

    _needs_init = is_subgraph and session_mode in ("fresh_per_call", "isolated")
    _needs_exit = is_subgraph

    # Exit intercept: redirect __end__ edges through _subgraph_exit
    _exit_intercept = False
    if _needs_exit and _has_end_edge and not _graph_exit:
        from framework.nodes.subgraph_init_node import make_subgraph_exit
        _subgraph_skeys = [
            n.get("session_key", n["id"])
            for n in graph_spec.get("nodes", [])
            if n.get("type", "") in _LLM_NODE_TYPES
        ]
        _exit_fn = make_subgraph_exit(session_mode=session_mode, subgraph_session_keys=_subgraph_skeys)
        builder.add_node("_subgraph_exit", _exit_fn)
        _exit_intercept = True
        logger.debug(
            f"[graph_builder] exit_intercept: _subgraph_exit created, "
            f"redirecting __end__ edges (mode={session_mode}, keys={_subgraph_skeys})"
        )

    # ── Step 3: add edges ────────────────────────────────────────────────────
    _routing_targets: dict[str, dict[str, str]] = defaultdict(dict)
    _named_conds: dict[str, list[tuple[str, object, str]]] = defaultdict(list)

    for edge in graph_spec.get("edges", []):
        src = edge["from"]
        dst = edge["to"]
        edge_type = edge.get("type")
        if dst == "__end__" and _exit_intercept:
            dst_node = "_subgraph_exit"
        else:
            dst_node = END if dst == "__end__" else dst
        max_retry = edge.get("max_retry")

        if not edge_type:
            if src == "__start__":
                builder.add_edge(START, dst_node)
            else:
                builder.add_edge(src, dst_node)
        elif edge_type == "routing_to":
            _routing_targets[src][dst] = dst_node
        else:
            fn = _maybe_limit(get_condition(edge_type), max_retry)
            _named_conds[src].append((edge_type, fn, dst_node))

    if _exit_intercept:
        builder.add_edge("_subgraph_exit", END)
        logger.debug("[graph_builder] exit_intercept: _subgraph_exit → END")

    # ── Entry side ────────────────────────────────────────────────────────────
    if _needs_init and _graph_entry and not _has_start_edge:
        from framework.nodes.subgraph_init_node import make_subgraph_init
        _init_fn = make_subgraph_init(session_mode, keep_fields=fresh_keep_fields)
        builder.add_node("_subgraph_init", _init_fn)
        builder.add_edge(START, "_subgraph_init")
        builder.add_edge("_subgraph_init", _graph_entry)
        logger.debug(f"[graph_builder] subgraph_init injected: START → _subgraph_init → {_graph_entry!r} (mode={session_mode})")
    elif _graph_entry and not _has_start_edge:
        builder.add_edge(START, _graph_entry)
        logger.debug(f"[graph_builder] dynamic entry: START → {_graph_entry!r}")

    # ── Exit side (explicit exit node, no __end__ edges) ─────────────────────
    if _needs_exit and _graph_exit and not _has_end_edge and not _exit_intercept:
        from framework.nodes.subgraph_init_node import make_subgraph_exit
        _subgraph_skeys = [
            n.get("session_key", n["id"])
            for n in graph_spec.get("nodes", [])
            if n.get("type", "") in _LLM_NODE_TYPES
        ]
        _exit_fn = make_subgraph_exit(session_mode=session_mode, subgraph_session_keys=_subgraph_skeys)
        builder.add_node("_subgraph_exit", _exit_fn)
        builder.add_edge(_graph_exit, "_subgraph_exit")
        builder.add_edge("_subgraph_exit", END)
        logger.debug(f"[graph_builder] subgraph_exit injected: {_graph_exit!r} → _subgraph_exit → END")
    elif _graph_exit and not _has_end_edge:
        builder.add_edge(_graph_exit, END)
        logger.debug(f"[graph_builder] dynamic exit: {_graph_exit!r} → END")

    # ── Conditional edges ─────────────────────────────────────────────────────
    all_cond_sources = set(_routing_targets) | set(_named_conds)

    for src in all_cond_sources:
        rt_map = _routing_targets.get(src, {})
        nc_list = _named_conds.get(src, [])

        route_map: dict[str, str] = {}
        priority_fns: dict[str, object] = {}
        no_routing_dst = None

        for ckey, fn, dst in nc_list:
            if ckey == "no_routing" and rt_map:
                no_routing_dst = dst
            else:
                route_map[ckey] = dst
                priority_fns[ckey] = fn

        for route_key, dst in rt_map.items():
            route_map[route_key] = dst

        if rt_map:
            if no_routing_dst is not None:
                route_map[""] = no_routing_dst
            else:
                route_map[""] = END

        has_routing = bool(rt_map)

        def _make_router(p_fns, rmap, has_rt, src_id):
            def _router(state):
                for ckey, fn in p_fns.items():
                    if fn(state):
                        log_graph_flow("route", src_id, f"{ckey} → {rmap[ckey]}")
                        return ckey
                if has_rt:
                    target = state.get("routing_target", "")
                    if target and target in rmap:
                        log_graph_flow("route", src_id, f"routing → {target}")
                        return target
                    fallback_dst = rmap.get("", END)
                    log_graph_flow("route", src_id, f"no_routing → {fallback_dst}")
                    return ""
                fallback_key = next(iter(rmap), "")
                log_graph_flow("route", src_id, f"fallback → {rmap.get(fallback_key)}")
                return fallback_key
            return _router

        builder.add_conditional_edges(
            src,
            _make_router(priority_fns, route_map, has_routing, src),
            route_map,
        )

    # ── Step 4: checkpointer ──────────────────────────────────────────────────
    if checkpointer is _DEFAULT:
        db_path = os.path.abspath(config.db_path)
        conn = await aiosqlite.connect(db_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=10000")
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

    compiled = builder.compile(checkpointer=checkpointer or None)
    compiled._llm_node_instances = _llm_node_instances
    return compiled


# ---------------------------------------------------------------------------
# Backward compat aliases (were in agent_loader.py)
# ---------------------------------------------------------------------------

from framework.registry import register_schema as _register_schema, get_all_schemas


def register_state_schema(name: str, cls):
    """向后兼容别名 — 请改用 framework.registry.register_schema()。"""
    _register_schema(name, cls)


def _get_state_schemas() -> dict:
    """向后兼容别名 — 请改用 framework.registry.get_all_schemas()。"""
    import framework.schema  # noqa: F401
    return get_all_schemas()
