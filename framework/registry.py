"""
框架级注册表 — framework/registry.py

三类注册表，将字符串名称映射到对应的 Python 对象：

1. 节点注册表（Node Registry）
   类型名 → 工厂函数。每次调用返回新实例（不缓存）。
   注册方式：@register_node("CLAUDE_CLI")
   查询方式：get_node_factory("CLAUDE_CLI")

2. 条件注册表（Condition Registry）
   条件名 → 谓词函数 (state → bool)，用于声明式图的条件边。
   注册方式：@register_condition("needs_gemini")
   查询方式：get_condition("needs_gemini")

3. Schema 注册表（Schema Registry）
   schema 名 → TypedDict 类，用于 LangGraph StateGraph 的 state schema。
   注册方式：register_schema("base_schema", BaseAgentState)
   查询方式：get_schema("base_schema") / get_all_schemas()

   所有 schema 名称以 _schema 结尾。框架内置 schema 在 framework/schema/ 下定义，
   业务图 schema 在各自目录的 state.py 中定义，均通过 register_schema() 自注册。
   entity.json 中通过 "state_schema": "<name>" 引用，不声明则默认 "base_schema"。

内置节点和条件通过 framework/builtins.py 注册（import 时自动执行）。
"""

import logging
from typing import Callable

from framework.config import AgentConfig

logger = logging.getLogger(__name__)

# NodeFactory: (config: AgentConfig, node_config: dict) -> node callable
NodeFactory = Callable[[AgentConfig, dict], object]

# ConditionFn: (state: dict) -> bool
ConditionFn = Callable[[dict], bool]

_NODE_REGISTRY: dict[str, NodeFactory] = {}
_CONDITION_REGISTRY: dict[str, ConditionFn] = {}


def register_node(name: str):
    """
    装饰器：将工厂函数注册为节点类型。

    @register_node("CLAUDE_SDK")
    def _(config, node_config):
        from framework.nodes.llm.claude import ClaudeSDKNode
        return ClaudeSDKNode(config, node_config)
    """
    def decorator(fn: NodeFactory) -> NodeFactory:
        _NODE_REGISTRY[name] = fn
        logger.debug(f"[registry] registered node type {name!r}")
        return fn
    return decorator


def get_node_factory(name: str) -> NodeFactory:
    """获取节点工厂。未知类型抛 ValueError。"""
    factory = _NODE_REGISTRY.get(name)
    if factory is None:
        known = sorted(_NODE_REGISTRY.keys())
        raise ValueError(
            f"Unknown node type {name!r}. Known types: {known}"
        )
    return factory


def register_condition(name: str):
    """
    装饰器：将谓词函数注册为条件名。

    @register_condition("needs_gemini")
    def _(state):
        return state.get("gemini_context", "").startswith("__PENDING__")
    """
    def decorator(fn: ConditionFn) -> ConditionFn:
        _CONDITION_REGISTRY[name] = fn
        logger.debug(f"[registry] registered condition {name!r}")
        return fn
    return decorator


def get_condition(name: str) -> ConditionFn:
    """获取条件谓词。未知条件抛 ValueError。"""
    fn = _CONDITION_REGISTRY.get(name)
    if fn is None:
        known = sorted(_CONDITION_REGISTRY.keys())
        raise ValueError(
            f"Unknown condition {name!r}. Known conditions: {known}"
        )
    return fn


# ---------------------------------------------------------------------------
# Schema Registry — state schema 注册表
# ---------------------------------------------------------------------------

_SCHEMA_REGISTRY: dict[str, type] = {}


def register_schema(name: str, cls: type) -> None:
    """注册 state schema（TypedDict 类）。

    所有 schema 在各自模块的顶层调用此函数自注册：
      框架内置：framework/schema/base.py, framework/schema/debate.py
      业务图：  blueprints/.../state.py

    Args:
        name: schema 名称，必须以 _schema 结尾（如 "base_schema"）
        cls:  TypedDict 子类
    """
    if not name.endswith("_schema"):
        raise ValueError(
            f"Schema 名称必须以 _schema 结尾，got: {name!r}"
        )
    _SCHEMA_REGISTRY[name] = cls
    logger.debug(f"[registry] registered schema {name!r} → {cls.__name__}")


def get_schema(name: str) -> type:
    """获取已注册的 state schema。未知 schema 抛 ValueError。"""
    cls = _SCHEMA_REGISTRY.get(name)
    if cls is None:
        known = sorted(_SCHEMA_REGISTRY.keys())
        raise ValueError(
            f"Unknown schema {name!r}. Known schemas: {known}"
        )
    return cls


def get_all_schemas() -> dict[str, type]:
    """返回所有已注册 schema 的副本。"""
    return dict(_SCHEMA_REGISTRY)
