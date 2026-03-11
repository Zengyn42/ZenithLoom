"""
框架级节点与条件注册表 — framework/registry.py

Registry 将字符串类型名映射到工厂函数（节点）或谓词函数（条件边）。

节点工厂：每次调用返回新实例（不缓存），支持同一类型在图中多次出现。
条件谓词：pure function，state → bool，用于声明式图的条件边。

用法（声明式 agent.json）：
  {"nodes": [{"id": "ceo", "type": "CLAUDE_CLI", ...}],
   "edges": [{"from": "ceo", "to": "worker", "condition": "needs_gemini"}]}

内置节点和条件通过 framework/builtins.py 注册（import 时自动执行）。
自定义节点：
  from framework.registry import register_node
  @register_node("MY_NODE")
  def _(config, node_config):
      return MyNode(config, node_config)
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

    @register_node("CLAUDE_CLI")
    def _(config, node_config):
        from framework.claude.node import ClaudeNode
        return ClaudeNode(config, node_config)
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
