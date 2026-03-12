"""
框架内置节点类型和条件注册 — framework/builtins.py

此模块注册所有开箱即用的节点类型和条件谓词到 registry。
import 时自动执行注册（无副作用，幂等）。

节点类型（NodeFactory：每次调用返回新实例）：
  CLAUDE_CLI       — ClaudeNode(AgentNode)，Claude Code SDK 实现
  GEMINI_CLI       — GeminiNode(AgentNode)，Gemini 单轮对话实现
  LOCAL_VLLM       — LlamaNode(AgentNode)，本地 vLLM/Ollama 实现（stub）
  GIT_SNAPSHOT     — GitSnapshotNode，提交前自动快照
  GIT_ROLLBACK     — GitRollbackNode，验证失败时回退
  VALIDATE         — ValidateNode，输出质量验证
  VRAM_FLUSH       — VramFlushNode，GPU 显存清洗
  SUBGRAPH_MAPPER  — SubgraphMapperNode，子图状态字段映射
  EXTERNAL_TOOL    — ExternalToolNode，通用外部 CLI 调用（gws / obsidian / cli-anything-* 等）

条件谓词（ConditionFn：state → bool）：
  always           — 总是 True
  on_error         — rollback_reason 非空
  no_error         — rollback_reason 为空
  no_routing       — routing_target 为空（无路由请求，走 __end__）

注意：routing_to 边类型不在此注册——其条件由 _build_declarative() 动态生成，
      基于每条边的 "to" 字段自动检查 state["routing_target"]。
"""

import logging

from framework.registry import register_condition, register_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 节点类型（懒导入避免循环依赖，每次调用返回新实例）
# ---------------------------------------------------------------------------

@register_node("CLAUDE_CLI")
def _(config, node_config):
    from framework.claude.node import ClaudeNode
    return ClaudeNode(config, node_config, system_prompt=node_config.get("system_prompt", ""))


@register_node("GEMINI_CLI")
def _(config, node_config):
    from framework.gemini.node import GeminiNode
    return GeminiNode(config, node_config)


@register_node("LOCAL_VLLM")
def _(config, node_config):
    from framework.llama.node import LlamaNode
    return LlamaNode(config, node_config)


@register_node("GIT_SNAPSHOT")
def _(config, node_config):
    from framework.nodes.git_nodes import GitSnapshotNode
    return GitSnapshotNode()


@register_node("GIT_ROLLBACK")
def _(config, node_config):
    from framework.nodes.git_nodes import GitRollbackNode
    return GitRollbackNode()


@register_node("VALIDATE")
def _(config, node_config):
    from framework.nodes.validate_node import ValidateNode
    return ValidateNode(config)


@register_node("VRAM_FLUSH")
def _(config, node_config):
    from framework.nodes.vram_flush_node import VramFlushNode
    return VramFlushNode()


@register_node("SUBGRAPH_MAPPER")
def _(config, node_config):
    from framework.nodes.subgraph_mapper import SubgraphMapperNode
    return SubgraphMapperNode(node_config)


@register_node("AGENT_REF")
def _(config, node_config):
    from framework.nodes.agent_ref_node import AgentRefNode
    return AgentRefNode(config, node_config)


@register_node("EXTERNAL_TOOL")
def _(config, node_config):
    from framework.nodes.external_tool_node import ExternalToolNode
    return ExternalToolNode(config, node_config)


# ---------------------------------------------------------------------------
# 条件谓词
# ---------------------------------------------------------------------------

@register_condition("always")
def _(state) -> bool:
    return True


@register_condition("on_error")
def _(state) -> bool:
    return bool(state.get("rollback_reason"))


@register_condition("no_error")
def _(state) -> bool:
    return not state.get("rollback_reason")


@register_condition("no_routing")
def _(state) -> bool:
    """没有路由请求时为 True（routing_target 为空）→ 走 __end__。"""
    return not state.get("routing_target", "")


logger.debug("[builtins] all built-in nodes and conditions registered")
