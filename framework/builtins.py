"""
框架内置节点类型和条件注册 — framework/builtins.py

此模块注册所有开箱即用的节点类型和条件谓词到 registry。
import 时自动执行注册（无副作用，幂等）。

节点类型（NodeFactory：每次调用返回新实例）：
  CLAUDE_CLI       — ClaudeSDKNode(AgentNode)，Claude Code SDK subprocess（别名 CLAUDE_SDK）
  GEMINI_CLI       — GeminiCLINode(AgentNode)，Gemini CLI subprocess（支持高级模型）
  GEMINI_API       — GeminiCodeAssistNode(AgentNode)，Gemini Code Assist HTTP API
  OLLAMA           — OllamaNode(AgentNode)，Ollama HTTP API（别名 LOCAL_VLLM）
  LOCAL_VLLM       — OllamaNode(AgentNode)，同 OLLAMA（向后兼容别名）
  GIT_SNAPSHOT     — GitSnapshotNode，提交前自动快照
  GIT_ROLLBACK     — GitRollbackNode，验证失败时回退
  VALIDATE         — ValidateNode，输出质量验证
  VRAM_FLUSH       — VramFlushNode，GPU 显存清洗
  SUBGRAPH_MAPPER  — SubgraphMapperNode，子图状态字段映射
  SUBGRAPH_REF     — SubgraphRefNode，外部子图节点（向后兼容别名 AGENT_REF）
  AGENT_REF        — SubgraphRefNode，同 SUBGRAPH_REF（向后兼容别名）
  EXTERNAL_TOOL    — ExternalToolNode，通用外部 CLI 调用（gws / obsidian / cli-anything-* 等）
  PROBE            — ProbeNode，服务存活探针（heartbeat 图专用，claude/gemini/ollama）
  SYSTEM_STATS     — SystemStatsNode，系统资源采集（CPU/内存/磁盘/GPU，支持阈值告警）
  AGENT_TASK       — HeartbeatNode，定期调用完整 Agent 主图（heartbeat 专用，持独立 EntityLoader）
  HEARTBEAT        — HeartbeatNode，同 AGENT_TASK（向后兼容别名）

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
    from framework.nodes.llm.claude import ClaudeSDKNode
    return ClaudeSDKNode(config, node_config, system_prompt=node_config.get("system_prompt", ""))


@register_node("CLAUDE_SDK")
def _(config, node_config):
    from framework.nodes.llm.claude import ClaudeSDKNode
    return ClaudeSDKNode(config, node_config, system_prompt=node_config.get("system_prompt", ""))


@register_node("GEMINI_CLI")
def _(config, node_config):
    from framework.nodes.llm.gemini import GeminiCLINode
    return GeminiCLINode(config, node_config)


@register_node("GEMINI_API")
def _(config, node_config):
    from framework.nodes.llm.gemini import GeminiCodeAssistNode
    return GeminiCodeAssistNode(config, node_config)


@register_node("OLLAMA")
def _(config, node_config):
    from framework.nodes.llm.ollama import OllamaNode
    return OllamaNode(config, node_config)


@register_node("LOCAL_VLLM")
def _(config, node_config):
    from framework.nodes.llm.ollama import OllamaNode
    return OllamaNode(config, node_config)


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
    from framework.nodes.subgraph.subgraph_mapper import SubgraphMapperNode
    return SubgraphMapperNode(node_config)


@register_node("SUBGRAPH_REF")
def _(config, node_config):
    from framework.nodes.subgraph.subgraph_ref_node import SubgraphRefNode
    return SubgraphRefNode(config, node_config)


@register_node("AGENT_REF")
def _(config, node_config):
    from framework.nodes.subgraph.subgraph_ref_node import SubgraphRefNode
    return SubgraphRefNode(config, node_config)


@register_node("DETERMINISTIC")
def _(config, node_config):
    from framework.nodes.deterministic_node import DeterministicNode
    return DeterministicNode(config, node_config)


@register_node("EXTERNAL_TOOL")
def _(config, node_config):
    from framework.nodes.external_tool_node import ExternalToolNode
    return ExternalToolNode(config, node_config)


@register_node("PROBE")
def _(config, node_config):
    from framework.nodes.heartbeat.probe_node import ProbeNode
    return ProbeNode(node_config)


@register_node("SYSTEM_STATS")
def _(config, node_config):
    from framework.nodes.heartbeat.system_stats_node import SystemStatsNode
    return SystemStatsNode(node_config)


@register_node("AGENT_TASK")
def _(config, node_config):
    from framework.nodes.heartbeat.heartbeat_node import HeartbeatNode
    return HeartbeatNode(node_config)


@register_node("HEARTBEAT")  # 向后兼容别名
def _(config, node_config):
    from framework.nodes.heartbeat.heartbeat_node import HeartbeatNode
    return HeartbeatNode(node_config)


@register_node("TASK_MONITOR")
def _(config, node_config):
    """TASK_MONITOR：后台子进程监控节点。

    由 AsyncTaskManager 通过 Heartbeat 动态注册，不直接在 entity.json 中声明。
    此注册确保 registry 中有该类型，避免 get_node_factory("TASK_MONITOR") 失败。
    """
    # TASK_MONITOR 不需要真正的节点实例 — 它通过 HeartbeatManager.register_monitor() 管理
    # 返回一个 no-op callable 满足 registry 接口
    async def _noop(state: dict) -> dict:
        return {}
    return _noop


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
