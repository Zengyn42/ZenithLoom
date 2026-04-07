"""
子图状态字段映射节点 — framework/nodes/subgraph_mapper.py

SUBGRAPH_MAPPER 节点类型：在子图和父图之间进行显式字段映射，
防止子图节点意外写入父图保留字段（如 rollback_reason、gemini_context）。

用法（声明式 entity.json 子图节点列表中）：
  {"id": "sub_entry", "type": "SUBGRAPH_MAPPER", "direction": "in",
   "map": {"messages": "sub_messages", "project_root": "sub_root"}}
  {"id": "sub_exit",  "type": "SUBGRAPH_MAPPER", "direction": "out",
   "map": {"sub_messages": "messages"},
   "merge_strategy": "append"}

direction:
  "in"   — 进入子图前：将父图 state 字段重命名/隔离
  "out"  — 退出子图后：将子图 state 字段写回父图

merge_strategy（仅 direction="out" 时有效）：
  "replace" — 目标字段直接覆盖（默认）
  "append"  — 仅适用于无 LangGraph reducer 的普通 list 字段，
               将源 list 追加到目标 list

messages 字段特殊处理：
  messages 使用 LangGraph add_messages reducer。
  direction="out" 时直接返回源 list，reducer 自动追加+去重，不手动 append。
"""

import logging

from framework.debug import is_debug

logger = logging.getLogger(__name__)

# 使用 LangGraph add_messages reducer 的字段（不手动追加，让 reducer 接管）
_REDUCER_FIELDS = frozenset({"messages"})


class SubgraphMapperNode:
    """
    子图状态字段映射节点（SUBGRAPH_MAPPER 节点类型）。

    node_config 字段：
      direction      str   "in" 或 "out"
      map            dict  {source_field: target_field}（direction="in" 时方向为父→子）
      merge_strategy str   "replace"（默认）或 "append"（direction="out" 且非 reducer 字段）
    """

    def __init__(self, node_config: dict):
        self._direction = node_config.get("direction", "out")
        self._map: dict[str, str] = node_config.get("map", {})
        self._merge_strategy: str = node_config.get("merge_strategy", "replace")

        if self._direction not in ("in", "out"):
            raise ValueError(
                f"SubgraphMapperNode: direction must be 'in' or 'out', "
                f"got {self._direction!r}"
            )

    def __call__(self, state: dict) -> dict:
        if is_debug():
            logger.debug(
                f"[subgraph_mapper] direction={self._direction} "
                f"map={self._map} strategy={self._merge_strategy}"
            )

        result: dict = {}

        # ── subgraph_topic 生命周期管理 ──────────────────────────────────
        # 入口（in）：若 subgraph_topic 为空，从 routing_context 写入主题锚点
        # 出口（out）：清空 subgraph_topic，防止残留污染父图
        if self._direction == "in":
            existing_topic = state.get("subgraph_topic", "")
            routing_ctx = state.get("routing_context", "")
            if not existing_topic and routing_ctx:
                result["subgraph_topic"] = routing_ctx
                logger.info(
                    f"[subgraph_mapper] subgraph_topic ← routing_context "
                    f"({len(routing_ctx)} chars)"
                )
            # 入口清空 previous_node_output，防止父图最后一条输出污染子图首节点
            result["previous_node_output"] = ""
            if is_debug():
                logger.debug("[subgraph_mapper] previous_node_output cleared on entry")
        elif self._direction == "out":
            result["subgraph_topic"] = ""
            result["previous_node_output"] = ""
            if is_debug():
                logger.debug("[subgraph_mapper] subgraph_topic + previous_node_output cleared on exit")

        # ── 声明式字段映射 ────────────────────────────────────────────────
        for source_field, target_field in self._map.items():
            value = state.get(source_field)
            if value is None:
                continue

            if self._direction == "out" and target_field in _REDUCER_FIELDS:
                # messages 等 reducer 字段：直接返回新消息列表，让 reducer 接管去重/追加
                result[target_field] = value
                logger.debug(
                    f"[subgraph_mapper] {source_field} → {target_field} "
                    f"(reducer field, len={len(value)})"
                )
            elif (
                self._direction == "out"
                and self._merge_strategy == "append"
                and isinstance(value, list)
                and target_field not in _REDUCER_FIELDS
            ):
                # 普通 list 字段 + append 策略：手动追加
                existing = state.get(target_field) or []
                if isinstance(existing, list):
                    result[target_field] = existing + value
                else:
                    result[target_field] = value
                logger.debug(
                    f"[subgraph_mapper] {source_field} → {target_field} "
                    f"(append, +{len(value)} items)"
                )
            else:
                # replace（默认）：直接覆盖
                result[target_field] = value
                logger.debug(
                    f"[subgraph_mapper] {source_field} → {target_field} (replace)"
                )

        return result
