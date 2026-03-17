"""
框架级心跳调度器 — framework/heartbeat.py

心跳图由 agent.json["heartbeat"]["graph"] 声明，使用与 agent 图完全相同的
DSL（nodes + edges），支持 PROBE、AGENT_RUN 等节点类型，并行执行独立节点。

  "heartbeat": {
    "interval_hours": 23,
    "prompt": "Heartbeat triggered.",
    "graph": {
      "nodes": [
        {"id": "probe_ollama", "type": "PROBE",      "name": "ollama"},
        {"id": "run_reporter", "type": "HEARTBEAT",  "agent_dir": "blueprints/functional_graphs/reporter",
         "prompt": "Generate daily report."}
      ],
      "edges": [
        {"from": "__start__",    "to": "probe_ollama"},
        {"from": "__start__",    "to": "run_reporter"},
        {"from": "probe_ollama", "to": "__end__"},
        {"from": "run_reporter", "to": "__end__"}
      ]
    }
  }

接口层调用方式：
  hb_graph, hb_cfg = await loader.build_heartbeat_graph()
  if hb_graph:
      await run_heartbeat_once(hb_graph, hb_cfg)
      asyncio.create_task(heartbeat_loop(hb_graph, hb_cfg))
"""

import asyncio
import logging

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 23 * 3600  # 23小时
_DEFAULT_PROMPT = "Heartbeat triggered."


async def heartbeat_loop(graph, cfg: dict) -> None:
    """后台无限循环心跳。"""
    interval = cfg.get("interval_hours", 23) * 3600
    while True:
        await asyncio.sleep(interval)
        await _invoke(graph, cfg)


async def run_heartbeat_once(graph, cfg: dict) -> bool:
    """启动时立即跑一次。"""
    return await _invoke(graph, cfg)


async def _invoke(graph, cfg: dict) -> bool:
    prompt = cfg.get("prompt", _DEFAULT_PROMPT)
    logger.info("[heartbeat] 开始...")
    try:
        await graph.ainvoke({"messages": [HumanMessage(content=prompt)]})
        logger.info("[heartbeat] 完成")
        return True
    except Exception as e:
        logger.error(f"[heartbeat] 失败: {e}")
        return False
