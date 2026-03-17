# backward-compat shim — real code moved to framework/nodes/heartbeat/heartbeat_node.py
from framework.nodes.heartbeat.heartbeat_node import HeartbeatNode
AgentRunNode = HeartbeatNode
__all__ = ["AgentRunNode", "HeartbeatNode"]
