# backward-compat shim — real code moved to framework/nodes/subgraph/subgraph_ref_node.py
from framework.nodes.subgraph.subgraph_ref_node import SubgraphRefNode
AgentRefNode = SubgraphRefNode
__all__ = ["AgentRefNode", "SubgraphRefNode"]
