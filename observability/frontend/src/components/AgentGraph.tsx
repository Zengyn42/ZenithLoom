/**
 * ZenithLoom Observability — Agent Graph Renderer
 * src/components/AgentGraph.tsx
 *
 * Renders the AgentGraph topology using ReactFlow + dagre auto-layout.
 * Nodes reflect live execution state (idle/running/done/error).
 * Supports subgraph group node expansion (SUBGRAPH_REF nodes with .subgraph).
 */

import React, { useCallback, useMemo } from 'react'
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeMouseHandler,
} from 'reactflow'
import 'reactflow/dist/style.css'
import dagre from '@dagrejs/dagre'
import type { NodeStatus, RawGraph, RawNodeSpec, RawEdgeSpec } from '../types'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_WIDTH = 180
const NODE_HEIGHT = 48
const GROUP_PADDING = 30

// Node type → display color (idle state)
const TYPE_COLORS: Record<string, string> = {
  CLAUDE_SDK: '#1e3a5f',
  CLAUDE_CLI: '#1e3a5f',
  GEMINI_API: '#1a3d2b',
  GEMINI_CLI: '#1a3d2b',
  OLLAMA: '#3d2b1a',
  LOCAL_VLLM: '#3d2b1a',
  HEARTBEAT: '#2b1a3d',
  VALIDATE: '#2b2b2b',
  GIT_SNAPSHOT: '#2b2b2b',
  GIT_ROLLBACK: '#2b2b2b',
  DETERMINISTIC: '#2b2b2b',
  EXTERNAL_TOOL: '#1a2b3d',
  SUBGRAPH_REF: '#3d1a2b',
}

// Status → border/glow style
const STATUS_STYLES: Record<NodeStatus, React.CSSProperties> = {
  idle: { borderColor: '#444', boxShadow: 'none' },
  running: {
    borderColor: '#3b82f6',
    boxShadow: '0 0 12px 3px rgba(59,130,246,0.6)',
    animation: 'pulse 1.2s ease-in-out infinite',
  },
  done: { borderColor: '#22c55e', boxShadow: '0 0 6px 1px rgba(34,197,94,0.4)' },
  error: { borderColor: '#ef4444', boxShadow: '0 0 8px 2px rgba(239,68,68,0.6)' },
}

// ---------------------------------------------------------------------------
// Dagre layout helper
// ---------------------------------------------------------------------------

function applyDagreLayout(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 80 })

  nodes.forEach((n) => {
    const w = (n.style?.width as number) ?? NODE_WIDTH
    const h = (n.style?.height as number) ?? NODE_HEIGHT
    g.setNode(n.id, { width: w, height: h })
  })
  edges.forEach((e) => g.setEdge(e.source, e.target))

  dagre.layout(g)

  return nodes.map((n) => {
    const pos = g.node(n.id)
    const w = (n.style?.width as number) ?? NODE_WIDTH
    const h = (n.style?.height as number) ?? NODE_HEIGHT
    return {
      ...n,
      position: {
        x: pos.x - w / 2,
        y: pos.y - h / 2,
      },
    }
  })
}

/** Sub-layout child nodes within a group. Returns positioned child nodes and group dimensions. */
function applySubgraphLayout(
  childNodes: Node[],
  childEdges: Edge[]
): { positionedChildren: Node[]; groupWidth: number; groupHeight: number } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 60 })

  childNodes.forEach((n) => g.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT }))
  childEdges.forEach((e) => g.setEdge(e.source, e.target))

  dagre.layout(g)

  let maxX = 0
  let maxY = 0
  const positioned = childNodes.map((n) => {
    const pos = g.node(n.id)
    const x = pos.x - NODE_WIDTH / 2 + GROUP_PADDING
    const y = pos.y - NODE_HEIGHT / 2 + GROUP_PADDING
    maxX = Math.max(maxX, x + NODE_WIDTH)
    maxY = Math.max(maxY, y + NODE_HEIGHT)
    return { ...n, position: { x, y } }
  })

  return {
    positionedChildren: positioned,
    groupWidth: maxX + GROUP_PADDING,
    groupHeight: maxY + GROUP_PADDING,
  }
}

// ---------------------------------------------------------------------------
// Custom node component
// ---------------------------------------------------------------------------

interface CustomNodeData {
  label: string
  nodeType: string
  status: NodeStatus
}

function AgentNode({ data }: { data: CustomNodeData }) {
  const bg = TYPE_COLORS[data.nodeType] ?? '#1c1c1c'
  const statusStyle = STATUS_STYLES[data.status]

  return (
    <div
      style={{
        background: bg,
        border: `2px solid ${statusStyle.borderColor ?? '#444'}`,
        boxShadow: statusStyle.boxShadow ?? 'none',
        borderRadius: 8,
        padding: '8px 14px',
        minWidth: NODE_WIDTH,
        minHeight: NODE_HEIGHT,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#e6edf3',
        fontSize: 13,
        fontWeight: 500,
        cursor: 'pointer',
        transition: 'border-color 0.3s, box-shadow 0.3s',
        animation: data.status === 'running' ? 'pulse 1.2s ease-in-out infinite' : 'none',
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 13 }}>{data.label}</div>
      <div
        style={{
          fontSize: 10,
          color: '#8b949e',
          marginTop: 2,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}
      >
        {data.nodeType || 'SUBGRAPH'}
      </div>
      <div
        style={{
          position: 'absolute',
          top: 4,
          right: 6,
          width: 6,
          height: 6,
          borderRadius: '50%',
          background:
            data.status === 'running'
              ? '#3b82f6'
              : data.status === 'done'
              ? '#22c55e'
              : data.status === 'error'
              ? '#ef4444'
              : '#555',
        }}
      />
    </div>
  )
}

const nodeTypes = { agentNode: AgentNode }

// ---------------------------------------------------------------------------
// Build nodes + edges from raw graph (with subgraph expansion)
// ---------------------------------------------------------------------------

function buildNodesAndEdges(
  graph: RawGraph,
  nodeStates: Record<string, NodeStatus>
): { rfNodes: Node[]; rfEdges: Edge[] } {
  const rfNodes: Node[] = []
  const rfEdges: Edge[] = []

  // Build top-level edges first (no change)
  graph.edges.forEach((e, idx) => {
    rfEdges.push({
      id: `${e.from}-${e.to}-${idx}`,
      source: e.from,
      target: e.to,
      animated: false,
      style: {
        stroke: e.type === 'routing_to' ? '#f59e0b' : e.type ? '#64748b' : '#3d4a5c',
        strokeWidth: 1.5,
        strokeDasharray: e.type ? '4 2' : undefined,
      },
      label: e.type ? e.type : undefined,
      labelStyle: { fill: '#8b949e', fontSize: 10 },
    })
  })

  graph.nodes.forEach((n) => {
    if (n.subgraph && n.subgraph.nodes.length > 0) {
      // Build child nodes
      const childNodes: Node[] = n.subgraph.nodes.map((cn) => {
        const childId = `${n.id}::${cn.id}`
        const stateKey = `${n.id}:${cn.id}`
        return {
          id: childId,
          type: 'agentNode',
          position: { x: 0, y: 0 },
          parentId: n.id,
          extent: 'parent' as const,
          data: {
            label: cn.id,
            nodeType: cn.type ?? (cn.agent_dir ? 'SUBGRAPH_REF' : ''),
            status: (nodeStates[stateKey] ?? 'idle') as NodeStatus,
          },
        }
      })

      const childEdges: Edge[] = n.subgraph.edges.map((ce, idx) => ({
        id: `${n.id}::${ce.from}-${ce.to}-${idx}`,
        source: `${n.id}::${ce.from}`,
        target: `${n.id}::${ce.to}`,
        animated: false,
        style: {
          stroke: ce.type === 'routing_to' ? '#f59e0b' : ce.type ? '#64748b' : '#3d4a5c',
          strokeWidth: 1.5,
          strokeDasharray: ce.type ? '4 2' : undefined,
        },
        label: ce.type ? ce.type : undefined,
        labelStyle: { fill: '#8b949e', fontSize: 10 },
      }))

      const { positionedChildren, groupWidth, groupHeight } = applySubgraphLayout(
        childNodes,
        childEdges
      )

      // Add group node
      rfNodes.push({
        id: n.id,
        type: 'default',
        position: { x: 0, y: 0 },
        data: { label: n.id },
        style: {
          background: 'rgba(61,26,43,0.3)',
          border: '2px dashed #7d3b5f',
          borderRadius: 12,
          width: groupWidth,
          height: groupHeight,
          fontSize: 12,
          color: '#c9a0b4',
          fontWeight: 600,
          padding: '4px 10px',
        },
      })

      positionedChildren.forEach((cn) => rfNodes.push(cn))
      childEdges.forEach((ce) => rfEdges.push(ce))
    } else {
      rfNodes.push({
        id: n.id,
        type: 'agentNode',
        position: { x: 0, y: 0 },
        data: {
          label: n.id,
          nodeType: n.type ?? (n.agent_dir ? 'SUBGRAPH_REF' : ''),
          status: (nodeStates[n.id] ?? 'idle') as NodeStatus,
        },
      })
    }
  })

  return { rfNodes, rfEdges }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface AgentGraphProps {
  graph: RawGraph
  nodeStates: Record<string, NodeStatus>
  onNodeClick?: (nodeId: string) => void
}

export function AgentGraph({ graph, nodeStates, onNodeClick }: AgentGraphProps) {
  // Build ReactFlow nodes+edges (including subgraph expansion)
  const { rfNodes: baseNodes, rfEdges } = useMemo(
    () => buildNodesAndEdges(graph, nodeStates),
    // Only re-build when graph structure changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graph.nodes, graph.edges]
  )

  // Apply dagre layout (top-level nodes only — child nodes already positioned)
  const topLevelNodes = useMemo(
    () => baseNodes.filter((n) => !n.parentId),
    [baseNodes]
  )
  const topLevelEdges = useMemo(
    () => rfEdges.filter((e) => !e.id.includes('::')),
    [rfEdges]
  )

  const layoutedTopLevel = useMemo(
    () => applyDagreLayout(topLevelNodes, topLevelEdges),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [graph.nodes, graph.edges]
  )

  // Merge layouted top-level positions back into all nodes
  const layoutedNodes = useMemo(() => {
    const posMap = new Map(layoutedTopLevel.map((n) => [n.id, n.position]))
    return baseNodes.map((n) => {
      if (!n.parentId && posMap.has(n.id)) {
        return { ...n, position: posMap.get(n.id)! }
      }
      return n
    })
  }, [layoutedTopLevel, baseNodes])

  // Merge current node statuses into layouted nodes
  const nodesWithStatus: Node[] = useMemo(
    () =>
      layoutedNodes.map((n) => {
        if (n.parentId) {
          // Child node: state key = parentId:childOrigId
          const childOrigId = n.id.split('::')[1] ?? n.id
          const stateKey = `${n.parentId}:${childOrigId}`
          return {
            ...n,
            data: {
              ...n.data,
              status: (nodeStates[stateKey] ?? 'idle') as NodeStatus,
            },
          }
        }
        return {
          ...n,
          data: {
            ...n.data,
            status: (nodeStates[n.id] ?? 'idle') as NodeStatus,
          },
        }
      }),
    [layoutedNodes, nodeStates]
  )

  const [nodes, , onNodesChange] = useNodesState(nodesWithStatus)
  const [edges, , onEdgesChange] = useEdgesState(rfEdges)

  // Keep nodes in sync with status changes without re-layout
  const currentNodes = useMemo(
    () =>
      nodes.map((n) => {
        if (n.parentId) {
          const childOrigId = n.id.split('::')[1] ?? n.id
          const stateKey = `${n.parentId}:${childOrigId}`
          return {
            ...n,
            data: {
              ...n.data,
              status: (nodeStates[stateKey] ?? 'idle') as NodeStatus,
            },
          }
        }
        return {
          ...n,
          data: {
            ...n.data,
            status: (nodeStates[n.id] ?? 'idle') as NodeStatus,
          },
        }
      }),
    [nodes, nodeStates]
  )

  const handleNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      if (onNodeClick) {
        onNodeClick(node.id)
      }
    },
    [onNodeClick]
  )

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.75; }
        }
      `}</style>
      <ReactFlow
        nodes={currentNodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
        panOnDrag
        zoomOnScroll
        style={{ background: '#0d1117' }}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} color="#21262d" gap={20} />
        <Controls showInteractive={false} style={{ background: '#161b22', border: '1px solid #30363d' }} />
        <MiniMap
          nodeColor={(n) => {
            const status = (n.data as CustomNodeData).status
            return status === 'running'
              ? '#3b82f6'
              : status === 'done'
              ? '#22c55e'
              : status === 'error'
              ? '#ef4444'
              : '#444'
          }}
          style={{ background: '#161b22', border: '1px solid #30363d' }}
        />
      </ReactFlow>
    </div>
  )
}
