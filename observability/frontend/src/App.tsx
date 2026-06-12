/**
 * ZenithLoom Observability — Root App Component
 * src/App.tsx
 */

import React, { useState } from 'react'
import { useStore } from './store/useStore'
import { useWebSocket } from './hooks/useWebSocket'
import { useGraphData } from './hooks/useGraphData'
import { AgentTabs } from './components/AgentTabs'
import { AgentGraph } from './components/AgentGraph'
import { StatusBar } from './components/StatusBar'
import { NodePanel } from './components/NodePanel'

export function App() {
  useWebSocket()

  const { agents, activeAgent, setActiveAgent, wsStatus, recentEvents } = useStore()
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const agentList = Object.values(agents)

  const activeAgentData = activeAgent ? agents[activeAgent] : null
  const graphInfo = useGraphData(activeAgent)

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        height: '100%',
        background: '#0d1117',
      }}
    >
      {/* Top tab bar */}
      <AgentTabs
        agents={agentList}
        activeAgent={activeAgent}
        onSelect={setActiveAgent}
        wsStatus={wsStatus}
      />

      {/* Main content area */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative', display: 'flex' }}>
        {!activeAgent && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#444',
              fontSize: 16,
            }}
          >
            {wsStatus === 'connecting' || wsStatus === 'disconnected'
              ? 'Connecting to ZenithLoom Observability Server…'
              : 'Waiting for agent data…'}
          </div>
        )}

        {activeAgent && graphInfo?.loading && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#8b949e',
              fontSize: 14,
            }}
          >
            Loading graph topology for <strong style={{ color: '#58a6ff', marginLeft: 6 }}>{activeAgent}</strong>…
          </div>
        )}

        {activeAgent && graphInfo?.error && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              gap: 8,
              color: '#ef4444',
            }}
          >
            <span style={{ fontSize: 16 }}>Failed to load graph topology</span>
            <span style={{ fontSize: 12, color: '#8b949e' }}>{graphInfo.error}</span>
          </div>
        )}

        {activeAgent && graphInfo?.graph && activeAgentData && (
          <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
            <AgentGraph
              key={activeAgent}
              graph={graphInfo.graph}
              nodeStates={activeAgentData.node_states}
              onNodeClick={(nodeId) => setSelectedNode(nodeId)}
            />
          </div>
        )}
        {selectedNode && activeAgent && (
          <NodePanel
            agentName={activeAgent}
            nodeId={selectedNode}
            nodeStatus={activeAgentData?.node_states[selectedNode] ?? 'idle'}
            recentEvents={recentEvents[activeAgent] ?? []}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>

      {/* Bottom status bar */}
      <StatusBar agent={activeAgentData} />
    </div>
  )
}
