/**
 * ZenithLoom Observability — Root App Component
 * src/App.tsx
 *
 * Two views, toggled by a view switcher tab bar:
 *   1. "Topology" — React Flow agent graph (original view)
 *   2. "Pixel Office" — PixiJS sprite office (v2, new)
 */

import React, { useState } from 'react'
import { useStore } from './store/useStore'
import { useWebSocket } from './hooks/useWebSocket'
import { useGraphData } from './hooks/useGraphData'
import { AgentTabs } from './components/AgentTabs'
import { AgentGraph } from './components/AgentGraph'
import { StatusBar } from './components/StatusBar'
import { NodePanel } from './components/NodePanel'
import { SpriteOffice } from './components/SpriteOffice'

type ViewMode = 'topology' | 'office'

const VIEW_TABS: { id: ViewMode; label: string }[] = [
  { id: 'topology', label: '⬡ Topology' },
  { id: 'office',   label: '🏢 Pixel Office' },
]

export function App() {
  useWebSocket()

  const { agents, activeAgent, setActiveAgent, wsStatus, recentEvents } = useStore()
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('topology')
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
      {/* Top: agent tab bar + view mode switcher */}
      <div style={{ display: 'flex', alignItems: 'stretch', borderBottom: '1px solid #21262d', flexShrink: 0 }}>
        {/* Agent tabs (left) */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <AgentTabs
            agents={agentList}
            activeAgent={activeAgent}
            onSelect={setActiveAgent}
            wsStatus={wsStatus}
          />
        </div>

        {/* View mode switcher (right) */}
        <div style={{ display: 'flex', alignItems: 'stretch', borderLeft: '1px solid #21262d' }}>
          {VIEW_TABS.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setViewMode(id)}
              style={{
                padding: '0 18px',
                background: viewMode === id ? '#1c2433' : 'transparent',
                border: 'none',
                borderBottom: viewMode === id ? '2px solid #3b82f6' : '2px solid transparent',
                color: viewMode === id ? '#c9d1d9' : '#6e7681',
                fontSize: 12,
                fontFamily: 'monospace',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'color 0.15s, background 0.15s',
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Main content area */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative', display: 'flex' }}>

        {/* ── Topology view ── */}
        {viewMode === 'topology' && (
          <>
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
                Loading graph topology for{' '}
                <strong style={{ color: '#58a6ff', marginLeft: 6 }}>{activeAgent}</strong>…
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
          </>
        )}

        {/* ── Sprite Office view ── */}
        {viewMode === 'office' && (
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <SpriteOffice />
          </div>
        )}
      </div>

      {/* Bottom status bar */}
      <StatusBar agent={activeAgentData} />
    </div>
  )
}
