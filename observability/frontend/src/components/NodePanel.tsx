/**
 * ZenithLoom Observability — Node Side Panel
 * src/components/NodePanel.tsx
 *
 * Shows details for a selected node: status, recent state_update events
 * with updates_preview fields, and run_start input_preview.
 */

import React from 'react'
import type { ObservEvent, NodeStatus } from '../types'

interface NodePanelProps {
  agentName: string
  nodeId: string
  nodeStatus: NodeStatus
  recentEvents: ObservEvent[]
  onClose: () => void
}

const STATUS_COLOR: Record<NodeStatus, string> = {
  idle: '#8b949e',
  running: '#3b82f6',
  done: '#22c55e',
  error: '#ef4444',
}

export function NodePanel({ agentName: _agentName, nodeId, nodeStatus, recentEvents, onClose }: NodePanelProps) {
  // Find the most recent run_start for this context (for input_preview)
  const latestRunStart = [...recentEvents]
    .reverse()
    .find((e) => e.event_type === 'run_start')

  // Find state_update events for this node
  const stateUpdates = recentEvents.filter(
    (e) => e.event_type === 'state_update' && (e.payload.node as string) === nodeId
  )

  return (
    <div
      style={{
        width: 320,
        background: '#161b22',
        borderLeft: '1px solid #30363d',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 14px',
          borderBottom: '1px solid #30363d',
          background: '#0d1117',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: STATUS_COLOR[nodeStatus] ?? '#555',
              display: 'inline-block',
            }}
          />
          <span
            style={{
              color: '#e6edf3',
              fontWeight: 600,
              fontSize: 13,
              fontFamily: 'monospace',
            }}
          >
            {nodeId}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              fontSize: 10,
              padding: '2px 6px',
              borderRadius: 4,
              background: STATUS_COLOR[nodeStatus] + '33',
              color: STATUS_COLOR[nodeStatus],
              textTransform: 'uppercase',
              fontWeight: 600,
              letterSpacing: '0.05em',
            }}
          >
            {nodeStatus}
          </span>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#8b949e',
              cursor: 'pointer',
              fontSize: 16,
              lineHeight: 1,
              padding: '2px 4px',
            }}
            title="Close panel"
          >
            ×
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '12px 14px' }}>

        {/* Input preview from latest run_start */}
        {latestRunStart && latestRunStart.payload.input_preview ? (
          <div style={{ marginBottom: 16 }}>
            <div style={{ color: '#8b949e', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
              Run Input Preview
            </div>
            <div
              style={{
                background: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: 6,
                padding: '8px 10px',
                color: '#c9d1d9',
                fontSize: 12,
                fontFamily: 'monospace',
                wordBreak: 'break-word',
                whiteSpace: 'pre-wrap',
              }}
            >
              {String(latestRunStart.payload.input_preview)}
            </div>
          </div>
        ) : null}

        {/* State updates */}
        <div>
          <div style={{ color: '#8b949e', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
            State Updates ({stateUpdates.length})
          </div>

          {stateUpdates.length === 0 && (
            <div style={{ color: '#444', fontSize: 12, fontStyle: 'italic' }}>
              No state updates yet
            </div>
          )}

          {[...stateUpdates].reverse().map((evt, i) => {
            const preview = evt.payload.updates_preview as Record<string, string> | undefined
            const keysChanged = evt.payload.keys_changed as string[] | undefined

            return (
              <div
                key={i}
                style={{
                  background: '#0d1117',
                  border: '1px solid #21262d',
                  borderRadius: 6,
                  padding: '8px 10px',
                  marginBottom: 8,
                  fontSize: 12,
                }}
              >
                <div style={{ color: '#6e7681', fontSize: 10, marginBottom: 4 }}>
                  {new Date(evt.timestamp * 1000).toLocaleTimeString()}
                </div>
                {preview && Object.keys(preview).length > 0 ? (
                  Object.entries(preview).map(([k, v]) => (
                    <div key={k} style={{ marginBottom: 4 }}>
                      <span style={{ color: '#58a6ff', fontFamily: 'monospace' }}>{k}</span>
                      <span style={{ color: '#8b949e' }}>: </span>
                      <span
                        style={{
                          color: '#c9d1d9',
                          fontFamily: 'monospace',
                          wordBreak: 'break-word',
                          whiteSpace: 'pre-wrap',
                        }}
                      >
                        {v}
                      </span>
                    </div>
                  ))
                ) : keysChanged && keysChanged.length > 0 ? (
                  <div style={{ color: '#8b949e' }}>
                    Keys changed: {keysChanged.join(', ')}
                  </div>
                ) : (
                  <div style={{ color: '#444', fontStyle: 'italic' }}>no preview</div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
