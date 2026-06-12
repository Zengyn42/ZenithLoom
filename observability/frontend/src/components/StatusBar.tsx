/**
 * ZenithLoom Observability — Status Bar
 * src/components/StatusBar.tsx
 *
 * Shows active run_id, thread_id, last event seq for the selected agent.
 */

import React from 'react'
import type { AgentSnapshot } from '../types'

interface StatusBarProps {
  agent: AgentSnapshot | null
}

export function StatusBar({ agent }: StatusBarProps) {
  if (!agent) return null

  const runningNodes = Object.entries(agent.node_states)
    .filter(([, s]) => s === 'running')
    .map(([k]) => k)

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 20,
        padding: '0 16px',
        background: '#161b22',
        borderTop: '1px solid #30363d',
        height: 32,
        fontSize: 11,
        color: '#8b949e',
        flexShrink: 0,
      }}
    >
      <span>
        <span style={{ color: '#6e7681' }}>thread: </span>
        <span style={{ color: '#58a6ff', fontFamily: 'monospace' }}>
          {agent.active_thread_id || '—'}
        </span>
      </span>
      <span>
        <span style={{ color: '#6e7681' }}>run: </span>
        <span style={{ color: '#f0883e', fontFamily: 'monospace' }}>
          {agent.active_run_id ? agent.active_run_id.slice(0, 8) + '…' : '—'}
        </span>
      </span>
      {runningNodes.length > 0 && (
        <span style={{ color: '#3b82f6' }}>
          ⚡ {runningNodes.join(', ')}
        </span>
      )}
      <div style={{ flex: 1 }} />
      <span>seq #{agent.last_seq}</span>
    </div>
  )
}
