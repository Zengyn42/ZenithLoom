/**
 * ZenithLoom Observability — Agent Tab Bar
 * src/components/AgentTabs.tsx
 */

import React from 'react'
import type { AgentSnapshot } from '../types'

interface AgentTabsProps {
  agents: AgentSnapshot[]
  activeAgent: string | null
  onSelect: (name: string) => void
  wsStatus: 'connecting' | 'connected' | 'disconnected'
}

export function AgentTabs({ agents, activeAgent, onSelect, wsStatus }: AgentTabsProps) {
  const wsColor =
    wsStatus === 'connected' ? '#22c55e' : wsStatus === 'connecting' ? '#f59e0b' : '#ef4444'

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        background: '#161b22',
        borderBottom: '1px solid #30363d',
        height: 48,
        gap: 4,
        flexShrink: 0,
      }}
    >
      {/* Logo / title */}
      <span
        style={{
          fontWeight: 700,
          fontSize: 15,
          color: '#58a6ff',
          marginRight: 16,
          letterSpacing: '-0.01em',
        }}
      >
        ⬡ ZenithLoom
      </span>

      {/* Agent tabs */}
      {agents.map((a) => (
        <button
          key={a.name}
          onClick={() => onSelect(a.name)}
          style={{
            padding: '4px 14px',
            borderRadius: 6,
            border: activeAgent === a.name ? '1px solid #388bfd' : '1px solid transparent',
            background: activeAgent === a.name ? '#1f3050' : 'transparent',
            color: activeAgent === a.name ? '#58a6ff' : '#8b949e',
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            transition: 'all 0.15s',
          }}
        >
          {/* Online indicator dot */}
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: '50%',
              background: a.online ? '#22c55e' : '#555',
              flexShrink: 0,
            }}
          />
          {a.name}
        </button>
      ))}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* WS connection status */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          fontSize: 11,
          color: '#8b949e',
        }}
      >
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: '50%',
            background: wsColor,
          }}
        />
        {wsStatus}
      </div>
    </div>
  )
}
