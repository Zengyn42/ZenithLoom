/**
 * ZenithLoom Observability v2 — Sprite Office React Component
 * src/components/SpriteOffice.tsx
 *
 * React wrapper around PixiJS canvas + state machine.
 * Connects to viewer WS (ws://127.0.0.1:8766/ws), drives sprite office.
 */

import React, { useEffect, useRef, useCallback, useState } from 'react'
import {
  createOfficeState,
  applyEvent,
  applySnapshot,
  runGC,
  type OfficeState,
  type ViewerEvent,
  GRID_COLS,
  GRID_ROWS,
} from '../sprite/stateMachine'
import { useViewerWS, type ViewerWsStatus } from '../hooks/useViewerWS'

// Canvas dimensions based on grid layout
const SLOT_W = 180
const SLOT_H = 120
const PADDING = 16
const CANVAS_W = GRID_COLS * SLOT_W + (GRID_COLS + 1) * PADDING
const CANVAS_H = GRID_ROWS * SLOT_H + (GRID_ROWS + 1) * PADDING

const WS_STATUS_COLORS: Record<ViewerWsStatus, string> = {
  connected: '#22c55e',
  connecting: '#fcd34d',
  disconnected: '#ef4444',
}

const WS_STATUS_LABELS: Record<ViewerWsStatus, string> = {
  connected: '● connected to viewer',
  connecting: '◌ connecting to viewer…',
  disconnected: '○ viewer disconnected — retrying',
}

export function SpriteOffice() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const officeStateRef = useRef<OfficeState>(createOfficeState())
  // PixiOffice is dynamically imported to avoid SSR issues
  const pixiOfficeRef = useRef<any>(null)
  const gcTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [wsStatus, setWsStatus] = useState<ViewerWsStatus>('connecting')
  const [spriteCount, setSpriteCount] = useState(0)

  // Initialize PixiJS
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let destroyed = false
    import('../sprite/pixiOffice').then(({ createPixiOffice }) => {
      if (destroyed) return
      createPixiOffice(canvas, CANVAS_W, CANVAS_H).then((office) => {
        if (destroyed) {
          office.destroy()
          return
        }
        pixiOfficeRef.current = office
        // Initial render
        office.render(officeStateRef.current)
      })
    })

    // GC timer: run every second
    gcTimerRef.current = setInterval(() => {
      runGC(officeStateRef.current)
      if (pixiOfficeRef.current) {
        pixiOfficeRef.current.render(officeStateRef.current)
        setSpriteCount(officeStateRef.current.sprites.size)
      }
    }, 1000)

    return () => {
      destroyed = true
      if (gcTimerRef.current) clearInterval(gcTimerRef.current)
      if (pixiOfficeRef.current) {
        pixiOfficeRef.current.destroy()
        pixiOfficeRef.current = null
      }
    }
  }, [])

  const handleMessage = useCallback((msg: ViewerEvent) => {
    const state = officeStateRef.current

    if (msg.type === 'snapshot') {
      applySnapshot(state, msg)
    } else if (msg.type === 'ping') {
      // keepalive — no-op
    } else {
      applyEvent(state, msg)
    }

    if (pixiOfficeRef.current) {
      pixiOfficeRef.current.render(state)
      setSpriteCount(state.sprites.size)
    }
  }, [])

  useViewerWS({
    onMessage: handleMessage,
    onStatusChange: setWsStatus,
  })

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: '#0d1117',
        overflow: 'auto',
      }}
    >
      {/* Header bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          padding: '8px 16px',
          borderBottom: '1px solid #21262d',
          flexShrink: 0,
        }}
      >
        <span style={{ color: '#c9d1d9', fontSize: 13, fontFamily: 'monospace', fontWeight: 600 }}>
          🏢 Pixel Office
        </span>
        <span
          style={{
            fontSize: 11,
            fontFamily: 'monospace',
            color: WS_STATUS_COLORS[wsStatus],
          }}
        >
          {WS_STATUS_LABELS[wsStatus]}
        </span>
        <span style={{ fontSize: 11, color: '#6e7681', fontFamily: 'monospace', marginLeft: 'auto' }}>
          {spriteCount} agent{spriteCount !== 1 ? 's' : ''} on floor
        </span>
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, overflow: 'auto', display: 'flex', justifyContent: 'center', padding: 16 }}>
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={{
            imageRendering: 'pixelated',
            border: '1px solid #21262d',
            borderRadius: 8,
          }}
        />
      </div>

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px 24px',
          padding: '8px 16px',
          borderTop: '1px solid #21262d',
          flexShrink: 0,
        }}
      >
        {[
          { label: 'hani', color: '#3b82f6' },
          { label: 'asa', color: '#f97316' },
          { label: 'jei', color: '#22c55e' },
          { label: 'dan', color: '#a855f7' },
        ].map(({ label, color }) => (
          <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: color, display: 'inline-block' }} />
            <span style={{ color: '#8b949e', fontSize: 11, fontFamily: 'monospace' }}>{label}</span>
          </span>
        ))}
        <span style={{ color: '#6e7681', fontSize: 10, fontFamily: 'monospace', marginLeft: 'auto' }}>
          ws://127.0.0.1:8766/ws · ZenithLoom Observability v2
        </span>
      </div>
    </div>
  )
}
