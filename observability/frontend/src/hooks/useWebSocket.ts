/**
 * ZenithLoom Observability — WebSocket hook
 * src/hooks/useWebSocket.ts
 *
 * Connects to /ws/events, dispatches events to the Zustand store.
 * Reconnects with exponential back-off on disconnect.
 */

import { useEffect, useRef } from 'react'
import { useStore } from '../store/useStore'
import type { InitMessage, ObservEvent } from '../types'

const WS_URL = `ws://${window.location.host}/ws/events`
const BACKOFF_INITIAL = 1000
const BACKOFF_MAX = 30000

export function useWebSocket() {
  const { handleInitMessage, handleEvent, setWsStatus } = useStore()
  const wsRef = useRef<WebSocket | null>(null)
  const backoffRef = useRef(BACKOFF_INITIAL)
  const unmountedRef = useRef(false)

  useEffect(() => {
    unmountedRef.current = false

    const connect = () => {
      if (unmountedRef.current) return
      setWsStatus('connecting')

      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        backoffRef.current = BACKOFF_INITIAL
        setWsStatus('connected')
      }

      ws.onmessage = (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data as string)
          if (data.type === 'init') {
            handleInitMessage((data as InitMessage).agents)
          } else if (data.type === 'ping') {
            // keepalive — ignore
          } else {
            handleEvent(data as ObservEvent)
          }
        } catch {
          // ignore parse errors
        }
      }

      ws.onclose = () => {
        setWsStatus('disconnected')
        if (!unmountedRef.current) {
          const delay = backoffRef.current
          backoffRef.current = Math.min(delay * 2, BACKOFF_MAX)
          setTimeout(connect, delay)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      unmountedRef.current = true
      wsRef.current?.close()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
}
