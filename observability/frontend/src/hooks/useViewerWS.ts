/**
 * ZenithLoom Observability v2 — Viewer WebSocket Hook
 * src/hooks/useViewerWS.ts
 *
 * Connects to the viewer WS server (ws://127.0.0.1:8766/ws)
 * with exponential backoff reconnect.
 *
 * Emits parsed ViewerEvent / snapshot objects to the provided callbacks.
 */

import { useEffect, useRef, useCallback } from 'react'
import type { ViewerEvent } from '../sprite/stateMachine'

export type ViewerWsStatus = 'connecting' | 'connected' | 'disconnected'

interface UseViewerWSOptions {
  url?: string
  onMessage: (msg: ViewerEvent) => void
  onStatusChange?: (status: ViewerWsStatus) => void
}

const DEFAULT_URL = 'ws://127.0.0.1:8766/ws'
const BASE_BACKOFF_MS = 1_000
const MAX_BACKOFF_MS = 30_000

export function useViewerWS({ url = DEFAULT_URL, onMessage, onStatusChange }: UseViewerWSOptions): void {
  const wsRef = useRef<WebSocket | null>(null)
  const backoffRef = useRef(BASE_BACKOFF_MS)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  const onMessageRef = useRef(onMessage)
  const onStatusRef = useRef(onStatusChange)
  onMessageRef.current = onMessage
  onStatusRef.current = onStatusChange

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current) {
      wsRef.current.onopen = null
      wsRef.current.onmessage = null
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      try { wsRef.current.close() } catch {}
      wsRef.current = null
    }

    onStatusRef.current?.('connecting')
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      backoffRef.current = BASE_BACKOFF_MS // reset backoff on successful connect
      onStatusRef.current?.('connected')
    }

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return
      try {
        const msg: ViewerEvent = JSON.parse(evt.data as string)
        onMessageRef.current(msg)
      } catch {
        // ignore malformed JSON
      }
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      onStatusRef.current?.('disconnected')
      wsRef.current = null
      // Exponential backoff reconnect
      reconnectTimer.current = setTimeout(() => {
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS)
        connect()
      }, backoffRef.current)
    }

    ws.onerror = () => {
      // onclose will fire immediately after, handles reconnect
    }
  }, [url])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) {
        wsRef.current.onopen = null
        wsRef.current.onmessage = null
        wsRef.current.onclose = null
        wsRef.current.onerror = null
        try { wsRef.current.close() } catch {}
        wsRef.current = null
      }
    }
  }, [connect])
}
