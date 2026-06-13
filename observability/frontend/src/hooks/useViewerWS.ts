/**
 * ZenithLoom Observability v2 — Viewer WebSocket Hook
 * src/hooks/useViewerWS.ts
 *
 * Connects to the server-side /ws proxy (same origin as the page) with
 * exponential backoff reconnect.  The URL is derived dynamically so that
 * any tunnel address, LAN IP, or localhost all work without rebuilding.
 *
 * Token auth (optional):
 *   Set VITE_OBSERV_TOKEN at build time to append ?token=<value> to the URL.
 *   Leave unset (or empty) to disable auth (matches server default).
 */

import { useEffect, useRef, useCallback } from 'react'
import type { ViewerEvent } from '../sprite/stateMachine'

export type ViewerWsStatus = 'connecting' | 'connected' | 'disconnected'

interface UseViewerWSOptions {
  url?: string
  onMessage: (msg: ViewerEvent) => void
  onStatusChange?: (status: ViewerWsStatus) => void
}

/** Build the default WS URL from the current page origin (works on any host/tunnel). */
function defaultViewerUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  let url = `${proto}//${window.location.host}/ws`
  const token = import.meta.env.VITE_OBSERV_TOKEN as string | undefined
  if (token) url += `?token=${encodeURIComponent(token)}`
  return url
}

const BASE_BACKOFF_MS = 1_000
const MAX_BACKOFF_MS = 30_000

export function useViewerWS({ url, onMessage, onStatusChange }: UseViewerWSOptions): void {
  // Resolve URL lazily so defaultViewerUrl() runs in browser context (not SSR).
  const resolvedUrl = url ?? defaultViewerUrl()
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
    const ws = new WebSocket(resolvedUrl)
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
  }, [resolvedUrl])

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
