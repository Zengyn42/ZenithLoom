/**
 * ZenithLoom Observability v2 — Sprite State Machine
 * src/sprite/stateMachine.ts
 *
 * Pure TypeScript — zero rendering / zero DOM / zero PixiJS dependencies.
 * Consumed by pixiOffice.ts for rendering, and by vitest tests directly.
 *
 * State diagram per sprite (agent:thread_id key):
 *   IDLE → ARRIVING → WORKING → BRIEF_PAUSE → DEPARTING → IDLE
 *   Any state → LEAVING (GC: 5 min no events)
 *   LEAVING → (removed from map)
 *
 * GC rules:
 *   - 30 s no events     → IDLE (blink animation hint)
 *   - 5 min no events    → LEAVING → remove after animation
 *   - MAX_SPRITES = 20   → evict longest-IDLE sprite on overflow
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SpriteState =
  | 'IDLE'
  | 'ARRIVING'
  | { kind: 'WORKING'; nodeType: string }
  | 'BRIEF_PAUSE'
  | 'DEPARTING'
  | 'LEAVING'

export type NodeType =
  | 'CLAUDE_SDK'
  | 'GEMINI_API'
  | 'GEMINI_CLI'
  | 'SUBGRAPH_REF'
  | 'HEARTBEAT'
  | string // fallback for unknown node types

export interface Sprite {
  /** Composite key: `${agent}:${thread_id}` */
  id: string
  agent: string
  threadId: string
  runId: string
  inputPreview: string
  state: SpriteState
  /** Timestamp of last event received (ms, from Date.now()) */
  lastEventMs: number
  /** Monotonic tick counter for LRU eviction ordering */
  lastActiveTick: number
  /** Workstation grid slot index (0-19), assigned on ARRIVING */
  slotIndex: number
}

export type AgentColor = 'blue' | 'orange' | 'green' | 'purple' | 'gray'

export const AGENT_COLORS: Record<string, AgentColor> = {
  hani: 'blue',
  asa: 'orange',
  jei: 'green',
  dan: 'purple',
}

export const MAX_SPRITES = 20
export const IDLE_TIMEOUT_MS = 30_000   // 30 s → IDLE
export const LEAVE_TIMEOUT_MS = 300_000 // 5 min → LEAVING

// Grid layout: 4 columns × 5 rows = 20 slots
export const GRID_COLS = 4
export const GRID_ROWS = 5

// ---------------------------------------------------------------------------
// ViewerWS event types (JSONL line payloads from viewer WS)
// ---------------------------------------------------------------------------

export interface ViewerEvent {
  ts: number
  agent: string
  event_type: string
  run_id?: string
  thread_id?: string
  input_preview?: string
  updates_preview?: string
  node_id?: string
  node_type?: string
  seq?: number
  // snapshot-specific
  type?: string
  active_runs?: ViewerEvent[]
}

// ---------------------------------------------------------------------------
// Office state
// ---------------------------------------------------------------------------

export interface OfficeState {
  sprites: Map<string, Sprite>
  /** Occupied slot indices */
  occupiedSlots: Set<number>
  /** Monotonic counter for LRU ordering */
  tick: number
}

export function createOfficeState(): OfficeState {
  return {
    sprites: new Map(),
    occupiedSlots: new Set(),
    tick: 0,
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function spriteId(agent: string, threadId: string): string {
  return `${agent}:${threadId}`
}

function nextFreeSlot(occupied: Set<number>): number {
  for (let i = 0; i < MAX_SPRITES; i++) {
    if (!occupied.has(i)) return i
  }
  return -1 // should not happen if eviction is done first
}

function evictLongestIdle(state: OfficeState): void {
  // Find the sprite that has been IDLE the longest (smallest lastActiveTick)
  let candidate: Sprite | null = null
  for (const sp of state.sprites.values()) {
    if (sp.state === 'IDLE' || sp.state === 'LEAVING') {
      if (!candidate || sp.lastActiveTick < candidate.lastActiveTick) {
        candidate = sp
      }
    }
  }
  if (candidate) {
    state.occupiedSlots.delete(candidate.slotIndex)
    state.sprites.delete(candidate.id)
  }
}

function getOrCreateSprite(
  state: OfficeState,
  agent: string,
  threadId: string,
  runId: string,
  inputPreview: string,
  nowMs: number,
): Sprite {
  const id = spriteId(agent, threadId)
  let sp = state.sprites.get(id)
  if (sp) return sp

  // Need to create — evict if at capacity
  if (state.sprites.size >= MAX_SPRITES) {
    evictLongestIdle(state)
  }

  const slot = nextFreeSlot(state.occupiedSlots)
  state.occupiedSlots.add(slot)
  state.tick++

  sp = {
    id,
    agent,
    threadId,
    runId,
    inputPreview,
    state: 'ARRIVING',
    lastEventMs: nowMs,
    lastActiveTick: state.tick,
    slotIndex: slot,
  }
  state.sprites.set(id, sp)
  return sp
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

/**
 * Apply a single ViewerEvent to the office state.
 * Mutates state in place (Map is reference-typed — consumers should shallow-copy sprites for react diffing).
 */
export function applyEvent(state: OfficeState, evt: ViewerEvent, nowMs: number = Date.now()): void {
  const { agent, event_type, thread_id = '', run_id = '', input_preview = '' } = evt

  switch (event_type) {
    case 'agent_restart': {
      // Wipe all sprites for this agent
      for (const [id, sp] of state.sprites.entries()) {
        if (sp.agent === agent) {
          state.occupiedSlots.delete(sp.slotIndex)
          state.sprites.delete(id)
        }
      }
      break
    }

    case 'run_start':
    case 'session_resume': {
      if (!thread_id) break
      const sp = getOrCreateSprite(state, agent, thread_id, run_id, input_preview, nowMs)
      sp.runId = run_id
      if (input_preview) sp.inputPreview = input_preview
      if (sp.state !== 'ARRIVING') sp.state = 'ARRIVING'
      sp.lastEventMs = nowMs
      state.tick++
      sp.lastActiveTick = state.tick
      break
    }

    case 'run_end': {
      if (!thread_id) break
      const id = spriteId(agent, thread_id)
      const sp = state.sprites.get(id)
      if (sp) {
        sp.state = 'DEPARTING'
        sp.lastEventMs = nowMs
        state.tick++
        sp.lastActiveTick = state.tick
      }
      break
    }

    case 'node_start': {
      if (!thread_id) break
      const id = spriteId(agent, thread_id)
      const sp = state.sprites.get(id)
      if (sp) {
        const nodeType = evt.node_type || inferNodeType(evt.node_id ?? '')
        sp.state = { kind: 'WORKING', nodeType }
        sp.lastEventMs = nowMs
        state.tick++
        sp.lastActiveTick = state.tick
      }
      break
    }

    case 'node_end': {
      if (!thread_id) break
      const id = spriteId(agent, thread_id)
      const sp = state.sprites.get(id)
      if (sp && typeof sp.state === 'object' && sp.state.kind === 'WORKING') {
        sp.state = 'BRIEF_PAUSE'
        sp.lastEventMs = nowMs
        state.tick++
        sp.lastActiveTick = state.tick
      }
      break
    }

    case 'state_update': {
      if (!thread_id) break
      const id = spriteId(agent, thread_id)
      const sp = state.sprites.get(id)
      if (sp) {
        sp.lastEventMs = nowMs
        state.tick++
        sp.lastActiveTick = state.tick
      }
      break
    }
  }
}

/**
 * Apply a snapshot message from viewer WS.
 * Replaces entire sprite state for all agents mentioned.
 */
export function applySnapshot(state: OfficeState, msg: ViewerEvent, nowMs: number = Date.now()): void {
  if (msg.type !== 'snapshot' || !msg.active_runs) return

  // Clear all sprites and slots
  state.sprites.clear()
  state.occupiedSlots.clear()

  for (const run of msg.active_runs) {
    const { agent, thread_id = '', run_id = '', input_preview = '' } = run
    if (!agent || !thread_id) continue

    // Evict if at capacity
    if (state.sprites.size >= MAX_SPRITES) {
      evictLongestIdle(state)
    }

    const slot = nextFreeSlot(state.occupiedSlots)
    state.occupiedSlots.add(slot)
    state.tick++

    const sp: Sprite = {
      id: spriteId(agent, thread_id),
      agent,
      threadId: thread_id,
      runId: run_id,
      inputPreview: input_preview,
      state: 'IDLE', // snapshot restore → directly in place, no ARRIVING animation
      lastEventMs: nowMs,
      lastActiveTick: state.tick,
      slotIndex: slot,
    }
    state.sprites.set(sp.id, sp)
  }
}

/**
 * Run GC pass. Call periodically (e.g. every second from animation loop).
 * - Sprites with DEPARTING state are transitioned to IDLE after BRIEF_PAUSE_MS if needed.
 * - Sprites with no event for IDLE_TIMEOUT_MS → state = 'IDLE'
 * - Sprites with no event for LEAVE_TIMEOUT_MS → state = 'LEAVING'
 * - LEAVING sprites past LEAVING_ANIM_MS → remove
 * Returns set of sprite IDs that were removed.
 */
export const BRIEF_PAUSE_MS = 200
export const DEPARTING_MS = 2_000  // 2s walk-back animation before IDLE
export const LEAVING_ANIM_MS = 1_000

export function runGC(state: OfficeState, nowMs: number = Date.now()): Set<string> {
  const removed = new Set<string>()
  for (const [id, sp] of state.sprites.entries()) {
    const elapsed = nowMs - sp.lastEventMs

    if (sp.state === 'LEAVING') {
      if (elapsed >= LEAVING_ANIM_MS) {
        state.occupiedSlots.delete(sp.slotIndex)
        state.sprites.delete(id)
        removed.add(id)
      }
      continue
    }

    if (elapsed >= LEAVE_TIMEOUT_MS) {
      sp.state = 'LEAVING'
      continue
    }

    if (elapsed >= IDLE_TIMEOUT_MS && sp.state !== 'IDLE' && sp.state !== 'DEPARTING') {
      sp.state = 'IDLE'
      continue
    }

    if (sp.state === 'DEPARTING' && elapsed >= DEPARTING_MS) {
      sp.state = 'IDLE'
      continue
    }

    if (sp.state === 'BRIEF_PAUSE' && elapsed >= BRIEF_PAUSE_MS) {
      sp.state = 'IDLE'
    }
  }
  return removed
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function inferNodeType(nodeId: string): NodeType {
  const lower = nodeId.toLowerCase()
  if (lower.includes('claude')) return 'CLAUDE_SDK'
  if (lower.includes('gemini')) return 'GEMINI_API'
  if (lower.includes('heartbeat') || lower.includes('probe')) return 'HEARTBEAT'
  if (lower.includes('subgraph') || lower.includes('ref')) return 'SUBGRAPH_REF'
  return nodeId || 'UNKNOWN'
}

export function agentColor(agent: string): AgentColor {
  return AGENT_COLORS[agent.toLowerCase()] ?? 'gray'
}

/**
 * Shallow snapshot of current sprites for React diffing.
 * Returns a plain array (sorted by slotIndex) of sprite copies.
 */
export function getSpritesSnapshot(state: OfficeState): Sprite[] {
  return Array.from(state.sprites.values())
    .sort((a, b) => a.slotIndex - b.slotIndex)
}
