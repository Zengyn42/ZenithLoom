/**
 * Vitest unit tests for sprite state machine (stateMachine.ts)
 * Tests cover all state transitions, snapshot restore, GC, and MAX_SPRITES eviction.
 */

import { describe, it, expect, beforeEach } from 'vitest'
import {
  createOfficeState,
  applyEvent,
  applySnapshot,
  runGC,
  getSpritesSnapshot,
  MAX_SPRITES,
  IDLE_TIMEOUT_MS,
  LEAVE_TIMEOUT_MS,
  LEAVING_ANIM_MS,
  BRIEF_PAUSE_MS,
  DEPARTING_MS,
  type OfficeState,
  type ViewerEvent,
} from './stateMachine'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const T0 = 1_000_000 // base timestamp ms

function mkEvt(event_type: string, extra: Partial<ViewerEvent> = {}): ViewerEvent {
  return {
    ts: T0 / 1000,
    agent: 'hani',
    event_type,
    thread_id: 't1',
    run_id: 'r1',
    input_preview: 'hello',
    ...extra,
  }
}

let state: OfficeState

beforeEach(() => {
  state = createOfficeState()
})

// ---------------------------------------------------------------------------
// 1. run_start → ARRIVING
// ---------------------------------------------------------------------------

describe('run_start → ARRIVING', () => {
  it('creates sprite in ARRIVING state', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    const sprites = getSpritesSnapshot(state)
    expect(sprites).toHaveLength(1)
    expect(sprites[0].state).toBe('ARRIVING')
    expect(sprites[0].agent).toBe('hani')
    expect(sprites[0].threadId).toBe('t1')
    expect(sprites[0].runId).toBe('r1')
  })

  it('assigns a slot index in range 0-19', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.slotIndex).toBeGreaterThanOrEqual(0)
    expect(sp.slotIndex).toBeLessThan(MAX_SPRITES)
  })
})

// ---------------------------------------------------------------------------
// 2. node_start → WORKING
// ---------------------------------------------------------------------------

describe('node_start → WORKING', () => {
  it('transitions ARRIVING sprite to WORKING with nodeType', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('node_start', { node_id: 'claude_main' }), T0 + 100)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.state).toEqual({ kind: 'WORKING', nodeType: 'CLAUDE_SDK' })
  })

  it('infers GEMINI_API for gemini nodes', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('node_start', { node_id: 'gemini_researcher' }), T0 + 100)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.state).toEqual({ kind: 'WORKING', nodeType: 'GEMINI_API' })
  })

  it('uses node_type field directly when present', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('node_start', { node_id: 'some_node', node_type: 'CLAUDE_SDK' }), T0 + 100)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.state).toEqual({ kind: 'WORKING', nodeType: 'CLAUDE_SDK' })
  })

  it('falls back to inferring from node_id when node_type missing', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('node_start', { node_id: 'gemini_research' }), T0 + 100)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.state).toEqual({ kind: 'WORKING', nodeType: 'GEMINI_API' })
  })

  it('ignores node_start for unknown thread', () => {
    applyEvent(state, mkEvt('node_start', { node_id: 'claude_main', thread_id: 'ghost' }), T0)
    expect(getSpritesSnapshot(state)).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// 3. node_end → BRIEF_PAUSE
// ---------------------------------------------------------------------------

describe('node_end → BRIEF_PAUSE', () => {
  it('transitions WORKING to BRIEF_PAUSE', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('node_start', { node_id: 'claude_main' }), T0 + 100)
    applyEvent(state, mkEvt('node_end'), T0 + 200)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.state).toBe('BRIEF_PAUSE')
  })

  it('BRIEF_PAUSE → IDLE after BRIEF_PAUSE_MS via GC', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('node_start', { node_id: 'n1' }), T0 + 100)
    applyEvent(state, mkEvt('node_end'), T0 + 200)
    // GC at T0+200 (0ms elapsed for the node_end) — should still be BRIEF_PAUSE
    runGC(state, T0 + 200)
    expect(getSpritesSnapshot(state)[0].state).toBe('BRIEF_PAUSE')
    // GC at T0+200+BRIEF_PAUSE_MS — should flip to IDLE
    runGC(state, T0 + 200 + BRIEF_PAUSE_MS + 1)
    expect(getSpritesSnapshot(state)[0].state).toBe('IDLE')
  })
})

// ---------------------------------------------------------------------------
// 4. run_end → DEPARTING
// ---------------------------------------------------------------------------

describe('run_end → DEPARTING', () => {
  it('transitions sprite to DEPARTING', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('run_end'), T0 + 1000)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.state).toBe('DEPARTING')
  })

  it('ignores run_end for unknown thread', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('run_end', { thread_id: 'ghost' }), T0 + 1000)
    expect(getSpritesSnapshot(state)[0].state).toBe('ARRIVING')
  })

  it('DEPARTING → IDLE after DEPARTING_MS via GC', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('run_end'), T0 + 1000)
    expect(getSpritesSnapshot(state)[0].state).toBe('DEPARTING')
    // Not yet
    runGC(state, T0 + 1000 + DEPARTING_MS - 1)
    expect(getSpritesSnapshot(state)[0].state).toBe('DEPARTING')
    // Now
    runGC(state, T0 + 1000 + DEPARTING_MS + 1)
    expect(getSpritesSnapshot(state)[0].state).toBe('IDLE')
  })
})

// ---------------------------------------------------------------------------
// 5. agent_restart wipes all sprites for that agent
// ---------------------------------------------------------------------------

describe('agent_restart clears agent sprites', () => {
  it('removes all sprites for the restarted agent', () => {
    applyEvent(state, mkEvt('run_start', { thread_id: 't1' }), T0)
    applyEvent(state, mkEvt('run_start', { thread_id: 't2' }), T0)
    applyEvent(state, mkEvt('agent_restart'), T0 + 100)
    expect(getSpritesSnapshot(state)).toHaveLength(0)
  })

  it('does not remove sprites for other agents', () => {
    applyEvent(state, mkEvt('run_start', { agent: 'asa', thread_id: 'ta' }), T0)
    applyEvent(state, mkEvt('run_start', { agent: 'hani', thread_id: 't1' }), T0)
    applyEvent(state, { ts: T0 / 1000, agent: 'hani', event_type: 'agent_restart' }, T0 + 100)
    const sprites = getSpritesSnapshot(state)
    expect(sprites).toHaveLength(1)
    expect(sprites[0].agent).toBe('asa')
  })

  it('frees slots so new sprites can occupy them', () => {
    applyEvent(state, mkEvt('run_start', { thread_id: 't1' }), T0)
    const oldSlot = getSpritesSnapshot(state)[0].slotIndex
    applyEvent(state, mkEvt('agent_restart'), T0 + 100)
    applyEvent(state, mkEvt('run_start', { thread_id: 'tnew' }), T0 + 200)
    const newSlot = getSpritesSnapshot(state)[0].slotIndex
    expect(newSlot).toBe(oldSlot) // slot 0 reused
  })
})

// ---------------------------------------------------------------------------
// 6. Snapshot restore
// ---------------------------------------------------------------------------

describe('applySnapshot restores state', () => {
  it('creates sprites in IDLE state (direct placement, no ARRIVING animation)', () => {
    applySnapshot(state, {
      ts: T0 / 1000,
      agent: '',
      event_type: '',
      type: 'snapshot',
      active_runs: [
        { ts: T0 / 1000, agent: 'hani', event_type: 'run_start', thread_id: 't1', run_id: 'r1', input_preview: 'test' },
        { ts: T0 / 1000, agent: 'asa', event_type: 'run_start', thread_id: 'ta', run_id: 'ra', input_preview: '' },
      ],
    }, T0)
    const sprites = getSpritesSnapshot(state)
    expect(sprites).toHaveLength(2)
    expect(sprites.every(s => s.state === 'IDLE')).toBe(true)
  })

  it('clears all existing sprites on snapshot', () => {
    applyEvent(state, mkEvt('run_start', { thread_id: 'old' }), T0)
    applySnapshot(state, {
      ts: T0 / 1000, agent: '', event_type: '', type: 'snapshot',
      active_runs: [
        { ts: T0 / 1000, agent: 'hani', event_type: 'run_start', thread_id: 'fresh', run_id: 'r2' },
      ],
    }, T0 + 1000)
    const sprites = getSpritesSnapshot(state)
    expect(sprites).toHaveLength(1)
    expect(sprites[0].threadId).toBe('fresh')
  })

  it('ignores empty snapshot (no active_runs)', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applySnapshot(state, { ts: T0 / 1000, agent: '', event_type: '', type: 'snapshot', active_runs: [] }, T0 + 100)
    expect(getSpritesSnapshot(state)).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// 7. GC: IDLE_TIMEOUT → IDLE
// ---------------------------------------------------------------------------

describe('GC idle timeout', () => {
  it('sets sprite to IDLE after IDLE_TIMEOUT_MS with no events', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('node_start', { node_id: 'n1' }), T0 + 100)
    // GC runs at node_start_time + IDLE_TIMEOUT_MS + 1
    runGC(state, T0 + 100 + IDLE_TIMEOUT_MS + 1)
    expect(getSpritesSnapshot(state)[0].state).toBe('IDLE')
  })

  it('does not touch sprites that received recent events', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('node_start', { node_id: 'n1' }), T0 + IDLE_TIMEOUT_MS - 100)
    runGC(state, T0 + IDLE_TIMEOUT_MS + 1)
    // lastEventMs updated — within timeout
    const sp = getSpritesSnapshot(state)[0]
    // The working node was started at T0+IDLE_TIMEOUT_MS-100, GC at T0+IDLE_TIMEOUT_MS+1 = 101ms elapsed
    expect(sp.state).toEqual({ kind: 'WORKING', nodeType: expect.any(String) })
  })
})

// ---------------------------------------------------------------------------
// 8. GC: LEAVE_TIMEOUT → LEAVING → remove
// ---------------------------------------------------------------------------

describe('GC leave timeout and removal', () => {
  it('transitions to LEAVING after LEAVE_TIMEOUT_MS', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    runGC(state, T0 + LEAVE_TIMEOUT_MS + 1)
    expect(getSpritesSnapshot(state)[0].state).toBe('LEAVING')
  })

  it('removes sprite after LEAVING_ANIM_MS in LEAVING state', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    runGC(state, T0 + LEAVE_TIMEOUT_MS + 1)
    // Run GC again after anim duration
    runGC(state, T0 + LEAVE_TIMEOUT_MS + LEAVING_ANIM_MS + 1)
    expect(getSpritesSnapshot(state)).toHaveLength(0)
  })

  it('returns removed IDs from runGC', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    runGC(state, T0 + LEAVE_TIMEOUT_MS + 1) // → LEAVING
    const removed = runGC(state, T0 + LEAVE_TIMEOUT_MS + LEAVING_ANIM_MS + 1)
    expect(removed.has('hani:t1')).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// 9. MAX_SPRITES cap: evicts longest-IDLE on overflow
// ---------------------------------------------------------------------------

describe('MAX_SPRITES eviction', () => {
  it('never exceeds MAX_SPRITES sprites', () => {
    // Fill up to MAX with IDLE sprites
    for (let i = 0; i < MAX_SPRITES; i++) {
      applyEvent(state, mkEvt('run_start', { thread_id: `t${i}` }), T0 + i)
    }
    expect(state.sprites.size).toBe(MAX_SPRITES)

    // Force all to IDLE
    runGC(state, T0 + IDLE_TIMEOUT_MS + 1)

    // Adding one more should evict the oldest IDLE
    applyEvent(state, mkEvt('run_start', { thread_id: 'overflow' }), T0 + 99999)
    expect(state.sprites.size).toBe(MAX_SPRITES)
    // New sprite should exist
    expect(state.sprites.has('hani:overflow')).toBe(true)
  })

  it('evicted sprite slot is freed and reassigned', () => {
    for (let i = 0; i < MAX_SPRITES; i++) {
      applyEvent(state, mkEvt('run_start', { thread_id: `t${i}` }), T0 + i)
    }
    runGC(state, T0 + IDLE_TIMEOUT_MS + 1)

    applyEvent(state, mkEvt('run_start', { thread_id: 'new' }), T0 + 99999)
    const newSprite = state.sprites.get('hani:new')!
    expect(newSprite.slotIndex).toBeGreaterThanOrEqual(0)
    expect(newSprite.slotIndex).toBeLessThan(MAX_SPRITES)
    // All slots still unique
    const slots = Array.from(state.sprites.values()).map(s => s.slotIndex)
    expect(new Set(slots).size).toBe(slots.length)
  })
})

// ---------------------------------------------------------------------------
// 10. Full lifecycle: ARRIVING → WORKING → BRIEF_PAUSE → WORKING → DEPARTING → IDLE
// ---------------------------------------------------------------------------

describe('full lifecycle flow', () => {
  it('walks through complete state machine lifecycle', () => {
    // run_start → ARRIVING
    applyEvent(state, mkEvt('run_start'), T0)
    expect(getSpritesSnapshot(state)[0].state).toBe('ARRIVING')

    // node_start → WORKING
    applyEvent(state, mkEvt('node_start', { node_id: 'claude_main', node_type: 'CLAUDE_SDK' }), T0 + 500)
    expect(getSpritesSnapshot(state)[0].state).toEqual({ kind: 'WORKING', nodeType: 'CLAUDE_SDK' })

    // node_end → BRIEF_PAUSE
    applyEvent(state, mkEvt('node_end'), T0 + 5000)
    expect(getSpritesSnapshot(state)[0].state).toBe('BRIEF_PAUSE')

    // GC: BRIEF_PAUSE → IDLE
    runGC(state, T0 + 5000 + BRIEF_PAUSE_MS + 1)
    expect(getSpritesSnapshot(state)[0].state).toBe('IDLE')

    // Another node_start → WORKING again
    applyEvent(state, mkEvt('node_start', { node_id: 'gemini_node', node_type: 'GEMINI_API' }), T0 + 6000)
    expect(getSpritesSnapshot(state)[0].state).toEqual({ kind: 'WORKING', nodeType: 'GEMINI_API' })

    // run_end → DEPARTING
    applyEvent(state, mkEvt('run_end'), T0 + 10000)
    expect(getSpritesSnapshot(state)[0].state).toBe('DEPARTING')

    // GC: DEPARTING → IDLE
    runGC(state, T0 + 10000 + DEPARTING_MS + 1)
    expect(getSpritesSnapshot(state)[0].state).toBe('IDLE')

    // GC: IDLE for 5 min → LEAVING
    runGC(state, T0 + 10000 + DEPARTING_MS + LEAVE_TIMEOUT_MS + 1)
    expect(getSpritesSnapshot(state)[0].state).toBe('LEAVING')

    // GC: LEAVING → removed
    const removed = runGC(state, T0 + 10000 + DEPARTING_MS + LEAVE_TIMEOUT_MS + LEAVING_ANIM_MS + 2)
    expect(removed.has('hani:t1')).toBe(true)
    expect(getSpritesSnapshot(state)).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// 11. session_resume treated like run_start
// ---------------------------------------------------------------------------

describe('session_resume', () => {
  it('creates sprite in ARRIVING state like run_start', () => {
    applyEvent(state, mkEvt('session_resume', { run_id: 'resumed', input_preview: 'ctx' }), T0)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.state).toBe('ARRIVING')
    expect(sp.runId).toBe('resumed')
    expect(sp.inputPreview).toBe('ctx')
  })

  it('updates existing sprite run_id on re-resume', () => {
    applyEvent(state, mkEvt('run_start'), T0)
    applyEvent(state, mkEvt('session_resume', { run_id: 'resumed-2' }), T0 + 100)
    const sp = getSpritesSnapshot(state)[0]
    expect(sp.runId).toBe('resumed-2')
  })
})
