/**
 * Vitest unit tests for movement.ts
 * Pure TS — no PixiJS, no DOM.
 *
 * Test grid: 10×10, one 2×2 desk obstacle at tiles (3,1)-(4,2)
 *   Seat for slot 0: (3,4)
 *   Spawn tile: (0,5)
 */

import { describe, it, expect, beforeEach } from 'vitest'
import {
  findPath, buildOfficeGrid, createMovementChar, setIntent, updateMovement,
  tileCenterPx, Direction, TILE_SIZE,
  type OfficeTileGrid, type MovementChar, type LayoutParams,
} from './movement'

// ---------------------------------------------------------------------------
// Test grid helpers
// ---------------------------------------------------------------------------

/**
 * 10×10 hand-built grid:
 *   blocked: (3,1),(4,1),(3,2),(4,2)  — simulates a desk obstacle
 *   seat[0]: (3,4) — below desk
 *   spawn:   (0,5)
 */
function makeGrid(): OfficeTileGrid {
  const blocked = new Set<string>(['3,1', '4,1', '3,2', '4,2'])
  const walkable = []
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      if (!blocked.has(`${c},${r}`)) walkable.push({ col: c, row: r })
    }
  }
  return {
    cols: 10, rows: 10,
    blocked,
    seatTiles: [{ col: 3, row: 4 }],
    spawnTile:  { col: 0, row: 5 },
    walkableTiles: walkable,
  }
}

function advanceUntil(
  chars: Map<string, MovementChar>,
  grid: OfficeTileGrid,
  predicate: (ch: MovementChar) => boolean,
  maxTicks = 500,
  dtPerTick = 0.05,
): boolean {
  for (let i = 0; i < maxTicks; i++) {
    updateMovement(chars, grid, dtPerTick)
    const ch = chars.values().next().value!
    if (predicate(ch)) return true
  }
  return false
}

// ---------------------------------------------------------------------------
// 1. Tile utilities
// ---------------------------------------------------------------------------

describe('tileCenterPx', () => {
  it('returns center of tile in display pixels', () => {
    const { px, py } = tileCenterPx({ col: 2, row: 3 })
    expect(px).toBe(2 * TILE_SIZE + TILE_SIZE / 2)
    expect(py).toBe(3 * TILE_SIZE + TILE_SIZE / 2)
  })
})

// ---------------------------------------------------------------------------
// 2. findPath — BFS correctness
// ---------------------------------------------------------------------------

describe('findPath', () => {
  let grid: OfficeTileGrid

  beforeEach(() => { grid = makeGrid() })

  it('returns empty path when start === end', () => {
    expect(findPath(1, 1, 1, 1, grid)).toHaveLength(0)
  })

  it('finds direct horizontal path', () => {
    const path = findPath(0, 0, 4, 0, grid)
    expect(path.length).toBeGreaterThan(0)
    // Last step must be destination
    expect(path[path.length - 1]).toEqual({ col: 4, row: 0 })
  })

  it('routes around blocked tiles', () => {
    // Desk blocks (3,1)-(4,2). Path from (2,1) to (5,1) must detour via row 0 or row 3
    const path = findPath(2, 1, 5, 1, grid)
    expect(path.length).toBeGreaterThan(0)
    // No step in blocked zone
    for (const t of path) {
      expect(grid.blocked.has(`${t.col},${t.row}`)).toBe(false)
    }
    expect(path[path.length - 1]).toEqual({ col: 5, row: 1 })
  })

  it('returns empty path when destination is blocked', () => {
    expect(findPath(0, 0, 3, 1, grid)).toHaveLength(0)
  })

  it('returns empty path when destination is unreachable (fully enclosed)', () => {
    // Enclose (5,5) by adding extra blocked tiles on all 4 sides
    const enclosed: OfficeTileGrid = {
      ...grid,
      blocked: new Set([...grid.blocked, '5,4', '5,6', '4,5', '6,5', '5,5']),
    }
    expect(findPath(0, 0, 5, 5, enclosed)).toHaveLength(0)
  })

  it('path excludes start but includes end', () => {
    const path = findPath(0, 0, 3, 0, grid)
    expect(path[0]).not.toEqual({ col: 0, row: 0 })
    expect(path[path.length - 1]).toEqual({ col: 3, row: 0 })
  })
})

// ---------------------------------------------------------------------------
// 3. buildOfficeGrid — desk blocking and seat tile positions
// ---------------------------------------------------------------------------

describe('buildOfficeGrid', () => {
  it('blocks desk tiles for all slots', () => {
    // Tiny 2×1 grid: 2 columns, 1 row of slots; 200×200 canvas
    const p: LayoutParams = {
      canvasW: 200, canvasH: 200,
      gridCols: 2, gridRows: 1,
      slotW: 90, slotH: 90, padding: 10,
      deskX: 10, deskY: 10, deskW: 32, deskH: 32,
      charX: 10, charW: 16, charY: 42, charH: 32,
    }
    const g = buildOfficeGrid(p)
    // Slot 0 origin (10,10), desk at (20,20)-(51,51)
    // At TILE_SIZE=32: floor(20/32)=0 to floor(51/32)=1
    expect(g.blocked.has('0,0')).toBe(true)
    expect(g.blocked.has('1,0')).toBe(true)
    // Slot 1 origin (110,10), desk at (120,20)-(151,51)
    // tile cols: floor(120/32)=3 to floor(151/32)=4
    expect(g.blocked.has('3,0')).toBe(true)
    expect(g.blocked.has('4,0')).toBe(true)
  })

  it('computes correct seat tiles below desk', () => {
    const p: LayoutParams = {
      canvasW: 400, canvasH: 200,
      gridCols: 2, gridRows: 1,
      slotW: 180, slotH: 140, padding: 10,
      deskX: 44, deskY: 22, deskW: 96, deskH: 64,
      charX: 76, charW: 32, charY: 36, charH: 64,
    }
    const g = buildOfficeGrid(p)
    // Slot 0 origin (10,10), seatPx=(10+76+16, 10+36+64)=(102,110)
    // tile: (floor(102/32), floor(110/32)) = (3,3)
    expect(g.seatTiles[0]).toEqual({ col: 3, row: 3 })
    // Slot 1 origin (200,10), seatPx=(296,110) → tile (9,3)
    expect(g.seatTiles[1]).toEqual({ col: 9, row: 3 })
  })

  it('all seat tiles are walkable', () => {
    const p: LayoutParams = {
      canvasW: 800, canvasH: 900,
      gridCols: 4, gridRows: 5,
      slotW: 184, slotH: 148, padding: 10,
      deskX: 44, deskY: 22, deskW: 96, deskH: 64,
      charX: 76, charW: 32, charY: 36, charH: 64,
    }
    const g = buildOfficeGrid(p)
    for (const seat of g.seatTiles) {
      expect(g.blocked.has(`${seat.col},${seat.row}`)).toBe(false)
    }
  })
})

// ---------------------------------------------------------------------------
// 4. createMovementChar
// ---------------------------------------------------------------------------

describe('createMovementChar', () => {
  it('starts at spawn tile with WORK intent', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid)
    expect(ch.tileCol).toBe(grid.spawnTile.col)
    expect(ch.tileRow).toBe(grid.spawnTile.row)
    expect(ch.intent).toBe('WORK')
    expect(ch.walkState.kind).toBe('AT_TILE')
    expect(ch.done).toBe(false)
  })

  it('starts at seat when startAtSeat=true', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid, true)
    expect(ch.tileCol).toBe(grid.seatTiles[0].col)
    expect(ch.tileRow).toBe(grid.seatTiles[0].row)
  })

  it('px/py match tile centre', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid)
    const { px, py } = tileCenterPx(grid.spawnTile)
    expect(ch.px).toBe(px)
    expect(ch.py).toBe(py)
  })
})

// ---------------------------------------------------------------------------
// 5. WORK intent: ARRIVING → walks to seat
// ---------------------------------------------------------------------------

describe('ARRIVING → walks to seat', () => {
  it('character reaches seat from spawn with WORK intent', () => {
    const grid = makeGrid()
    const chars = new Map<string, MovementChar>()
    chars.set('hani:t1', createMovementChar('hani:t1', 0, grid))

    const reached = advanceUntil(chars, grid, (ch) =>
      ch.walkState.kind === 'AT_TILE' &&
      ch.tileCol === grid.seatTiles[0].col &&
      ch.tileRow === grid.seatTiles[0].row
    )
    expect(reached).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// 6. setIntent('WORK') preempts wander immediately
// ---------------------------------------------------------------------------

describe('setIntent WORK preempts wander', () => {
  it('redirects path to seat when WORK intent set mid-wander', () => {
    const grid = makeGrid()
    const seat = grid.seatTiles[0]
    // Start char at spawn, manually put it in WALKING toward (9,9) — opposite of seat
    const ch = createMovementChar('hani:t1', 0, grid, false)
    ch.intent = 'IDLE'
    ch.walkState = {
      kind: 'WALKING',
      path: [{ col: 1, row: 5 }, { col: 2, row: 5 }, { col: 9, row: 9 }],
      progress: 0,
    }

    // Preempt with WORK intent
    setIntent(ch, 'WORK', grid)

    // Path must now lead to seat (or char is already at seat)
    if (ch.walkState.kind === 'WALKING') {
      const ws = ch.walkState as { kind: 'WALKING'; path: any[] }
      const dest = ws.path[ws.path.length - 1]
      expect(dest).toEqual(seat)
    } else {
      // AT_TILE and already at seat — also valid
      expect(ch.tileCol).toBe(seat.col)
      expect(ch.tileRow).toBe(seat.row)
    }
  })

  it('sets AT_TILE immediately if already at seat', () => {
    const grid = makeGrid()
    const seat = grid.seatTiles[0]
    const ch = createMovementChar('hani:t1', 0, grid, true)
    ch.intent = 'IDLE'

    setIntent(ch, 'WORK', grid)

    expect(ch.intent).toBe('WORK')
    expect(ch.walkState.kind).toBe('AT_TILE')
    expect(ch.tileCol).toBe(seat.col)
    expect(ch.tileRow).toBe(seat.row)
  })
})

// ---------------------------------------------------------------------------
// 7. LEAVE intent: character walks to spawn and marks done
// ---------------------------------------------------------------------------

describe('LEAVE intent', () => {
  it('character reaches spawn and sets done=true', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid, true)  // start at seat
    setIntent(ch, 'LEAVE', grid)
    const chars = new Map([['hani:t1', ch]])

    const done = advanceUntil(chars, grid, (c) => c.done, 500)
    expect(done).toBe(true)
    expect(ch.tileCol).toBe(grid.spawnTile.col)
    expect(ch.tileRow).toBe(grid.spawnTile.row)
  })

  it('sets done=true immediately if already at spawn', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid)  // starts at spawn
    setIntent(ch, 'LEAVE', grid)
    expect(ch.done).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// 8. IDLE wander: character leaves seat and roams
// ---------------------------------------------------------------------------

describe('IDLE wander', () => {
  it('character eventually moves away from seat during IDLE', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid, true)
    ch.intent = 'IDLE'
    ch.seatTimer = 0  // no initial rest
    ch.wanderTimer = 0
    const chars = new Map([['hani:t1', ch]])

    const moved = advanceUntil(chars, grid, (c) =>
      c.walkState.kind === 'WALKING', 200)
    expect(moved).toBe(true)
  })

  it('wander does not step on blocked desk tiles', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid, true)
    ch.intent = 'IDLE'
    ch.seatTimer = 0
    ch.wanderTimer = 0
    ch.wanderLimit = 999  // keep wandering
    const chars = new Map([['hani:t1', ch]])

    // Simulate 60 seconds
    for (let i = 0; i < 1200; i++) {
      updateMovement(chars, grid, 0.05)
      expect(grid.blocked.has(`${ch.tileCol},${ch.tileRow}`)).toBe(false)
    }
  })
})

// ---------------------------------------------------------------------------
// 9. Direction tracking during walk
// ---------------------------------------------------------------------------

describe('direction tracking', () => {
  it('sets direction correctly when walking horizontally', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid)
    // Use IDLE intent so no repath to seat overrides the manual path
    ch.intent = 'IDLE'
    ch.walkState = { kind: 'WALKING', path: [{ col: 1, row: 5 }, { col: 2, row: 5 }], progress: 0 }

    updateMovement(new Map([['hani:t1', ch]]), grid, 0.01)

    expect(ch.dir).toBe(Direction.RIGHT)
  })

  it('sets DOWN direction when walking down', () => {
    const grid = makeGrid()
    const ch = createMovementChar('hani:t1', 0, grid)
    // Use IDLE intent so no repath overrides the manual path
    ch.intent = 'IDLE'
    ch.walkState = { kind: 'WALKING', path: [{ col: 0, row: 6 }], progress: 0 }

    updateMovement(new Map([['hani:t1', ch]]), grid, 0.01)

    expect(ch.dir).toBe(Direction.DOWN)
  })
})
