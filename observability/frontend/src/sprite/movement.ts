/**
 * ZenithLoom Observability v2 — Pixel Office Movement Engine
 * src/sprite/movement.ts
 *
 * Pure TypeScript — zero PixiJS / zero DOM dependencies. Fully testable.
 *
 * Ported from pixel-agents (MIT, Pablo De Lucca):
 *   tileMap.ts   → findPath (BFS, 4-connected, identical algorithm)
 *   characters.ts → updateCharacter state machine (IDLE/WALK/TYPE states),
 *                   wander logic, seat-rest timer, in-walk repath on intent change
 *   constants.ts  → WALK_SPEED_PX_PER_SEC=48→96 (×2 scale), WANDER timings adjusted
 *
 * Changes from pixel-agents:
 *   - No isometric / tile-type system; single boolean walkability grid
 *   - Intent (WORK | IDLE | LEAVE) replaces isActive flag + CharacterState.TYPE
 *   - SEAT_REST reduced to 5–15 s (pixel-agents uses 120–240 s)
 *   - WANDER_PAUSE_MAX reduced to 8 s (was 20 s) for livelier office
 *   - TILE_SIZE = 32 display px (= 16 source px × SCALE 2)
 *   - Spawn/exit tile = left-edge middle row (simulated door)
 */

// ---------------------------------------------------------------------------
// Constants (ported from pixel-agents/webview-ui/src/constants.ts)
// ---------------------------------------------------------------------------

export const TILE_SIZE = 32                  // display pixels per tile
export const WALK_SPEED_PX_PER_SEC = 96     // 48 px/s at 1× → 96 at 2× scale
export const WALK_FRAME_DT = 0.15           // seconds per walk frame (unchanged)
export const TYPE_FRAME_DT  = 0.30          // seconds per type frame (unchanged)
export const WANDER_PAUSE_MIN_SEC = 1.0     // was 2.0 — faster first step
export const WANDER_PAUSE_MAX_SEC = 8.0     // was 20.0 — more lively
export const WANDER_MOVES_MIN = 2           // was 3
export const WANDER_MOVES_MAX = 5           // was 6
export const SEAT_REST_MIN_SEC = 5.0        // spec: 5-15 s (was 120 s)
export const SEAT_REST_MAX_SEC = 15.0       // spec (was 240 s)
export const MAX_DELTA_SEC = 0.1            // unchanged

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TilePos { col: number; row: number }

/** Matches pixel-agents Direction enum (used to pick sprite row / flip). */
export const Direction = { DOWN: 0, LEFT: 1, RIGHT: 2, UP: 3 } as const
export type Direction = (typeof Direction)[keyof typeof Direction]

/** Intent driven by the state machine's SpriteState. */
export type MovementIntent = 'WORK' | 'IDLE' | 'LEAVE'

type WalkState =
  | { kind: 'AT_TILE' }
  | { kind: 'WALKING'; path: TilePos[]; progress: number }

export interface MovementChar {
  spriteId: string
  slotIndex: number
  tileCol: number
  tileRow: number
  /** Display-pixel x of character centre (interpolated during walk). */
  px: number
  /** Display-pixel y of character feet (interpolated during walk). */
  py: number
  dir: Direction
  walkState: WalkState
  intent: MovementIntent
  wanderTimer: number
  wanderCount: number
  wanderLimit: number
  seatTimer: number
  /** True once character has reached the exit tile (LEAVE intent, AT_TILE at spawn). */
  done: boolean
}

/** Pre-computed tile grid; build once with buildOfficeGrid() and reuse. */
export interface OfficeTileGrid {
  cols: number
  rows: number
  blocked: Set<string>
  seatTiles: TilePos[]      // index = slotIndex
  spawnTile: TilePos        // entry/exit point (simulated door)
  walkableTiles: TilePos[]  // all non-blocked in-bounds tiles
}

/** Layout parameters passed from the renderer layer. */
export interface LayoutParams {
  canvasW: number; canvasH: number
  gridCols: number; gridRows: number
  slotW: number; slotH: number; padding: number
  /** All coordinates below are relative to slot origin (top-left of slot). */
  deskX: number; deskY: number; deskW: number; deskH: number
  charX: number; charW: number; charY: number; charH: number
}

// ---------------------------------------------------------------------------
// Grid building
// ---------------------------------------------------------------------------

function tileKey(c: number, r: number): string { return `${c},${r}` }

export function buildOfficeGrid(p: LayoutParams): OfficeTileGrid {
  const cols = Math.floor(p.canvasW / TILE_SIZE)
  const rows = Math.floor(p.canvasH / TILE_SIZE)
  const blocked = new Set<string>()
  const slotCount = p.gridCols * p.gridRows
  const seatTiles: TilePos[] = []

  for (let si = 0; si < slotCount; si++) {
    const sc = si % p.gridCols
    const sr = Math.floor(si / p.gridCols)
    const ox = p.padding + sc * (p.slotW + p.padding)
    const oy = p.padding + sr * (p.slotH + p.padding)

    // Block all tiles covered by the desk rectangle
    const dx1 = ox + p.deskX
    const dy1 = oy + p.deskY
    const tc1 = Math.floor(dx1 / TILE_SIZE)
    const tc2 = Math.floor((dx1 + p.deskW - 1) / TILE_SIZE)
    const tr1 = Math.floor(dy1 / TILE_SIZE)
    const tr2 = Math.floor((dy1 + p.deskH - 1) / TILE_SIZE)
    for (let tc = tc1; tc <= tc2; tc++) {
      for (let tr = tr1; tr <= tr2; tr++) {
        blocked.add(tileKey(tc, tr))
      }
    }

    // Seat tile = tile at character foot position (below desk)
    const spx = ox + p.charX + p.charW / 2
    const spy = oy + p.charY + p.charH
    seatTiles.push({ col: Math.floor(spx / TILE_SIZE), row: Math.floor(spy / TILE_SIZE) })
  }

  const walkableTiles: TilePos[] = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!blocked.has(tileKey(c, r))) walkableTiles.push({ col: c, row: r })
    }
  }

  const spawnTile: TilePos = { col: 0, row: Math.floor(rows / 2) }

  return { cols, rows, blocked, seatTiles, spawnTile, walkableTiles }
}

// ---------------------------------------------------------------------------
// Pathfinding — BFS on 4-connected grid (direct port of pixel-agents tileMap.ts)
// ---------------------------------------------------------------------------

export function findPath(
  startCol: number, startRow: number,
  endCol: number, endRow: number,
  grid: OfficeTileGrid,
): TilePos[] {
  if (startCol === endCol && startRow === endRow) return []

  const walkable = (c: number, r: number) =>
    c >= 0 && r >= 0 && c < grid.cols && r < grid.rows &&
    !grid.blocked.has(tileKey(c, r))

  if (!walkable(endCol, endRow)) return []

  const startKey = tileKey(startCol, startRow)
  const endKey   = tileKey(endCol, endRow)
  const visited  = new Set<string>([startKey])
  const parent   = new Map<string, string>()
  const queue: TilePos[] = [{ col: startCol, row: startRow }]
  const dirs = [{ dc: 0, dr: -1 }, { dc: 0, dr: 1 }, { dc: -1, dr: 0 }, { dc: 1, dr: 0 }]

  while (queue.length > 0) {
    const curr = queue.shift()!
    const ck = tileKey(curr.col, curr.row)
    if (ck === endKey) {
      const path: TilePos[] = []
      let k = endKey
      while (k !== startKey) {
        const [c, r] = k.split(',').map(Number)
        path.unshift({ col: c, row: r })
        k = parent.get(k)!
      }
      return path
    }
    for (const { dc, dr } of dirs) {
      const nc = curr.col + dc
      const nr = curr.row + dr
      const nk = tileKey(nc, nr)
      if (!visited.has(nk) && walkable(nc, nr)) {
        visited.add(nk)
        parent.set(nk, ck)
        queue.push({ col: nc, row: nr })
      }
    }
  }
  return []
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function tileCenterPx(tile: TilePos): { px: number; py: number } {
  return { px: tile.col * TILE_SIZE + TILE_SIZE / 2, py: tile.row * TILE_SIZE + TILE_SIZE / 2 }
}

function dirBetween(fc: number, fr: number, tc: number, tr: number): Direction {
  const dc = tc - fc; const dr = tr - fr
  if (dc > 0) return Direction.RIGHT
  if (dc < 0) return Direction.LEFT
  if (dr > 0) return Direction.DOWN
  return Direction.UP
}

function rand(min: number, max: number): number { return min + Math.random() * (max - min) }
function randInt(min: number, max: number): number {
  return min + Math.floor(Math.random() * (max - min + 1))
}

// ---------------------------------------------------------------------------
// Character creation
// ---------------------------------------------------------------------------

export function createMovementChar(
  spriteId: string,
  slotIndex: number,
  grid: OfficeTileGrid,
  startAtSeat = false,
): MovementChar {
  const tile = startAtSeat
    ? (grid.seatTiles[slotIndex] ?? grid.spawnTile)
    : grid.spawnTile
  const { px, py } = tileCenterPx(tile)
  return {
    spriteId, slotIndex,
    tileCol: tile.col, tileRow: tile.row,
    px, py,
    dir: Direction.DOWN,
    walkState: { kind: 'AT_TILE' },
    intent: 'WORK',
    wanderTimer: 0,
    wanderCount: 0,
    wanderLimit: randInt(WANDER_MOVES_MIN, WANDER_MOVES_MAX),
    seatTimer: rand(SEAT_REST_MIN_SEC, SEAT_REST_MAX_SEC),
    done: false,
  }
}

// ---------------------------------------------------------------------------
// Intent setter — may trigger immediate repath (pixel-agents setAgentActive logic)
// ---------------------------------------------------------------------------

export function setIntent(
  ch: MovementChar,
  intent: MovementIntent,
  grid: OfficeTileGrid,
): void {
  if (ch.intent === intent) return
  ch.intent = intent

  if (intent === 'WORK') {
    // Preempt wander: repath to seat immediately (matches pixel-agents mid-walk repath)
    const seat = grid.seatTiles[ch.slotIndex]
    if (!seat) return
    if (ch.tileCol === seat.col && ch.tileRow === seat.row) {
      ch.walkState = { kind: 'AT_TILE' }
    } else {
      const path = findPath(ch.tileCol, ch.tileRow, seat.col, seat.row, grid)
      if (path.length > 0) ch.walkState = { kind: 'WALKING', path, progress: 0 }
    }

  } else if (intent === 'LEAVE') {
    const sp = grid.spawnTile
    if (ch.tileCol === sp.col && ch.tileRow === sp.row) {
      ch.done = true
    } else {
      const path = findPath(ch.tileCol, ch.tileRow, sp.col, sp.row, grid)
      if (path.length > 0) {
        ch.walkState = { kind: 'WALKING', path, progress: 0 }
      } else {
        ch.done = true
      }
    }
  }
  // IDLE: don't repath; let updateMovement handle wander timer countdown
}

// ---------------------------------------------------------------------------
// Main update — call once per frame with dt in seconds
// ---------------------------------------------------------------------------

export function updateMovement(
  chars: Map<string, MovementChar>,
  grid: OfficeTileGrid,
  dt: number,
): void {
  const cdt = Math.min(dt, MAX_DELTA_SEC)
  for (const ch of chars.values()) {
    if (ch.done) continue
    if (ch.walkState.kind === 'WALKING') {
      _stepWalk(ch, grid, cdt)
    } else {
      _stepAtTile(ch, grid, cdt)
    }
  }
}

// ── AT_TILE handler ─────────────────────────────────────────────────────────

function _stepAtTile(ch: MovementChar, grid: OfficeTileGrid, dt: number): void {
  const seat  = grid.seatTiles[ch.slotIndex]
  const spawn = grid.spawnTile

  if (ch.intent === 'WORK') {
    const atSeat = seat && ch.tileCol === seat.col && ch.tileRow === seat.row
    if (!atSeat && seat) {
      const path = findPath(ch.tileCol, ch.tileRow, seat.col, seat.row, grid)
      if (path.length > 0) ch.walkState = { kind: 'WALKING', path, progress: 0 }
    }
    // Already at seat: just sit (no timer here — state machine owns that)
    return
  }

  if (ch.intent === 'LEAVE') {
    const atSpawn = ch.tileCol === spawn.col && ch.tileRow === spawn.row
    if (atSpawn) { ch.done = true; return }
    const path = findPath(ch.tileCol, ch.tileRow, spawn.col, spawn.row, grid)
    if (path.length > 0) ch.walkState = { kind: 'WALKING', path, progress: 0 }
    else ch.done = true
    return
  }

  // IDLE: wander countdown
  const atSeat = seat && ch.tileCol === seat.col && ch.tileRow === seat.row

  if (atSeat && ch.seatTimer > 0) {
    // Resting at seat before wandering
    ch.seatTimer -= dt
    return
  }

  ch.wanderTimer -= dt
  if (ch.wanderTimer > 0) return

  // Time to move — should we return to seat for a rest?
  if (ch.wanderCount >= ch.wanderLimit && seat) {
    if (atSeat) {
      // Reset rest
      ch.wanderCount = 0
      ch.wanderLimit = randInt(WANDER_MOVES_MIN, WANDER_MOVES_MAX)
      ch.seatTimer   = rand(SEAT_REST_MIN_SEC, SEAT_REST_MAX_SEC)
      ch.wanderTimer = rand(WANDER_PAUSE_MIN_SEC, WANDER_PAUSE_MAX_SEC)
      return
    }
    const path = findPath(ch.tileCol, ch.tileRow, seat.col, seat.row, grid)
    if (path.length > 0) {
      ch.walkState = { kind: 'WALKING', path, progress: 0 }
      ch.wanderTimer = rand(WANDER_PAUSE_MIN_SEC, WANDER_PAUSE_MAX_SEC)
      return
    }
  }

  // Pick a random walkable tile and walk to it
  if (grid.walkableTiles.length > 0) {
    const target = grid.walkableTiles[Math.floor(Math.random() * grid.walkableTiles.length)]
    const path = findPath(ch.tileCol, ch.tileRow, target.col, target.row, grid)
    if (path.length > 0) {
      ch.walkState = { kind: 'WALKING', path, progress: 0 }
      ch.wanderCount++
    }
  }
  ch.wanderTimer = rand(WANDER_PAUSE_MIN_SEC, WANDER_PAUSE_MAX_SEC)
}

// ── WALKING handler ─────────────────────────────────────────────────────────

function _stepWalk(ch: MovementChar, grid: OfficeTileGrid, dt: number): void {
  const ws = ch.walkState as { kind: 'WALKING'; path: TilePos[]; progress: number }

  // Mid-walk repath when WORK intent kicks in (pixel-agents pattern)
  if (ch.intent === 'WORK') {
    const seat = grid.seatTiles[ch.slotIndex]
    if (seat) {
      const dest = ws.path[ws.path.length - 1]
      if (!dest || dest.col !== seat.col || dest.row !== seat.row) {
        const newPath = findPath(ch.tileCol, ch.tileRow, seat.col, seat.row, grid)
        if (newPath.length > 0) { ws.path = newPath; ws.progress = 0 }
      }
    }
  }

  if (ws.path.length === 0) {
    // Arrived at destination
    const c = tileCenterPx({ col: ch.tileCol, row: ch.tileRow })
    ch.px = c.px; ch.py = c.py
    ch.dir = Direction.DOWN
    ch.walkState = { kind: 'AT_TILE' }

    if (ch.intent === 'LEAVE') {
      const atSpawn = ch.tileCol === grid.spawnTile.col && ch.tileRow === grid.spawnTile.row
      if (atSpawn) ch.done = true
    } else if (ch.intent === 'IDLE') {
      ch.wanderTimer = rand(WANDER_PAUSE_MIN_SEC, WANDER_PAUSE_MAX_SEC)
    }
    return
  }

  const next = ws.path[0]
  ch.dir = dirBetween(ch.tileCol, ch.tileRow, next.col, next.row)
  ws.progress += (WALK_SPEED_PX_PER_SEC / TILE_SIZE) * dt

  const from = tileCenterPx({ col: ch.tileCol, row: ch.tileRow })
  const to   = tileCenterPx(next)
  const t    = Math.min(ws.progress, 1)
  ch.px = from.px + (to.px - from.px) * t
  ch.py = from.py + (to.py - from.py) * t

  if (ws.progress >= 1) {
    ch.tileCol = next.col; ch.tileRow = next.row
    ch.px = to.px;         ch.py = to.py
    ws.path.shift(); ws.progress = 0
  }
}
