/**
 * ZenithLoom Observability v2 — PixiJS Pixel Office Renderer
 * src/sprite/pixiOffice.ts
 *
 * ── Sprite sheet frame layout (pixel-agents, MIT) ──────────────────────────
 *   Sheet: 112×96 px  ·  Frame: 16×32 px  ·  Grid: 7 cols × 3 rows
 *   Row 0 (Y=0 ): DOWN   Row 1 (Y=32): UP   Row 2 (Y=64): RIGHT (LEFT=flip)
 *   Cols 0-2: walk [0,1,2,1]@0.15s  ·  Cols 3-4: type [3,4]@0.30s  ·  Col 1: idle
 *
 * ── Architecture ─────────────────────────────────────────────────────────────
 *   Stage
 *     bgFloor   TilingSprite (full canvas, floor_0 tiled)
 *     slotLayer Container (20 static workstations: floor/desk/chair/PC labels)
 *     charLayer Container (N world-space character sprites, walk between tiles)
 *
 *   Movement engine (movement.ts, pure TS, no PIXI):
 *     buildOfficeGrid()  →  OfficeTileGrid (BFS-walkable tile map)
 *     createMovementChar / setIntent / updateMovement  →  dt game loop
 *
 *   PIXI.Ticker drives the game loop (updateMovement + visual sync every frame).
 *   render(officeState) syncs intents from the state machine; does not animate.
 */

import * as PIXI from 'pixi.js'
import {
  type Sprite as SpriteData,
  type OfficeState,
  type SpriteState,
  getSpritesSnapshot,
  agentColor,
  GRID_COLS,
  GRID_ROWS,
} from './stateMachine'
import {
  buildOfficeGrid,
  createMovementChar,
  setIntent,
  updateMovement,
  tileCenterPx,
  Direction,
  type MovementChar,
  type OfficeTileGrid,
  type MovementIntent,
} from './movement'

// ---------------------------------------------------------------------------
// Layout constants (must stay in sync with movement.ts LayoutParams)
// ---------------------------------------------------------------------------

const SCALE = 2

const CHAR_SRC_W = 16
const CHAR_SRC_H = 32
export const CHAR_W = CHAR_SRC_W * SCALE        // 32 px
export const CHAR_H = CHAR_SRC_H * SCALE        // 64 px

const DESK_W = 48 * SCALE                       // 96
const DESK_H = 32 * SCALE                       // 64
const PC_W   = 16 * SCALE                       // 32
const CHAIR_H = 16 * SCALE                      // 32

export const SLOT_W   = 184
export const SLOT_H   = 148
export const PADDING  = 10

const DESK_X  = Math.round((SLOT_W - DESK_W) / 2)         // 44
const DESK_Y  = 22
const PC_X    = DESK_X + Math.round((DESK_W - PC_W) / 2)  // 76
const PC_Y    = DESK_Y - 20
const CHAIR_X = Math.round((SLOT_W - CHAR_W) / 2)         // 76
const CHAIR_Y = DESK_Y + DESK_H - CHAIR_H + 4

const CHAR_X  = Math.round((SLOT_W - CHAR_W) / 2)         // 76 (default/seat offset)
const CHAR_Y  = DESK_Y + 14                               // 36

const LABEL_Y  = SLOT_H - 28
const LABEL2_Y = SLOT_H - 14

export const CANVAS_W = GRID_COLS * SLOT_W + (GRID_COLS + 1) * PADDING  // 766
export const CANVAS_H = GRID_ROWS * SLOT_H + (GRID_ROWS + 1) * PADDING  // 800

// Animation speeds (frames per 60fps ticker tick)
const WALK_SPEED = 1 / (0.15 * 60)   // ≈ 0.111
const TYPE_SPEED = 1 / (0.30 * 60)   // ≈ 0.056
const PC_SPEED   = 0.04

const AGENT_HALO_COLOR: Record<string, number> = {
  blue: 0x3b82f6, orange: 0xf97316, green: 0x22c55e, purple: 0xa855f7, gray: 0x6b7280,
}

const CHAR_ASSIGNMENT: Record<string, number> = { hani: 0, asa: 1, jei: 2, dan: 3 }
function charIndex(agent: string): number {
  return CHAR_ASSIGNMENT[agent.toLowerCase()] ?? (agent.charCodeAt(0) % 6)
}

// ---------------------------------------------------------------------------
// Asset types
// ---------------------------------------------------------------------------

interface CharTextures {
  walkDown:   PIXI.Texture[]  // cols [0,1,2,1], row 0
  walkUp:     PIXI.Texture[]  // cols [0,1,2,1], row 1
  walkRight:  PIXI.Texture[]  // cols [0,1,2,1], row 2
  typingDown: PIXI.Texture[]  // cols [3,4], row 0
  idleDown:   PIXI.Texture[]  // col 1, row 0 (static)
}

interface LoadedAssets {
  chars:    CharTextures[]
  floor:    PIXI.Texture
  desk:     PIXI.Texture
  chair:    PIXI.Texture
  pcFrames: PIXI.Texture[]
}

function subTexture(base: PIXI.Texture, col: number, row: number): PIXI.Texture {
  return new PIXI.Texture({
    source: base.source,
    frame:  new PIXI.Rectangle(col * CHAR_SRC_W, row * CHAR_SRC_H, CHAR_SRC_W, CHAR_SRC_H),
  })
}

async function loadAssets(): Promise<LoadedAssets> {
  const charPromises = Array.from({ length: 6 }, (_, i) =>
    PIXI.Assets.load<PIXI.Texture>(`/sprites/characters/char_${i}.png`),
  )
  const [floorTex, deskTex, chairTex, pc1, pc2, pc3, ...charSheets] = await Promise.all([
    PIXI.Assets.load<PIXI.Texture>('/sprites/floors/floor_0.png'),
    PIXI.Assets.load<PIXI.Texture>('/sprites/furniture/DESK_FRONT.png'),
    PIXI.Assets.load<PIXI.Texture>('/sprites/furniture/CUSHIONED_CHAIR_FRONT.png'),
    PIXI.Assets.load<PIXI.Texture>('/sprites/furniture/PC_FRONT_ON_1.png'),
    PIXI.Assets.load<PIXI.Texture>('/sprites/furniture/PC_FRONT_ON_2.png'),
    PIXI.Assets.load<PIXI.Texture>('/sprites/furniture/PC_FRONT_ON_3.png'),
    ...charPromises,
  ])

  const chars: CharTextures[] = charSheets.map((s) => ({
    walkDown:   [0, 1, 2, 1].map((c) => subTexture(s, c, 0)),
    walkUp:     [0, 1, 2, 1].map((c) => subTexture(s, c, 1)),
    walkRight:  [0, 1, 2, 1].map((c) => subTexture(s, c, 2)),
    typingDown: [3, 4].map((c)       => subTexture(s, c, 0)),
    idleDown:   [subTexture(s, 1, 0)],
  }))

  return { chars, floor: floorTex, desk: deskTex, chair: chairTex, pcFrames: [pc1, pc2, pc3] }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function stateLabel(state: SpriteState): string {
  if (typeof state === 'object' && state.kind === 'WORKING') return `⚙ ${state.nodeType.replace(/_/g, ' ')}`
  const m: Record<string, string> = {
    IDLE: 'idle', ARRIVING: '→', BRIEF_PAUSE: '⏸', DEPARTING: '←', LEAVING: '🚪',
  }
  return m[state as string] ?? (state as string)
}

function stateToIntent(state: SpriteState): MovementIntent {
  if (state === 'ARRIVING') return 'WORK'
  if (typeof state === 'object' && state.kind === 'WORKING') return 'WORK'
  if (state === 'BRIEF_PAUSE') return 'WORK'
  if (state === 'DEPARTING' || state === 'LEAVING') return 'LEAVE'
  return 'IDLE'  // IDLE
}

// ---------------------------------------------------------------------------
// SlotRenderer — static workstation furniture (no character, no halo)
// ---------------------------------------------------------------------------

class SlotRenderer {
  readonly container: PIXI.Container
  private pcAnim:    PIXI.AnimatedSprite
  private nameLabel: PIXI.Text
  private stateTag:  PIXI.Text

  constructor(private assets: LoadedAssets) {
    this.container = new PIXI.Container()

    const floor = new PIXI.TilingSprite({ texture: assets.floor, width: SLOT_W, height: SLOT_H })
    floor.tileScale.set(SCALE)

    const desk = new PIXI.Sprite(assets.desk)
    desk.scale.set(SCALE); desk.x = DESK_X; desk.y = DESK_Y

    const chair = new PIXI.Sprite(assets.chair)
    chair.scale.set(SCALE); chair.x = CHAIR_X; chair.y = CHAIR_Y

    this.pcAnim = new PIXI.AnimatedSprite(assets.pcFrames)
    this.pcAnim.scale.set(SCALE)
    this.pcAnim.x = PC_X; this.pcAnim.y = PC_Y
    this.pcAnim.animationSpeed = PC_SPEED
    this.pcAnim.loop = true
    this.pcAnim.visible = false

    this.nameLabel = new PIXI.Text({ text: '', style: { fontSize: 9, fill: 0xc9d1d9, fontFamily: 'monospace' } })
    this.nameLabel.x = 6; this.nameLabel.y = LABEL_Y

    this.stateTag = new PIXI.Text({ text: '', style: { fontSize: 8, fill: 0x6e7681, fontFamily: 'monospace' } })
    this.stateTag.x = 6; this.stateTag.y = LABEL2_Y

    this.container.addChild(floor, desk, this.pcAnim, chair, this.nameLabel, this.stateTag)
  }

  update(sp: SpriteData | null): void {
    if (!sp) {
      this.pcAnim.visible = false; this.pcAnim.stop()
      this.nameLabel.text = ''; this.stateTag.text = ''
      return
    }
    this.nameLabel.text = `${sp.agent}:${sp.threadId.slice(-6)}`
    this.stateTag.text = stateLabel(sp.state)
    const working = typeof sp.state === 'object' && sp.state.kind === 'WORKING'
    if (working && !this.pcAnim.visible) {
      this.pcAnim.visible = true; this.pcAnim.play()
    } else if (!working && this.pcAnim.visible) {
      this.pcAnim.visible = false; this.pcAnim.stop()
    }
  }

  destroy(): void { this.container.destroy({ children: true }) }
}

// ---------------------------------------------------------------------------
// CharacterSprite — world-space, follows movementChar px/py
// ---------------------------------------------------------------------------

class CharacterSprite {
  readonly container: PIXI.Container
  private anim:   PIXI.AnimatedSprite
  private halo:   PIXI.Graphics
  private bubble: PIXI.Text

  private lastDir: Direction = Direction.DOWN
  private lastIntent: MovementIntent = 'WORK'
  private lastIsWalking = false

  constructor(private assets: LoadedAssets, agent: string) {
    this.container = new PIXI.Container()

    this.halo = new PIXI.Graphics()

    this.anim = new PIXI.AnimatedSprite(this.assets.chars[0].idleDown)
    this.anim.scale.set(SCALE)
    this.anim.loop = true

    this.bubble = new PIXI.Text({
      text: '',
      style: { fontSize: 8, fill: 0xfcd34d, fontFamily: 'monospace' },
    })
    this.bubble.y = -16
    this.bubble.visible = false

    this.container.addChild(this.halo, this.anim, this.bubble)
    this._drawHalo(agent)
  }

  /** Sync visual to current movement + sprite state. Called each frame. */
  sync(ch: MovementChar, sp: SpriteData): void {
    // Position: centre-x, feet-y → sprite top-left
    this.container.x = ch.px - CHAR_W / 2
    this.container.y = ch.py - CHAR_H

    // Halo at feet
    this.halo.x = CHAR_W / 2
    this.halo.y = CHAR_H

    const isWalking = ch.walkState.kind === 'WALKING'
    const working   = typeof sp.state === 'object' && sp.state.kind === 'WORKING'
    const leaving   = ch.intent === 'LEAVE'

    // Fade when leaving
    this.container.alpha = leaving && ch.done ? 0 : (leaving ? Math.max(0.2, this.container.alpha - 0.005) : 1)

    // Rebuild animation only when direction / state changes
    const dirChanged    = ch.dir !== this.lastDir
    const walkChanged   = isWalking !== this.lastIsWalking
    const intentChanged = ch.intent !== this.lastIntent

    if (dirChanged || walkChanged || intentChanged) {
      this.lastDir = ch.dir
      this.lastIsWalking = isWalking
      this.lastIntent = ch.intent
      this._setAnim(ch, working)
    }

    // Bubble
    if (working) {
      const nodeType = (sp.state as { kind: 'WORKING'; nodeType: string }).nodeType
      this.bubble.text = nodeType.replace(/_/g, '·')
      this.bubble.visible = true
    } else {
      this.bubble.visible = false
    }
  }

  destroy(): void { this.container.destroy({ children: true }) }

  private _drawHalo(agent: string): void {
    const color = AGENT_HALO_COLOR[agentColor(agent)] ?? 0x6b7280
    this.halo.clear()
    this.halo.ellipse(0, 0, CHAR_W / 2, 5).fill({ color, alpha: 0.55 })
  }

  private _setAnim(ch: MovementChar, working: boolean): void {
    const tex = this.assets.chars[charIndex(
      ch.spriteId.split(':')[0] ?? 'unknown',
    )]

    if (working && !ch.walkState.kind === undefined) {
      // Should never be walking+working simultaneously, but handle gracefully
    }

    let frames: PIXI.Texture[]
    let speed = 0
    let flipX = false

    if (ch.walkState.kind === 'WALKING') {
      switch (ch.dir) {
        case Direction.UP:    frames = tex.walkUp;    break
        case Direction.RIGHT: frames = tex.walkRight; break
        case Direction.LEFT:  frames = tex.walkRight; flipX = true; break
        default:              frames = tex.walkDown;  break
      }
      speed = WALK_SPEED
    } else if (working) {
      frames = tex.typingDown
      speed  = TYPE_SPEED
    } else {
      frames = tex.idleDown
    }

    if (this.anim.textures !== frames) {
      this.anim.textures = frames
    }
    this.anim.scale.x = flipX ? -SCALE : SCALE
    this.anim.x       = flipX ? CHAR_W : 0

    if (speed > 0) {
      this.anim.animationSpeed = speed
      if (!this.anim.playing) this.anim.play()
    } else {
      this.anim.stop()
      this.anim.currentFrame = 0
    }
  }
}

// ---------------------------------------------------------------------------
// PixiOffice
// ---------------------------------------------------------------------------

export class PixiOffice {
  private app!:          PIXI.Application
  private assets!:       LoadedAssets
  private grid!:         OfficeTileGrid
  private slotLayer!:    PIXI.Container
  private charLayer!:    PIXI.Container
  private slotRenders:   Map<number, SlotRenderer>   = new Map()
  private charSprites:   Map<string, CharacterSprite> = new Map()
  private movementChars: Map<string, MovementChar>   = new Map()
  private latestSprites: Map<string, SpriteData>     = new Map()

  async init(canvas: HTMLCanvasElement, width: number, height: number): Promise<void> {
    this.app = new PIXI.Application()
    await this.app.init({ canvas, width, height, backgroundColor: 0x0d1117, antialias: false, resolution: 1 })

    this.assets = await loadAssets()

    // Build tile grid for pathfinding
    this.grid = buildOfficeGrid({
      canvasW: width, canvasH: height,
      gridCols: GRID_COLS, gridRows: GRID_ROWS,
      slotW: SLOT_W, slotH: SLOT_H, padding: PADDING,
      deskX: DESK_X, deskY: DESK_Y, deskW: DESK_W, deskH: DESK_H,
      charX: CHAR_X, charW: CHAR_W, charY: CHAR_Y, charH: CHAR_H,
    })

    // Background floor
    const bgFloor = new PIXI.TilingSprite({ texture: this.assets.floor, width, height })
    bgFloor.tileScale.set(SCALE)
    this.app.stage.addChild(bgFloor)

    // Slot furniture layer
    this.slotLayer = new PIXI.Container()
    this.app.stage.addChild(this.slotLayer)

    // Character layer (world-space, on top)
    this.charLayer = new PIXI.Container()
    this.app.stage.addChild(this.charLayer)

    // Ticker: dt game loop
    this.app.ticker.add(({ deltaMS }) => {
      const dt = Math.min(deltaMS / 1000, 0.1)
      updateMovement(this.movementChars, this.grid, dt)
      this._syncVisuals()
    })
  }

  /** Sync intents from state machine. Called from React component on WS events / GC ticks. */
  render(officeState: OfficeState): void {
    if (!this.app || !this.assets) return

    const sprites = getSpritesSnapshot(officeState)
    const activeIds = new Set(sprites.map((s) => s.id))

    // Remove stale characters
    for (const [id, charSp] of this.charSprites.entries()) {
      if (!activeIds.has(id)) {
        this.charLayer.removeChild(charSp.container)
        charSp.destroy()
        this.charSprites.delete(id)
        this.movementChars.delete(id)
        this.latestSprites.delete(id)
      }
    }

    // Update / create for active sprites
    for (const sp of sprites) {
      this.latestSprites.set(sp.id, sp)

      // Movement char
      let mch = this.movementChars.get(sp.id)
      if (!mch) {
        // Snapshot restore: start at seat; live arrival: start at spawn
        const startAtSeat = sp.state === 'IDLE' || sp.state === 'BRIEF_PAUSE'
        mch = createMovementChar(sp.id, sp.slotIndex, this.grid, startAtSeat)
        this.movementChars.set(sp.id, mch)
      }

      // Translate SpriteState → intent
      const intent = stateToIntent(sp.state)
      setIntent(mch, intent, this.grid)

      // Character visual
      let charSp = this.charSprites.get(sp.id)
      if (!charSp) {
        charSp = new CharacterSprite(this.assets, sp.agent)
        this.charLayer.addChild(charSp.container)
        this.charSprites.set(sp.id, charSp)
      }

      // Slot furniture
      let slotR = this.slotRenders.get(sp.slotIndex)
      if (!slotR) {
        slotR = new SlotRenderer(this.assets)
        const pos = this._slotPos(sp.slotIndex)
        slotR.container.x = pos.x; slotR.container.y = pos.y
        this.slotLayer.addChild(slotR.container)
        this.slotRenders.set(sp.slotIndex, slotR)
      }
      slotR.update(sp)
    }

    // Clear slot renderers that are no longer occupied
    const occupiedSlots = new Set(sprites.map((s) => s.slotIndex))
    for (const [idx, slotR] of this.slotRenders.entries()) {
      if (!occupiedSlots.has(idx)) {
        this.slotLayer.removeChild(slotR.container)
        slotR.destroy()
        this.slotRenders.delete(idx)
      }
    }
  }

  destroy(): void {
    this.app?.destroy(false, { children: true, texture: true })
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private _syncVisuals(): void {
    for (const [id, mch] of this.movementChars.entries()) {
      const sp  = this.latestSprites.get(id)
      const csp = this.charSprites.get(id)
      if (sp && csp) csp.sync(mch, sp)
    }
  }

  private _slotPos(slotIndex: number): { x: number; y: number } {
    const col = slotIndex % GRID_COLS
    const row = Math.floor(slotIndex / GRID_COLS)
    return { x: PADDING + col * (SLOT_W + PADDING), y: PADDING + row * (SLOT_H + PADDING) }
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export async function createPixiOffice(
  canvas: HTMLCanvasElement,
  _w: number,
  _h: number,
): Promise<PixiOffice> {
  const office = new PixiOffice()
  await office.init(canvas, CANVAS_W, CANVAS_H)
  return office
}
