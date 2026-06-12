/**
 * ZenithLoom Observability v2 — PixiJS Pixel Office Renderer
 * src/sprite/pixiOffice.ts
 *
 * Renders the sprite office on a PixiJS Application canvas.
 * Consumes pure state machine from stateMachine.ts.
 *
 * ── Frame layout (from pixel-agents source, characters.ts + pngDecoder.ts) ──
 *
 *   Sprite sheet: 112 × 96 px per character PNG (char_0.png … char_5.png)
 *   Frame size  : 16 × 32 px per frame
 *   Grid        : 7 columns × 3 rows
 *
 *   Row 0 (Y=0  ): DOWN  direction (facing viewer)
 *   Row 1 (Y=32 ): UP    direction (facing away)
 *   Row 2 (Y=64 ): RIGHT direction (LEFT = horizontally mirrored RIGHT)
 *
 *   Columns per row:
 *     col 0–2 : walk animation  — sequence [0,1,2,1] looping, 0.15 s/frame
 *     col 1   : idle static pose (walk neutral frame)
 *     col 3–4 : typing animation — [3,4] alternating, 0.30 s/frame
 *     col 5–6 : reading animation — [5,6] alternating, 0.30 s/frame (unused here)
 *
 *   State → animation mapping used in this renderer:
 *     ARRIVING           → walk DOWN  [0,1,2,1] @ 0.10
 *     WORKING            → typing DOWN [3,4]    @ 0.05
 *     BRIEF_PAUSE / IDLE → static DOWN col 1
 *     DEPARTING          → walk RIGHT [0,1,2,1] @ 0.10 (heading away)
 *     LEAVING            → walk RIGHT + alpha fade
 *
 * ── Workstation slot layout (SCALE=2, slot 180×148 px) ────────────────────
 *
 *   Background : TilingSprite with floor_0.png (16×16 → 32×32 tiled)
 *   Desk       : DESK_FRONT (48×32 → 96×64), centred at y=24
 *   PC         : PC_FRONT_ON_1/2/3 (16×32 → 32×64), on desk, hidden unless WORKING
 *   Chair      : CUSHIONED_CHAIR_FRONT (16×16 → 32×32), at desk bottom
 *   Character  : AnimatedSprite (16×32 → 32×64), in front of desk
 *   Halo       : coloured circle at character feet (agent brand colour)
 *   Labels     : name + state text row
 *   Node bubble: small text above character when WORKING
 *
 *   Z-order: floor < desk < pc < chair < halo < character < labels < bubble
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
  MAX_SPRITES,
} from './stateMachine'

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const SCALE = 2

// Source sprite dimensions (1x)
const CHAR_SRC_W = 16
const CHAR_SRC_H = 32
const CHAR_COLS = 7
const CHAR_ROWS = 3

// Scaled dimensions
const CHAR_W = CHAR_SRC_W * SCALE        // 32
const CHAR_H = CHAR_SRC_H * SCALE        // 64
const DESK_W = 48 * SCALE                // 96
const DESK_H = 32 * SCALE                // 64
const PC_W   = 16 * SCALE                // 32
const PC_H   = 32 * SCALE                // 64
const CHAIR_W = 16 * SCALE               // 32
const CHAIR_H = 16 * SCALE               // 32

export const SLOT_W   = 184
export const SLOT_H   = 148
export const PADDING  = 10

export const CANVAS_W = GRID_COLS * SLOT_W + (GRID_COLS + 1) * PADDING  // 766
export const CANVAS_H = GRID_ROWS * SLOT_H + (GRID_ROWS + 1) * PADDING  // 800

// Per-slot positions (relative to slot origin)
const DESK_X  = Math.round((SLOT_W - DESK_W) / 2)        // 44
const DESK_Y  = 22
const PC_X    = DESK_X + Math.round((DESK_W - PC_W) / 2) // 76 → on desk centre
const PC_Y    = DESK_Y - 20                               // partially above desk
const CHAR_X  = Math.round((SLOT_W - CHAR_W) / 2)         // 76
const CHAR_Y  = DESK_Y + 14                               // in front of desk
const CHAIR_X = Math.round((SLOT_W - CHAIR_W) / 2)        // 76
const CHAIR_Y = DESK_Y + DESK_H - CHAIR_H + 4            // at desk bottom

const HALO_CX = CHAR_X + CHAR_W / 2
const HALO_CY = CHAR_Y + CHAR_H - 4

const LABEL_Y  = SLOT_H - 28
const LABEL2_Y = SLOT_H - 14

// Animation speeds (AnimatedSprite animationSpeed = frames per ticker tick at 60fps)
const WALK_SPEED = 1 / (0.15 * 60)   // ≈ 0.111  (0.15s per frame)
const TYPE_SPEED = 1 / (0.30 * 60)   // ≈ 0.056  (0.30s per frame)
const PC_SPEED   = 0.04               // slow screen flicker

// Agent colour map (same as state machine)
const AGENT_HALO_COLOR: Record<string, number> = {
  blue:   0x3b82f6,
  orange: 0xf97316,
  green:  0x22c55e,
  purple: 0xa855f7,
  gray:   0x6b7280,
}

// char_N.png index per agent name
const CHAR_ASSIGNMENT: Record<string, number> = {
  hani: 0, asa: 1, jei: 2, dan: 3,
}
function charIndex(agent: string): number {
  return CHAR_ASSIGNMENT[agent.toLowerCase()] ??
    (agent.charCodeAt(0) % 6)
}

// ---------------------------------------------------------------------------
// Asset types
// ---------------------------------------------------------------------------

interface CharTextures {
  walkDown:   PIXI.Texture[]   // [0,1,2,1]
  typingDown: PIXI.Texture[]   // [3,4]
  idleDown:   PIXI.Texture[]   // [1]  (single frame → loop=false)
  walkRight:  PIXI.Texture[]   // row 2, [0,1,2,1]
}

interface LoadedAssets {
  chars:    CharTextures[]          // index 0–5
  floor:    PIXI.Texture
  desk:     PIXI.Texture
  chair:    PIXI.Texture
  pcFrames: PIXI.Texture[]          // [on1, on2, on3]
}

// ---------------------------------------------------------------------------
// Asset loader
// ---------------------------------------------------------------------------

function subTexture(base: PIXI.Texture, col: number, row: number): PIXI.Texture {
  return new PIXI.Texture({
    source: base.source,
    frame:  new PIXI.Rectangle(col * CHAR_SRC_W, row * CHAR_SRC_H, CHAR_SRC_W, CHAR_SRC_H),
  })
}

async function loadAssets(): Promise<LoadedAssets> {
  // Load all character sheets in parallel
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

  const chars: CharTextures[] = charSheets.map((sheet) => ({
    walkDown:   [0, 1, 2, 1].map((c) => subTexture(sheet, c, 0)),
    typingDown: [3, 4].map((c) =>       subTexture(sheet, c, 0)),
    idleDown:   [subTexture(sheet, 1, 0)],
    walkRight:  [0, 1, 2, 1].map((c) => subTexture(sheet, c, 2)),
  }))

  return {
    chars,
    floor: floorTex,
    desk:  deskTex,
    chair: chairTex,
    pcFrames: [pc1, pc2, pc3],
  }
}

// ---------------------------------------------------------------------------
// Slot state label helpers
// ---------------------------------------------------------------------------

function stateLabel(state: SpriteState): string {
  if (typeof state === 'object' && state.kind === 'WORKING') {
    return `⚙ ${state.nodeType.replace(/_/g, ' ')}`
  }
  const map: Record<string, string> = {
    IDLE:        'idle',
    ARRIVING:    '→ arriving',
    BRIEF_PAUSE: '⏸',
    DEPARTING:   '← departing',
    LEAVING:     '🚪 leaving',
  }
  return map[state as string] ?? (state as string)
}

function stateColor(state: SpriteState): number {
  if (typeof state === 'object') return 0x3b82f6
  const map: Record<string, number> = {
    IDLE:        0x4b5563,
    ARRIVING:    0xfcd34d,
    BRIEF_PAUSE: 0xfbbf24,
    DEPARTING:   0xef4444,
    LEAVING:     0x374151,
  }
  return map[state as string] ?? 0x6b7280
}

// ---------------------------------------------------------------------------
// SlotRenderer: manages PixiJS objects for one workstation slot
// ---------------------------------------------------------------------------

class SlotRenderer {
  readonly container: PIXI.Container

  private floor:     PIXI.TilingSprite
  private desk:      PIXI.Sprite
  private chair:     PIXI.Sprite
  private pcAnim:    PIXI.AnimatedSprite
  private charAnim:  PIXI.AnimatedSprite
  private halo:      PIXI.Graphics
  private nameLabel: PIXI.Text
  private stateTag:  PIXI.Text
  private bubble:    PIXI.Text

  private currentSpriteId = ''
  private currentState: SpriteState = 'IDLE'
  private charIdx = 0

  constructor(private assets: LoadedAssets) {
    this.container = new PIXI.Container()

    // Floor
    this.floor = new PIXI.TilingSprite({
      texture: assets.floor,
      width:   SLOT_W,
      height:  SLOT_H,
    })
    this.floor.tileScale.set(SCALE)

    // Desk
    this.desk = new PIXI.Sprite(assets.desk)
    this.desk.scale.set(SCALE)
    this.desk.x = DESK_X
    this.desk.y = DESK_Y

    // Chair
    this.chair = new PIXI.Sprite(assets.chair)
    this.chair.scale.set(SCALE)
    this.chair.x = CHAIR_X
    this.chair.y = CHAIR_Y

    // PC monitor (hidden by default)
    this.pcAnim = new PIXI.AnimatedSprite(assets.pcFrames)
    this.pcAnim.scale.set(SCALE)
    this.pcAnim.x = PC_X
    this.pcAnim.y = PC_Y
    this.pcAnim.animationSpeed = PC_SPEED
    this.pcAnim.loop = true
    this.pcAnim.visible = false

    // Halo (filled in drawHalo)
    this.halo = new PIXI.Graphics()

    // Character (start with idle texture)
    this.charAnim = new PIXI.AnimatedSprite(assets.chars[0].idleDown)
    this.charAnim.scale.set(SCALE)
    this.charAnim.x = CHAR_X
    this.charAnim.y = CHAR_Y
    this.charAnim.loop = true
    this.charAnim.visible = false

    // Labels
    this.nameLabel = new PIXI.Text({
      text: '',
      style: { fontSize: 9, fill: 0xc9d1d9, fontFamily: 'monospace' },
    })
    this.nameLabel.x = 6
    this.nameLabel.y = LABEL_Y

    this.stateTag = new PIXI.Text({
      text: '',
      style: { fontSize: 8, fill: 0x6e7681, fontFamily: 'monospace' },
    })
    this.stateTag.x = 6
    this.stateTag.y = LABEL2_Y

    this.bubble = new PIXI.Text({
      text: '',
      style: { fontSize: 8, fill: 0xfcd34d, fontFamily: 'monospace' },
    })
    this.bubble.x = CHAR_X
    this.bubble.y = CHAR_Y - 14
    this.bubble.visible = false

    // Z-order
    this.container.addChild(
      this.floor,
      this.desk,
      this.pcAnim,
      this.chair,
      this.halo,
      this.charAnim,
      this.nameLabel,
      this.stateTag,
      this.bubble,
    )
  }

  /** Update to reflect a live sprite from state machine. */
  update(sp: SpriteData): void {
    const charChanged = sp.id !== this.currentSpriteId
    const stateChanged = charChanged || JSON.stringify(sp.state) !== JSON.stringify(this.currentState)

    if (charChanged) {
      this.currentSpriteId = sp.id
      this.charIdx = charIndex(sp.agent)
      this.charAnim.visible = true
      this.nameLabel.text = `${sp.agent}:${sp.threadId.slice(-6)}`
      this._drawHalo(sp.agent)
    }

    if (stateChanged) {
      this.currentState = sp.state
      this._applyState(sp.state, sp)
    }

    // LEAVING fade
    if (sp.state === 'LEAVING') {
      this.container.alpha = Math.max(0, this.container.alpha - 0.01)
    } else {
      this.container.alpha = 1
    }
  }

  /** Show empty workstation (no active agent). */
  showEmpty(): void {
    this.currentSpriteId = ''
    this.charAnim.visible = false
    this.pcAnim.visible = false
    this.pcAnim.stop()
    this.halo.clear()
    this.nameLabel.text = ''
    this.stateTag.text = ''
    this.bubble.visible = false
    this.container.alpha = 1
  }

  destroy(): void {
    this.container.destroy({ children: true })
  }

  // ── Private ───────────────────────────────────────────────────────────────

  private _drawHalo(agent: string): void {
    const color = AGENT_HALO_COLOR[agentColor(agent)] ?? 0x6b7280
    this.halo.clear()
    this.halo.ellipse(HALO_CX, HALO_CY, CHAR_W / 2, 6)
    this.halo.fill({ color, alpha: 0.5 })
  }

  private _applyState(state: SpriteState, sp: SpriteData): void {
    const textures = this.assets.chars[this.charIdx]
    this.stateTag.text = stateLabel(state)
    this.stateTag.style.fill = stateColor(state) as any

    if (state === 'ARRIVING') {
      this._setAnim(textures.walkDown, WALK_SPEED)
      this.pcAnim.visible = false
      this.pcAnim.stop()
      this.bubble.visible = false

    } else if (typeof state === 'object' && state.kind === 'WORKING') {
      this._setAnim(textures.typingDown, TYPE_SPEED)
      this.pcAnim.visible = true
      this.pcAnim.play()
      this.bubble.text = state.nodeType.replace(/_/g, '·')
      this.bubble.visible = true

    } else if (state === 'DEPARTING' || state === 'LEAVING') {
      this._setAnim(textures.walkRight, WALK_SPEED, /*flipX=*/true)
      this.pcAnim.visible = false
      this.pcAnim.stop()
      this.bubble.visible = false

    } else {
      // IDLE / BRIEF_PAUSE
      this._setAnim(textures.idleDown, 0)
      this.pcAnim.visible = false
      this.pcAnim.stop()
      this.bubble.visible = false
    }
  }

  private _setAnim(textures: PIXI.Texture[], speed: number, flipX = false): void {
    this.charAnim.textures = textures
    this.charAnim.scale.x = flipX ? -SCALE : SCALE
    this.charAnim.x = flipX ? CHAR_X + CHAR_W : CHAR_X
    if (speed > 0) {
      this.charAnim.animationSpeed = speed
      this.charAnim.loop = true
      this.charAnim.play()
    } else {
      this.charAnim.stop()
      this.charAnim.currentFrame = 0
    }
  }
}

// ---------------------------------------------------------------------------
// PixiOffice
// ---------------------------------------------------------------------------

export class PixiOffice {
  private app!: PIXI.Application
  private assets!: LoadedAssets
  private slots: Map<number, SlotRenderer> = new Map()
  private bgFloor!: PIXI.TilingSprite

  async init(canvas: HTMLCanvasElement, width: number, height: number): Promise<void> {
    this.app = new PIXI.Application()
    await this.app.init({
      canvas,
      width,
      height,
      backgroundColor: 0x0d1117,
      antialias: false,
      resolution: 1,
    })

    this.assets = await loadAssets()

    // Full-canvas floor tile background
    this.bgFloor = new PIXI.TilingSprite({
      texture: this.assets.floor,
      width,
      height,
    })
    this.bgFloor.tileScale.set(SCALE)
    this.app.stage.addChild(this.bgFloor)
  }

  /** Sync rendering state from office state machine snapshot. */
  render(officeState: OfficeState): void {
    if (!this.app || !this.assets) return

    const sprites = getSpritesSnapshot(officeState)
    const activeSlots = new Set(sprites.map((s) => s.slotIndex))

    // Remove renderers for vacated slots
    for (const [idx, renderer] of this.slots.entries()) {
      if (!activeSlots.has(idx)) {
        this.app.stage.removeChild(renderer.container)
        renderer.destroy()
        this.slots.delete(idx)
      }
    }

    // Update / create renderers for active sprites
    for (const sp of sprites) {
      let renderer = this.slots.get(sp.slotIndex)
      if (!renderer) {
        renderer = new SlotRenderer(this.assets)
        const pos = this._slotPosition(sp.slotIndex)
        renderer.container.x = pos.x
        renderer.container.y = pos.y
        this.app.stage.addChild(renderer.container)
        this.slots.set(sp.slotIndex, renderer)
      }
      renderer.update(sp)
    }
  }

  destroy(): void {
    this.app?.destroy(false, { children: true, texture: true })
  }

  private _slotPosition(slotIndex: number): { x: number; y: number } {
    const col = slotIndex % GRID_COLS
    const row = Math.floor(slotIndex / GRID_COLS)
    return {
      x: PADDING + col * (SLOT_W + PADDING),
      y: PADDING + row * (SLOT_H + PADDING),
    }
  }
}

// ---------------------------------------------------------------------------
// Factory (consumed by SpriteOffice.tsx)
// ---------------------------------------------------------------------------

export async function createPixiOffice(
  canvas: HTMLCanvasElement,
  _width: number,
  _height: number,
): Promise<PixiOffice> {
  const office = new PixiOffice()
  await office.init(canvas, CANVAS_W, CANVAS_H)
  return office
}
