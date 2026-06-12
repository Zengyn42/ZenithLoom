/**
 * ZenithLoom Observability v2 — PixiJS Pixel Office Renderer
 * src/sprite/pixiOffice.ts
 *
 * Renders the sprite office on a PixiJS Application canvas.
 * Consumes the pure state machine from stateMachine.ts.
 *
 * Grid: 4 columns × 5 rows = 20 workstation slots.
 * Each slot renders:
 *   - A desk (pixel art rectangle)
 *   - A sprite character (procedural colored pixel blob)
 *   - An activity indicator (color/animation hint per state)
 *   - Agent name + thread_id label
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

const SLOT_W = 180
const SLOT_H = 120
const PADDING = 16

const DESK_COLOR = 0x1e2a38
const DESK_BORDER = 0x2d4a6a
const FLOOR_COLOR = 0x0d1117

// Agent color palettes (body, glow)
const PALETTE: Record<string, [number, number]> = {
  blue:   [0x3b82f6, 0x60a5fa],
  orange: [0xf97316, 0xfb923c],
  green:  [0x22c55e, 0x4ade80],
  purple: [0xa855f7, 0xc084fc],
  gray:   [0x6b7280, 0x9ca3af],
}

// Activity indicator colors per state kind
const STATE_COLOR: Record<string, number> = {
  IDLE:        0x374151,
  ARRIVING:    0xfcd34d,
  WORKING_CLAUDE_SDK:    0x3b82f6,
  WORKING_GEMINI_API:    0x22c55e,
  WORKING_GEMINI_CLI:    0x16a34a,
  WORKING_SUBGRAPH_REF:  0xa855f7,
  WORKING_HEARTBEAT:     0xf97316,
  WORKING_DEFAULT:       0x94a3b8,
  BRIEF_PAUSE: 0xfbbf24,
  DEPARTING:   0xef4444,
  LEAVING:     0x1f2937,
}

function stateIndicatorColor(state: SpriteState): number {
  if (typeof state === 'object' && state.kind === 'WORKING') {
    const key = `WORKING_${state.nodeType}`
    return STATE_COLOR[key] ?? STATE_COLOR.WORKING_DEFAULT
  }
  return STATE_COLOR[state as string] ?? STATE_COLOR.IDLE
}

function stateLabel(state: SpriteState): string {
  if (typeof state === 'object' && state.kind === 'WORKING') {
    return `⚙ ${state.nodeType.replace('_', ' ')}`
  }
  const labels: Record<string, string> = {
    IDLE:        '💤 idle',
    ARRIVING:    '→ arriving',
    BRIEF_PAUSE: '⏸ pause',
    DEPARTING:   '← depart',
    LEAVING:     '🚪 leaving',
  }
  return labels[state as string] ?? state as string
}

// ---------------------------------------------------------------------------
// Slot sprite container
// ---------------------------------------------------------------------------

interface SlotDisplay {
  container: PIXI.Container
  body: PIXI.Graphics
  indicator: PIXI.Graphics
  label: PIXI.Text
  sublabel: PIXI.Text
  spriteId: string
}

// ---------------------------------------------------------------------------
// PixiOffice class
// ---------------------------------------------------------------------------

export class PixiOffice {
  private app: PIXI.Application
  private slots: Map<number, SlotDisplay> = new Map()
  private animTick = 0

  constructor(canvas: HTMLCanvasElement, width: number, height: number) {
    this.app = new PIXI.Application()
    // Initialization is async; caller must await init()
    ;(this as any)._canvas = canvas
    ;(this as any)._w = width
    ;(this as any)._h = height
  }

  async init(canvas: HTMLCanvasElement, width: number, height: number): Promise<void> {
    await this.app.init({
      canvas,
      width,
      height,
      backgroundColor: FLOOR_COLOR,
      antialias: false,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
    })
    this.app.ticker.add(() => this._animate())
  }

  /** Sync rendering state from office state machine snapshot. */
  render(officeState: OfficeState): void {
    const sprites = getSpritesSnapshot(officeState)
    const activeIds = new Set(sprites.map(s => s.id))

    // Remove displays for sprites no longer in state
    for (const [slotIdx, display] of this.slots.entries()) {
      if (!activeIds.has(display.spriteId)) {
        this.app.stage.removeChild(display.container)
        display.container.destroy({ children: true })
        this.slots.delete(slotIdx)
      }
    }

    // Update / create displays for each sprite
    for (const sp of sprites) {
      let display = this.slots.get(sp.slotIndex)
      if (!display || display.spriteId !== sp.id) {
        // Remove old display at this slot (different sprite)
        if (display) {
          this.app.stage.removeChild(display.container)
          display.container.destroy({ children: true })
        }
        display = this._createSlotDisplay(sp)
        this.slots.set(sp.slotIndex, display)
        this.app.stage.addChild(display.container)
      }
      this._updateSlotDisplay(display, sp)
    }
  }

  destroy(): void {
    this.app.destroy(false, { children: true, texture: true })
  }

  // ── Private ──────────────────────────────────────────────────────────────

  private _slotPosition(slotIndex: number): { x: number; y: number } {
    const col = slotIndex % GRID_COLS
    const row = Math.floor(slotIndex / GRID_COLS)
    return {
      x: PADDING + col * (SLOT_W + PADDING),
      y: PADDING + row * (SLOT_H + PADDING),
    }
  }

  private _createSlotDisplay(sp: SpriteData): SlotDisplay {
    const container = new PIXI.Container()
    const { x, y } = this._slotPosition(sp.slotIndex)
    container.x = x
    container.y = y

    // Desk background
    const desk = new PIXI.Graphics()
    desk.roundRect(0, 0, SLOT_W, SLOT_H, 6)
    desk.fill({ color: DESK_COLOR })
    desk.stroke({ color: DESK_BORDER, width: 1 })
    container.addChild(desk)

    // Character body (pixel blob: 16×20)
    const body = new PIXI.Graphics()
    container.addChild(body)

    // Activity indicator dot
    const indicator = new PIXI.Graphics()
    container.addChild(indicator)

    // Agent:thread label
    const label = new PIXI.Text({
      text: '',
      style: {
        fontSize: 11,
        fill: 0xc9d1d9,
        fontFamily: 'monospace',
      },
    })
    label.x = 8
    label.y = SLOT_H - 32
    container.addChild(label)

    // State sublabel
    const sublabel = new PIXI.Text({
      text: '',
      style: {
        fontSize: 10,
        fill: 0x6e7681,
        fontFamily: 'monospace',
      },
    })
    sublabel.x = 8
    sublabel.y = SLOT_H - 18
    container.addChild(sublabel)

    return { container, body, indicator, label, sublabel, spriteId: sp.id }
  }

  private _updateSlotDisplay(display: SlotDisplay, sp: SpriteData): void {
    const { body, indicator, label, sublabel } = display
    const color = agentColor(sp.agent)
    const [bodyColor] = PALETTE[color] ?? PALETTE.gray
    const indicatorColor = stateIndicatorColor(sp.state)

    // Draw pixel character body (centered horizontally)
    body.clear()
    const bx = SLOT_W / 2 - 8
    const by = 10

    // Head (8×8)
    body.rect(bx + 4, by, 8, 8).fill({ color: bodyColor })
    // Body (8×10)
    body.rect(bx + 4, by + 8, 8, 10).fill({ color: bodyColor })
    // Arms
    body.rect(bx, by + 8, 4, 8).fill({ color: bodyColor })
    body.rect(bx + 12, by + 8, 4, 8).fill({ color: bodyColor })
    // Legs
    body.rect(bx + 4, by + 18, 3, 8).fill({ color: bodyColor })
    body.rect(bx + 9, by + 18, 3, 8).fill({ color: bodyColor })

    // Working animation: pulsing glow
    if (typeof sp.state === 'object' && sp.state.kind === 'WORKING') {
      const [, glowColor] = PALETTE[color] ?? PALETTE.gray
      const pulse = 0.5 + 0.5 * Math.sin(this.animTick * 0.1)
      body.rect(bx - 2, by - 2, 20, 32)
        .fill({ color: glowColor, alpha: pulse * 0.25 })
    }

    // Activity indicator (dot, top-right of slot)
    indicator.clear()
    indicator.circle(SLOT_W - 14, 14, 6).fill({ color: indicatorColor })

    // Labels
    label.text = `${sp.agent}:${sp.threadId.slice(-6)}`
    sublabel.text = stateLabel(sp.state)
  }

  private _animate(): void {
    this.animTick++
  }
}

// ---------------------------------------------------------------------------
// Factory function for React integration
// ---------------------------------------------------------------------------

export async function createPixiOffice(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): Promise<PixiOffice> {
  const office = new PixiOffice(canvas, width, height)
  await office.init(canvas, width, height)
  return office
}
