# AI Test Suite — ApexCoder Coding Capability Benchmarks

> Last updated: 2026-04-17
> Status: Game 1 (Snake Battle) complete; Game 2 (Othello) and Game 3 (Planet Wars) pending

---

## Purpose

Quantitatively evaluate ApexCoder's coding capability by having it implement multiple AI battle games from scratch. Each game covers a different algorithmic paradigm, forming a comprehensive capability assessment matrix.

## Unified Delivery Standard

Each game must include:

| Component | Requirement |
|-----------|-------------|
| Game engine | Full rules implementation, headless mode support |
| AI tiers | At least 3 levels: Random → Greedy/Aggressive → Strategic |
| Benchmark | Win-rate matrix, round-robin tournament, performance timing |
| Visualization | Key-frame PNG + animated GIF replay |
| Tests | Unit tests + integration tests (AI win-rate assertions) |
| Design doc | Algorithm choices, architecture decisions, performance analysis |

## Game 1: Snake Battle (Dual Snake) ✅

**Location**: `EdenGateway/CompanyTests/CoderTest/snake_battle/`

**Rules**: 30×30 grid, two snakes move simultaneously, eat food to grow, collision detection (head-to-head: longer snake wins; body bite: determined by tail_portion; self-bite: truncation, not death).

**AI Tiers**:
| AI | Algorithm | Complexity |
|----|-----------|-----------|
| RandomAI | Random safe direction | O(1) |
| GreedyAI | Nearest food greedy | O(F) |
| AggressiveAI | Intercept opponent when distance < 6 + food greedy | O(F) |
| StrategicAI | 4-tier priority state machine: ABSOLUTE_SURVIVAL → SPACE_MAXIMIZE (Time-Stamped BFS Voronoi) → STRATEGIC_FEED → ATTACK (Minimax) | O(W×H) |

**Benchmark Baseline** (40 games, 30×30):
| Match | StrategicAI Win Rate |
|-------|---------------------|
| vs RandomAI | ~100% |
| vs GreedyAI | ~95% |
| vs AggressiveAI | ~75-82% |

**Algorithms tested**: Voronoi partitioning, Flood Fill, Minimax search, state machine design

**Pending**: PURSUIT mode (debate proposal finalized, pending ApexCoder implementation)

**Visualization tool**: `snake_battle/visualize.py` — key-frame PNG + GIF animation

---

## Game 2: Othello (Reversi) ⏳

**Location**: `EdenGateway/CompanyTests/CoderTest/othello/` (to be created)

**Rules**: Standard 8×8 Othello rules. Players take turns placing pieces; must flip opponent's pieces; skip when no legal move; ends when board is full or both players have no moves; most pieces wins.

**AI Tier Design**:
| AI | Algorithm | Complexity |
|----|-----------|-----------|
| RandomAI | Random legal move | O(1) |
| GreedyAI | Maximum flip count greedy | O(N) |
| PositionalAI | Position weight table (corners > edges > center) | O(N) |
| StrategicAI | Alpha-Beta pruning + evaluation function (corner stability, mobility, edge pieces) | O(b^d) |

**Algorithms tested**: Alpha-Beta pruning, evaluation function design, bitboard optimization, iterative deepening

**Delivery requirement**: same unified standard + StrategicAI vs GreedyAI win rate ≥ 80%

---

## Game 3: Planet Wars ⏳

**Location**: `EdenGateway/CompanyTests/CoderTest/planet_wars/` (to be created)

**Rules**: Based on Google AI Challenge 2010. Planets distributed on a 2D star map, each planet has ownership (P1/P2/neutral) and garrison count. Both players issue orders simultaneously, dispatching fleets from their planets to capture others. Fleet travel takes time (distance ÷ speed). Planets auto-produce troops each turn. Win by eliminating the opponent or highest troop count at timeout.

**AI Tier Design**:
| AI | Algorithm | Complexity |
|----|-----------|-----------|
| RandomAI | Random source and target | O(1) |
| GreedyAI | Attack weakest reachable planet | O(P²) |
| RushAI | Full force attack opponent's home planet | O(P) |
| StrategicAI | Multi-objective optimization: growth rate maximization + front-line defense + timing (exploit vulnerability when opponent's fleet is in transit) | O(P² × T) |

**Algorithms tested**: multi-objective resource allocation, time planning (fleet in transit), game theory prediction, heuristic evaluation

**Delivery requirement**: same unified standard + StrategicAI vs GreedyAI win rate ≥ 70%

---

## Capability Assessment Matrix

| Algorithm Capability | Snake Battle | Othello | Planet Wars |
|---------------------|:---:|:---:|:---:|
| Search tree (Minimax/AB) | ✓ | ✓✓✓ | ✓ |
| Spatial reasoning (Voronoi/Flood Fill) | ✓✓✓ | - | ✓ |
| Evaluation function design | ✓✓ | ✓✓✓ | ✓✓ |
| State machine / mode switching | ✓✓✓ | ✓ | ✓✓ |
| Resource allocation / multi-objective optimization | ✓ | - | ✓✓✓ |
| Time planning / prediction | - | ✓ | ✓✓✓ |
| Performance optimization (< 100ms) | ✓✓ | ✓✓ | ✓ |

## Workflow

```
Requirements → technical_architect evaluation
  ├── Rules clear (e.g. Othello) → dispatch directly to ApexCoder
  └── Design needed (e.g. Planet Wars engine) → debate_design → debate conclusion → ApexCoder
ApexCoder codes → technical_architect validates (benchmark + visualization) → record results in this document
```
