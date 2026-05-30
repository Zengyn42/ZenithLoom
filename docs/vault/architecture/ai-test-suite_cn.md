# AI Test Suite — ApexCoder 编码能力测试标准

> 最后更新: 2026-04-17
> 状态: Game 1 (Snake Battle) 完成，Game 2 (Othello) 和 Game 3 (Planet Wars) 待实现

---

## 目的

通过让 ApexCoder 从零实现多个 AI 对战游戏，量化评估其编码能力。每个游戏覆盖不同的算法范式，形成综合能力评估矩阵。

## 统一交付标准

每个游戏必须包含：

| 组件 | 要求 |
|------|------|
| 游戏引擎 | 完整规则实现，headless 模式支持 |
| AI 梯度 | 至少 3 级: Random → Greedy/Aggressive → Strategic |
| Benchmark | 胜率矩阵、round-robin 锦标赛、性能计时 |
| 可视化 | 关键帧 PNG + 动画 GIF 回放 |
| 测试 | 单元测试 + 集成测试（AI 胜率断言） |
| 设计文档 | 算法选择、架构决策、性能分析 |

## Game 1: Snake Battle（双蛇对战）✅

**位置**: `EdenGateway/CompanyTests/CoderTest/snake_battle/`

**规则**: 30×30 网格，两条蛇同时移动，吃食物增长，碰撞判定（头对头长蛇赢、咬身体按 tail_portion 判定、自咬截断不死）。

**AI 层级**:
| AI | 算法 | 复杂度 |
|----|------|--------|
| RandomAI | 随机安全方向 | O(1) |
| GreedyAI | 最近食物贪心 | O(F) |
| AggressiveAI | 距离 < 6 时拦截对手 + 食物贪心 | O(F) |
| StrategicAI | 4 层优先级状态机: ABSOLUTE_SURVIVAL → SPACE_MAXIMIZE (Time-Stamped BFS Voronoi) → STRATEGIC_FEED → ATTACK (Minimax) | O(W×H) |

**Benchmark 基线** (40 games, 30×30):
| 对局 | StrategicAI 胜率 |
|------|-----------------|
| vs RandomAI | ~100% |
| vs GreedyAI | ~95% |
| vs AggressiveAI | ~75-82% |

**考验的算法能力**: Voronoi 分区、Flood Fill、Minimax 搜索、状态机设计

**待实现**: PURSUIT 模式（辩论方案已定型，待 ApexCoder 编码）

**可视化工具**: `snake_battle/visualize.py` — 关键帧 PNG + GIF 动画

---

## Game 2: Othello（黑白棋）⏳

**位置**: `EdenGateway/CompanyTests/CoderTest/othello/`（待创建）

**规则**: 标准 8×8 黑白棋规则。双方轮流下子，必须翻转对方棋子，无合法步时跳过，棋盘满或双方无步时结束，棋子多者胜。

**AI 层级设计**:
| AI | 算法 | 复杂度 |
|----|------|--------|
| RandomAI | 随机合法步 | O(1) |
| GreedyAI | 最大翻转数贪心 | O(N) |
| PositionalAI | 位置权重表（角 > 边 > 中心）| O(N) |
| StrategicAI | Alpha-Beta 剪枝 + 评估函数（角稳定性、行动力、边界棋子）| O(b^d) |

**考验的算法能力**: Alpha-Beta 剪枝、评估函数设计、位棋盘优化、迭代加深

**交付要求**: 同统一标准 + StrategicAI vs GreedyAI 胜率 ≥ 80%

---

## Game 3: Planet Wars（星球大战）⏳

**位置**: `EdenGateway/CompanyTests/CoderTest/planet_wars/`（待创建）

**规则**: 基于 Google AI Challenge 2010。2D 星图上分布若干星球，每颗星球有归属（P1/P2/中立）和驻军数。双方同时下令，从己方星球派舰队攻占其他星球。舰队飞行需要时间（距离 ÷ 速度）。星球每回合自动产兵。全灭对方或超时比总兵力。

**AI 层级设计**:
| AI | 算法 | 复杂度 |
|----|------|--------|
| RandomAI | 随机选源和目标 | O(1) |
| GreedyAI | 攻击最弱可达星球 | O(P²) |
| RushAI | 全力攻对手母星 | O(P) |
| StrategicAI | 多目标优化: 增长率最大化 + 前线防御 + 时机判断（对手舰队在途时趁虚） | O(P² × T) |

**考验的算法能力**: 多目标资源分配、时间规划（舰队在途）、博弈预测、启发式评估

**交付要求**: 同统一标准 + StrategicAI vs GreedyAI 胜率 ≥ 70%

---

## 能力评估矩阵

| 算法能力 | Snake Battle | Othello | Planet Wars |
|---------|:---:|:---:|:---:|
| 搜索树 (Minimax/AB) | ✓ | ✓✓✓ | ✓ |
| 空间推理 (Voronoi/Flood Fill) | ✓✓✓ | - | ✓ |
| 评估函数设计 | ✓✓ | ✓✓✓ | ✓✓ |
| 状态机 / 模式切换 | ✓✓✓ | ✓ | ✓✓ |
| 资源分配 / 多目标优化 | ✓ | - | ✓✓✓ |
| 时间规划 / 预测 | - | ✓ | ✓✓✓ |
| 性能优化 (< 100ms) | ✓✓ | ✓✓ | ✓ |

## 工作流

```
需求 → Hani 评估
  ├── 规则明确（如 Othello）→ 直接下发 ApexCoder
  └── 需要设计（如 Planet Wars 引擎）→ debate_design → 辩论结论 → ApexCoder
ApexCoder 编码 → Hani 验证（benchmark + 可视化）→ 记录结果到本文档
```
