# ColonyCoder 模型对比实验 — 2026-04-13

> 测试任务：双蛇对战游戏 (Snake Battle)，curses 终端 UI，两个 AI 蛇自动对战

## 实验设置

- **ColonyCoder 架构**：plan (Claude+Gemini 辩论) → execute (本地 LLM code_gen) → qa (Claude E2E)
- **ApexCoder 架构**：单 Claude Opus 4.6 节点，全工具集
- **executor 子图**：inject_task_context → code_gen (可换模型) → run_tests → test_route
- **硬件**：WSL2, Ollama 本地推理

## 对比结果

| | ApexCoder (Claude Opus) | ColonyCoder (Qwen3.5:27b) | ColonyCoder (Gemma4:31b) |
|---|---|---|---|
| **总耗时** | **2 分 3 秒** | ~50 分钟（含 QA 超时循环） | ~15 分钟（planner 8 分 + executor 4 分） |
| **executor 耗时** | N/A | ~13 分钟 | **3 分 53 秒** |
| **tool call 迭代** | N/A | 25/25（用完） | **8/25** |
| **tool call 稳定性** | N/A | 崩溃（参数丢失 `path`） | 稳定（无崩溃） |
| **代码行数** | 517 行 | 938 行 | 261 行 |
| **代码质量** | 高 | 中 | 低 |
| **能实际运行** | 能 | 能（QA 修 bug 后） | 大概率崩（addch 参数错） |
| **单元测试** | 自验 20 局模拟 | 36 个 pass | pass（但测试太浅） |
| **QA 发现 bug** | N/A | 2 个（1 major, 1 minor） | 未进入 QA |
| **API 成本** | Claude Opus token（高） | Claude(辩论+QA) + Gemini(免费) + Ollama(免费) | 同左 |

### Qwen3-Coder-Next (79.7B) — 测试进行中

在实验结束时启动了 `qwen3-coder-next:q4_K_M` (79.7B) 的 executor 测试，tool calling 验证通过（参数完整且顺序正确），结果待补充。

## 关键发现

### 1. ApexCoder 远超 ColonyCoder

ApexCoder（单 Claude Opus 节点）在所有维度碾压 ColonyCoder：
- 速度快 25 倍
- 代码质量更高（A* pathfinding、proper 初始化、自验 20 局）
- 无需 QA 循环，一次通过

**结论**：对于这个复杂度的任务，单个强模型比多弱模型协作更有效。

### 2. Qwen3.5:27b 的 tool calling 有严重问题

**根本原因**：Ollama 将 Qwen3.5 映射到 JSON 格式的 tool calling pipeline，但 Qwen3.5 被训练使用 XML 格式（`<function=name><parameter=key>value</parameter></function>`）。已知 Ollama issue #14493。

具体表现：
- 反复全量重写文件（没有 edit 能力，每次 write_file 都重新生成 500+ 行）
- 迭代后期（msgs > 40）开始丢参数（`write_file(['content'])` 缺 `path`）
- 25 次迭代用完，效率极低

**缓解措施（已实施）**：
- 在 `_call_with_tools` 中加了 try/except TypeError，tool call 参数错误时返回错误信息给模型重试，不再直接崩溃

### 3. Gemma4:31b 稳定但代码质量差

- Tool calling 稳定（8 次迭代完成，无崩溃）
- 但生成的代码有明显错误（`addch` 参数顺序错、蛇初始长度只有 1、硬编码尺寸）
- 单元测试通过 ≠ 代码能跑（测试太浅，没测 curses 渲染）

### 4. ColonyCoder QA 发现了 ApexCoder 可能遗漏的 bug

ColonyCoder 的 Claude QA 发现了一个真实逻辑 bug：
- 单蛇死后游戏不结束（条件是 `and` 应该是 `or`）
- Qwen3.5 成功修复了这个 bug（在 QA feedback 后）

ApexCoder 没有独立 QA 阶段，无法确认是否有类似问题。

### 5. QA 超时是 ColonyCoder 的结构性瓶颈

`run_e2e.sh` 有 120 秒硬超时，但 Claude QA 写的 curses E2E 测试（pty 模拟）总时间超 120 秒。

**已修复**：在 QA Claude 的 system prompt 中加入速度约束：
- 单个测试 < 10 秒
- 总测试 < 90 秒
- curses 测试用快速 smoke test，不等自然结束

## 架构观察

### ColonyCoder 适合什么场景

ColonyCoder 的多子图协作架构在以下场景可能有价值：
1. **需要多视角审查的任务** — 辩论阶段确实能发现设计问题（Gemini 的评审质量不错）
2. **需要独立 QA 的任务** — QA 子图发现了真实 bug
3. **成本敏感的场景** — 代码生成用免费的本地模型，只在辩论和 QA 用云端模型

### ColonyCoder 需要改进的方向

1. **本地模型选择** — Qwen3.5:27b 的 tool calling 不可靠；Gemma4:31b 稳定但代码差；需要测试 Qwen3-Coder-Next 等更适合编码的模型
2. **executor 效率** — 模型反复全量重写文件，浪费迭代次数。需要考虑上下文压缩或分步生成
3. **QA 测试速度** — 已修复（prompt 约束）
4. **tool call 容错** — 已修复（try/except TypeError）

## 文件索引

| 文件 | 内容 |
|------|------|
| `run_colony_coder_debug.py` | ColonyCoder debug runner（用 DebugConsoleReporter） |
| `run_apex_coder_debug.py` | ApexCoder debug runner |
| `run_executor_test.py` | 只跑 executor 子图的快速测试（跳过 planner） |
| `framework/debug_reporter.py` | DebugConsoleReporter 通用类 |
| `blueprints/functional_graphs/colony_coder_executor/entity.json` | executor 配置（model 字段可换） |
| `blueprints/functional_graphs/colony_coder_qa/entity.json` | QA 配置（含速度约束 prompt） |
| `/tmp/snake_battle_apex/snake_battle.py` | ApexCoder 生成的游戏（517 行） |
| `/tmp/snake_battle_v3/snake_battle.py` | ColonyCoder(Qwen3.5) 生成的游戏（694 行，修 bug 后） |
| `/tmp/snake_battle_test/snake_battle.py` | ColonyCoder(Gemma4) 生成的游戏（261 行） |
