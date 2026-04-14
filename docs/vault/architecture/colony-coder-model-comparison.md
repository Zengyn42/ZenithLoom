# ColonyCoder vs ApexCoder 完整对比实验 — 2026-04-13

> 测试任务：双蛇对战游戏 (Snake Battle)，curses 终端 UI，两个 AI 蛇自动对战
> 状态：实验完成，结论已确认

## 实验设置

- **ColonyCoder 架构**：plan (Claude+Gemini 辩论) → execute (本地 LLM code_gen) → qa (Claude E2E)
- **ApexCoder 架构**：单 Claude Opus 4.6 节点，全工具集
- **executor 子图**：inject_task_context → code_gen (可换模型) → run_tests → test_route
- **硬件**：WSL2, RTX 5090 (32GB VRAM), 128GB RAM, Ollama 本地推理
- **Debug 工具**：`DebugConsoleReporter` + `astream(subgraphs=True)`

---

## 对比结果

### 性能指标

| | ApexCoder (Claude Opus) | ColonyCoder (Qwen3.5:27b) | ColonyCoder (Gemma4:31b) | ColonyCoder (QwenCoder:80B tuned) |
|---|---|---|---|---|
| **总耗时** | **2 分 3 秒** | ~50 分钟（含 QA 超时循环） | ~15 分钟（planner 8 + executor 4） | ~20 分钟（planner 8 + executor 12） |
| **executor 耗时** | N/A | ~13 分钟 | **3 分 53 秒** | 12 分 1 秒 |
| **tool call 迭代** | N/A | 25/25（用完） | **8/25** | 25/25（用完） |
| **tool call 稳定性** | N/A | 崩溃（参数丢失） | 稳定 | 稳定 |
| **代码行数** | 517 行 | 938 行 | 261 行 | 454 行 |

### 实际运行测试（pty smoke test, 6 秒）

| | ApexCoder | Qwen3.5:27b | Gemma4:31b | QwenCoder:80B |
|---|---|---|---|---|
| **能跑吗** | ❌ 24 行终端放不下 | ✅ 能跑 | ✅ 能跑 | ✅ 能跑 |
| **6 秒跑几帧** | 0（terminal too small） | 56 | 56 | 56 |
| **蛇吃到食物** | 否 | ✅（长到 8） | ✅（长到 10） | ✅（长到 7-9） |
| **AI 策略可见** | 否 | 是 | 是 | 是 |

注：ApexCoder 在 35 行终端下可运行，但 49 帧内蛇未吃到任何食物（Alpha 长度始终为 4）。

### QA 验证

| | ApexCoder | ColonyCoder (Qwen3.5:27b) |
|---|---|---|
| **QA 方式** | 自己写 headless 测试验证自己 | 独立 Claude QA 子图 |
| **发现 bug** | 0（但有 bug 未发现） | 2 个（1 major, 1 minor） |
| **真实环境验证** | ❌（headless only, 绕过 curses） | ✅（Claude QA 用 pty 启动游戏） |

---

## 关键发现

### 1. ApexCoder 代码有 3 个 bug，自己全没发现

**Bug 1: `appendleft` body 构建方向反了**

```python
# line 49-51
dy, dx = DELTA[OPPOSITE[direction]]
for i in range(INITIAL_LENGTH):
    self.body.appendleft((start_y + dy * i, start_x + dx * i))
```

Alpha 从 `(12, 5)` 面朝 RIGHT 生成，body 应该是 `[head=(12,5), (12,4), (12,3), (12,2)]`。
但 `appendleft` 翻转了顺序：`[head=(12,2), (12,3), (12,4), (12,5)]`。
Head 在最左端，面朝 RIGHT → 下一步 `(12,3)` 是自己的 body → 撞死。

Headless 测试下 AI 绕过了（选 UP 避开），但蛇的初始姿态不符合预期。

**Bug 2: 硬编码 `BOARD_H=24` 需要 28 行终端**

```python
BOARD_H = 24
min_rows = offset_y + BOARD_H + 2  # = 28
```

标准终端 24 行放不下。QwenCoder 和 Gemma4 都适配了终端尺寸，ApexCoder 没有。

**Bug 3: 蛇不吃食物**

即使在足够大的终端（35 行），49 帧后两条蛇长度仍为 4（初始值），说明 AI 寻路逻辑有问题。
可能与 Bug 1 相关 — head 位置不对导致 BFS 路径错误。

### 2. ApexCoder 验证是假的

ApexCoder 声称做了 3 种验证：
1. "100-frame headless test: Both snakes alive ✅"
2. "20-game simulation: Alpha won 7, Beta won 13 ✅"
3. "Syntax check: PASS ✅"

实际分析 session 日志发现：

```
[13] Write snake_battle.py
[17] Bash: python3 -c "import snake_battle; Game().tick()..."  ← headless, 绕过 curses
[19] Bash: python3 -c "Run 20 games..."                        ← 也是 headless
[21] Bash: syntax check
[23] "Done, ALL TESTS PASSED"
```

**所有验证都是 headless 的** — `import snake_battle; Game().tick()` 绕过了 curses UI。
headless 模式下 Bug 1 被 AI 的 fallback 逻辑掩盖了（选择 UP 避开自己 body）。
但真实 curses 运行时，这些 bug 导致游戏无法正常工作。

**核心问题：ApexCoder 自己写测试验证自己** — 它写的 headless 测试恰好绕过了 bug 的代码路径。

### 3. ColonyCoder 的 QA 架构优势

ColonyCoder 的 QA 子图是**独立的 Claude session**，不知道代码是怎么写的，只从用户视角测试。
它确实发现了一个真实 bug（`and` vs `or` 逻辑错误），ApexCoder 没有发现类似问题。

**独立 QA 的价值**：
- 不会因为"我知道代码是怎么写的"而写出绕过 bug 的测试
- 从用户视角（启动游戏、观察行为）而不是开发者视角（import + call）
- 即使 QA 测试超时的问题（已修复），QA 的**判断**是准确的（E2E_VERDICT: PASS 确实代码修好了）

### 4. Ollama 模型 tool calling 稳定性

| 模型 | 参数 | VRAM | Tool calling | 迭代效率 |
|------|------|------|---|---|
| Qwen3.5:27b | 27.8B | ~18GB 全 GPU | ❌ 参数丢失崩溃 | 25/25 用完 |
| Gemma4:31b | 31.3B | ~20GB 全 GPU | ✅ 稳定 | **8/25** |
| QwenCoder:80B (tuned) | 79.7B | 25GB GPU + 30GB RAM | ✅ 稳定 | 25/25 用完 |

**Qwen3.5:27b 的根本问题**：Ollama 用 JSON 格式 tool calling，但 Qwen3.5 训练用的是 XML 格式。已知 Ollama issue #14493。

**QwenCoder:80B 的问题**：model 太大，52% GPU offload，推理慢。通过 Modelfile 限制 `num_gpu=22` + `num_ctx=8192` 可以让 GPU 参与（6-9% utilization），但仍然慢。

**Gemma4:31b 最稳定**但代码质量最差（261 行，功能简陋）。

### 5. 所有本地模型的共同问题：不会停

Qwen3.5 和 QwenCoder 都用完 25 次迭代。tool loop 终止条件是模型返回纯文本（无 tool call），但模型写完代码后继续发 `bash_exec` 检查（ls、cat、python3 -c...），不给纯文本结束信号。

QwenCoder:80B 的 tool call 序列：
```
1-6.   write_file × 5 + bash_exec × 1  (写代码+测试)
7-25.  bash_exec × 19 连发              (无意义的重复检查)
```

迭代 7 就完成了代码，后面 18 次 bash_exec 全是浪费。

### 6. Context Explosion 问题

Hani 的辩论研究（`Vault/设计细节/Colony Coder Context Explosion Fix`）完全命中：
- 25 次迭代累积 500+ 行代码 × 多轮 → context 爆炸
- Qwen3.5 在 context 后期丢参数（`write_file(['content'])` 缺 `path`）
- 最简单修复：`max_iterations: 25 → 8` + 依靠外层 `test_route → code_gen` 循环自然重置 context

---

## 架构结论

### ApexCoder 需要改进的方向

1. **加独立 QA** — 不能自己写测试验证自己。可以 spawn 一个独立的 code-reviewer agent 用 pty 真实启动程序
2. **PROTOCOL.md 缺少"必须在真实环境运行"规则** — 现有规则强调 eval-first 和 PUA 铁律，但没有要求"必须在目标环境里跑一次"。headless 测试不等于真实测试
3. **验证应该包含终端适配测试** — curses 程序必须在标准 24x80 终端能跑

### ColonyCoder 需要改进的方向

1. **减 `max_iterations` 到 8** — 1 行改动，最高 ROI
2. **QA 测试速度约束** — ✅ 已修复（prompt 加了 90 秒限制）
3. **tool call 容错** — ✅ 已修复（try/except TypeError）
4. **模型选择** — Gemma4:31b 最稳定但代码质量差；Qwen3.5:27b 质量高但不稳定；QwenCoder:80B 在当前硬件上太慢
5. **Context 管理** — 考虑 Hani 的 4 层防御方案（session reset、deterministic summary、git snapshot、replace_lines）

### "自己验证自己"是根本性缺陷

> ApexCoder 的 3 个 bug 全部在 headless 自测中被"通过"了。
> ColonyCoder 的 QA 子图发现了 1 个真实 bug 并成功修复。
>
> 这不是 ApexCoder 的 prompt 或 skill 不够好 — 是**架构层面的缺陷**。
> 让开发者自己验证自己的代码，等于让学生自己出题自己答。
> 独立的 QA 角色（即使质量不完美）比没有 QA 好。

---

## 修复记录

### 已实施

| 修复 | 文件 | 内容 |
|------|------|------|
| Ollama tool call 容错 | `framework/nodes/llm/ollama.py` | catch TypeError on missing args, 返回错误给模型重试 |
| QA 测试速度约束 | `colony_coder_qa/entity.json` | prompt 加 "每个测试<10s, 总计<90s" |
| DebugConsoleReporter | `framework/debug_reporter.py` | 通用子图 debug 可视化 |

### 待实施

| 修复 | 预计改动 | 优先级 |
|------|---------|--------|
| `max_iterations: 25 → 8` | 1 行 | 高 |
| ApexCoder 加独立 QA agent | 新增 agent prompt | 高 |
| ApexCoder PROTOCOL.md 加"真实环境验证"规则 | ~10 行 | 中 |
| Context 管理（session reset + summary） | ~30 行 | 中 |
| `replace_lines` 工具 | ~15 行 | 低 |

---

## 文件索引

| 文件 | 内容 |
|------|------|
| `run_colony_coder_debug.py` | ColonyCoder debug runner |
| `run_apex_coder_debug.py` | ApexCoder debug runner |
| `run_executor_test.py` | executor 快速测试（跳过 planner） |
| `framework/debug_reporter.py` | DebugConsoleReporter |
| `Modelfile.qwen-coder-tuned` | QwenCoder:80B GPU 层数调优 |
| `/tmp/snake_battle_apex/snake_battle.py` | ApexCoder 生成（517 行，3 个 bug） |
| `/tmp/snake_battle_v3/snake_battle.py` | ColonyCoder/Qwen3.5 生成（694 行，QA 修了 1 个 bug） |
| `/tmp/snake_battle_test/snake_battle.py` | Gemma4/QwenCoder 生成的版本 |
| `Vault/设计细节/Colony Coder Context Explosion Fix` | Hani 的 context 管理研究 |

---

## 核心洞见

> **"谁来验证验证者？"** — 代码质量不取决于写代码的模型多强，而取决于验证流程的独立性。
> ApexCoder（Claude Opus）写出了有 bug 的代码且自测通过；ColonyCoder 的本地模型写出了更多 bug，但独立 QA 子图发现并修复了它们。
> 在 AI 编码系统中，**架构（独立 QA）比模型能力更重要**。
