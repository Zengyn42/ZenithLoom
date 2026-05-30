# ColonyCoder vs ApexCoder Full Comparison Experiment — 2026-04-13

> Test task: Dual Snake Battle game (Snake Battle), curses terminal UI, two AI snakes battle automatically
> Status: Experiment complete, conclusions confirmed

## Experiment Setup

- **ColonyCoder architecture**: plan (Claude+Gemini debate) → execute (local LLM code_gen) → qa (Claude E2E)
- **ApexCoder architecture**: single Claude Opus 4.6 node, full tool set
- **executor subgraph**: inject_task_context → code_gen (swappable model) → run_tests → test_route
- **Hardware**: WSL2, RTX 5090 (32GB VRAM), 128GB RAM, Ollama local inference
- **Debug tools**: `DebugConsoleReporter` + `astream(subgraphs=True)`

---

## Comparison Results

### Performance Metrics

| | ApexCoder (Claude Opus) | ColonyCoder (Qwen3.5:27b) | ColonyCoder (Gemma4:31b) | ColonyCoder (QwenCoder:80B tuned) |
|---|---|---|---|---|
| **Total time** | **2 min 3 sec** | ~50 min (including QA timeout loops) | ~15 min (planner 8 + executor 4) | ~20 min (planner 8 + executor 12) |
| **executor time** | N/A | ~13 min | **3 min 53 sec** | 12 min 1 sec |
| **tool call iterations** | N/A | 25/25 (used up) | **8/25** | 25/25 (used up) |
| **tool call stability** | N/A | Crashed (missing args) | Stable | Stable |
| **lines of code** | 517 | 938 | 261 | 454 |

### Real Runtime Test (pty smoke test, 6 seconds)

| | ApexCoder | Qwen3.5:27b | Gemma4:31b | QwenCoder:80B |
|---|---|---|---|---|
| **Runs?** | ❌ 24-row terminal too small | ✅ Runs | ✅ Runs | ✅ Runs |
| **Frames in 6 sec** | 0 (terminal too small) | 56 | 56 | 56 |
| **Snake ate food** | No | ✅ (grew to 8) | ✅ (grew to 10) | ✅ (grew to 7-9) |
| **AI strategy visible** | No | Yes | Yes | Yes |

Note: ApexCoder can run in a 35-row terminal, but in 49 frames the snake ate no food (Alpha length stayed at 4).

### QA Validation

| | ApexCoder | ColonyCoder (Qwen3.5:27b) |
|---|---|---|
| **QA method** | Self-wrote headless tests to validate own code | Independent Claude QA subgraph |
| **Bugs found** | 0 (but bugs existed undetected) | 2 (1 major, 1 minor) |
| **Real environment validation** | ❌ (headless only, bypasses curses) | ✅ (Claude QA launches game via pty) |

---

## Key Findings

### 1. ApexCoder Code Had 3 Bugs, All Undetected by Itself

**Bug 1: `appendleft` body build direction reversed**

```python
# line 49-51
dy, dx = DELTA[OPPOSITE[direction]]
for i in range(INITIAL_LENGTH):
    self.body.appendleft((start_y + dy * i, start_x + dx * i))
```

Alpha facing RIGHT starting at `(12, 5)`, body should be `[head=(12,5), (12,4), (12,3), (12,2)]`.
But `appendleft` reverses the order: `[head=(12,2), (12,3), (12,4), (12,5)]`.
Head is at the leftmost end, facing RIGHT → next step `(12,3)` is its own body → crash.

In headless tests the AI worked around it (chose UP to avoid body), but the snake's initial posture was wrong.

**Bug 2: Hard-coded `BOARD_H=24` requires 28-row terminal**

```python
BOARD_H = 24
min_rows = offset_y + BOARD_H + 2  # = 28
```

Standard 24-row terminal cannot fit. QwenCoder and Gemma4 adapted to terminal size; ApexCoder did not.

**Bug 3: Snake doesn't eat food**

Even in a large enough terminal (35 rows), in 49 frames both snakes remained at length 4 (initial value), indicating a problem with AI pathfinding logic.
Likely related to Bug 1 — wrong head position causes BFS path errors.

### 2. ApexCoder Validation Was Fake

ApexCoder claimed 3 types of validation:
1. "100-frame headless test: Both snakes alive ✅"
2. "20-game simulation: Alpha won 7, Beta won 13 ✅"
3. "Syntax check: PASS ✅"

Analyzing session logs reveals:

```
[13] Write snake_battle.py
[17] Bash: python3 -c "import snake_battle; Game().tick()..."  ← headless, bypasses curses
[19] Bash: python3 -c "Run 20 games..."                        ← also headless
[21] Bash: syntax check
[23] "Done, ALL TESTS PASSED"
```

**All validation was headless** — `import snake_battle; Game().tick()` bypasses the curses UI.
In headless mode Bug 1 was masked by AI fallback logic (choosing UP to avoid own body).
But in real curses runtime, these bugs caused the game to not work properly.

**Core issue: ApexCoder wrote tests to validate its own code** — the headless tests it wrote happened to bypass the code paths containing the bugs.

### 3. ColonyCoder's QA Architecture Advantage

ColonyCoder's QA subgraph is an **independent Claude session** that doesn't know how the code was written and tests only from the user's perspective.
It genuinely found a real bug (`and` vs `or` logic error) that ApexCoder did not find.

**Value of independent QA**:
- Won't write tests that bypass bugs because "I know how the code was written"
- Tests from user perspective (launch game, observe behavior) not developer perspective (import + call)
- Even when QA timeout issues existed (now fixed), QA's **judgment** was accurate (E2E_VERDICT: PASS did mean the code was actually fixed)

### 4. Ollama Model Tool Calling Stability

| Model | Parameters | VRAM | Tool calling | Iteration efficiency |
|-------|-----------|------|---|---|
| Qwen3.5:27b | 27.8B | ~18GB full GPU | ❌ Missing args, crashes | 25/25 used up |
| Gemma4:31b | 31.3B | ~20GB full GPU | ✅ Stable | **8/25** |
| QwenCoder:80B (tuned) | 79.7B | 25GB GPU + 30GB RAM | ✅ Stable | 25/25 used up |

**Qwen3.5:27b root problem**: Ollama uses JSON format tool calling, but Qwen3.5 was trained with XML format. Known Ollama issue #14493.

**QwenCoder:80B problem**: model too large, 52% GPU offload, slow inference. Limiting `num_gpu=22` + `num_ctx=8192` via Modelfile lets GPU participate (6-9% utilization), but still slow.

**Gemma4:31b most stable** but worst code quality (261 lines, minimal features).

### 5. Common Problem with All Local Models: They Don't Stop

Qwen3.5 and QwenCoder both used all 25 iterations. The tool loop termination condition is the model returning plain text (no tool calls), but after writing code the models keep issuing `bash_exec` checks (ls, cat, python3 -c...) without giving a plain text end signal.

QwenCoder:80B tool call sequence:
```
1-6.   write_file × 5 + bash_exec × 1  (write code + tests)
7-25.  bash_exec × 19 in rapid succession  (pointless repeated checks)
```

Iteration 7 completed the code; the following 18 bash_exec calls were all wasteful.

### 6. Context Explosion Problem

The technical_architect's debate research (`Vault/design-details/Colony Coder Context Explosion Fix`) fully identified:
- 25 iterations accumulate 500+ lines of code × multiple rounds → context explosion
- Qwen3.5 drops args in later context (`write_file(['content'])` missing `path`)
- Simplest fix: `max_iterations: 25 → 8` + rely on outer `test_route → code_gen` loop to naturally reset context

---

## Architecture Conclusions

### ApexCoder Improvement Directions

1. **Add independent QA** — cannot validate its own code. Can spawn an independent code-reviewer agent that uses pty to actually launch the program
2. **PROTOCOL.md missing "must run in real environment" rule** — existing rules emphasize eval-first and PUA rules, but don't require "must run once in the target environment". Headless tests do not equal real tests
3. **Validation should include terminal adaptation tests** — curses programs must run in a standard 24x80 terminal

### ColonyCoder Improvement Directions

1. **Reduce `max_iterations` to 8** — 1-line change, highest ROI
2. **QA test speed constraint** — ✅ Already fixed (prompt adds 90 second limit)
3. **Tool call error tolerance** — ✅ Already fixed (try/except TypeError)
4. **Model selection** — Gemma4:31b most stable but worst code quality; Qwen3.5:27b high quality but unstable; QwenCoder:80B too slow on current hardware
5. **Context management** — consider the technical_architect's 4-layer defense approach (session reset, deterministic summary, git snapshot, replace_lines)

### "Self-Validation" is a Fundamental Flaw

> ApexCoder's 3 bugs all "passed" in headless self-testing.
> ColonyCoder's QA subgraph found 1 real bug and successfully fixed it.
>
> This is not about ApexCoder's prompt or skills being insufficient — it is an **architectural flaw**.
> Having the developer validate their own code is like having a student both write and answer their own exam.
> An independent QA role (even imperfect) is better than no QA.

---

## Fix Log

### Already Implemented

| Fix | File | Content |
|-----|------|---------|
| Ollama tool call error tolerance | `framework/nodes/llm/ollama.py` | catch TypeError on missing args, return error to model for retry |
| QA test speed constraint | `colony_coder_qa/entity.json` | prompt adds "each test <10s, total <90s" |
| DebugConsoleReporter | `framework/debug_reporter.py` | general subgraph debug visualization |

### Pending Implementation

| Fix | Estimated changes | Priority |
|-----|------------------|---------|
| `max_iterations: 25 → 8` | 1 line | High |
| ApexCoder add independent QA agent | New agent prompt | High |
| ApexCoder PROTOCOL.md add "real environment validation" rule | ~10 lines | Medium |
| Context management (session reset + summary) | ~30 lines | Medium |
| `replace_lines` tool | ~15 lines | Low |

---

## File Index

| File | Content |
|------|---------|
| `run_colony_coder_debug.py` | ColonyCoder debug runner |
| `run_apex_coder_debug.py` | ApexCoder debug runner |
| `run_executor_test.py` | executor quick test (skip planner) |
| `framework/debug_reporter.py` | DebugConsoleReporter |
| `Modelfile.qwen-coder-tuned` | QwenCoder:80B GPU layer tuning |
| `/tmp/snake_battle_apex/snake_battle.py` | ApexCoder generated (517 lines, 3 bugs) |
| `/tmp/snake_battle_v3/snake_battle.py` | ColonyCoder/Qwen3.5 generated (694 lines, QA fixed 1 bug) |
| `/tmp/snake_battle_test/snake_battle.py` | Gemma4/QwenCoder generated versions |
| `Vault/design-details/Colony Coder Context Explosion Fix` | technical_architect's context management research |

---

## Core Insight

> **"Who validates the validators?"** — Code quality doesn't depend on how powerful the model that writes the code is, but on the independence of the validation process.
> ApexCoder (Claude Opus) wrote buggy code and self-testing passed; ColonyCoder's local models wrote more bugs, but the independent QA subgraph found and fixed them.
> In AI coding systems, **architecture (independent QA) matters more than model capability**.
