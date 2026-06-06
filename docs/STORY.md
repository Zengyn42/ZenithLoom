# ZenithLoom and Claude Pipeline: Two Answers to the Same Question

> They solved the same problem independently. ZenithLoom is an alternative, not a derivative.

---

## The Problem

Building with a single LLM call hits a ceiling fast. Real tasks require planning, execution, verification, and revision — across multiple steps, multiple tools, sometimes multiple models. The industry converged on a shared insight:

**You need to orchestrate multiple agents. The question is how.**

Anthropic answered this question from the inside out, building orchestration natively into Claude Code and the Claude API. ZenithLoom answered the same question from the outside in, building an orchestration layer that works across any LLM backend.

Different starting points. Remarkably similar solutions.

---

## The Same Core Architecture, Reached Independently

Both systems arrived at the same set of architectural decisions:

### 1. Agents are declared, not coded

**Claude pipeline:** Subagents defined as markdown files in `.claude/subagents/` — name, description, system prompt, tool list, permission mode.

**ZenithLoom:** Agents defined as `entity.json` + persona markdown files — node type, persona files, tool list, permission mode.

Neither requires you to write orchestration code. You declare what an agent is; the system decides when and how to run it.

### 2. The orchestrating agent signals intent; the system routes

**Claude pipeline:** The orchestrating Claude decides to spawn a subagent based on the task at hand. The decision emerges from the LLM's reasoning.

**ZenithLoom:** The main LLM emits a structured routing signal on the first line of its reply:
```json
{"route": "debate_brainstorm", "context": "microservices vs monolith"}
```
The framework routes to the declared subgraph. The LLM reasons; the framework executes.

### 3. Subagents run in isolated context

**Claude pipeline:** Each subagent gets its own context window. The parent's full conversation history doesn't bleed in.

**ZenithLoom:** `SubgraphInputState` explicitly defines what flows from parent to subgraph — routing context, workspace, a few declared fields. The parent's `messages` stay out. The subgraph's internal messages are cleaned up on exit.

### 4. Capabilities are packaged as skills

**Claude pipeline:** Agent Skills — filesystem-packaged capabilities with progressive loading. Metadata loads first, instructions on trigger, resources on demand.

**ZenithLoom:** The Skills system — same progressive disclosure pattern. Skills in `.claude/skills/` directories, loaded via `add_dirs` in node config.

### 5. Intermediate results live outside the context window

**Claude pipeline:** Dynamic Workflows use JavaScript script variables to hold intermediate results. The orchestrator's context window stays clean.

**ZenithLoom:** Subgraph conclusions write to dedicated state fields (`debate_conclusion`, `apex_conclusion`, `knowledge_result`) — not to `messages`. The parent LLM sees conclusions injected into its next prompt, then synthesizes and responds.

---

## Same Solutions, Different Constraints

The convergence isn't coincidence. Both systems are solving the same underlying physics of LLMs:

- **Non-determinism**: You can't hardcode every decision path. The LLM must have agency over routing.
- **Context decay**: Stuffing all intermediate results into one conversation window degrades quality and balloons cost.
- **Composability**: Complex pipelines need to be built from reusable, testable pieces.

When you're solving the same physical constraints, the solution space narrows to the same answers.

---

## Where They Differ — and Why That Matters

ZenithLoom and Claude's pipeline are not the same product. The differences are deliberate:

| Dimension | Claude Pipeline | ZenithLoom |
|---|---|---|
| **LLM backends** | Claude only | Claude, Gemini, Ollama, Grok — mix in one graph |
| **Graph topology** | Implicit (LLM decides dynamically) | Explicit (declared in entity.json, observable, diffable) |
| **State persistence** | Per-call, within session | SQLite checkpoint — survives restarts, resumable across days |
| **Deployment** | Claude Code / Anthropic API | Discord, CLI, Google Chat — same agent, any surface |
| **Vendor dependency** | Anthropic ecosystem | None — swap LLM backends without changing the graph |
| **Pipeline visibility** | Script runs, topology invisible | Full graph render, Mermaid topology, AgentGraph representation |

Claude's pipeline is the right answer if you're all-in on Anthropic's ecosystem and need massive parallel scale (up to 1000 concurrent agents).

ZenithLoom is the right answer if you need:
- **Multi-LLM pipelines** — Claude for reasoning, Gemini for critique, local model for privacy-sensitive steps
- **Observable topology** — a graph you can inspect, diff, version, and debug
- **Persistent state** — agents that remember across sessions, not just within one
- **Platform independence** — the same agent on Discord, CLI, and any future surface

---

## The One-Sentence Version

ZenithLoom is what Claude's pipeline would look like if it were designed to be backend-agnostic, topology-observable, and persistent — solving the same orchestration problem without the ecosystem lock-in.

---

*ZenithLoom*
