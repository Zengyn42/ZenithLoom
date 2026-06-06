# The Convergence Story: ZenithLoom and the Direction of Agent Orchestration

> How an independently-built orchestration engine arrived at the same architectural conclusions as Anthropic — before Anthropic shipped them.

---

## The Problem Everyone Was Solving (2024)

A single LLM call is not enough.

Anyone who tried to build something real with Claude or GPT-4 in 2024 ran into the same wall: the model is brilliant at reasoning, but one turn of conversation can't plan, execute, verify, and revise a complex task. You needed something more — but what?

The first answer was simple: **chain the calls**. LangChain popularized prompt chaining. But chaining in Python is invisible. You can't see the pipeline. You can't inspect it while it runs. You can't hand it to someone else without handing them the code.

The real question wasn't "how do I chain LLM calls?" It was: **how do I make the pipeline itself a first-class citizen?**

---

## ZenithLoom's Bet (Early 2025)

ZenithLoom was built around a specific hypothesis:

> **The pipeline should be declarative, observable, and composable — the same way Kubernetes made infrastructure declarative.**

This led to five concrete design decisions:

**1. The graph lives in a config file, not in code.**

Every ZenithLoom agent is defined by `entity.json` — a JSON declaration of nodes and edges. Not Python. Not a builder API. A file you can read, diff, version, and hand to another system. The code that *executes* the graph (LangGraph) is separate from the code that *defines* it.

**2. The LLM decides where to route — by emitting a signal.**

Instead of hardcoding "always go to node B after node A," ZenithLoom lets the LLM emit a routing signal on the first line of its reply:
```json
{"route": "debate_brainstorm", "context": "should we use microservices?"}
```
The graph has edges pre-declared for every possible route. The LLM picks which edge to activate. **The topology is fixed; the path through it is dynamic.**

**3. Subgraphs are composable pipelines.**

A "debate" isn't a prompt. It's a graph: Gemini proposes, Claude critiques, Claude concludes. That graph is a node in the parent graph. The parent doesn't know or care about the debate's internals — it just sees a `debate_conclusion` field appear in state when it's done.

**4. Context between subgraphs is explicitly isolated.**

When a subgraph runs, it gets a filtered view of parent state — not the full conversation history. The parent's `messages` don't flow in. Only routing context, workspace, and a few explicit fields pass through. When the subgraph exits, its internal messages are cleaned up. The parent's context stays clean.

**5. The same agent runs anywhere.**

One `entity.json` definition. Discord, CLI, Google Chat — the graph doesn't change. Only the connector does.

---

## Then Anthropic Shipped

Between mid-2025 and mid-2026, Anthropic released a sequence of features that validated each of these bets, one by one.

### June 2025 — Claude Agent SDK

Anthropic released the Claude Agent SDK, formalizing their approach to multi-agent orchestration. The core pattern: declare subagents with system prompts, tool restrictions, and permission modes. The orchestrating Claude decides when to spawn them.

**The parallel:** ZenithLoom's `SUBGRAPH_REF` nodes. Declare the subgraph. Let the LLM signal when to activate it.

### Mid-2025 — Custom Subagents (`.claude/subagents/*.md`)

Claude Code introduced a declarative subagent format: markdown files in `.claude/subagents/` declaring name, description, system prompt, tool restrictions, and permission mode.

**The parallel:** ZenithLoom's `entity.json` + persona files. The fields map almost exactly:

| Claude Subagent | ZenithLoom Node |
|---|---|
| `description:` | `routing_hint` in entity.json |
| `system_prompt:` | `persona_files: [ROLE.md, PROTOCOL.md]` |
| `tools:` list | `tools: [Read, Write, Bash]` |
| `permission_mode:` | `permission_mode:` (same vocabulary) |
| Isolated context window | `SubgraphInputState` (same isolation) |

Same vocabulary. Same structure. Independent implementations.

### Late 2025 — Agent Skills

Anthropic introduced Agent Skills: filesystem-packaged capabilities with progressive loading. Level 1 loads metadata (~100 tokens). Level 2 loads instructions when triggered. Level 3 loads resources on demand.

**The parallel:** ZenithLoom's Skills system, already in production. Same progressive disclosure pattern. Skills live in `.claude/skills/` and are loaded by `add_dirs` in node config.

### May 2026 — Dynamic Workflows

Claude Code's Dynamic Workflows: Claude writes a JavaScript orchestration script. The script defines which agents run in which phase, in parallel or sequential. Intermediate results live in script variables — not the context window.

**The parallel:** ZenithLoom's subgraph conclusion fields (`debate_conclusion`, `apex_conclusion`, `knowledge_result`). When a subgraph finishes, it writes its conclusion to a dedicated state field — not to the conversation `messages`. The parent LLM sees the conclusion injected into its next prompt, then responds. **Intermediate results outside the context window is exactly what ZenithLoom does with state fields.**

---

## Where the Thinking Converges

Map ZenithLoom's design decisions to Anthropic's shipped features:

| ZenithLoom design decision | When ZenithLoom built it | Claude equivalent | When Anthropic shipped it |
|---|---|---|---|
| Declarative agent definition (entity.json) | Early 2025 | Custom Subagents (.claude/subagents/) | Mid 2025 |
| Routing signal from LLM (first-line JSON) | Early 2025 | Subagent spawning decision | 2025 |
| Context isolation between subgraphs | Early 2025 | Subagent isolated context window | Mid 2025 |
| Skills system (progressive capability loading) | 2025 | Agent Skills | Late 2025 |
| Permission modes (bypassPermissions/plan/acceptEdits) | Early 2025 | Same exact vocabulary | Throughout 2025 |
| Intermediate results in state fields (not messages) | Early 2025 | Script variables in Dynamic Workflows | May 2026 |
| Subgraph composition (graph inside a node) | Early 2025 | Agent hierarchies in Dynamic Workflows | May 2026 |

---

## Why This Is Not Hindsight

A fair challenge: couldn't any project claim alignment with industry trends after the fact?

The answer lies in *why* both ZenithLoom and Anthropic made the same choices. It wasn't coincidence — it was physics.

**LLMs have two fundamental defects: non-determinism and context decay.** Every robust agent architecture must address both. ZenithLoom's design decisions weren't inspired by watching Anthropic — they were forced by the same underlying constraints:

- The early Claude API didn't support native Tool Use. ZenithLoom didn't adapt to API limitations; it worked *around* them by building an external DAG engine and hard context isolation. When Anthropic later added native subagents and structured outputs, they were filling the exact vacuum ZenithLoom had patched from the outside. **Need first, native support later — that's the signal of genuine foresight.**

- ZenithLoom's routing protocol (plain-text JSON, language-agnostic, decoupled from execution) is the same philosophy as MCP (Model Context Protocol): strip scheduling logic out of execution code and turn it into a portable, cross-process protocol. LangGraph remains deeply coupled to the Python runtime. ZenithLoom was on the protocol path before MCP existed.

The convergence isn't coincidence. It's what correct answers to the same physical constraints look like.

---

## What ZenithLoom Bets on Next

Dynamic Workflows solved one problem: scale. 16 concurrent agents, 1000 total. But they introduced a new one: **the pipeline is invisible again.** Claude writes a JS script. The script runs. You can't see the topology. You can't diff it. You can't render it. You can't version it.

ZenithLoom's answer — being built right now — is the **AgentGraph representation layer**: a typed, serializable, diffable graph object that sits between the config file and the execution engine. The graph becomes a first-class citizen.

This is the next convergence bet. If Anthropic's pattern holds, some version of "the pipeline as a versioned, inspectable artifact" will be a native Claude feature within 12–18 months.

---

## The One-Sentence Version

ZenithLoom is what you build when you decide that **the agent pipeline itself — not just the agents — is the product**: something that can be declared, inspected, composed, versioned, and run across any LLM backend or interface, in the same way Kubernetes made infrastructure a declarative artifact rather than a collection of shell scripts.

Anthropic reached the same conclusion. They just built the execution environment first and are working backward to the representation layer. ZenithLoom built the representation layer first.

---

*ZenithLoom*
