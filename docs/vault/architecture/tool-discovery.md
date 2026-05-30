# Automated Tool Discovery and Evaluation System — Architecture Design

> Status: Design draft | Date: 2026-03-22

## I. Goals

User describes a requirement (e.g. "find an AI tool that can write Google Slides"), and the system automatically searches, filters, clones, tests, and outputs an evaluation report.

## II. Pipeline Topology

```
__start__ → query_expand → search_aggregate → candidate_filter → sandbox_eval → report_gen → __end__
```

| Node | Framework Node Type | Provider | Responsibility |
|------|--------------------|---------|--------------:|
| `query_expand` | `CLAUDE_CLI` | Claude | Understand natural language requirements, generate search keywords (translation, synonym expansion) |
| `search_aggregate` | `DETERMINISTIC` | None | GitHub API + web search + dedup, pure Python |
| `candidate_filter` | `OLLAMA` | llama3.2:3b | Batch scoring (0-10), filter Top-K by relevance |
| `sandbox_eval` | `EXTERNAL_TOOL` | Docker | Clone → install → test, container isolation |
| `report_gen` | `CLAUDE_CLI` | Claude | Generate user-facing evaluation report |

## III. Integration Method

Mounted to the main agent graph (`technical_architect`) via `SUBGRAPH_REF`.

Main graph changes:
- `entity.json`: +1 node (`tool_discovery`), +2 edges
- `command_registry.py`: +1 line `_r("!discover", ...)`
- `base_interface.py`: `handle_command()` adds `!discover` branch

Auto-routing via `routing_hint`; Claude triggers automatically when it detects user search intent.

## IV. State Schema

```python
class ToolDiscoveryState(TypedDict):
    # Inherited BaseAgentState required fields
    messages: Annotated[list[BaseMessage], _keep_last_2]
    routing_target: str
    routing_context: str
    workspace: str
    project_root: str
    node_sessions: Annotated[dict, _merge_dict]
    # Discovery-specific fields (JSON serialized strings)
    user_query: str              # original natural language requirement
    search_intent: str           # {keywords, github_queries, web_queries}
    raw_candidates: str          # [{repo, stars, license, description, ...}]
    filtered_candidates: str     # Top-K [{..., relevance_score, rationale}]
    evaluation_results: str      # [{install_ok, test_pass_rate, ...}]
    discovery_config: str        # {depth: 5, timeout: 300}
    discovery_errors: str        # accumulated error log
```

## V. Data Structures

### CandidateRepo

```python
{
    "repo": "owner/name",
    "url": "https://github.com/...",
    "stars": int,
    "forks": int,
    "last_commit": "ISO8601",
    "license": "MIT",
    "language": "Python",
    "description": str,
    "readme_snippet": str,      # first 500 chars
    "topics": list[str],
    "open_issues": int,
    "archived": bool,
}
```

### EvalResult

```python
{
    "repo": "owner/name",
    "clone_ok": bool,
    "install_ok": bool,
    "install_time_s": float,
    "test_count": int,
    "test_pass_rate": float,    # 0.0-1.0
    "dependency_count": int,
    "code_lines": int,
    "api_surface": int,         # number of exported functions/classes
    "has_docs": bool,
    "has_examples": bool,
    "project_type": "python|node|rust|go|unknown",
    "eval_mode": "docker|venv|static",
    "errors": list[str],
}
```

## VI. Sandbox Design

`sandbox_eval` uses Docker one-shot containers (`--rm`):

- `--read-only` + tmpfs `/workspace`: prevents persistent writes
- Network whitelist: only allows pypi.org / npmjs.com / github.com
- Resource limits: `--memory=1g --cpus=1 --pids-limit=256`
- Per-repo hard timeout: 300s
- Fallback: without Docker, uses `venv` + static analysis, report marked `⚠️ Restricted Evaluation`

Security: static scan of `setup.py` / `pyproject.toml` for `os.system` / `subprocess` / `eval` calls before installation.

## VII. File Structure

### New Files

```
blueprints/functional_graphs/tool_discovery/
├── entity.json                  # graph definition
└── ROLE.md                     # LLM node persona

framework/schema/tool_discovery.py              # State schema
framework/nodes/tool_discovery/
├── search_aggregator.py        # DETERMINISTIC: GitHub API + web search
└── sandbox_runner.py           # EXTERNAL_TOOL: Docker sandbox orchestration

docker/
├── Dockerfile.sandbox-python   # Python sandbox image
└── Dockerfile.sandbox-node     # Node.js sandbox image
```

### Modified Files

```
blueprints/role_agents/technical_architect/entity.json   # +1 node, +2 edges
framework/command_registry.py                           # +1 line registration
framework/base_interface.py                             # +1 command handling branch
```

## VIII. entity.json Structure

```json
{
  "name": "ToolDiscovery",
  "routing_hint": "Use when user requests discovery/search/evaluation of open source tools",
  "graph": {
    "state_schema": "tool_discovery_schema",
    "nodes": [
      {"id": "query_expand",      "type": "CLAUDE_CLI", "permission_mode": "plan"},
      {"id": "search_aggregate",  "type": "DETERMINISTIC"},
      {"id": "candidate_filter",  "type": "OLLAMA", "model": "llama3.2:3b"},
      {"id": "sandbox_eval",      "type": "EXTERNAL_TOOL"},
      {"id": "report_gen",        "type": "CLAUDE_CLI", "permission_mode": "plan"}
    ],
    "edges": [
      {"from": "__start__",        "to": "query_expand"},
      {"from": "query_expand",     "to": "search_aggregate"},
      {"from": "search_aggregate", "to": "candidate_filter"},
      {"from": "candidate_filter", "to": "sandbox_eval"},
      {"from": "sandbox_eval",     "to": "report_gen"},
      {"from": "report_gen",       "to": "__end__"}
    ]
  }
}
```

## IX. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Malicious code execution | High | Docker sandbox + read-only + network whitelist + pre-install static scan |
| GitHub API rate limit | Medium | Require `GITHUB_TOKEN`; cache search results for 1h |
| Evaluation too slow | Medium | Default depth=3; progressive feedback; per-step hard timeout |
| Docker unavailable | Medium | Degrade to venv + static analysis, mark restricted evaluation |
| Ollama scoring unstable | Low | Require JSON output; invalid scores fall back to 5 |

## X. Implementation Priority

| Phase | Content | Estimated Work |
|-------|---------|---------------|
| P1 | State schema + entity.json skeleton | 2h |
| P2 | search_aggregator.py (GitHub API + dedup) | 3h |
| P3 | sandbox_runner.py + Dockerfiles | 6h |
| P4 | Report prompt + ROLE.md | 2h |
| P5 | Main graph integration + Discord command | 1h |
| P6 | E2E tests + security validation | 3h |
