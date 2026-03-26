# 自动化工具发现与评估系统 — 架构设计

> 状态：设计稿 | 日期：2026-03-22

## 一、目标

用户描述需求（如"找一个AI工具能写Google Slides"），系统自动搜索、筛选、克隆、测试、输出评估报告。

## 二、流水线拓扑

```
__start__ → query_expand → search_aggregate → candidate_filter → sandbox_eval → report_gen → __end__
```

| 节点 | 框架节点类型 | Provider | 职责 |
|------|-------------|----------|------|
| `query_expand` | `CLAUDE_CLI` | Claude | 理解自然语言需求，生成搜索关键词（中→英翻译、同义扩展） |
| `search_aggregate` | `DETERMINISTIC` | 无 | GitHub API + Web 搜索 + 去重，纯 Python |
| `candidate_filter` | `OLLAMA` | llama3.2:3b | 批量打分（0-10），按相关性筛选 Top-K |
| `sandbox_eval` | `EXTERNAL_TOOL` | Docker | 克隆 → 安装 → 测试，容器隔离 |
| `report_gen` | `CLAUDE_CLI` | Claude | 生成面向用户的评估报告 |

## 三、集成方式

通过 `SUBGRAPH_REF` 挂载到主 Agent 图（`technical_architect`）。

主图改动：
- `entity.json`：+1 node（`tool_discovery`），+2 edges
- `command_registry.py`：+1 行 `_r("!discover", ...)`
- `base_interface.py`：`handle_command()` 添加 `!discover` 分支

自动路由通过 `routing_hint` 实现，Claude 看到用户搜索意图时自动触发。

## 四、State Schema

```python
class ToolDiscoveryState(TypedDict):
    # 继承 BaseAgentState 必需字段
    messages: Annotated[list[BaseMessage], _keep_last_2]
    routing_target: str
    routing_context: str
    workspace: str
    project_root: str
    node_sessions: Annotated[dict, _merge_dict]
    # Discovery 专属字段（JSON 序列化字符串）
    user_query: str              # 原始自然语言需求
    search_intent: str           # {keywords, github_queries, web_queries}
    raw_candidates: str          # [{repo, stars, license, description, ...}]
    filtered_candidates: str     # Top-K [{..., relevance_score, rationale}]
    evaluation_results: str      # [{install_ok, test_pass_rate, ...}]
    discovery_config: str        # {depth: 5, timeout: 300}
    discovery_errors: str        # 累积错误日志
```

## 五、数据结构

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
    "readme_snippet": str,      # 前 500 字
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
    "api_surface": int,         # 导出函数/类数量
    "has_docs": bool,
    "has_examples": bool,
    "project_type": "python|node|rust|go|unknown",
    "eval_mode": "docker|venv|static",
    "errors": list[str],
}
```

## 六、沙箱设计

`sandbox_eval` 使用 Docker 一次性容器（`--rm`）：

- `--read-only` + tmpfs `/workspace`：防止持久化写入
- 网络白名单：仅允许 pypi.org / npmjs.com / github.com
- 资源限制：`--memory=1g --cpus=1 --pids-limit=256`
- 每个 repo 硬超时 300s
- 降级方案：无 Docker 时用 `venv` + 静态分析，报告标注 `⚠️ 受限评估`

安全措施：安装前静态扫描 `setup.py` / `pyproject.toml` 中的 `os.system` / `subprocess` / `eval` 调用。

## 七、文件结构

### 新建

```
blueprints/functional_graphs/tool_discovery/
├── entity.json                  # 图定义
└── ROLE.md                     # LLM 节点 persona

framework/schema/tool_discovery.py              # State schema
framework/nodes/tool_discovery/
├── search_aggregator.py        # DETERMINISTIC：GitHub API + Web 搜索
└── sandbox_runner.py           # EXTERNAL_TOOL：Docker 沙箱编排

docker/
├── Dockerfile.sandbox-python   # Python 沙箱镜像
└── Dockerfile.sandbox-node     # Node.js 沙箱镜像
```

### 修改

```
blueprints/role_agents/technical_architect/entity.json   # +1 node, +2 edges
framework/command_registry.py                           # +1 行注册
framework/base_interface.py                             # +1 命令处理分支
```

## 八、entity.json 结构

```json
{
  "name": "ToolDiscovery",
  "routing_hint": "当用户要求发现/搜索/评估开源工具时使用",
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

## 九、风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| 恶意代码执行 | 🔴 | Docker 沙箱 + read-only + 网络白名单 + 安装前静态扫描 |
| GitHub API Rate Limit | 🟡 | 要求 `GITHUB_TOKEN`；搜索结果缓存 1h |
| 评估耗时过长 | 🟡 | 默认 depth=3；渐进式反馈；每步硬超时 |
| Docker 不可用 | 🟡 | 降级到 venv + 静态分析，标注受限评估 |
| Ollama 打分不稳定 | 🟢 | 要求输出 JSON；异常分数 fallback 到 5 |

## 十、实现优先级

| Phase | 内容 | 预计工作量 |
|-------|------|-----------|
| P1 | State schema + entity.json 骨架 | 2h |
| P2 | search_aggregator.py（GitHub API + 去重） | 3h |
| P3 | sandbox_runner.py + Dockerfiles | 6h |
| P4 | Report prompt + ROLE.md | 2h |
| P5 | 主图集成 + Discord 命令 | 1h |
| P6 | E2E 测试 + 安全验证 | 3h |
