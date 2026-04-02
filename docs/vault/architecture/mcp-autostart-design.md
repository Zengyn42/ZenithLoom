# MCP 自启动机制 — 设计决策记录

> 创建时间: 2026-04-01
> 状态: 设计阶段，待实现
> 影响范围: agent_loader.py、所有 entity.json、heartbeat、agent_mail 及未来所有 MCP

---

## 背景

当前问题：
- heartbeat MCP 的启动逻辑硬编码在 `agent_loader.py` 的 `start_heartbeat()` 方法里
- agent_mail MCP 需要类似的自启动能力
- 未来每增加一个 MCP，都需要在 agent_loader 里单独处理

**决策：统一化所有 MCP 的自启动逻辑，由 blueprint/entity.json 声明依赖，框架自动处理。**

---

## 核心设计

### entity.json 新增 `mcps` 字段

```json
{
  "name": "asa",
  "blueprint": "...",
  "mcps": [
    {
      "name": "heartbeat",
      "module": "mcp_servers.heartbeat",
      "transport": "sse",
      "host": "127.0.0.1",
      "port": 8100,
      "pid_file": "data/heartbeat/mcp.pid"
    },
    {
      "name": "agent_mail",
      "module": "mcp_servers.agent_mail",
      "transport": "sse",
      "host": "127.0.0.1",
      "port": 8200,
      "pid_file": "data/agent_mail/mail.pid"
    }
  ]
}
```

每个 MCP 声明：`module`（Python 模块）、`transport`、`host:port`、`pid_file`（用于存活检测）。

---

## 启动逻辑（通用，适用所有 MCP）

```
agent 启动
    ↓
遍历 entity.json 的 mcps 列表
    ↓
对每个 MCP：
    _is_running(pid_file)?
        ├── 否 → 加文件锁 → double-check → 启动（detach）→ 释放锁 → 等待就绪
        └── 是 → 跳过启动
    ↓
连接（SSE）→ 注册工具到 TOOL_REGISTRY
```

### 文件锁防竞争（解决 heartbeat 的 race condition 问题）

```python
lock_path = pid_file.with_suffix(".launch.lock")
with open(lock_path, "w") as lf:
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not _is_running(pid_file):        # double-check
            _launch(module, host, port, pid_file)
    except BlockingIOError:
        pass  # 别人正在启动，等连接重试即可
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
```

---

## agent_loader.py 改造

### 现在（硬编码）

```python
async def start_heartbeat(self): ...       # heartbeat 专用
async def start_mcp_servers(self): ...     # obsidian 等专用
async def _connect_heartbeat_proxy_only(): # EXTERNAL_TOOL 专用
```

### 改造后（通用）

```python
async def start_mcps(self):
    """遍历 entity.json 的 mcps 字段，逐一确保运行并连接。"""
    for mcp_conf in self.entity.get("mcps", []):
        proxy = await MCPLauncher.ensure_and_connect(mcp_conf)
        if proxy:
            self._mcp_proxies[mcp_conf["name"]] = proxy
            # 注册工具到对应的 LLM tool registry
            self._register_mcp_tools(mcp_conf["name"], proxy)
```

`start_heartbeat()` 保留但内部走 `start_mcps()` 统一路径，最终废弃。

---

## MCPLauncher（新模块）

位置：`framework/mcp_launcher.py`

职责：
- `ensure_and_connect(mcp_conf)` — 检查 + 自启动 + 连接，返回 proxy
- `_is_running(pid_file)` — 检查进程存活
- `_launch(module, host, port, pid_file)` — detach 启动 + 写 PID 文件
- `_wait_ready(url, timeout=10)` — 轮询等待 SSE 端点就绪

---

## 各 entity.json 更新计划

| Agent | 当前 MCP 配置 | 迁移后 mcps 字段 |
|-------|-------------|----------------|
| asa | `"heartbeat": [...]` | heartbeat + agent_mail |
| jei | 无（但错误连了 heartbeat）| agent_mail |
| hani | 无 | agent_mail |

---

## 与现有设计的关系

- **heartbeat 的 race condition 修复**：统一使用文件锁，根治竞争问题
- **jei zombie 问题修复**：jei 的 mcps 字段不包含 heartbeat，彻底杜绝错误连接
- **agent_mail 启动**：和 heartbeat 完全同等地位，第一个启动的 agent 拉起 mail server

---

## 待实现

- [ ] `framework/mcp_launcher.py` — 通用 MCPLauncher
- [ ] `agent_loader.py` — `start_mcps()` 替换 `start_heartbeat()`
- [ ] `asa/identity.json` — 迁移 heartbeat 配置到 mcps 字段，增加 agent_mail
- [ ] `jei/identity.json` — 增加 agent_mail（无 heartbeat）
- [ ] `hani/identity.json` — 增加 agent_mail
- [ ] `mcp_servers/agent_mail/server.py` — 完善 SIGUSR1 通知逻辑
