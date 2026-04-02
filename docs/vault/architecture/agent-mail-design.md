# Agent Mail — 设计决策记录

> 创建时间: 2026-04-01
> 状态: 设计阶段，待实现

---

## 背景

当前问题：Jei 的 EXTERNAL_TOOL 执行超过 120s 后，需要将监控任务委托给 Asa（集团唯一监工）。
根本需求：Agent 间需要可靠的异步消息通信机制。

---

## 设计原则

1. **Asa 是唯一 heartbeat owner**：其他 agent 不启动 heartbeat MCP server，不在启动时连接 heartbeat proxy
2. **异步解耦**：发送方发完即走，不等接收方在线
3. **持久化优先**：agent 重启后收件箱不丢
4. **MCP 接口**：与框架现有 MCP 架构同构，工具调用方式一致

---

## 核心设计："邮件收件箱"模型

每个 agent 有独立收件箱。通信通过三个动词完成：

```
send_mail(to, subject, body)   → 写入收件人 inbox
fetch_inbox(agent_name)        → 读取自己未处理的邮件
ack_mail(mail_id)              → 标记已处理
```

### 消息结构

```json
{
  "mail_id": "uuid",
  "from_agent": "jei",
  "to_agent": "asa",
  "subject": "monitor_delegate",
  "body": {
    "task_id": "tool_abc123",
    "pid": 9876,
    "output_path": "/path/to/output",
    "hard_timeout": 600,
    "notify_channel": "1488233657742393358"
  },
  "created_at": "2026-04-01T22:00:00Z",
  "acked_at": null
}
```

---

## 架构方案

### 选型过程

| 方案 | 结论 |
|------|------|
| Discord 频道作总线 | ❌ 关键路径不该绑外部网络 |
| SQLite + inotify | ⚠️ 可行但 asyncio 集成有坑 |
| Agent Bus MCP Server（新建） | ⚠️ 架构一致但 Asa 无工具调用能力 |
| Unix Domain Socket | ✅ 低延迟但无持久化 |
| **Agent Mail MCP Server** | ✅ 最终选择 |
| Supervisor LLM（Grok 提案） | ❌ 传输层不该用 LLM，幻觉风险，单点故障 |

### 参考项目

- [mcp_agent_mail](https://github.com/Dicklesworthstone/mcp_agent_mail)（1.9k stars）：设计方向一致，但功能过重（file reservations、git 追踪、Beads 集成、Python 3.14）
- 决策：**参考其接口设计，自建轻量版** `mcp_servers/agent_mail/`

---

## 实现规划

### 存储

每个 agent 自己的 SQLite db（如 `asa.db`）里有 `mailbox` 表，或统一用 `shared.db`。
倾向：**shared.db**，路径 `data/agent_mail/mail.db`，单一事实来源，查询跨 agent 方便。

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS mailbox (
    mail_id     TEXT PRIMARY KEY,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,       -- JSON string
    created_at  TEXT NOT NULL,
    acked_at    TEXT DEFAULT NULL
);

CREATE INDEX idx_inbox ON mailbox (to_agent, acked_at);
```

### MCP 工具接口

```python
@mcp.tool()
async def send_mail(to: str, subject: str, body: dict) -> dict

@mcp.tool()
async def fetch_inbox(agent_name: str, unread_only: bool = True) -> list[dict]

@mcp.tool()
async def ack_mail(mail_id: str) -> dict

@mcp.tool()
async def list_agents() -> list[dict]   # agent 发现
```

### 收发分离：启动时不强制连接 MCP

**核心原则：读直接 SQL，写才走 MCP。**

```
读路径（收邮件）：agent 进程内 background task 直接读 mail.db
                  → 无 MCP 长连接，无额外进程依赖
                  → mail server 挂了，收件箱照常可读

写路径（发邮件）：懒加载，首次需要发邮件时才连接 mail MCP server
                  → 发完即断，无持久连接
```

| Agent | 启动时连接 mail MCP？ | 理由 |
|-------|---------------------|------|
| Asa | ❌ 直接读 SQLite（background task） | 监工，持续轮询收件箱 |
| Jei | ❌ PENDING 时懒加载连接 | 只在超时委托时发一封 |
| Hani | ❌ 按需懒加载 | 需要协调时才发 |

**mail MCP server 只是写的入口**，不是所有 agent 都需要在启动时与它建立连接。

### 触发机制（agent 如何感知新邮件）

每个 agent 进程内启动一个 asyncio background task，每 **1 秒**直接查询 `mail.db`：

```sql
SELECT * FROM mailbox WHERE to_agent = ? AND acked_at IS NULL ORDER BY created_at ASC
```

收到新邮件 → 触发 `_on_mail_received(mail)` 回调 → 框架层处理
（例：Asa 收到 `monitor_delegate` → 调 `heartbeat_register_monitor`）。

### Agent 发现（见下节）

---

## Agent 发现机制

### 静态发现：目录扫描

`EdenGateway/agents/` 目录本身就是注册表：

```
EdenGateway/agents/
├── asa/identity.json    → name: "asa"
├── hani/identity.json   → name: "hani"
└── jei/identity.json    → name: "jei"
```

`list_agents()` 工具扫描此目录，返回所有已知 agent。
优点：离线 agent 也能被发现；无需运行时注册。

### 动态状态：心跳注册

agent 进程启动时向 mail server 调 `register_agent(name, pid)`，
shutdown 时调 `unregister_agent(name)`，
mail server 维护 `online_since` / `last_seen` 字段。

结合两者：**静态知道有谁，动态知道谁在线。**

---

## 与现有架构的集成点

### 修复 Jei 启动时错误绑定 heartbeat 的问题

`agent_loader.py` 第 344-348 行需修改：
```python
# 当前（错误）：有 EXTERNAL_TOOL 就连 heartbeat
if self._has_external_tool_nodes():
    return await self._connect_heartbeat_proxy_only()

# 修改后：有 EXTERNAL_TOOL 但配置了 mail_delegate，走 agent mail 路径
# Jei 启动时不连 heartbeat，PENDING 时通过 mail 委托给 Asa
```

### ExternalToolNode._on_timeout() 新增委托逻辑

```python
# PENDING 发生时，发邮件给 asa
await send_mail(
    to="asa",
    subject="monitor_delegate",
    body={"task_id": task_id, "pid": proc.pid, ...}
)
```

### Asa 收件箱处理

Asa 的 mailbox watcher 收到 `monitor_delegate` → 直接调 `heartbeat_register_monitor` → 注册完成后发邮件回 notify_channel 对应的 agent。

---

## 待决策

- [x] ~~每个 agent 启动时是否必须连接 mail MCP？~~ → **是，统一走 MCP 自启动机制**
- [x] ~~懒加载 vs 启动时连接？~~ → **启动时连接，由 entity.json mcps 字段声明**
- [x] ~~注册 PID 走 MCP 还是直接 SQL？~~ → **连接时通过 MCP 工具注册，连接即注册**
- [ ] shared.db 还是各自 db？→ 倾向 shared.db（`data/agent_mail/mail.db`）
- [ ] `list_agents()` 是否需要"在线状态"？

## 关联文档

- [MCP 自启动机制](./mcp-autostart-design.md) — agent_mail 启动逻辑属于框架级通用设计的一部分
