# BootstrapBuilder - 0号管家 基础 Builder Agent 计划

## Context

在空的 `BootstrapBuilder/` 目录里搭建"无垠智穹 0号管家"。
核心是 LangGraph 状态机（Claude 主控 + Gemini 战略顾问），通过 SQLite 持久化记忆，
支持多个接口无缝共享同一个会话状态。

**最核心的设计原则：** 所有接口共用同一个 `thread_id` + SQLite DB → 记忆自动同步，切换接口不丢失上下文。

---

## 接口优先级

| 优先级 | 接口 | 机制 | 备注 |
|--------|------|------|------|
| P0 | Discord | discord.py bot → LangGraph `astream()` | 远程唯一入口 |
| P1 | VSCode Terminal | CLI 脚本 + 固定 thread_id | VSCode 集成终端直连，看代码同时对话 |
| P1 | Tmux | `libtmux` 管理 named session `bootstrap_boss` | detach/reattach 不丢历史 |
| P2 | PixelOffice | **无需显式接入** | 监听 Claude Code JSONL transcript 自动可视化 |

> PixelOffice 是纯观测层（VSCode 扩展），只要在 VSCode 终端运行 CLI，它会自动把 agent 活动渲染成像素小人。

---

## 框架选型：LangGraph + SQLite WAL

- SqliteSaver 天然提供 thread_id → 会话映射
- `astream()` 支持 Discord 异步流式
- `PRAGMA journal_mode=WAL` 解决 Discord + CLI 并发访问锁问题

---

## 目录结构

```
BootstrapBuilder/
├── agent/
│   ├── __init__.py
│   ├── core.py          # LangGraph 状态机 + 引擎实例
│   └── tools.py         # consult_ceo_gemini 等 tool 定义
├── interfaces/
│   ├── discord_bot.py   # P0: Discord 远程接口（async）
│   └── cli.py           # P1: 本地 CLI + Tmux 集成
├── main.py              # 入口：python main.py [discord|cli|tmux]
├── requirements.txt
├── .env.example
└── PLAN.md              # 本文件
```

---

## Discord 关键设计（参考 openclaw）

1. **Draft Streaming** — 先发占位消息，边生成边 edit，节流 1200ms
2. **Fence-Aware Chunking** — 代码块感知分块，保证 ``` fence 不被截断
3. **Long Session Sliding Window** — `trim_messages()` 保留最近 40 条，System Prompt 永远保留

---

## 会话同步示意图

```
Discord Bot ──┐
              ├──→ builder_engine (LangGraph)
VSCode CLI ───┤         ↕
              │    SQLite WAL (cyber_bootstrap.db)
Tmux CLI ─────┘    thread_id: "boss_bootstrap_session_01"

PixelOffice → 自动监听 VSCode JSONL（无需集成代码）
```

---

## 验证步骤

1. `python main.py cli` → 本地对话，看到流式输出
2. Ctrl+C 重启 → 历史记忆保留
3. `python main.py tmux` → 创建 tmux session `bootstrap_boss`，detach/reattach 正常
4. `python main.py discord` → Discord 发消息，Bot 回复，与 CLI 共享历史
5. VSCode 终端运行 CLI → PixelOffice 自动显示像素小人
