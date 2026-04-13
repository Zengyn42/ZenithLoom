# PrismRagMCP Inbox Architecture

> Date: 2026-04-09
> Status: Approved (architecture decision, not yet implemented)

## Summary

LLM 产出的知识笔记（.md）通过**生产者-消费者模型**入库 Obsidian Vault。
任何 LLM 都能写笔记（无论有没有 MCP），整理和索引由专职 Agent（Jei）统一处理。

## Design Principle

**写入门槛为零，整理交给专人。**

- 生产者（Hani / Asa / 任何 LLM）不需要理解图谱结构、标签体系、embedding 机制
- 消费者（Jei + PrismRagMCP）负责所有知识工程：分类、打标签、embedding、图整理
- 写入和索引解耦：LLM 不调 `embed()`，embedding 是基础设施自动处理

## Architecture

```
生产者（任何 LLM，有没有 MCP 都行）
    │
    ├── 有 MCP：调 PrismRagMCP.write_note() → 直接入库 + 自动 embedding
    │
    └── 无 MCP：用 Write 工具 → ~/Foundation/Vault/inbox/*.md（纯文件投递）
                    │
                    ▼
              ~/Foundation/Vault/inbox/          ← 未分类投递箱
                    │
                    ▼
              消费者（Jei，有 PrismRagMCP）
                    │
                    ├── 定时扫描 inbox/ 中的新 .md
                    ├── 理解内容 → 打标签 → 选择分类目录
                    ├── 调 PrismRagMCP 工具写入正式 Vault 位置
                    ├── 自动 embedding + 图整理
                    └── 原文件移到 inbox/processed/（或删除）
```

## Two Modes of Writing

### Mode 1: With PrismRagMCP (direct)

LLM 拥有 PrismRagMCP 工具时，直接调用入库：

```
LLM 调用 prism_write_note(
    path="Architecture/session-mode-design.md",
    content="...",
    tags=["session", "subgraph", "langgraph"]
)
    │
    ▼
PrismRagMCP 内部：
    1. 写入文件到 Vault 正式目录
    2. 自动补 frontmatter（created_at, author, tags）
    3. 触发 embedding pipeline（chunk → embed → upsert vector store）
    4. 更新知识图谱索引
```

**适用场景**：Jei（知识管理员）、未来升级后的 Hani/Asa

### Mode 2: Without PrismRagMCP (inbox drop)

LLM 没有 PrismRagMCP 时，用框架原生 Write 工具自由写入 inbox：

```
LLM 调用 Write(
    path="~/Foundation/Vault/inbox/2026-04-09-hani-session-design-notes.md",
    content="..."   ← 格式随意，纯文本也行
)
```

**约定**：
- 路径：`~/Foundation/Vault/inbox/`
- 文件名建议：`YYYY-MM-DD-<topic>.md`（但不强制）
- 内容格式：完全自由，LLM 想怎么写就怎么写
- Frontmatter：可有可无（消费者会补全）

**适用场景**：当前的 Hani、Asa（无 MCP 权限）

### Consumer: Jei's Inbox Processing

Jei 通过 heartbeat 或手动触发，定期处理 inbox：

```
1. 扫描 inbox/ 中的 .md 文件（排除 processed/ 子目录）
2. 逐个读取内容
3. 调 prism_list_tags() 获取现有标签体系
4. 基于内容理解：
   - 选择目标目录（Architecture/ Projects/ Decisions/ etc.）
   - 生成标签列表（优先复用现有标签，必要时创建新标签）
   - 补全 frontmatter（author 从文件名或内容推断）
5. 调 prism_write_note() 写入正式位置
6. 自动 embedding（PrismRagMCP 内部处理）
7. 将原文件移到 inbox/processed/（保留溯源）
```

## PrismRagMCP Tool Interface (Conceptual)

### LLM-facing tools (LLM 调用)

| Tool | Description |
|---|---|
| `prism_write_note(path, content, tags)` | 写入笔记到正式 Vault 位置 + frontmatter 规范化 |
| `prism_patch_note(path, section, content)` | Section-based 局部修改 |
| `prism_read_note(path)` | 读取完整笔记 |
| `prism_search(query)` | 混合搜索：向量语义搜索 + 关键词搜索 |
| `prism_list_tags()` | 返回现有标签体系（供写入时参考） |
| `prism_list_files(directory)` | 列出目录下的文件 |
| `prism_get_links(path)` | 查询双向链接关系 |

### Internal (LLM 不调用，PrismRagMCP 自动处理)

| Internal Process | Trigger |
|---|---|
| Chunk + embed + upsert vector store | `write_note()` / `patch_note()` 调用后自动触发 |
| 标签索引更新 | 标签变更时自动触发 |
| 知识图谱更新（链接关系） | 笔记写入/修改后自动触发 |

**LLM 永远不需要知道 embedding 的存在。** 它写笔记、搜笔记、管标签。向量化是透明的基础设施。

## Agent Roles

| Agent | MCP 权限 | 写入方式 | 知识管理职责 |
|---|---|---|---|
| **Hani** | 无 PrismRagMCP | Write → inbox/ | 生产者：记录设计决策、技术经验 |
| **Asa** | 无 PrismRagMCP | Write → inbox/ | 生产者：记录管理流程、运营笔记 |
| **Jei** | 有 PrismRagMCP | 直接 write_note() | 消费者 + 生产者：整理 inbox、管理知识库 |
| **未来 Agent** | 视配置 | 有 MCP 则直接，无则 inbox | 取决于是否分配 MCP |

## Directory Convention

```
~/Foundation/Vault/
├── inbox/                    ← 未分类投递箱（任何 LLM 可写）
│   ├── 2026-04-09-hani-xxx.md
│   ├── 2026-04-10-asa-yyy.md
│   └── processed/            ← Jei 整理后的原文件归档
├── Architecture/             ← 正式分类目录（Jei 通过 MCP 管理）
├── Projects/
├── Decisions/
├── Operations/
└── ...
```

## Key Design Decisions

### Why not give all agents MCP?

- **关注点分离**：Hani 是技术架构师，不应该关心知识库的分类体系和标签一致性
- **质量控制**：Jei 作为知识管理员，可以保证标签不混乱、分类一致、embedding 质量
- **容错**：即使 MCP 服务挂了，Hani/Asa 仍然可以通过 inbox 产出知识，不丢失

### Why inbox + consumer, not direct write + async index?

- **LLM 不知道 Vault 的目录结构**：如果让 Hani 直接写 `Architecture/xxx.md`，它需要先了解整个 Vault 的分类体系。这不是它的职责。
- **标签一致性**：如果每个 LLM 自己打标签，标签会碎片化（一个叫 `#arch`，另一个叫 `#architecture`）。集中管理更可控。
- **渐进式升级**：先 inbox 模式跑通，以后给 Hani/Asa 加 MCP 是无缝的（多一个 write 路径，inbox 仍然可用）。

### Why move to processed/ instead of delete?

- **溯源**：可以追溯原始内容和 Jei 整理后的版本差异
- **防误删**：Jei 整理错了可以从 processed/ 恢复
- **审计**：可以定期检查 Jei 的整理质量

## Future: PrismRagMCP = Obsidian MCP + RAG

PrismRagMCP 是当前 `mcp_servers/obsidian/` 和未来 vector embedding 系统的整合：

| 当前 (Obsidian MCP) | 未来 (PrismRagMCP) |
|---|---|
| 纯文件读写 | 文件读写 + 自动 embedding |
| grep 文本搜索 | 向量语义搜索 + 关键词混合搜索 |
| 手动管标签 | 写入时自动建议标签（参考现有体系）|
| 无知识图谱 | 双向链接 + 标签 → 知识图谱 |
| 无 inbox 机制 | inbox + consumer pipeline |

## See Also

- `mcp_servers/obsidian/` — 当前 Obsidian MCP 实现
- `blueprints/functional_graphs/knowledge_shelf/` — Jei 的知识库子图
- `blueprints/role_agents/knowledge_curator/` — Jei 的角色定义
