# Google Docs Skill

生成专业文档。所有操作通过路由执行，不直接运行命令。

---

## 路径一：Pandoc 本地渲染（推荐）→ render_docs

将 Markdown 内容转换为带专业样式的 DOCX 文件。

### 使用方式

将完整的 Markdown 内容放在路由信号的 context 中：

```json
{"route": "render_docs", "context": "# 文档标题\n\n## 第一章\n\n正文内容..."}
```

框架会自动将 context 传给 Pandoc，返回生成的 DOCX 文件路径。

### Markdown 写作规范

```markdown
---
title: 文档标题
author: Jei · 无垠智穹
date: 2026-03-23
---

# 第一章

正文内容。**重点** 用粗体标注。

## 1.1 小节

- 要点一
- 要点二

> 关键结论或重要发现用 blockquote 高亮。

### 1.1.1 细节

| 指标 | Q4 | Q1 | 变化 |
|------|----|----|------|
| 笔记数 | 930 | 1,247 | +34% |
```

### 写作规则

- 用 heading 层级组织结构（最多 3 层：H1 → H2 → H3）
- 关键结论用 `>` blockquote 高亮
- 表格用 pipe 格式，标注数据来源
- 代码用 fenced code block（带语言标识）
- 内容来自 Vault 时，在脚注或文末标注笔记路径

---

## 路径二：Google Docs API → gws_docs

在 Google Drive 中直接创建和编辑文档。适合需要实时协作的场景。

### 使用方式

将 gws 命令放在路由信号的 context 中：

```json
{"route": "gws_docs", "context": "gws docs documents create --json '{\"title\": \"文档标题\"}'"}
```

### 常用命令

创建文档：
```
gws docs documents create --json '{"title": "文档标题"}'
```

快速追加文本：
```
gws docs +write --document DOCUMENT_ID --text '要追加的文本'
```

获取文档内容：
```
gws docs documents get --params '{"documentId": "DOCUMENT_ID"}'
```

结构化编辑（batchUpdate）：
```
gws docs documents batchUpdate --params '{"documentId": "ID"}' --json '{"requests": [...]}'
```

### 注意事项

- **index 计算**：每次插入后，后续内容的 index 会偏移
- **换行符**：每段文本末尾需要 `\n`
- **batchUpdate 顺序**：requests 按顺序执行，先插入文本再设置样式
- **出错回滚**：整个 batchUpdate 原子执行，任一请求失败全部回滚
