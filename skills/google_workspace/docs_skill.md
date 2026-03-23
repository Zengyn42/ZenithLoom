# Google Docs Skill

生成专业文档。支持两条路径：
1. **Pandoc 本地渲染**（推荐）— 生成带专业样式的 DOCX 文件
2. **Google Docs API** — 通过 `gws` CLI 直接操作 Google Docs

---

## 路径一：Pandoc 本地渲染（推荐）

将 Markdown 内容转换为带专业样式的 DOCX 文件。

### 工作流

1. 将内容写为 Markdown 格式（含 YAML frontmatter）
2. 保存到临时文件（如 `/tmp/doc_content.md`）
3. 路由到 `render_docs` 节点，或直接调用脚本

### 调用方式

```bash
bash skills/google_workspace/scripts/render_docs.sh \
  /tmp/doc_content.md \    # Markdown 文件
  /tmp/output.docx          # 输出路径
```

### Markdown 写作规范

```markdown
---
title: 文档标题
author: Jei · 无垠智穹
date: 2026-03-22
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

## 路径二：Google Docs API（gws CLI）

通过 `gws` CLI 直接在 Google Drive 中创建和编辑文档。适合需要实时协作的场景。

## 常用命令

### 创建文档

```bash
gws docs documents create --json '{"title": "文档标题"}'
```

返回 JSON 中 `documentId` 即为新建的文档 ID。

### 获取文档内容

```bash
gws docs documents get --params '{"documentId": "DOCUMENT_ID"}'
```

### 快速追加文本（Helper）

```bash
gws docs +write --document DOCUMENT_ID --text '要追加的文本'
```

文本会插入到文档末尾。适合快速写入，不需要复杂格式。

### 结构化编辑（batchUpdate）

所有富文本修改通过 `batchUpdate` 完成：

```bash
gws docs documents batchUpdate --params '{"documentId": "DOCUMENT_ID"}' --json '{
  "requests": [
    {
      "insertText": {
        "location": {"index": 1},
        "text": "标题文本\n"
      }
    }
  ]
}'
```

### 常用 batchUpdate 请求类型

**插入文本：**
```json
{"insertText": {"location": {"index": N}, "text": "文本\n"}}
```
index 从 1 开始（0 是文档开头之前的特殊位置）。

**删除内容：**
```json
{"deleteContentRange": {"range": {"startIndex": N, "endIndex": M}}}
```

**设置段落样式（标题）：**
```json
{
  "updateParagraphStyle": {
    "range": {"startIndex": N, "endIndex": M},
    "paragraphStyle": {"namedStyleType": "HEADING_1"},
    "fields": "namedStyleType"
  }
}
```
可用样式: `TITLE`, `SUBTITLE`, `HEADING_1` ~ `HEADING_6`, `NORMAL_TEXT`

**设置文字样式：**
```json
{
  "updateTextStyle": {
    "range": {"startIndex": N, "endIndex": M},
    "textStyle": {"bold": true, "fontSize": {"magnitude": 14, "unit": "PT"}},
    "fields": "bold,fontSize"
  }
}
```

**插入表格：**
```json
{
  "insertTable": {
    "rows": 3,
    "columns": 3,
    "location": {"index": N}
  }
}
```

## 工作流程：Markdown → Google Doc

1. 用 `documents create` 创建空白文档
2. 解析 Markdown 内容，按结构拆分（标题、段落、列表、代码块）
3. 用 `batchUpdate` 批量插入文本
4. 再用 `batchUpdate` 批量设置样式（标题级别、粗体、代码字体等）
5. 用 `documents get` 确认最终结果

## 注意事项

- **index 计算**：每次插入后，后续内容的 index 会偏移。建议从文档末尾往前插入，或一次 batchUpdate 搞定
- **换行符**：每段文本末尾需要 `\n`
- **batchUpdate 顺序**：requests 按顺序执行，先插入文本再设置样式
- **出错回滚**：整个 batchUpdate 原子执行，任一请求失败全部回滚
- **快速写入**：纯文本场景直接用 `gws docs +write`，比 batchUpdate 简单得多
