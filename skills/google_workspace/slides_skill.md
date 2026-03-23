# Google Slides Skill

生成专业 Slides 演示文稿。所有操作通过路由执行，不直接运行命令。

---

## 路径一：Presenton 本地渲染（推荐）→ render_slides

生成漂亮的、可编辑的 PPTX 文件。Presenton 有专业设计引擎，自动处理布局、配色、排版。

### 使用方式

将完整的 slides 内容文本放在路由信号的 context 中：

```json
{"route": "render_slides", "context": "你整理好的 slides 内容文本"}
```

框架会自动将 context 传给 Presenton 渲染引擎，返回生成的 PPTX 文件路径。

### 内容写作格式

将内容组织为清晰的文本，包含主题、要点、数据。Presenton 的 AI 引擎会自动拆分为 slides 并设计布局。

示例 context 内容：
```
知识库季度总结 - 2026 Q1

关键发现：
- Vault 笔记总量增长 34%，达到 1,247 篇
- 知识图谱覆盖 12 个领域，交叉引用密度提升 2.1x
- 最活跃领域：AI 架构设计、系统集成

数据概览：
Q4 笔记数: 930 → Q1 笔记数: 1,247
Q4 标签数: 89 → Q1 标签数: 142
Q4 引用密度: 1.8 → Q1 引用密度: 3.9

下一步行动：
1. 建立跨领域知识桥接
2. 引入自动摘要生成
3. 优化标签分类体系
```

---

## 路径二：Google Slides API → gws_slides

在 Google Drive 中直接创建和编辑 Slides。适合需要实时协作的场景。

### 使用方式

将 gws 命令放在路由信号的 context 中：

```json
{"route": "gws_slides", "context": "gws slides presentations create --json '{\"title\": \"演示标题\"}'"}
```

### 常用命令

创建演示文稿：
```
gws slides presentations create --json '{"title": "演示标题"}'
```

获取演示文稿信息：
```
gws slides presentations get --params '{"presentationId": "PRESENTATION_ID"}'
```

添加内容（batchUpdate）：
```
gws slides presentations batchUpdate --params '{"presentationId": "ID"}' --json '{"requests": [...]}'
```

可用 Layout: `TITLE`, `TITLE_AND_BODY`, `TITLE_AND_TWO_COLUMNS`, `TITLE_ONLY`, `BLANK`, `SECTION_HEADER`, `BIG_NUMBER`

---

## Slides 内容设计规则

无论使用哪条路径，生成 slides 内容时必须遵循：

### 结构规则
- **标题页**：只放标题 + 副标题，不要塞内容
- **内容页**：每页最多 **3 个要点**，每个要点不超过 **15 个字**
- **章节分隔**：每 3-4 页插入一个 section header 页
- **数据页**：用表格或 big_number 布局，标注数据来源
- **结尾页**：必须有 call-to-action（下一步行动）

### 文字规则
- 每页总文字量不超过 **40 字**（标题除外）
- 用短句、关键词，不要写完整段落
- 数据用数字说话，不要用形容词

### 来源标注
- 内容来自 Vault 时，在 speaker notes 或脚注中标注笔记路径
- 数据引用标注来源和日期
