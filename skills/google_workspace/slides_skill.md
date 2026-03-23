# Google Slides Skill

生成专业 Slides 演示文稿。支持两条路径：
1. **Presenton 本地渲染**（推荐）— 生成漂亮的可编辑 PPTX 文件
2. **Google Slides API** — 通过 `gws` CLI 直接操作 Google Slides

---

## 路径一：Presenton 本地渲染（推荐）

生成漂亮的、可编辑的 PPTX 文件。Presenton 有专业设计引擎，自动处理布局、配色、排版。

### 工作流

1. 整理内容为纯文本（从 Vault 知识或用户需求）
2. 将内容写入临时文件（如 `/tmp/slides_content.txt`）
3. 路由到 `render_slides` 节点，或直接调用脚本

### 调用方式

```bash
bash skills/google_workspace/scripts/render_slides.sh \
  /tmp/slides_content.txt \   # 内容文件
  /tmp/output.pptx \           # 输出路径
  10 \                         # slides 数量
  modern                       # 模板: general, modern, swift
```

需要设置环境变量：
- `PRESENTON_API_KEY` — Presenton API key
- `PRESENTON_API_URL` — Presenton 服务地址（默认 http://localhost:5000）

### 内容写作格式

将内容组织为清晰的文本，包含主题、要点、数据。Presenton 的 AI 引擎会自动拆分为 slides 并设计布局。

示例输入内容：
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

## Slides 内容设计规则

无论使用哪条路径，生成 slides 内容时必须遵循以下规则：

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

---

## 路径二：Google Slides API（gws CLI）

通过 `gws` CLI 直接在 Google Drive 中创建和编辑 Slides。适合需要实时协作或直接在 Google Workspace 中操作的场景。

## 常用命令

### 创建演示文稿

```bash
gws slides presentations create --json '{"title": "演示标题"}'
```

返回 JSON 中 `presentationId` 即为新建的文稿 ID。

### 获取演示文稿信息

```bash
gws slides presentations get --params '{"presentationId": "PRESENTATION_ID"}'
```

### 添加幻灯片和内容（batchUpdate）

所有内容修改通过 `batchUpdate` 完成，一次可提交多个请求：

```bash
gws slides presentations batchUpdate --params '{"presentationId": "PRESENTATION_ID"}' --json '{
  "requests": [
    {
      "createSlide": {
        "slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"},
        "insertionIndex": 1
      }
    }
  ]
}'
```

### 常用 batchUpdate 请求类型

**创建幻灯片：**
```json
{"createSlide": {"slideLayoutReference": {"predefinedLayout": "LAYOUT"}, "insertionIndex": N}}
```
可用 Layout: `TITLE`, `TITLE_AND_BODY`, `TITLE_AND_TWO_COLUMNS`, `TITLE_ONLY`, `BLANK`, `SECTION_HEADER`, `BIG_NUMBER`

**插入文本：**
```json
{"insertText": {"objectId": "SHAPE_ID", "text": "文本内容", "insertionIndex": 0}}
```

**删除文本：**
```json
{"deleteText": {"objectId": "SHAPE_ID", "textRange": {"type": "ALL"}}}
```

**创建文本框：**
```json
{
  "createShape": {
    "objectId": "自定义ID",
    "shapeType": "TEXT_BOX",
    "elementProperties": {
      "pageObjectId": "PAGE_ID",
      "size": {"width": {"magnitude": 400, "unit": "PT"}, "height": {"magnitude": 50, "unit": "PT"}},
      "transform": {"scaleX": 1, "scaleY": 1, "translateX": 50, "translateY": 100, "unit": "PT"}
    }
  }
}
```

**设置文本样式：**
```json
{
  "updateTextStyle": {
    "objectId": "SHAPE_ID",
    "textRange": {"type": "ALL"},
    "style": {"fontSize": {"magnitude": 18, "unit": "PT"}, "bold": true},
    "fields": "fontSize,bold"
  }
}
```

## 工作流程

1. 先用 `presentations create` 创建空白演示文稿
2. 用 `presentations get` 获取结构（了解现有页面和元素的 objectId）
3. 用 `batchUpdate` 批量添加幻灯片、文本、样式
4. 每次 batchUpdate 后检查返回的 replies 确认成功

## 注意事项

- objectId 必须全局唯一（建议用 `slide_1`、`title_1` 等有意义的命名）
- 插入文本前先通过 `get` 确认 shape 的 objectId
- batchUpdate 的 requests 按顺序执行，可以在一次调用中完成多步操作
- 出错时整个 batchUpdate 回滚，不会部分生效
