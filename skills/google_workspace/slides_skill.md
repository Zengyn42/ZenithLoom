# Google Slides Skill

通过 `gws` CLI 操作 Google Slides 演示文稿。

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
