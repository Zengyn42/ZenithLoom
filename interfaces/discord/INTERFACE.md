# Discord Interface — 框架契约

本文档由框架在 Discord 启动时自动注入，适用于所有通过 Discord 运行的 Agent。

## 发送文件到 Discord

在回复文本中任意位置包含以下标记，框架自动将文件发送到当前频道，**无需知道频道 ID**：

```
[SEND_FILE: /absolute/path/to/file]
```

规则：
- 路径必须是 WSL 绝对路径
- 同一条回复可包含多个 `[SEND_FILE: ...]` 标记
- 标记从显示文本中自动剥离，用户只看到文件附件
- 文件不存在时框架发送警告

示例：
```
视频已生成完成！[SEND_FILE: /tmp/output/video.mp4]
```

## 用户身份

消息格式为 `用户名 (DISCORD): 消息内容`。`用户名` 即当前发言用户。
