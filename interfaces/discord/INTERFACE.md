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

## Discord 频道工具

需要查询频道信息时，路由到 `discord_tools` 节点，通过 `context` 指定操作：

| 操作 | context 格式 | 说明 |
|------|-------------|------|
| 读取历史 | `history` 或 `history limit=N` | 默认最近 **20** 条，最多 200 |
| 搜索消息 | `search q=关键词` 或 `search q=关键词 limit=N` | 默认搜最近 100 条 |
| 列出频道 | `channels` | 列出服务器所有文字频道 |
| 查询用户 | `user id=<USER_ID>` | 查询用户名、身份组、加入时间 |

示例：
```json
{"route": "discord_tools", "context": "history limit=20"}
{"route": "discord_tools", "context": "search q=视频生成"}
{"route": "discord_tools", "context": "channels"}
{"route": "discord_tools", "context": "user id=286003878997262337"}
```

## 用户身份

消息格式为 `用户名 (DISCORD): 消息内容`。`用户名` 即当前发言用户。
