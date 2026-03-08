# Hani 命令手册

## Discord 命令

| 命令 | 说明 |
|------|------|
| `!session` | 显示当前 session ID |
| `!sessions` | 列出所有保存的 sessions |
| `!memory` | 查看 session 消息数和 DB 大小 |
| `!compact [N]` | 压缩 session，保留最近 N 条（默认 20） |
| `!reset confirm` | 清空当前 session 全部记忆（需确认） |
| `!clear` | 清空 session（无需确认） |
| `!tokens` | 查看 token 消耗统计 |
| `!tokens reset` | 重置 token 计数 |
| `!setproject <路径>` | 设置工作目录（Git 时间机器在此生效） |
| `!project` | 查看当前项目目录 |
| `!debug` | 查看 debug 模式状态 |
| `!whoami` | 显示你的 Discord ID |
| `!help_hani` | 显示帮助信息 |

## 特殊触发

- 消息中包含 `@Gemini` 可强制触发 Gemini 首席架构师咨询（3轮对抗）

## Debug 模式

启动时设置环境变量 `DEBUG=1` 开启：
- 终端输出完整 prompt/response
- 打印 consult_gemini 信号路由
- 显示 session ID 和 prompt 长度

## 配置

所有配置通过 `.env` 文件设置，`HANI_` 前缀：

```
HANI_WORKSPACE=/path/to/project      # 默认工作目录
HANI_TOOLS=Read,Write,Edit,Bash,...   # 工具列表
HANI_PERMISSION_MODE=bypassPermissions
HANI_MAX_RETRIES=2                    # git rollback 重试上限
HANI_DB_PATH=./hani.db                # SQLite 路径
HANI_SESSIONS_FILE=./sessions.json    # Named sessions 映射
HANI_THREAD_ID=hani_session_01        # LangGraph thread ID
DISCORD_BOT_TOKEN=...                 # Discord bot token
DISCORD_ALLOWED_USERS=123,456         # 白名单（逗号分隔 ID）
```
