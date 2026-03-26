# 后台任务监控 Skill

当你调用的外部工具（如 render_slides、render_docs）执行时间超过 120 秒时，
工具会返回一条 `[PENDING]` 消息，包含子进程的 PID 和相关信息。

**你必须立即使用 `heartbeat_register_monitor` 工具注册监控。**

---

## 流程

1. 你路由到外部工具（如 `render_slides`）
2. 工具超过 120s 未返回 → 你收到 `[PENDING]` 消息，包含：
   - `task_id`: 任务唯一标识
   - `pid`: 子进程 PID
   - `output_path`: 输出文件路径
   - `hard_timeout`: 最大允许运行时间（秒）
3. **你立即调用 `heartbeat_register_monitor`**，传入上述参数
4. 告诉用户任务已转入后台监控，完成后会自动通知
5. 监控系统每 60 秒检查一次 PID，完成后自动通知用户

## 调用示例

当你收到类似这样的消息：

```
[PENDING] 命令执行超过 120s，子进程仍在后台运行。
task_id: tool_abc123
pid: 12345
output_path: /path/to/output
hard_timeout: 600
```

你应该立即调用：

```
heartbeat_register_monitor(task_id="tool_abc123", pid=12345, output_path="/path/to/output", hard_timeout=600)
```

然后回复用户："幻灯片正在后台生成中，完成后我会自动通知你。"

## 其他可用工具

- `heartbeat_my_monitors()` — 查看你当前正在监控的所有后台任务
