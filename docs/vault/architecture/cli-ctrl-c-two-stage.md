# Plan: CLI Ctrl+C 两段式中断 + 软停止关键词拦截

## Context

CLI 目前是单线程阻塞模型：`invoke_agent` 运行时 stdin 被占用，用户无法输入任何内容中断。
Ctrl+C 无论何时按下都直接退出程序，体验差。

目标：
- Ctrl+C 在 agent 运行中 → 取消当前任务，回到提示符
- Ctrl+C 在空闲时 → 正常退出（现有行为不变）
- 软停止关键词（stop/wait/停/停止）在 CLI 输入时 → 拦截，不发给 agent，打印提示

Discord 侧 `_channel_tasks` 模式已验证可行，CLI 侧参考相同思路。

---

## 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `interfaces/cli.py` | 主要改动：task 跟踪、两段式 Ctrl+C、软停止关键词 |
| `framework/base_interface.py` | 无需改动（`invoke_agent` 已是 async，可被 cancel） |

---

## 实现方案

### 1. `_CliInterface.__init__` 新增字段

```python
self._current_agent_task: asyncio.Task | None = None
```

---

### 2. 主循环改造（`run()` 方法，第 112-155 行）

#### 2a. 软停止关键词拦截（`!` 命令判断之后、`invoke_agent` 之前）

```python
_SOFT_STOP_WORDS = {"stop", "wait", "停", "停止"}
if user_input.lower() in _SOFT_STOP_WORDS:
    print("没有正在运行的任务。")
    continue
```

#### 2b. 用 `asyncio.create_task()` 包装 `invoke_agent`

替换原来的：
```python
response = await self.invoke_agent(user_input)
```

改为：
```python
agent_task = asyncio.create_task(self.invoke_agent(user_input))
self._current_agent_task = agent_task
try:
    response = await agent_task
    if not self._streaming or self._last_stream_chunk_count == 0:
        print(response, end="", flush=True)
except (asyncio.CancelledError, KeyboardInterrupt):
    if not agent_task.done():
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
    print("\n已停止。\n")
    continue          # 回到输入提示符，不退出
except Exception as e:
    print(f"\n[错误] {e}", file=sys.stderr)
finally:
    self._current_agent_task = None
```

#### 2c. 空闲时 Ctrl+C（input() 阶段）保持退出行为

```python
except (KeyboardInterrupt, EOFError):
    print(f"\n\n{agent_name} 待命中，再见。")
    await loader.stop_heartbeat()
    break
```
此处 `_current_agent_task` 必为 `None`（task 运行中不会执行到这里），无需额外判断。

---

## 完整改动后的 `run()` 结构（伪代码）

```
while True:
  try:
    user_input = await run_in_executor(input)   # Ctrl+C here → exit
  except KeyboardInterrupt/EOFError → exit

  if not user_input: continue
  if quit keyword → exit

  if user_input starts with "!":
    handle_command(); continue

  if user_input in SOFT_STOP_WORDS:
    print("没有正在运行的任务。"); continue

  # 正常对话
  task = create_task(invoke_agent(user_input))
  _current_agent_task = task
  try:
    response = await task                        # Ctrl+C here → cancel, continue
  except CancelledError/KeyboardInterrupt → print "已停止", continue
  finally: _current_agent_task = None

  log_snapshot()
```

---

## 验证方法

1. **正常对话**：发一条消息，等待回复完整输出，确认无回归
2. **运行中 Ctrl+C**：发一条消息（触发长回复），立即按 Ctrl+C
   - 预期：打印「已停止。」，回到 `> ` 提示符，程序不退出
3. **空闲时 Ctrl+C**：在 `> ` 提示符直接按 Ctrl+C
   - 预期：打印「待命中，再见。」，程序退出
4. **软停止关键词**：分别输入 `stop`、`STOP`、`Wait`、`停`、`停止`
   - 预期：打印「没有正在运行的任务。」，回到提示符，不发给 agent
5. **快照记录**：正常完成对话后确认 `log_snapshot()` 仍被调用
6. **运行测试**：`python -m pytest test_commands.py -v`，确认无测试回归
