"""
无垠智穹 — Google Chat 接口

gws 的两种角色（均通过此模块体现接口层那一侧）：
  接口层  — gws events +subscribe 监听 GChat 消息 → invoke_agent() → gws chat +send 回复
  工具节点 — gws gmail / drive 等作为 EXTERNAL_TOOL 节点在图内部使用（见 framework/nodes/）

用法：
  由 main.py 注入 AgentLoader 后调用 run_gchat(loader)。

agent.json 配置：
  "gchat_space":       "spaces/AAAA..."
  "gchat_gcp_project": "my-gcp-project"
  "gchat_event_types": "google.workspace.chat.message.v1.created"  # 可选，有默认值

注意：
  gws events +subscribe 的实际 NDJSON 字段路径需通过以下命令实测确认：
    gws events +subscribe --target //chat.googleapis.com/<space> --once
  预期结构：event["data"]["message"]["text"] 和 event["data"]["space"]["name"]
  Bot 自身消息通过 sender.type == "BOT" 过滤，防止回声循环。
"""

import asyncio
import json
import logging
import subprocess

from framework.base_interface import BaseInterface

logger = logging.getLogger("gchat_bot")


def _run_gws_send(space: str, text: str) -> None:
    """Synchronous helper: send a GChat message via gws (called in thread pool)."""
    subprocess.run(
        ["gws", "chat", "+send", "--space", space, "--text", text],
        capture_output=True,
    )


def _stream_gws_events(cmd: list[str], queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """
    Blocking reader: run gws events +subscribe and push each NDJSON line into queue.
    Runs in a thread pool so it doesn't block the event loop.
    Pushes None as sentinel when the process exits.
    """
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            line = line.strip()
            if line:
                loop.call_soon_threadsafe(queue.put_nowait, line)
    except Exception as e:
        logger.error("[GChat] 事件流异常: %s", e)
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel — signals EOF


class GChatInterface(BaseInterface):
    """
    GChat 接口：通过 gws events +subscribe 接收消息，经 LangGraph 处理，
    通过 gws chat +send 回复。继承 BaseInterface 的通用命令处理逻辑。
    """

    async def run(self) -> None:
        await self.setup()
        config = self._loader.load_config()

        if not config.gchat_space:
            raise RuntimeError(
                "GChatInterface: gchat_space 未配置。"
                "请在 agent.json 中设置 gchat_space。"
            )

        logger.info(
            "[GChat] 启动中... space=%s project=%s",
            config.gchat_space,
            config.gchat_gcp_project,
        )

        cmd = [
            "gws", "events", "+subscribe",
            "--target", f"//chat.googleapis.com/{config.gchat_space}",
            "--event-types", config.gchat_event_types,
            "--project", config.gchat_gcp_project,
            "--poll-interval", "5",
        ]

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        # Run blocking gws process in thread pool, push lines into queue
        asyncio.get_event_loop().run_in_executor(
            None, _stream_gws_events, cmd, queue, loop
        )

        logger.info("[GChat] 事件监听已启动，等待消息...")

        while True:
            raw_line = await queue.get()
            if raw_line is None:
                logger.warning("[GChat] 事件流已关闭")
                break

            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.debug("[GChat] 跳过非 JSON 行: %s", raw_line[:80])
                continue

            if self._is_bot_message(event):
                continue

            msg_text, space_id = self._extract_chat_event(event)
            if not msg_text or not space_id:
                continue

            logger.info("[GChat] 收到消息: %s", msg_text[:80])

            try:
                if msg_text.startswith("!"):
                    parts = msg_text.split(maxsplit=1)
                    cmd_str = parts[0].lower()
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    reply = await self.handle_command(cmd_str, arg)
                    if reply is None:
                        reply = f"未知命令：{cmd_str}"
                else:
                    reply = await self.invoke_agent(msg_text)
            except Exception as e:
                logger.error("[GChat] 处理消息出错: %s", e, exc_info=True)
                reply = f"出错了：{e}"

            for chunk in self.split_fence_aware(reply, max_chars=4000):
                await asyncio.to_thread(_run_gws_send, space_id, chunk)

    @staticmethod
    def _extract_chat_event(event: dict) -> tuple[str, str]:
        """
        从 CloudEvent NDJSON 提取 (message_text, space_name)。

        字段路径基于 Google Workspace Events API CloudEvent 规范预期结构：
          event["data"]["message"]["text"]
          event["data"]["space"]["name"]

        注意：实际字段路径需通过 `gws events +subscribe --once` 实测确认。
        若结构不符，此方法返回 ("", "") — 事件将被跳过，不影响主循环。
        """
        try:
            data = event.get("data", {})
            msg_text = data["message"]["text"].strip()
            space_name = data["space"]["name"].strip()
            return msg_text, space_name
        except (KeyError, TypeError, AttributeError):
            return "", ""

    @staticmethod
    def _is_bot_message(event: dict) -> bool:
        """检查消息发送者是否为 BOT，防止自我回声。"""
        try:
            sender = event["data"]["message"]["sender"]
            return sender.get("type", "").upper() == "BOT"
        except (KeyError, TypeError):
            return False


async def run_gchat(loader) -> None:
    """GChat 接口入口（异步）。由 main.py 的 asyncio.run() 调用。"""
    iface = GChatInterface(loader)
    await iface.run()
