"""
Chrome 异步执行器 — framework/utils/chrome_executor.py

用于统一管理对底层 Python Playwright (Headless Chrome) 桥接脚本的调用、
标准输出(stdout)的实时截获、以及标准错误(stderr)中 Session URL 等令牌的解析。
任何需要操作 Chrome 的 Node（无论是 LLM 还是普通工具节点）都可以组合使用此执行器。
"""

import asyncio
import logging
import os
import sys
import uuid

from framework.debug import is_debug

logger = logging.getLogger(__name__)


class ChromeExecutor:
    """
    负责启动底层桥接脚本并处理异步流式通信。
    """

    def __init__(self, script_path: str, timeout: int = 180, model_name: str = "chrome", user_data_dir: str | None = None):
        """
        初始化执行器。
        :param script_path: 底层桥接脚本的绝对路径 (例如 ~/ChromeHeadless/grok_playwright_bridge.py)
        :param timeout: 超时时间 (秒)
        :param model_name: 用于日志的前缀名称
        :param user_data_dir: Chrome 持久化配置目录 (传入绝对路径)
        """
        self._bridge_script = os.path.expanduser(script_path)
        self._timeout = timeout
        self._model = model_name
        self._user_data_dir = user_data_dir
        self._bridge_python = sys.executable

    async def execute(
        self,
        prompt: str,
        session_id: str = "",
        cwd: str | None = None,
        stream_cb=None
    ) -> tuple[str, str]:
        """
        执行底层脚本并截获输入输出。
        :param prompt: 发送给底层脚本的输入（对于 LLM 节点，这通常是拼装好的对话记录）
        :param session_id: 当前会话的 URL（如果存在的话）
        :param cwd: 工作目录
        :param stream_cb: 用于接收流式字符回调的函数，接收 (text_chunk, is_thinking)
        :return: (最终完整的文本结果, 新的_session_url或uuid)
        """
        cmd = [
            self._bridge_python,
            self._bridge_script,
            prompt
        ]
        
        if session_id and session_id.startswith("http"):
            cmd.extend(["--url", session_id])
            
        if self._user_data_dir:
            cmd.extend(["--user-data-dir", self._user_data_dir])
            
        cmd.extend(["--timeout", str(self._timeout)])
        
        print(f"DEBUG: Starting bridge with command: {' '.join(cmd)}", flush=True)
        logger.info(f"[{self._model}] full_cmd: {' '.join(cmd)}")
        logger.info(f"[{self._model}] executing bridge script (url={session_id[:30] if session_id else 'new'})")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or None
        )
        print(f"DEBUG: Bridge process started (PID: {proc.pid})", flush=True)

        result_text = ""
        stderr_lines = []
        new_session_id = session_id
        
        async def read_stdout():
            nonlocal result_text
            while True:
                line = await proc.stdout.readline()
                if not line: break
                text = line.decode(errors="replace").strip()
                if not text: continue
                print(f"DEBUG: Bridge output: {text}", flush=True)
                if stream_cb:
                    stream_cb(text, False)
                result_text += text + "\n"

        async def read_stderr():
            nonlocal new_session_id
            while True:
                line = await proc.stderr.readline()
                if not line: break
                text = line.decode(errors="replace").strip()
                if text:
                    stderr_lines.append(text)
                    logger.debug(f"[{self._model}/stderr] {text}")
                    if text.startswith("DEBUG:"):
                        print(text, flush=True)
                    # Parse JSON tokens from stderr
                    if text.startswith("{") and "__session_url__" in text:
                        try:
                            import json
                            data = json.loads(text)
                            if "__session_url__" in data:
                                new_session_id = data["__session_url__"]
                        except:
                            pass

        try:
            await asyncio.gather(
                read_stdout(),
                read_stderr(),
                asyncio.wait_for(proc.wait(), timeout=self._timeout + 30)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"{self._model} Bridge 超时 ({self._timeout}s, prompt_len={len(prompt)})")

        if proc.returncode != 0:
            stderr_text = "\n".join(stderr_lines[-20:]) if stderr_lines else ""
            raise RuntimeError(f"{self._model} Bridge 退出码 {proc.returncode}: {stderr_text[:500]}")

        clean_result = result_text.strip()
        
        if is_debug():
            logger.debug(f"[{self._model}] output_preview={clean_result[:200]!r}")

        # 如果没有 session_id 并且也未能从 stderr 解析到 URL，则给个随机标识
        if not new_session_id:
            new_session_id = str(uuid.uuid4())
        
        return clean_result, new_session_id
