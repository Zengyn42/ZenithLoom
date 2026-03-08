"""
无垠智穹 0号管家 - LLM Provider 抽象层

设计原则：
  - 唯一对外接口 complete() / acomplete()
  - 当前实现：ClaudeCLIProvider + GeminiCLIProvider（subprocess）
  - 未来占位：ClaudeAPIProvider + GeminiAPIProvider（API key）
  - 环境变量 CLAUDE_PROVIDER=cli|api 切换，零代码改动

P2: acomplete 预留异步接口，底层默认用 run_in_executor 包同步调用，
    未来 API provider 可直接 override 为原生 async。
"""

import asyncio
import os
from abc import ABC, abstractmethod


# ==========================================
# 抽象基类
# ==========================================
class LLMProvider(ABC):

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """同步调用，返回纯文本回复。"""
        ...

    async def acomplete(self, prompt: str, **kwargs) -> str:
        """
        P2: 异步接口预留。
        默认实现：把同步 complete() 推到线程池，不阻塞 event loop。
        未来 API provider 直接 override 此方法用原生 async client。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.complete(prompt, **kwargs)
        )


# ==========================================
# Claude CLI Provider
# ==========================================
class ClaudeCLIProvider(LLMProvider):
    """
    通过 subprocess 调用 `claude -p` 的 provider。
    cwd 参数实现多项目 session 隔离（不同目录 = 不同 Claude session）。
    """

    def __init__(
        self,
        cwd: str | None = None,
        skip_permissions: bool = False,
        tools: list[str] | None = None,
        timeout: int = 120,
    ):
        self.cwd = cwd
        self.skip_permissions = skip_permissions
        self.tools = tools or []
        self.timeout = timeout

    def complete(self, prompt: str, **kwargs) -> str:
        from agent.cli_wrapper import call_claude
        return call_claude(
            prompt,
            cwd=kwargs.get("cwd", self.cwd),
            tools=kwargs.get("tools", self.tools),
            skip_permissions=kwargs.get("skip_permissions", self.skip_permissions),
            timeout=self.timeout,
            resume_session_id=kwargs.get("resume_session_id", ""),
        )


# ==========================================
# Gemini CLI Provider
# ==========================================
class GeminiCLIProvider(LLMProvider):
    """通过 subprocess 调用 `gemini -p` 的 provider。"""

    def __init__(
        self,
        extensions: list[str] | None = None,
        yolo: bool = False,
        timeout: int = 60,
    ):
        self.extensions = extensions or []
        self.yolo = yolo
        self.timeout = timeout

    def complete(self, prompt: str, **kwargs) -> str:
        from agent.cli_wrapper import call_gemini
        return call_gemini(
            prompt,
            extensions=kwargs.get("extensions", self.extensions),
            yolo=kwargs.get("yolo", self.yolo),
            timeout=self.timeout,
        )


# ==========================================
# API Provider 占位（未来）
# ==========================================
class ClaudeAPIProvider(LLMProvider):
    """未来：使用 ANTHROPIC_API_KEY 的原生 SDK provider。"""

    def complete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "ClaudeAPIProvider 尚未实现。"
            "请设置 CLAUDE_PROVIDER=cli，或提供 ANTHROPIC_API_KEY 后实现此类。"
        )

    async def acomplete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("ClaudeAPIProvider async 尚未实现。")


class GeminiAPIProvider(LLMProvider):
    """未来：使用 GOOGLE_API_KEY 的原生 SDK provider。"""

    def complete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "GeminiAPIProvider 尚未实现。"
            "请设置 GEMINI_PROVIDER=cli，或提供 GOOGLE_API_KEY 后实现此类。"
        )

    async def acomplete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("GeminiAPIProvider async 尚未实现。")


# ==========================================
# 工厂函数（读 .env 决定用哪个）
# ==========================================
def get_claude_provider() -> LLMProvider:
    """
    根据 CLAUDE_PROVIDER 环境变量返回对应 provider。
    默认 cli。CLAUDE_SKIP_PERMISSIONS=true 开启无权限确认模式。
    """
    mode = os.getenv("CLAUDE_PROVIDER", "cli").lower()
    if mode == "api":
        return ClaudeAPIProvider()

    tools_raw = os.getenv("CLAUDE_TOOLS", "Bash,Read,Write,Edit,Glob,Grep")
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
    skip = os.getenv("CLAUDE_SKIP_PERMISSIONS", "true").lower() == "true"

    return ClaudeCLIProvider(skip_permissions=skip, tools=tools)


def get_gemini_provider() -> LLMProvider:
    """
    根据 GEMINI_PROVIDER 环境变量返回对应 provider。
    默认 cli。GEMINI_YOLO=true 开启自动 approve 模式。
    """
    mode = os.getenv("GEMINI_PROVIDER", "cli").lower()
    if mode == "api":
        return GeminiAPIProvider()

    yolo = os.getenv("GEMINI_YOLO", "false").lower() == "true"
    return GeminiCLIProvider(yolo=yolo)
