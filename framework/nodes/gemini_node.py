"""
框架级 Gemini 顾问节点 — GeminiNode

Phase 2: 直接调用 Gemini Code Assist API（cloudcode-pa.googleapis.com）。
复用 Gemini CLI 的 OAuth 凭据（~/.gemini/oauth_creds.json），
无需 API key，支持 gemini-2.5-pro / gemini-2.5-flash 等全量模型。

认证流程：
  1. 读取 ~/.gemini/oauth_creds.json（Gemini CLI 登录后生成）
  2. Token 过期时用 client_id/secret + refresh_token 自动续期
  3. 首次调用时通过 loadCodeAssist 获取 project_id（缓存）
  4. 直接 POST 到 cloudcode-pa.googleapis.com/v1internal:generateContent
"""

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from framework.config import AgentConfig
from framework.nodes.claude_node import ClaudeNode

logger = logging.getLogger(__name__)

# Gemini CLI 公开 OAuth 客户端（已在 oauth2.js 中声明为安全的 installed-app 凭据）
_OAUTH_CLIENT_ID = (
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
)
_OAUTH_CLIENT_SECRET = "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"
_TOKEN_URI = "https://oauth2.googleapis.com/token"
_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"

_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"
_CODE_ASSIST_VERSION = "v1internal"

_GEMINI_SYSTEM = (
    "你是无垠智穹的首席架构师（Gemini）。\n"
    "请根据问题给出极致架构建议。直接输出建议，不需要客套。"
)


class _CodeAssistClient:
    """
    轻量级 Code Assist API 客户端。
    - 自动处理 token 过期续期
    - 缓存 project_id（每次启动只请求一次）
    - 维护对话历史（multi-turn）
    """

    def __init__(self, model: str):
        self._model = model
        self._access_token: str = ""
        self._token_expiry: float = 0.0  # unix timestamp
        self._project_id: str = ""
        self._history: list[dict] = []  # [{role, parts: [{text}]}]

    def _load_creds(self) -> dict:
        if not _CREDS_PATH.exists():
            raise FileNotFoundError(
                f"Gemini CLI 凭据不存在: {_CREDS_PATH}\n"
                "请先运行 `gemini` 并完成登录。"
            )
        return json.loads(_CREDS_PATH.read_text())

    def _ensure_token(self) -> str:
        """返回有效的 access_token，必要时自动刷新。"""
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        creds = self._load_creds()

        # 尝试用现有 access_token（expiry_date 是毫秒）
        expiry_ms = creds.get("expiry_date", 0)
        if creds.get("access_token") and time.time() * 1000 < expiry_ms - 60000:
            self._access_token = creds["access_token"]
            self._token_expiry = expiry_ms / 1000
            logger.debug("[gemini_client] 使用现有 access_token")
            return self._access_token

        # 用 refresh_token 换取新 token
        logger.info("[gemini_client] 刷新 access_token...")
        refresh_token = creds.get("refresh_token", "")
        if not refresh_token:
            raise ValueError("~/.gemini/oauth_creds.json 中缺少 refresh_token")

        data = urllib.parse.urlencode(
            {
                "client_id": _OAUTH_CLIENT_ID,
                "client_secret": _OAUTH_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode()
        req = urllib.request.Request(_TOKEN_URI, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                token_data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"OAuth token 刷新失败 ({e.code}): {e.read().decode()[:200]}"
            )

        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = time.time() + expires_in

        # 更新本地缓存文件
        creds["access_token"] = self._access_token
        creds["expiry_date"] = int(self._token_expiry * 1000)
        _CREDS_PATH.write_text(json.dumps(creds, indent=2))
        logger.info("[gemini_client] access_token 已刷新并缓存")
        return self._access_token

    def _get_project_id(self, token: str) -> str:
        """通过 loadCodeAssist 获取 cloudaicompanionProject（缓存）。"""
        if self._project_id:
            return self._project_id

        url = f"{_CODE_ASSIST_ENDPOINT}/{_CODE_ASSIST_VERSION}:loadCodeAssist"
        data = json.dumps(
            {
                "cloudaicompanionProject": None,
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                },
            }
        ).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"loadCodeAssist 失败 ({e.code}): {e.read().decode()[:300]}"
            )

        self._project_id = result.get("cloudaicompanionProject", "")
        logger.info(f"[gemini_client] project_id={self._project_id}")
        return self._project_id

    def _chat_sync(self, user_text: str, system: str = "") -> str:
        """
        同步版本（供 run_in_executor 调用）。
        发送一轮对话，自动维护历史，返回模型回复文本。
        """
        token = self._ensure_token()
        project_id = self._get_project_id(token)

        self._history.append(
            {"role": "user", "parts": [{"text": user_text}]}
        )

        request_body: dict = {"contents": list(self._history)}
        if system:
            request_body["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{_CODE_ASSIST_ENDPOINT}/{_CODE_ASSIST_VERSION}:generateContent"
        data = json.dumps(
            {
                "model": self._model,
                "project": project_id,
                "request": request_body,
            }
        ).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"generateContent 失败 ({e.code}): {e.read().decode()[:300]}"
            )

        candidates = result.get("response", {}).get("candidates", [])
        if not candidates:
            raise RuntimeError(f"generateContent 返回无 candidates: {result}")

        reply = candidates[0]["content"]["parts"][0]["text"]
        self._history.append(
            {"role": "model", "parts": [{"text": reply}]}
        )
        return reply.strip()

    async def chat(self, user_text: str, system: str = "") -> str:
        """异步包装，避免阻塞 event loop。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._chat_sync, user_text, system)

    def reset(self):
        """清空对话历史（新话题时调用）。"""
        self._history.clear()


class GeminiNode:
    """
    Gemini 顾问节点（3轮对抗咨询）。

    Phase 2: 直接调用 Code Assist API，Gemini 自管对话历史。
    每个 LangGraph thread_id 对应一个独立的 _CodeAssistClient（持久多轮）。
    """

    def __init__(self, config: AgentConfig, claude_node: ClaudeNode):
        self.config = config
        self.claude_node = claude_node
        gemini_model = os.getenv("HANI_GEMINI_MODEL", "gemini-2.5-flash")
        self._clients: dict[str, _CodeAssistClient] = {}  # thread_id → client
        self._default_model = gemini_model

    def _get_client(self, thread_id: str) -> _CodeAssistClient:
        if thread_id not in self._clients:
            self._clients[thread_id] = _CodeAssistClient(self._default_model)
        return self._clients[thread_id]

    async def consult(
        self,
        topic: str,
        context: str,
        session_id: str = "",
    ) -> str:
        """
        3轮对抗咨询（Gemini 自管 multi-turn 历史）：
          Round 1: Gemini 首次回答
          Round 2: Claude 挑刺 → Gemini 修订
          Round 3: Claude 深度挑刺 → Gemini 最终建议
        """
        client = self._get_client(session_id or "default")
        client.reset()  # 每次咨询都是独立话题

        try:
            # Round 1
            logger.info("[gemini] Round 1/3: Gemini 首次回答")
            g1 = await client.chat(
                f"问题：{topic}\n当前上下文：{context}",
                system=_GEMINI_SYSTEM,
            )

            # Round 2: Claude 挑刺
            logger.info("[gemini] Round 2/3: Claude 挑刺")
            critique_prompt = (
                f"以下是 Gemini 架构师对「{topic}」的建议，请找出其中的逻辑漏洞、"
                f"遗漏的边界情况或过于理想化的假设（简明扼要，3点以内）：\n\n{g1}"
            )
            critique, _ = await self.claude_node.call_claude(critique_prompt)

            logger.info("[gemini] Round 2/3: Gemini 修订")
            g2 = await client.chat(
                f"Hani 的质疑（请针对以下质疑修订你的建议）：\n{critique}"
            )

            # Round 3: Claude 深度挑刺
            logger.info("[gemini] Round 3/3: Claude 深度挑刺")
            nitpick_prompt = (
                f"对以下修订后的架构建议进行最后一轮深度审查。\n"
                f"重点关注：实施复杂度、潜在的单点故障、与现有环境的兼容性。\n\n{g2}"
            )
            nitpick, _ = await self.claude_node.call_claude(nitpick_prompt)

            logger.info("[gemini] Round 3/3: Gemini 最终建议")
            g_final = await client.chat(
                f"Hani 的最终审查意见：\n{nitpick}\n\n"
                "请给出你的最终建议（这将直接被采纳执行）："
            )

            logger.info("[gemini] 3轮咨询完成")
            return g_final

        except Exception as e:
            logger.error(f"[gemini] Code Assist API 失败: {e}")
            return f"[Gemini 咨询失败: {e}]"
