"""
框架级 Gemini 顾问节点 — framework/gemini/node.py

Phase 2: 直接调用 Gemini Code Assist API（cloudcode-pa.googleapis.com）。
复用 Gemini CLI 的 OAuth 凭据（~/.gemini/oauth_creds.json），
无需 API key，支持 gemini-2.5-pro / gemini-2.5-flash 等全量模型。

Session 管理（自包含）：
  每次 consult() / chat() 接受 session_id（UUID）：
    - 空 UUID → 创建新 Gemini session（ConversationRecord），生成新 UUID
    - 非空 UUID → 从 ~/.gemini/tmp/ 加载已有 session，resume 对话
  调用结束后自动将更新后的 ConversationRecord 写回磁盘（Gemini CLI 兼容格式）。
  返回 (result, session_id)，由调用方（图 wrapper）写回 BaseAgentState.node_sessions。

安全守则（防止账号/钱包被打）：
  - 每次 API 调用前强制 jitter sleep 2~5s
  - 403 / 429 立刻抛出 GeminiQuotaError，不重试
"""

import asyncio
import json
import logging
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from framework.config import AgentConfig
from framework.debug import is_debug
import framework.gemini.gemini_session as gem_sess
from framework.gemini.gemini_session import ConversationRecord

logger = logging.getLogger(__name__)

# Gemini CLI 公开 OAuth 客户端凭据
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

_JITTER_MIN = 2.0
_JITTER_MAX = 5.0


class GeminiQuotaError(RuntimeError):
    """
    403 / 429 触发时抛出。
    LangGraph 节点捕获后应立即停止图，不得重试。
    """


# ---------------------------------------------------------------------------
# _CodeAssistClient — 底层 HTTP 客户端
# ---------------------------------------------------------------------------

class _CodeAssistClient:
    """
    轻量级 Code Assist API 客户端。
    - 自动处理 token 过期续期
    - 缓存 project_id
    - 维护对话 _history（API contents 格式）
    - 每次 generateContent 前强制 jitter sleep
    """

    def __init__(self, model: str):
        self._model = model
        self._access_token: str = ""
        self._token_expiry: float = 0.0
        self._project_id: str = ""
        self._history: list[dict] = []

    def _load_creds(self) -> dict:
        if not _CREDS_PATH.exists():
            raise FileNotFoundError(
                f"Gemini CLI 凭据不存在: {_CREDS_PATH}\n"
                "请先运行 `gemini` 并完成登录。"
            )
        return json.loads(_CREDS_PATH.read_text())

    def _ensure_token(self) -> str:
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        creds = self._load_creds()
        expiry_ms = creds.get("expiry_date", 0)
        if creds.get("access_token") and time.time() * 1000 < expiry_ms - 60000:
            self._access_token = creds["access_token"]
            self._token_expiry = expiry_ms / 1000
            return self._access_token

        logger.info("[gemini.client] 刷新 access_token...")
        refresh_token = creds.get("refresh_token", "")
        if not refresh_token:
            raise ValueError("~/.gemini/oauth_creds.json 中缺少 refresh_token")

        data = urllib.parse.urlencode({
            "client_id": _OAUTH_CLIENT_ID,
            "client_secret": _OAUTH_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }).encode()
        req = urllib.request.Request(_TOKEN_URI, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                token_data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OAuth token 刷新失败 ({e.code}): {e.read().decode()[:200]}")

        self._access_token = token_data["access_token"]
        self._token_expiry = time.time() + token_data.get("expires_in", 3600)
        creds["access_token"] = self._access_token
        creds["expiry_date"] = int(self._token_expiry * 1000)
        _CREDS_PATH.write_text(json.dumps(creds, indent=2))
        logger.info("[gemini.client] access_token 已刷新并缓存")
        return self._access_token

    def _get_project_id(self, token: str) -> str:
        if self._project_id:
            return self._project_id

        url = f"{_CODE_ASSIST_ENDPOINT}/{_CODE_ASSIST_VERSION}:loadCodeAssist"
        data = json.dumps({
            "cloudaicompanionProject": None,
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"loadCodeAssist 失败 ({e.code}): {e.read().decode()[:300]}")

        self._project_id = result.get("cloudaicompanionProject", "")
        logger.info(f"[gemini.client] project_id={self._project_id}")
        return self._project_id

    def _chat_sync(self, user_text: str, system: str = "") -> str:
        delay = random.uniform(_JITTER_MIN, _JITTER_MAX)
        logger.info(f"[gemini.client] model={self._model} jitter={delay:.1f}s")
        if is_debug():
            logger.debug(f"[gemini.client] prompt_preview={user_text[:100]!r}")
        time.sleep(delay)

        token = self._ensure_token()
        project_id = self._get_project_id(token)

        self._history.append({"role": "user", "parts": [{"text": user_text}]})

        request_body: dict = {"contents": list(self._history)}
        if system:
            request_body["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{_CODE_ASSIST_ENDPOINT}/{_CODE_ASSIST_VERSION}:generateContent"
        data = json.dumps({
            "model": self._model,
            "project": project_id,
            "request": request_body,
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            if e.code in (403, 429):
                raise GeminiQuotaError(
                    f"Gemini API {e.code} — 图已停止，请检查账号状态或等待配额重置。\n{body}"
                )
            raise RuntimeError(f"generateContent 失败 ({e.code}): {body}")

        candidates = result.get("response", {}).get("candidates", [])
        if not candidates:
            raise RuntimeError(f"generateContent 返回无 candidates: {result}")

        reply = candidates[0]["content"]["parts"][0]["text"]
        self._history.append({"role": "model", "parts": [{"text": reply}]})
        if is_debug():
            logger.debug(f"[gemini.client] reply_preview={reply[:100]!r}")
        return reply.strip()

    async def chat(self, user_text: str, system: str = "") -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._chat_sync, user_text, system)

    def reset(self):
        self._history.clear()


# ---------------------------------------------------------------------------
# GeminiNode — 自包含 session-aware 节点
# ---------------------------------------------------------------------------

class GeminiNode:
    """
    Gemini 顾问节点（自包含 session 管理）。

    统一接口：
      consult(topic, context, session_id="") → (result, session_id)
      chat(message, session_id="", system="") → (reply, session_id)

    session_id 为空 → 创建新 Gemini session；
    非空 → 从 ~/.gemini/tmp/{project_id}/chats/ 恢复已有 session。
    每次调用结束后自动持久化 ConversationRecord（Gemini CLI 兼容格式）。

    调用方（图 wrapper）只需：
      1. 从 state.node_sessions["gemini_main"] 取 session_id 传入
      2. 把返回的 session_id 写回 state.node_sessions["gemini_main"]
    """

    def __init__(self, config: AgentConfig, llm_node):
        self.config = config
        self.llm_node = llm_node  # ClaudeNode / LlamaNode，用于 consult 的挑刺轮

        self._default_model = getattr(config, "gemini_model", None) or "gemini-2.5-flash"

        # project_id: 从 workspace 生成 slug，定位 ~/.gemini/tmp/{project_id}/
        self._project_id = gem_sess.get_project_id(config.workspace or os.getcwd())
        logger.info(f"[gemini] project_id={self._project_id}")

        # 按 session_id UUID 索引（避免每次重建 token 缓存）
        self._clients: dict[str, _CodeAssistClient] = {}
        self._records: dict[str, ConversationRecord] = {}

    def _get_client(self, session_id: str) -> _CodeAssistClient:
        if session_id not in self._clients:
            self._clients[session_id] = _CodeAssistClient(self._default_model)
        return self._clients[session_id]

    def _load_or_create(self, session_id: str) -> tuple[ConversationRecord, _CodeAssistClient]:
        """
        按 session_id 加载或创建 ConversationRecord + client。
        将磁盘上的历史同步到 client._history（resume 语义）。
        """
        # 内存缓存命中
        if session_id and session_id in self._records:
            if is_debug():
                logger.debug(f"[gemini] cache hit for session {session_id[:8]}")
            return self._records[session_id], self._get_client(session_id)

        # 磁盘加载
        if session_id:
            record = gem_sess.load_session(session_id, self._project_id)
            if record is not None:
                client = self._get_client(session_id)
                client._history = gem_sess.to_api_history(record)
                self._records[session_id] = record
                logger.info(
                    f"[gemini] resumed {session_id[:8]} "
                    f"({len(client._history) // 2} turns)"
                )
                return record, client
            logger.warning(
                f"[gemini] session {session_id[:8]} not found on disk, creating new"
            )

        # 新建
        record = gem_sess.new_session(self._project_id, self._default_model)
        client = self._get_client(record.sessionId)
        self._records[record.sessionId] = record
        logger.info(f"[gemini] new session {record.sessionId[:8]}")
        return record, client

    def _persist(self, record: ConversationRecord, client: _CodeAssistClient) -> None:
        """将 client._history 追加到 record 并保存到磁盘。"""
        gem_sess.append_history(record, client._history, model=self._default_model)
        gem_sess.save_session(record, self._project_id)

    async def consult(
        self,
        topic: str,
        context: str,
        session_id: str = "",
    ) -> tuple[str, str]:
        """
        3 轮对抗咨询。返回 (最终建议, session_id)。
        GeminiQuotaError（403/429）直接穿透。
        """
        record, client = self._load_or_create(session_id)
        sid = record.sessionId

        try:
            logger.info("[gemini] Round 1/3: Gemini 首次回答")
            g1 = await client.chat(
                f"问题：{topic}\n当前上下文：{context}",
                system=_GEMINI_SYSTEM,
            )
            if is_debug():
                logger.debug(f"[gemini] g1_preview={g1[:150]!r}")

            logger.info("[gemini] Round 2/3: Claude 挑刺")
            critique_prompt = (
                f"以下是 Gemini 架构师对「{topic}」的建议，请找出其中的逻辑漏洞、"
                f"遗漏的边界情况或过于理想化的假设（简明扼要，3点以内）：\n\n{g1}"
            )
            critique, _ = await self.llm_node.call_llm(critique_prompt)

            logger.info("[gemini] Round 2/3: Gemini 修订")
            g2 = await client.chat(
                f"Hani 的质疑（请针对以下质疑修订你的建议）：\n{critique}"
            )

            logger.info("[gemini] Round 3/3: Claude 深度挑刺")
            nitpick_prompt = (
                f"对以下修订后的架构建议进行最后一轮深度审查。\n"
                f"重点关注：实施复杂度、潜在的单点故障、与现有环境的兼容性。\n\n{g2}"
            )
            nitpick, _ = await self.llm_node.call_llm(nitpick_prompt)

            logger.info("[gemini] Round 3/3: Gemini 最终建议")
            g_final = await client.chat(
                f"Hani 的最终审查意见：\n{nitpick}\n\n"
                "请给出你的最终建议（这将直接被采纳执行）："
            )

            logger.info("[gemini] 3轮咨询完成")
            self._persist(record, client)
            return g_final, sid

        except GeminiQuotaError:
            raise
        except Exception as e:
            logger.error(f"[gemini] consult 失败: {e}")
            self._persist(record, client)
            return f"[Gemini 咨询失败: {e}]", sid

    async def chat(
        self,
        message: str,
        session_id: str = "",
        system: str = "",
    ) -> tuple[str, str]:
        """
        单轮直接对话，返回 (reply, session_id)。
        适合不需要 Claude 挑刺的直接 Gemini 对话场景。
        """
        record, client = self._load_or_create(session_id)
        sid = record.sessionId

        try:
            reply = await client.chat(message, system=system or _GEMINI_SYSTEM)
            self._persist(record, client)
            return reply, sid
        except GeminiQuotaError:
            raise
        except Exception as e:
            logger.error(f"[gemini] chat 失败: {e}")
            self._persist(record, client)
            return f"[Gemini chat 失败: {e}]", sid
