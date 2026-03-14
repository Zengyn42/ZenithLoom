"""
框架级 Gemini 节点 — framework/gemini/node.py

GeminiNode(AgentNode)  →  GEMINI_CLI 节点类型
  Gemini 作为咨询 agent，重写 __call__()：
    - 读取 state["routing_context"] 作为提问
    - 调用 call_llm() 获取回复
    - 清除 routing_target / routing_context
    - 增加 consult_count

安全守则（防止账号/钱包被打）：
  - 每次 API 调用前强制 jitter sleep，时长随 prompt 长度线性缩放：
      1 char → 1s，2000+ chars → 20s，再叠加 ±2s 随机抖动
    （Claude 不需要此机制：SDK 子进程内部自行处理 rate limiting）
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

from langchain_core.messages import AIMessage

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.agent_node import AgentNode
from framework.resource_lock import acquire_resource
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

# Gemini jitter 机制（Claude 不需要，SDK 子进程内部自行处理 rate limiting）：
# prompt 长度线性映射到等待时间，再叠加随机抖动，防止 429。
# 1 char → 1s base，2000+ chars → 20s base，±2s 随机噪声，最低 0.5s。
_JITTER_MIN_S = 1.0
_JITTER_MAX_S = 20.0
_JITTER_SCALE_LEN = 2000
_JITTER_NOISE = 2.0


def _jitter_secs(text_len: int) -> float:
    t = min(max(text_len, 1) / _JITTER_SCALE_LEN, 1.0)
    base = _JITTER_MIN_S + t * (_JITTER_MAX_S - _JITTER_MIN_S)
    return max(0.5, base + random.uniform(-_JITTER_NOISE, _JITTER_NOISE))


class GeminiQuotaError(RuntimeError):
    """
    403 / 429 触发时抛出。
    LangGraph 节点捕获后应立即停止图，不得重试。
    """


# ---------------------------------------------------------------------------
# _CodeAssistClient — 底层 HTTP 客户端（内部用）
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
            raise RuntimeError(
                f"OAuth token 刷新失败 ({e.code}): {e.read().decode()[:200]}"
            )

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
            raise RuntimeError(
                f"loadCodeAssist 失败 ({e.code}): {e.read().decode()[:300]}"
            )

        self._project_id = result.get("cloudaicompanionProject", "")
        logger.info(f"[gemini.client] project_id={self._project_id}")
        return self._project_id

    def _chat_sync(self, user_text: str, system: str = "") -> str:
        delay = _jitter_secs(len(user_text))
        logger.info(f"[gemini.client] model={self._model} jitter={delay:.1f}s (prompt_len={len(user_text)})")
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

        result = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:300]
                if e.code == 429 and attempt == 0:
                    logger.warning(f"[gemini.client] 429，60s 后重试一次...")
                    time.sleep(60)
                    continue
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
# _GeminiSessionMixin — 共享 session 管理逻辑
# ---------------------------------------------------------------------------

class _GeminiSessionMixin:
    """Gemini session 管理 mixin：按 session_id UUID 索引 client 和 ConversationRecord。"""

    # 内存中最多保留的 session 数量（LRU 淘汰旧 session，防止辩论等场景内存泄漏）
    _SESSION_CACHE_LIMIT = 1

    def _init_session_state(self, node_config: dict):
        # 模型从 node_config 读取（"model" → 声明式节点字段）
        self._default_model = (
            node_config.get("model")
            or node_config.get("gemini_model")
            or "gemini-3.1-pro"
        )
        self._clients: dict[str, _CodeAssistClient] = {}
        self._records: dict[str, ConversationRecord] = {}
        logger.info(f"[gemini] default_model={self._default_model}")

    def _get_client(self, session_id: str) -> _CodeAssistClient:
        if session_id not in self._clients:
            self._clients[session_id] = _CodeAssistClient(self._default_model)
        return self._clients[session_id]

    def _load_or_create(
        self, session_id: str, workspace: str = ""
    ) -> tuple[ConversationRecord, _CodeAssistClient]:
        # project_id 从 workspace 实时计算（不缓存，workspace 可能随 session 变化）
        project_id = gem_sess.get_project_id(workspace or os.getcwd())

        if session_id and session_id in self._records:
            if is_debug():
                logger.debug(f"[gemini] cache hit for session {session_id[:8]}")
            return self._records[session_id], self._get_client(session_id)

        if session_id:
            record = gem_sess.load_session(session_id, project_id)
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

        record = gem_sess.new_session(project_id, self._default_model)
        client = self._get_client(record.sessionId)
        self._evict_old_sessions()
        self._records[record.sessionId] = record
        logger.info(f"[gemini] new session {record.sessionId[:8]}")
        return record, client

    def _evict_old_sessions(self) -> None:
        """LRU 淘汰：当缓存超限时，移除最旧的 session（保留最近的）。"""
        while len(self._records) >= self._SESSION_CACHE_LIMIT:
            oldest_key = next(iter(self._records))
            del self._records[oldest_key]
            self._clients.pop(oldest_key, None)
            if is_debug():
                logger.debug(f"[gemini] evicted session {oldest_key[:8]} from cache")

    def _persist(
        self, record: ConversationRecord, client: _CodeAssistClient, workspace: str = ""
    ) -> None:
        project_id = gem_sess.get_project_id(workspace or os.getcwd())
        gem_sess.append_history(record, client._history, model=self._default_model)
        gem_sess.save_session(record, project_id)


# ---------------------------------------------------------------------------
# GeminiNode — GEMINI_CLI 节点类型（继承 AgentNode）
# ---------------------------------------------------------------------------

class GeminiNode(_GeminiSessionMixin, AgentNode):
    """
    Gemini 咨询 agent（GEMINI_CLI 节点类型）。

    重写 __call__()：
      - 读取 state["routing_context"] 作为提问
      - 清除 routing_target / routing_context
      - 增加 consult_count

    不调用 super().__call__()，因为：
      - prompt 来自 routing_context，不是 messages[-1]
      - Gemini 不需要信号解析（不会再路由到其他节点）
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        AgentNode.__init__(self, config, node_config)
        self._init_session_state(node_config)
        # node_config 中的 system_prompt 优先；否则使用框架默认
        self._system_prompt: str = node_config.get("system_prompt") or _GEMINI_SYSTEM

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> tuple[str, str]:
        """单轮 Gemini 对话，返回 (reply, session_id)。"""
        workspace = cwd or ""
        record, client = self._load_or_create(session_id, workspace)
        sid = record.sessionId
        try:
            reply = await client.chat(prompt, system=self._system_prompt)
            self._persist(record, client, workspace)
            return reply, sid
        except GeminiQuotaError:
            raise
        except Exception as e:
            logger.error(f"[gemini.node] call_llm 失败: {e}")
            self._persist(record, client, workspace)
            return f"[Gemini 失败: {e}]", sid

    async def __call__(self, state: dict) -> dict:
        """
        读取 routing_context 作为 prompt，调用 Gemini，清除路由信号。
        """
        routing_context = state.get("routing_context", "")
        msgs = state.get("messages", [])

        ns = dict(state.get("node_sessions") or {})
        session_id = ns.get(self._session_key, "")
        workspace = state.get("workspace", "")

        if routing_context:
            prompt = routing_context
            prompt_src = "routing_context"
        elif len(msgs) > 1 and not session_id:
            # 无 session 的多轮（首次进入辩论）：完整历史格式化进 prompt
            parts = []
            for m in msgs:
                role = "议题" if getattr(m, "type", "") == "human" else "发言"
                parts.append(f"[{role}]:\n{m.content}")
            prompt = "\n\n---\n\n".join(parts)
            topic = msgs[0].content if msgs and getattr(msgs[0], "type", "") == "human" else ""
            if topic:
                prompt += f"\n\n---\n\n[原始要求（请严格遵守）]:\n{topic}\n请基于以上讨论继续发言。"
            prompt_src = f"full_history ({len(msgs)} msgs)"
        else:
            # 有 session（resume）：只传最新消息，历史已在 session 中
            prompt = msgs[-1].content if msgs else ""
            prompt_src = "messages[-1]" if not session_id else f"resume({session_id[:8]})"

        if is_debug():
            logger.debug(
                f"[{self._node_id}] prompt_src={prompt_src} "
                f"prompt_preview={prompt[:120]!r} "
                f"session_id={session_id[:8] if session_id else 'new'}"
            )

        try:
            async with acquire_resource(
                self._resource_lock,
                timeout=self._resource_timeout,
                holder=self._node_id,
            ):
                reply, new_session_id = await self.call_llm(
                    prompt, session_id=session_id, cwd=workspace or None
                )
        except GeminiQuotaError as e:
            logger.error(f"[{self._node_id}] Gemini 配额耗尽，降级跳过: {e}")
            return {
                "messages": [AIMessage(content="[Gemini 暂不可用，已跳过本轮咨询]")],
                "routing_target": "",
                "routing_context": "",
                "consult_count": state.get("consult_count", 0) + 1,
                "node_sessions": ns,
            }

        ns[self._session_key] = new_session_id or session_id
        logger.info(f"[{self._node_id}] done, consult_count→{state.get('consult_count', 0) + 1}")
        return {
            "messages": [AIMessage(content=reply)],
            "routing_target": "",     # 清除路由信号，让 validate 正常路由
            "routing_context": "",
            "consult_count": state.get("consult_count", 0) + 1,
            "node_sessions": ns,
        }
