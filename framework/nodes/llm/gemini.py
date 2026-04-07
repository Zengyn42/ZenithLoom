"""
框架级 Gemini 节点 — framework/nodes/llm/gemini.py

两种实现：
  GeminiCodeAssistNode(LlmNode) — Code Assist HTTP API（GEMINI_API 节点类型）
  GeminiCLINode(LlmNode)        — Gemini CLI subprocess（GEMINI_CLI 节点类型）

共同接口：
  call_llm(prompt, session_id, tools, cwd) -> (text, new_session_id)
  __call__(state) -> dict（读取 routing_context / messages，更新 node_sessions）

安全守则（防止账号/钱包被打）：
  - 每次 API 调用前强制 jitter sleep，时长随 prompt 长度线性缩放：
      1 char → 1s，2000+ chars → 20s，再叠加 ±2s 随机抖动
    （Claude 不需要此机制：SDK 子进程内部自行处理 rate limiting）
  - 403 / 429 立刻抛出 GeminiQuotaError，不重试

permission_mode 实现：
  Gemini CLI 通过 --yolo 标志控制自动批准行为，仅支持二档：
    plan 模式   → 不传 --yolo → CLI 无 stdin 交互 → 写操作自动被拒绝（read-only）
    其他模式    → 传 --yolo → 自动批准所有操作
  Gemini CLI 无法区分 default / acceptEdits / bypassPermissions，三者行为相同。
  GeminiCodeAssistNode 通过 HTTP API 调用，不执行本地文件操作，permission_mode 不影响其行为。
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
from framework.debug import is_debug, log_node_thinking, get_debug_output_file, log_node_output_to_file, get_graph_scope
from framework.nodes.llm.llm_node import LlmNode as AgentNode, get_channel_send_callback
from framework.resource_lock import acquire_resource
from framework.token_guard import TokenLimitExceeded, check_before_llm
import framework.nodes.llm.gemini_session as gem_sess
from framework.nodes.llm.gemini_session import ConversationRecord

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


def _jitter_secs(text_len: int, multiplier: float = 1.0) -> float:
    """计算 jitter 延迟秒数。multiplier=0 则返回 0（无延迟）。"""
    if multiplier <= 0:
        return 0.0
    t = min(max(text_len, 1) / _JITTER_SCALE_LEN, 1.0)
    base = _JITTER_MIN_S + t * (_JITTER_MAX_S - _JITTER_MIN_S)
    return max(0.5, (base + random.uniform(-_JITTER_NOISE, _JITTER_NOISE)) * multiplier)


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

    def __init__(self, model: str, jitter_multiplier: float = 1.0):
        self._model = model
        self._jitter_multiplier = jitter_multiplier
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
        delay = _jitter_secs(len(user_text), self._jitter_multiplier)
        if delay > 0:
            logger.info(f"[gemini.client] model={self._model} jitter={delay:.1f}s (prompt_len={len(user_text)} mult={self._jitter_multiplier})")
            if is_debug():
                logger.debug(f"[gemini.client] prompt_preview={user_text[:100]!r}")
            time.sleep(delay)
        else:
            logger.info(f"[gemini.client] model={self._model} jitter=OFF (prompt_len={len(user_text)})")
            if is_debug():
                logger.debug(f"[gemini.client] prompt_preview={user_text[:100]!r}")

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
            or "gemini-2.5-pro"
        )
        self._jitter_multiplier: float = float(node_config.get("jitter_multiplier", 1.0))
        self._clients: dict[str, _CodeAssistClient] = {}
        self._records: dict[str, ConversationRecord] = {}
        logger.info(f"[gemini] default_model={self._default_model} jitter_mult={self._jitter_multiplier}")

    def _get_client(self, session_id: str) -> _CodeAssistClient:
        if session_id not in self._clients:
            self._clients[session_id] = _CodeAssistClient(self._default_model, self._jitter_multiplier)
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
                # workspace 变更时更新 projectHash
                if record.projectHash != project_id:
                    logger.info(
                        f"[gemini] workspace changed for {session_id[:8]}: "
                        f"{record.projectHash} → {project_id}"
                    )
                    record.projectHash = project_id
                    gem_sess.save_session(record, project_id)
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
# GeminiCodeAssistNode — GEMINI_API 节点类型（Code Assist HTTP API）
# ---------------------------------------------------------------------------

class GeminiCodeAssistNode(_GeminiSessionMixin, AgentNode):
    """
    Gemini Code Assist API 节点（GEMINI_API 节点类型）。

    通过 Code Assist HTTP API 调用 Gemini，模型范围受 API 白名单限制
    （gemini-2.5-pro / gemini-2.5-flash）。

    重写 __call__()：
      - 读取 state["routing_context"] 作为提问
      - 清除 routing_target / routing_context
      - enable_routing=true 时，解析输出中的路由信号并写入 state
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        AgentNode.__init__(self, config, node_config)
        self._enable_routing: bool = bool(node_config.get("enable_routing", False))
        self._init_session_state(node_config)
        # node_config 中的 system_prompt 优先；否则使用框架默认
        base_prompt: str = node_config.get("system_prompt") or _GEMINI_SYSTEM
        skill_content = self._load_skill_content()
        self._system_prompt = f"{base_prompt}\n\n{skill_content}" if skill_content else base_prompt

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
        history: list | None = None,
    ) -> tuple[str, str]:
        """单轮 Gemini 对话，返回 (reply, session_id)。history 由 session 管理，忽略。"""
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

        session_id = (state.get("node_sessions") or {}).get(self._session_key, "")
        workspace = state.get("workspace", "")

        # 若 routing_context 是文件路径，自动读取文件内容作为 prompt
        if routing_context and routing_context.startswith("/") and "\n" not in routing_context:
            import os as _os
            if _os.path.isfile(routing_context):
                try:
                    with open(routing_context, "r", encoding="utf-8") as _f:
                        routing_context = _f.read()
                except Exception:
                    pass

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

        # ── Token 安全阀 ──
        # routing_context 不在 msgs 中，history 计全部；否则 prompt 已含 msgs[-1]，只取前缀避免双重计数
        _guard_history = list(msgs) if routing_context else list(msgs[:-1]) if msgs else []
        try:
            check_before_llm(prompt=prompt, history=_guard_history, node_id=self._node_id, limit=self._token_limit)
        except TokenLimitExceeded as exc:
            logger.error(str(exc))
            return {
                "messages": [AIMessage(content=f"⛔ {exc}")],
                "routing_target": "",
                "routing_context": "",
                "node_sessions": {self._session_key: session_id},
            }

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
                "node_sessions": {self._session_key: session_id},
            }

        logger.info(f"[{self._node_id}] done")
        _model_name = getattr(self, "_default_model", "")
        _prompt_preview = prompt[:120] if prompt else ""
        if is_debug():
            log_node_thinking(node_id=self._node_id, output_text=reply,
                              model=_model_name, prompt_preview=_prompt_preview)
        elif get_debug_output_file():
            log_node_output_to_file(node_id=self._node_id, output_text=reply,
                                    model=_model_name, prompt_preview=_prompt_preview)

        # ── 路由信号检测（enable_routing=true 时）──
        if self._enable_routing:
            signal = self._signal_parser.parse(reply)
            routing_target = signal.get("route", "") if signal else ""
            routing_context = signal.get("context", "") if signal else ""
            if routing_target:
                logger.info(f"[{self._node_id}] routing signal: target={routing_target!r}")
                return {
                    "messages": [AIMessage(content=reply)],
                    "routing_target": routing_target,
                    "routing_context": routing_context,
                    "node_sessions": {self._session_key: new_session_id or session_id},
                }

        result = {
            "messages": [AIMessage(content=reply)],
            "routing_target": "",
            "routing_context": "",
            "rollback_reason": "",
            "node_sessions": {self._session_key: new_session_id or session_id},
        }
        # output_field 映射（子图末尾节点用）
        if self._output_field and reply:
            result[self._output_field] = reply
        # previous_node_output：供下一节点读取上文结论
        if reply:
            result["previous_node_output"] = reply
        return result


# 向后兼容别名
GeminiNode = GeminiCodeAssistNode


# ---------------------------------------------------------------------------
# GeminiCLINode — GEMINI_CLI 节点类型（Gemini CLI subprocess）
# ---------------------------------------------------------------------------

class _GeminiCapacityError(RuntimeError):
    """Gemini CLI 模型容量不足（429 / CAPACITY_EXHAUSTED）。"""
    pass


# 模型降级链：优先 Pro，再 Flash；仅包含 CLI 实际可用的模型
# gemini-1.5-pro / gemini-1.5-flash 在 CLI 返回 ModelNotFoundError，已移除
_MODEL_FALLBACK_CHAIN: list[str] = [
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

# stderr 中出现这些关键词则判定为容量/配额错误
_CAPACITY_KEYWORDS = ("capacity", "429", "quota", "exhausted", "overloaded", "rate_limit")

# 跨实例共享：记录不可用的模型及其失败时间戳
# 同一进程内所有 GeminiCLINode 实例共享，避免重复探测
_unavailable_models: dict[str, float] = {}  # model -> time.monotonic() of failure
_UNAVAILABLE_COOLDOWN = 300  # 5 分钟后重新尝试

# 跨实例共享：记录 session 实际使用的模型
# 降级后 session 绑定的模型与 node_config 配置的不同，resume 时需要用实际模型
_session_effective_model: dict[str, str] = {}  # session_id -> actual model used


class GeminiCLINode(AgentNode):
    """
    Gemini CLI subprocess 节点（GEMINI_CLI 节点类型）。

    通过 `gemini -p "prompt" -m model -o json --yolo` 子进程调用，
    支持 Gemini CLI 可用的所有模型（包括 gemini-3-pro-preview 等高级模型）。

    模型降级机制：
      当主模型返回容量不足（429 / CAPACITY_EXHAUSTED）或超时时，
      自动按降级链尝试下一个模型：
        3-pro → 2.5-pro → 1.5-pro → 3-flash → 2.5-flash → 1.5-flash
      仅对新 session 生效；resume 已有 session 时不降级（session 绑定模型）。

    Session 管理由 Gemini CLI 自身处理：
      - 新建：不传 --resume → CLI 自动创建 session
      - 续接：--resume session_id → CLI 恢复已有 session
      - session_id 从 JSON 输出的 "session_id" 字段获取
    """

    # 最小超时：60 秒（短 prompt 的基线）
    _DEFAULT_TIMEOUT = 60
    # 动态超时上限：300 秒
    _MAX_TIMEOUT = 300
    # 每多少字符增加 1 秒（25k chars → +125s，总 ~185s）
    _TIMEOUT_CHARS_PER_SEC = 200

    def __init__(self, config: AgentConfig, node_config: dict):
        super().__init__(config, node_config)
        self._model = (
            node_config.get("model")
            or node_config.get("gemini_model")
            or "gemini-2.5-pro"
        )
        self._enable_routing: bool = bool(node_config.get("enable_routing", False))
        base_prompt: str = node_config.get("system_prompt") or _GEMINI_SYSTEM
        skill_content = self._load_skill_content()
        self._system_prompt = f"{base_prompt}\n\n{skill_content}" if skill_content else base_prompt
        self._timeout: int = node_config.get("timeout") or self._DEFAULT_TIMEOUT
        logger.info(f"[gemini-cli] node={self._node_id} model={self._model} timeout={self._timeout}s enable_routing={self._enable_routing}")

    def _build_fallback_chain(self) -> list[str]:
        """从配置的主模型开始，返回降级链（跳过已知不可用的模型）。"""
        now = time.monotonic()
        # 清理过期记录
        expired = [m for m, t in _unavailable_models.items() if now - t > _UNAVAILABLE_COOLDOWN]
        for m in expired:
            del _unavailable_models[m]
            logger.info(f"[gemini-cli] 模型 {m} 冷却期结束，重新可用")

        if self._model in _MODEL_FALLBACK_CHAIN:
            idx = _MODEL_FALLBACK_CHAIN.index(self._model)
            chain = [m for m in _MODEL_FALLBACK_CHAIN[idx:] if m not in _unavailable_models]
            if not chain:
                # 所有模型都不可用，忽略缓存强制重试
                logger.warning("[gemini-cli] 所有模型均标记为不可用，忽略缓存重试")
                return _MODEL_FALLBACK_CHAIN[idx:]
            return chain
        # 自定义模型不在链中：只用它自己，不降级
        return [self._model]

    async def _run_cli(
        self,
        full_prompt: str,
        model: str,
        session_id: str = "",
        cwd: str | None = None,
        allowed_mcp_servers: list[str] | None = None,
    ) -> tuple[str, str]:
        """
        执行单次 Gemini CLI 调用。

        容量/配额错误 → raise _GeminiCapacityError
        其他错误     → raise RuntimeError
        成功         → return (reply, session_id)
        """
        # 用 stdin 传 prompt 而不是 -p 参数。
        # 原因：-p 的值经 yargs 解析时，若 prompt 以 "--" 开头（如 Claude 的 markdown ---
        # 分隔线），yargs 会把它误当 end-of-flags marker → "Not enough arguments following: p"。
        # stdin 完全绕过参数解析，是 Gemini CLI 官方支持的输入方式。
        cmd = [
            "gemini",
            "-m", model,
            "-o", "json",
        ]
        # permission_mode 控制 --yolo：
        #   plan 模式 → 不传 --yolo，文件操作因无交互输入而被拒绝（read-only）
        #   其他模式 → 传 --yolo，自动批准所有操作
        if not self.is_plan_mode:
            cmd.append("--yolo")
        elif is_debug():
            logger.debug(f"[gemini-cli] plan mode → 不传 --yolo（read-only mode）")
        if session_id:
            cmd.extend(["--resume", session_id])
        if allowed_mcp_servers:
            cmd.extend(["--allowed-mcp-server-names"] + allowed_mcp_servers)

        effective_timeout = min(
            self._MAX_TIMEOUT,
            max(self._timeout, len(full_prompt) // self._TIMEOUT_CHARS_PER_SEC),
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or None,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=full_prompt.encode()), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise _GeminiCapacityError(
                f"model={model} 超时 ({effective_timeout}s, prompt_len={len(full_prompt)})"
            )
        except FileNotFoundError:
            raise  # 保持原类型，让调用者区分"CLI 未安装"与"模型错误"

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace")[:500]
            # 同时检查 stdout，CLI 可能把错误写进 JSON output
            stdout_err = stdout_bytes.decode(errors="replace")[:300]
            combined_lower = (stderr_text + stdout_err).lower()
            if any(kw in combined_lower for kw in _CAPACITY_KEYWORDS):
                raise _GeminiCapacityError(
                    f"model={model} 容量不足: {stderr_text}"
                )
            raise RuntimeError(
                f"Gemini CLI 退出码 {proc.returncode}: {stderr_text}"
            )

        stdout_text = stdout_bytes.decode(errors="replace")
        # Gemini CLI may print warnings (e.g. "MCP issues detected...") before
        # the JSON payload.  Strip everything before the first '{'.
        json_start = stdout_text.find("{")
        if json_start > 0:
            stdout_text = stdout_text[json_start:]
        try:
            data = json.loads(stdout_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Gemini CLI JSON 解析失败: {e}\nstdout={stdout_text[:300]}"
            )

        reply = data.get("response", "")
        new_sid = data.get("session_id", session_id)
        return reply.strip(), new_sid

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
        history: list | None = None,
    ) -> tuple[str, str]:
        """调用 Gemini CLI subprocess，返回 (reply, session_id)。history 由 session 管理，忽略。

        tools: 基类 _select_tools() 产出的工具名列表。
          对 Gemini CLI，映射为 --allowed-mcp-server-names（MCP server 粒度过滤）。
          工具名会被当作 MCP server name 直接传递。
        """
        # tools → allowed_mcp_servers（Gemini CLI 按 MCP server name 过滤）
        allowed_mcp_servers = tools if tools else None
        # plan 模式（辩论节点等）无 workspace 时，强制用 /tmp 作为 cwd，
        # 避免 Gemini CLI 扫描代码库目录后把任务误解为项目编码协助
        if self.is_plan_mode and not cwd:
            cwd = "/tmp"
        # 首次调用（无 session）时嵌入 system prompt
        if not session_id and self._system_prompt:
            full_prompt = (
                f"[System Instructions]\n{self._system_prompt}\n\n"
                f"[Task]\n{prompt}"
            )
        else:
            full_prompt = prompt

        # resume 已有 session：确保 session 文件在当前 project 目录 + 通知目录变更
        if session_id:
            # 1. 文件迁移：确保 CLI 的 --resume 能找到 session 文件
            current_project_id = gem_sess.get_project_id(cwd or os.getcwd())
            gem_sess._find_session_file(session_id, current_project_id)

            # 2. 上下文更新：检测 workspace 是否变化，注入目录变更通知
            #    session 的 projectHash 记录创建时的 project，若与当前不同说明 workspace 变了
            record = gem_sess.load_session(session_id, current_project_id)
            if record and record.projectHash != current_project_id:
                if self.is_plan_mode:
                    # plan 模式（辩论节点）：跳过 workspace 变更注入，避免干扰纯文本讨论
                    record.projectHash = current_project_id
                    gem_sess.save_session(record, current_project_id)
                    logger.debug(
                        f"[gemini-cli] plan mode → skipped cwd injection "
                        f"for session {session_id[:8]}"
                    )
                else:
                    full_prompt = (
                        f"[重要：工作目录已变更]\n"
                        f"你的工作目录已从旧项目切换到：{cwd}\n"
                        f"请以新目录为准进行所有后续操作。\n\n"
                        f"{full_prompt}"
                    )
                    record.projectHash = current_project_id
                    gem_sess.save_session(record, current_project_id)
                    logger.info(
                        f"[gemini-cli] workspace changed → injected cwd notice "
                        f"for session {session_id[:8]}"
                    )

            effective_model = _session_effective_model.get(session_id, self._model)
            sid_short = session_id[:8]
            # 构建 resume 降级链：实际模型优先，然后是其余可用模型
            resume_chain = [effective_model]
            for m in self._build_fallback_chain():
                if m != effective_model:
                    resume_chain.append(m)

            for j, model in enumerate(resume_chain):
                if j == 0:
                    logger.info(
                        f"[gemini-cli] resume model={model} sid={sid_short} "
                        f"prompt_len={len(full_prompt)}"
                    )
                else:
                    logger.warning(
                        f"[gemini-cli] resume 降级: sid={sid_short} 换模型 → {model}"
                    )
                try:
                    reply, new_sid = await self._run_cli(
                        full_prompt, model, session_id=session_id, cwd=cwd,
                        allowed_mcp_servers=allowed_mcp_servers,
                    )
                    _session_effective_model[new_sid or session_id] = model
                    _unavailable_models.pop(model, None)
                    if j > 0:
                        logger.warning(f"[gemini-cli] resume 降级成功: → {model}")
                    logger.info(f"[gemini-cli] done sid={new_sid[:8] if new_sid else '?'}")
                    return reply, new_sid
                except _GeminiCapacityError as e:
                    _unavailable_models[model] = time.monotonic()
                    logger.warning(f"[gemini-cli] {e} (标记不可用 {_UNAVAILABLE_COOLDOWN}s)")
                    continue
                except RuntimeError as e:
                    # 非容量错误（如 session 不兼容）→ 跳过此模型但不标记为不可用
                    logger.warning(f"[gemini-cli] resume model={model} 失败: {e}")
                    continue

            # 所有模型 resume 均失败 → 放弃旧 session，创建新 session 继续
            logger.warning(
                f"[gemini-cli] resume 全部失败 sid={sid_short}，"
                f"已尝试 {len(resume_chain)} 个模型 → 放弃旧 session，新建"
            )
            # 递归调用自身，session_id="" 走新建路径
            return await self.call_llm(prompt, session_id="", tools=tools, cwd=cwd, history=history)

        # 新 session：按降级链尝试
        chain = self._build_fallback_chain()
        last_error: Exception | None = None

        for i, model in enumerate(chain):
            if i == 0:
                logger.info(
                    f"[gemini-cli] model={model} prompt_len={len(full_prompt)}"
                )
            else:
                logger.warning(
                    f"[gemini-cli] 降级尝试: {self._model} → {model} ({i+1}/{len(chain)})"
                )
            try:
                reply, new_sid = await self._run_cli(
                    full_prompt, model, cwd=cwd,
                    allowed_mcp_servers=allowed_mcp_servers,
                )
                _unavailable_models.pop(model, None)  # 成功则清除不可用标记
                if new_sid:
                    _session_effective_model[new_sid] = model  # 记录 session 实际模型
                if i > 0:
                    logger.warning(
                        f"[gemini-cli] 降级成功: {self._model} → {model}"
                    )
                if is_debug():
                    logger.debug(f"[gemini-cli] reply_preview={reply[:100]!r}")
                logger.info(f"[gemini-cli] done model={model} sid={new_sid[:8] if new_sid else '?'}")
                return reply, new_sid
            except FileNotFoundError:
                raise RuntimeError(
                    "gemini CLI 未安装或不在 PATH 中。"
                    "请先运行: npm install -g @google/gemini-cli"
                )
            except _GeminiCapacityError as e:
                _unavailable_models[model] = time.monotonic()
                logger.warning(f"[gemini-cli] {e} (标记不可用 {_UNAVAILABLE_COOLDOWN}s)")
                last_error = e
                continue
            except RuntimeError as e:
                # 非容量错误（如退出码异常、JSON 解析失败）→ 跳过此模型，不标记为不可用
                logger.warning(f"[gemini-cli] model={model} 失败，尝试下一个: {e}")
                last_error = e
                continue

        # 全部降级失败
        raise RuntimeError(
            f"Gemini CLI 所有模型均不可用 (tried {len(chain)}): {last_error}"
        )

    async def __call__(self, state: dict) -> dict:
        """
        读取 routing_context / messages 作为 prompt，调用 Gemini CLI。
        """
        routing_context = state.get("routing_context", "")
        msgs = state.get("messages", [])

        session_id = (state.get("node_sessions") or {}).get(self._session_key, "")
        workspace = state.get("workspace", "")

        # 若 routing_context 是文件路径，自动读取文件内容作为 prompt
        if routing_context and routing_context.startswith("/") and "\n" not in routing_context:
            import os as _os
            if _os.path.isfile(routing_context):
                try:
                    with open(routing_context, "r", encoding="utf-8") as _f:
                        routing_context = _f.read()
                except Exception:
                    pass

        if routing_context:
            prompt = routing_context
            prompt_src = "routing_context"
        elif len(msgs) > 1 and not session_id:
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
            prompt = msgs[-1].content if msgs else ""
            prompt_src = "messages[-1]" if not session_id else f"resume({session_id[:8]})"

        # ── subgraph_topic 只读注入 ───────────────────────────────────────
        # SubgraphMapperNode 负责写入/清空；Gemini 节点只读取并注入 prompt
        _subgraph_topic = state.get("subgraph_topic", "")
        if _subgraph_topic and not routing_context:
            prompt = f"【当前主题·严格围绕此展开】\n{_subgraph_topic}\n\n{prompt}"

        # ── previous_node_output 注入（仅在子图内有效）────────────────────
        # 跨 session_key 传递上一节点（Claude/Gemini 交替）的结论，确保辩论连续。
        _prev_output = state.get("previous_node_output", "") if _subgraph_topic else ""
        if _prev_output:
            prompt = f"【前一节点输出·请基于此继续】\n{_prev_output}\n\n{prompt}"

        if is_debug():
            logger.debug(
                f"[{self._node_id}] prompt_src={prompt_src} "
                f"prompt_preview={prompt[:120]!r} "
                f"session_id={session_id[:8] if session_id else 'new'}"
            )

        # ── Token 安全阀 ──
        # routing_context 不在 msgs 中，history 计全部；否则 prompt 已含 msgs[-1]，只取前缀避免双重计数
        _guard_history = list(msgs) if routing_context else list(msgs[:-1]) if msgs else []
        try:
            check_before_llm(prompt=prompt, history=_guard_history, node_id=self._node_id, limit=self._token_limit)
        except TokenLimitExceeded as exc:
            logger.error(str(exc))
            return {
                "messages": [AIMessage(content=f"⛔ {exc}")],
                "routing_target": "",
                "routing_context": "",
                "node_sessions": {self._session_key: session_id},
            }

        try:
            async with acquire_resource(
                self._resource_lock,
                timeout=self._resource_timeout,
                holder=self._node_id,
            ):
                reply, new_session_id = await self.call_llm(
                    prompt, session_id=session_id, cwd=workspace or None
                )
        except Exception as e:
            logger.error(f"[{self._node_id}] Gemini CLI 失败: {e}")
            return {
                "messages": [AIMessage(content=f"[Gemini CLI 失败: {e}]")],
                "routing_target": "",
                "routing_context": "",
                "node_sessions": {self._session_key: session_id},
            }

        logger.info(f"[{self._node_id}] done")

        # 立即推送到 Discord（不等 outer astream updates 事件，那是在子图 ainvoke 返回后才批量触发）
        ch_cb = get_channel_send_callback()
        if ch_cb and reply:
            scope = get_graph_scope()
            scope_str = " › ".join(scope) if scope else ""
            header = f"\n⚙️ **{scope_str} › {self._node_id}**\n" if scope_str else f"\n⚙️ **{self._node_id}**\n"
            try:
                await ch_cb(header + reply + "\n")
            except Exception:
                pass

        _model_name = getattr(self, "_model", "")
        _prompt_preview = prompt[:120] if prompt else ""
        if is_debug():
            log_node_thinking(node_id=self._node_id, output_text=reply,
                              model=_model_name, prompt_preview=_prompt_preview)
        elif get_debug_output_file():
            log_node_output_to_file(node_id=self._node_id, output_text=reply,
                                    model=_model_name, prompt_preview=_prompt_preview)

        # ── 路由信号检测（enable_routing=true 时）──
        if self._enable_routing:
            signal = self._signal_parser.parse(reply)
            routing_target = signal.get("route", "") if signal else ""
            routing_context = signal.get("context", "") if signal else ""
            if routing_target:
                logger.info(f"[{self._node_id}] routing signal: target={routing_target!r}")
                return {
                    "messages": [AIMessage(content=reply)],
                    "routing_target": routing_target,
                    "routing_context": routing_context,
                    "node_sessions": {self._session_key: new_session_id or session_id},
                }

        result = {
            "messages": [AIMessage(content=reply)],
            "routing_target": "",
            "routing_context": "",
            "rollback_reason": "",
            "node_sessions": {self._session_key: new_session_id or session_id},
        }
        # output_field 映射（子图末尾节点用）
        if self._output_field and reply:
            result[self._output_field] = reply
        # previous_node_output：供下一节点读取上文结论
        if reply:
            result["previous_node_output"] = reply
        return result
