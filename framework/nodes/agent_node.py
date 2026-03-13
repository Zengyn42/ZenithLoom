"""
框架级通用 Agent 节点（抽象基类）— framework/nodes/agent_node.py

AgentNode 是所有 LLM 节点的抽象基类，封装图协议逻辑。
子类只需实现 call_llm()，其余框架行为由基类统一处理。

继承体系：
  AgentNode（base class）
    ├── __call__(state) → dict      ← LangGraph 节点，处理路由、tool_rules、信号检测
    └── call_llm(prompt, ...) → (str, str)  ← 抽象方法，子类实现
  ClaudeNode(AgentNode)            ← Claude CLI SDK 实现
  GeminiNode(AgentNode)            ← Gemini Code Assist API 实现
  LlamaNode(AgentNode)             ← 本地 vLLM/Ollama 实现

node_config（来自 agent.json）驱动行为：
  first_turn_suffix      str   首轮 prompt 末尾附加字符串，如 "Hani:"
  user_msg_prefix        str   用户消息前缀，如 "老板: "
  gemini_mention_pattern str   触发强制咨询的 @Gemini 正则，如 "@[Gg]emini"
  tombstone_enabled      bool  是否注入 .tombstone 耻辱柱内容
  tool_rules             list  [{"pattern": ..., "flags": [...], "tools": [...]}]
  resource_lock          str   持锁资源名，如 "GPU_0_VRAM_22GB"（可选）
  resource_timeout       float 资源锁超时秒数（默认 300）
  signal_parser          str   信号解析器类型（默认 "json_line"）
  id                     str   节点 ID，用于 node_sessions 键名（默认 "claude_main"）
"""

import json
import logging
import os
import re
from abc import abstractmethod

from langchain_core.messages import AIMessage

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.resource_lock import acquire_resource
from framework.signal_parser import get_signal_parser

logger = logging.getLogger(__name__)

# project_meta 文件读取
_IN_PROGRESS_RE = re.compile(r"^## In Progress", re.MULTILINE)
PLAN_MAX_CHARS = 3000
TASKS_MAX_CHARS = 2000


class AgentNode:
    """
    所有 LLM 节点的抽象基类。

    子类实现 call_llm()；基类 __call__() 处理所有框架级逻辑：
      - node_sessions UUID 路由（读取/写入 state["node_sessions"]）
      - 资源锁（acquire_resource）
      - 动态注入（rollback warning、Gemini 建议、project_meta）
      - consult_gemini 信号检测（路由到 gemini_advisor）
      - tool_rules 关键词匹配
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        self.config = config
        self.node_config = node_config
        # 保留 _cfg 兼容旧内部引用
        self._cfg = node_config

        # 节点 ID（用于 node_sessions 字典的键）
        self._node_id = node_config.get("id", "claude_main")
        # session_key：多个节点共享同一 session 时使用（如 debate 子图的 Claude 节点）
        self._session_key = node_config.get("session_key", self._node_id)

        # 资源锁
        self._resource_lock = node_config.get("resource_lock")
        self._resource_timeout = float(node_config.get("resource_timeout", 300))

        # 信号解析器
        self._signal_parser = get_signal_parser(
            node_config.get("signal_parser", "json_line")
        )

        # 预编译 @Gemini 触发正则
        mention_pat = node_config.get("gemini_mention_pattern")
        self._gemini_mention_re: re.Pattern | None = (
            re.compile(mention_pat, re.IGNORECASE) if mention_pat else None
        )

        # 预编译 tool_rules
        self._tool_rules: list[tuple[re.Pattern, list[str]]] = [
            (
                re.compile(
                    rule["pattern"],
                    flags=sum(getattr(re, f, 0) for f in rule.get("flags", [])),
                ),
                rule["tools"],
            )
            for rule in node_config.get("tool_rules", [])
        ]

    @abstractmethod
    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
    ) -> tuple[str, str]:
        """
        调用具体 LLM，返回 (text, new_session_id)。
        session_id 空 → 新建 session；非空 → resume 已有 session。
        子类必须实现。
        """

    async def __call__(self, state: dict) -> dict:
        msgs = state["messages"]
        latest_input = msgs[-1].content
        # project_root（!setproject）优先；退回 per-session workspace
        project_root = state.get("project_root", "") or state.get("workspace", "") or None

        # node_sessions UUID 路由（session_key 允许多节点共享 session）
        ns = dict(state.get("node_sessions") or {})
        session_id = ns.get(self._session_key) or (
            state.get("claude_session_id", "")
            if self._node_id == "claude_main"
            else ""
        )

        if is_debug():
            logger.debug(
                f"[{self._node_id}] session_id={session_id[:8] if session_id else 'new'} "
                f"input_len={len(latest_input)}"
            )

        # ── 框架层动态注入 ──────────────────────────────────────────────────
        rollback_warning = self._build_rollback_warning(state)
        gemini_section = self._build_gemini_section(state)
        project_section = _build_project_section(state)
        extra = self._build_extra_injections(state, latest_input)

        dynamic_injections = "".join(
            filter(None, [rollback_warning, extra, gemini_section, project_section])
        )

        # ── 消息格式化 ─────────────────────────────────────────────────────
        user_msg = self._format_user_msg(latest_input, state)

        # ── Prompt 组装 ────────────────────────────────────────────────────
        parts = [p for p in [dynamic_injections, user_msg] if p]
        prompt = "\n\n".join(parts)
        if not session_id:
            suffix = self._cfg.get("first_turn_suffix", "")
            if suffix:
                prompt += f"\n\n{suffix}"

        tools = self._select_tools(latest_input)

        if is_debug():
            logger.debug(f"[{self._node_id}] prompt_len={len(prompt)} tools={tools}")

        # ── LLM 调用（持资源锁）────────────────────────────────────────────
        async with acquire_resource(
            self._resource_lock,
            timeout=self._resource_timeout,
            holder=self._node_id,
        ):
            raw_output, new_session_id = await self.call_llm(
                prompt,
                session_id=session_id,
                tools=tools,
                cwd=project_root,
            )

        if is_debug():
            logger.debug(f"[{self._node_id}] raw_output_preview={raw_output[:200]!r}")

        # ── 路由信号检测（用注册的 SignalParser）────────────────────────────
        # 信号格式：{"route": "<node_id>", "context": "<question|background>"}
        signal = self._signal_parser.parse(raw_output)
        routing_target = signal.get("route", "") if signal else ""
        routing_context = signal.get("context", "") if signal else ""

        ns[self._session_key] = new_session_id or session_id

        if routing_target:
            logger.info(f"[{self._node_id}] routing signal: target={routing_target!r}")
            result: dict = {
                "messages": [AIMessage(content=raw_output)],
                "routing_target": routing_target,
                "routing_context": routing_context,
                "node_sessions": ns,
            }
        else:
            result = {
                "messages": [AIMessage(content=raw_output)],
                "routing_target": "",
                "routing_context": "",
                "consult_count": 0,
                "rollback_reason": "",
                "retry_count": 0,
                "node_sessions": ns,
            }

        # 向后兼容：claude_main 同步写 claude_session_id
        if self._node_id == "claude_main":
            result["claude_session_id"] = new_session_id or session_id
        return result

    # ── 框架层内部方法 ──────────────────────────────────────────────────────

    def _build_rollback_warning(self, state: dict) -> str:
        rollback_reason = state.get("rollback_reason", "")
        if not rollback_reason:
            return ""
        last_commit = state.get("last_stable_commit", "")
        commit_short = last_commit[:8] if last_commit else "上一个稳定版本"
        warning = (
            f"⚠️ 【系统警告·时光倒流】\n"
            f"上一次操作触发了验证失败（原因：{rollback_reason}）。\n"
            f"系统已执行 git reset --hard，文件已恢复到 {commit_short}。\n"
            f"请换一种思路，不要重复同样的错误。\n"
        )
        if is_debug():
            logger.debug(
                f"[{self._node_id}] rollback_warning injected: reason={rollback_reason!r}"
            )
        return warning

    def _build_gemini_section(self, state: dict) -> str:
        # Gemini 的回复已通过 AIMessage 进入 messages，Claude 在对话历史中直接看到。
        # 此方法保留供子类扩展；基类不再注入 routing_context。
        return ""

    def _build_extra_injections(self, state: dict, user_input: str) -> str:
        """耻辱柱注入（tombstone_enabled: true 时激活）。"""
        if not self._cfg.get("tombstone_enabled", False):
            return ""
        from framework.nodes.git_nodes import read_tombstone
        project_root = state.get("project_root", "") or ""
        tombstone_raw = read_tombstone(project_root)
        if not tombstone_raw:
            return ""
        if is_debug():
            logger.debug(
                f"[{self._node_id}] tombstone injected ({len(tombstone_raw)} chars)"
            )
        return (
            f"⛔ 【跨时空耻辱柱·绝对禁止重蹈】\n{tombstone_raw}\n"
            f"绝对不要重复以上任何模式。\n"
        )

    def _format_user_msg(self, user_input: str, state: dict) -> str:
        """用户消息前缀 + @Gemini 强制咨询检测。"""
        prefix = self._cfg.get("user_msg_prefix", "")
        if self._gemini_mention_re and self._gemini_mention_re.search(user_input):
            topic = self._gemini_mention_re.sub("", user_input).strip() or user_input
            logger.info(f"[{self._node_id}] @Gemini trigger: topic={topic!r}")
            return (
                f"{prefix}{user_input}\n\n"
                f"【系统指令·强制咨询】用户明确要求咨询 Gemini，关于：{topic}\n"
                f"请把这个问题组装成专业提问，"
                f"**第一行且只有第一行**输出以下 JSON，不要任何前缀或解释：\n"
                f'{{"route": "gemini_advisor", "context": "<提炼后的问题>|<相关项目状态>"}}'
            )
        return f"{prefix}{user_input}" if prefix else user_input

    def _select_tools(self, user_input: str) -> list[str] | None:
        """tool_rules 关键词匹配后动态追加工具。node_config.tools 优先于顶层 config.tools。"""
        tools = list(self._cfg.get("tools") or self.config.tools)
        for pattern, extra in self._tool_rules:
            if pattern.search(user_input):
                for t in extra:
                    if t not in tools:
                        tools.append(t)
        return tools


# ── 框架工具函数 ──────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _read_project_file(
    path: str, is_tasks: bool = False, max_chars: int = 3000
) -> str:
    if not path:
        return ""
    full_path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    if not os.path.exists(full_path):
        return ""

    with open(full_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    if is_tasks:
        match = _IN_PROGRESS_RE.search(content)
        if match:
            content = content[match.start():]
        if len(content) > max_chars:
            content = "...(older tasks truncated)\n" + content[-max_chars:]
    else:
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(rest truncated)"

    return content.strip()


def _build_project_section(state: dict) -> str:
    meta = state.get("project_meta") or {}
    root = state.get("project_root") or ""
    sections = []

    plan_file = meta.get("plan", "")
    if plan_file:
        path = os.path.join(root, plan_file) if root else plan_file
        content = _read_project_file(path, is_tasks=False, max_chars=PLAN_MAX_CHARS)
        if content:
            sections.append(f"[当前项目架构 — {plan_file}]\n{content}")

    tasks_file = meta.get("tasks", "")
    if tasks_file:
        path = os.path.join(root, tasks_file) if root else tasks_file
        content = _read_project_file(path, is_tasks=True, max_chars=TASKS_MAX_CHARS)
        if content:
            sections.append(
                f"[当前任务列表 — {tasks_file}（尾部/In Progress 截取）]\n{content}"
            )

    if is_debug() and sections:
        logger.debug(f"[agent_node] project_section sections={len(sections)}")

    return "\n\n".join(sections)


# 向后兼容别名
AgentClaudeNode = AgentNode
