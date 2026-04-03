"""
框架级 LLM 节点抽象基类 — framework/nodes/llm/llm_node.py

LlmNode 是所有 LLM 节点的抽象基类，封装图协议逻辑。
子类只需实现 call_llm()，其余框架行为由基类统一处理。

继承体系：
  LlmNode（base class）
    ├── __call__(state) → dict      ← LangGraph 节点，处理路由、tool_rules、信号检测
    └── call_llm(prompt, ...) → (str, str)  ← 抽象方法，子类实现
  ClaudeSDKNode(LlmNode)         ← Claude CLI SDK 实现
  GeminiCodeAssistNode(LlmNode)  ← Gemini Code Assist HTTP API 实现
  GeminiCLINode(LlmNode)         ← Gemini CLI subprocess 实现
  OllamaNode(LlmNode)            ← 本地 Ollama 实现

permission_mode — LlmNode 级权限控制抽象：
  entity.json 中声明 "permission_mode": "<mode>"，LlmNode 基类解析并存储到
  self._permission_mode。各子类负责将其映射到 provider 原生机制。

  可用模式：
    "default"            标准模式 — 工具操作需用户确认
    "plan"               只读/规划模式 — 禁止一切写入/执行操作，仅可推理和读取
    "acceptEdits"        自动接受文件编辑，其他操作仍需确认
    "bypassPermissions"  跳过所有权限检查，全自动执行

  各子类实现：
    ClaudeSDKNode   → SDK 原生支持四种模式：permission_mode 参数直传 + disallowed_tools
    GeminiCLINode   → 二档控制：plan 时不传 --yolo（CLI 无 stdin → 写操作失败）；
                      其余模式传 --yolo（自动批准）。CLI 无法区分 default/acceptEdits/bypass
    OllamaNode      → plan 时 system_prompt 注入禁写指令 + _call_with_tools 过滤写入类工具；
                      其余模式正常调用。Ollama 无原生权限机制

  基类提供的辅助接口：
    self._permission_mode   str         当前模式值
    self.is_plan_mode       property    是否为 plan 模式
    self._get_disallowed_tools()        plan 模式自动合并 _WRITE_TOOLS 到禁用列表
    _WRITE_TOOLS            frozenset   写入/执行类工具名集合

  优先级：node_config["permission_mode"] > config.permission_mode > "default"

node_config（来自 entity.json）驱动行为：
  permission_mode        str   权限模式（见上方详细说明）
  first_turn_suffix      str   首轮 prompt 末尾附加字符串，如 "Hani:"
  user_msg_prefix        str   用户消息前缀，如 "老板: "
  gemini_mention_pattern str   触发强制咨询的 @Gemini 正则，如 "@[Gg]emini"
  tombstone_enabled      bool  是否注入 .tombstone 耻辱柱内容
  tool_rules             list  [{"pattern": ..., "flags": [...], "tools": [...]}]
  resource_lock          str   持锁资源名，如 "GPU_0_VRAM_22GB"（可选）
  resource_timeout       float 资源锁超时秒数（默认 300）
  signal_parser          str   信号解析器类型（默认 "json_line"）
  resume_prompt          str   共享 session 时 resume 使用的固定指令（避免重复前一节点输出）
  id                     str   节点 ID，用于 node_sessions 键名（默认 "claude_main"）
  skills                 list  SkillRegistry skill 名称列表，如 ["blender-index", "gws-gam"]；
                               基类在 _load_skill_content() 中自动加载并拼接到现有 parts，
                               对 Gemini / Ollama 节点注入 system_prompt，Claude 节点由
                               子类决定注入时机（_load_skill_content 可选调用）
"""

import contextvars
import json
import logging
import os
import re
from abc import abstractmethod
from pathlib import Path

from langchain_core.messages import AIMessage

from framework.config import AgentConfig
from framework.debug import is_debug, log_node_thinking
from framework.resource_lock import acquire_resource
from framework.signal_parser import get_signal_parser
from framework.token_guard import TokenLimitExceeded, check_before_llm, get_default_limit

logger = logging.getLogger(__name__)

# ── Streaming callback ────────────────────────────────────────────────────────
# ContextVar: set before invoking the graph, inherited by all awaited coroutines.
# Callback signature: (text: str) -> None
_stream_cb: contextvars.ContextVar = contextvars.ContextVar("claude_stream_cb", default=None)


def set_stream_callback(fn) -> None:
    """Set (or clear) the streaming callback for the current async context."""
    _stream_cb.set(fn)


def get_stream_callback():
    """Return the current streaming callback, or None if not set."""
    return _stream_cb.get()


# project_meta 文件读取
_IN_PROGRESS_RE = re.compile(r"^## In Progress", re.MULTILINE)
PLAN_MAX_CHARS = 3000
TASKS_MAX_CHARS = 2000


class LlmNode:
    """
    所有 LLM 节点的抽象基类。

    子类实现 call_llm()；基类 __call__() 处理所有框架级逻辑：
      - node_sessions UUID 路由（读取/写入 state["node_sessions"]）
      - 资源锁（acquire_resource）
      - 动态注入（rollback warning、Gemini 建议、project_meta）
      - 路由信号检测（routing_target）
      - tool_rules 关键词匹配
      - permission_mode 权限控制（plan / default / acceptEdits / bypassPermissions）

    permission_mode 控制层级（声明式，entity.json 中配置）：
      "plan"              只读/规划模式 — 可推理和读取，禁止所有写入/执行操作
      "default"           标准模式 — 工具操作需确认
      "acceptEdits"       自动接受文件编辑
      "bypassPermissions" 跳过所有权限检查

    各子类负责将 permission_mode 映射到各自 provider 的原生机制：
      ClaudeSDKNode  → SDK permission_mode 参数 + disallowed_tools
      GeminiCLINode  → --yolo 控制（plan 时不传 --yolo）
      OllamaNode     → system_prompt 注入禁写指令 + 不传 tool 定义
    """

    # 写入/执行类工具 — plan 模式下自动禁止
    # 子类通过 _get_disallowed_tools() 获取，按需传给各自 provider
    _WRITE_TOOLS = frozenset([
        "Write", "Edit", "MultiEdit", "Bash",
        "TodoWrite", "NotebookEdit", "Agent",
    ])

    def __init__(self, config: AgentConfig, node_config: dict):
        self.config = config
        self.node_config = node_config
        # 保留 _cfg 兼容旧内部引用
        self._cfg = node_config

        # 节点 ID（用于 node_sessions 字典的键）
        self._node_id = node_config.get("id", "claude_main")
        # session_key：多个节点共享同一 session 时使用（如 debate 子图的 Claude 节点）
        self._session_key = node_config.get("session_key", self._node_id)

        # permission_mode：节点级 > 顶层配置级 > 默认值
        self._permission_mode: str = (
            node_config.get("permission_mode")
            or getattr(config, "permission_mode", None)
            or "default"
        )

        # Token 安全阀：node_config["token_limit"] > 按 type 默认值 > 环境变量
        node_type = node_config.get("type", "")
        configured_limit = node_config.get("token_limit")
        self._token_limit: int = (
            int(configured_limit) if configured_limit is not None
            else get_default_limit(node_type)
        )

        # 资源锁
        self._resource_lock = node_config.get("resource_lock")
        self._resource_timeout = float(node_config.get("resource_timeout", 300))

        # output_field：子图末尾节点用，把 LLM 输出自动写入指定 state 字段
        # 如 "debate_conclusion"、"apex_conclusion"、"knowledge_result" 等
        self._output_field: str | None = node_config.get("output_field")

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

        # add_dirs：声明额外项目目录，扫描其中的 .claude/skills/ 供子类使用
        # Claude 节点：原生传给 ClaudeAgentOptions(add_dirs=...)
        # Gemini / Ollama 节点：_load_skill_content() 结果注入 system_prompt
        self._add_dirs: list[Path] = [
            Path(d) for d in node_config.get("add_dirs", [])
        ]

        # skill_files：显式声明的 skill 文件路径列表（相对于 blueprint_dir 或绝对路径）
        self._skill_files: list[str] = node_config.get("skill_files", [])

        # skills：通过 SkillRegistry 注入的 skill 名称列表（声明式，优先于 skill_files）
        # 示例：["blender-index", "opencli-grok"]
        self._skill_names: list[str] = node_config.get("skills", [])

    @property
    def is_plan_mode(self) -> bool:
        """当前节点是否处于 plan（只读/规划）模式。"""
        return self._permission_mode == "plan"

    def _get_disallowed_tools(self) -> list[str]:
        """基于 permission_mode 返回应禁用的工具列表。

        plan 模式：禁用所有写入/执行类工具（_WRITE_TOOLS）。
        其他模式：仅返回 node_config 中显式声明的 disallowed_tools。
        子类可直接使用此方法获取禁用列表，传给各自 provider。
        """
        base = list(self._cfg.get("disallowed_tools") or [])
        if self.is_plan_mode:
            base = list(set(base) | self._WRITE_TOOLS)
        return base

    def _load_skill_content(self) -> str:
        """
        加载 skill 内容，两种来源合并：
        1. skill_files：显式声明的文件路径（相对于 blueprint_dir 或绝对路径）
        2. add_dirs：扫描 .claude/skills/*/SKILL.md（Claude 原生机制）

        供非 Claude 节点（Gemini、Ollama）在 __init__ 中调用，
        将 skill 内容追加到 system_prompt 末尾。
        Claude 节点通过 ClaudeAgentOptions(add_dirs=...) 原生加载，无需调用此方法。
        """
        parts = []

        # 1. 显式 skill_files（相对路径基于 process cwd 解析）
        for sf in self._skill_files:
            p = Path(sf)
            if p.exists():
                parts.append(p.read_text(encoding="utf-8").strip())
                logger.info(f"[{self._node_id}] loaded skill: {p}")
            else:
                logger.warning(f"[{self._node_id}] skill file not found: {p}")

        # 2. add_dirs 扫描
        for d in self._add_dirs:
            skills_dir = d / ".claude" / "skills"
            if skills_dir.exists():
                for skill_md in sorted(skills_dir.rglob("SKILL.md")):
                    parts.append(skill_md.read_text(encoding="utf-8").strip())

        # 3. skills（名称 → SkillRegistry）——声明式注入，兼容所有 LLM 节点
        if self._skill_names:
            from framework.skill_registry import SkillRegistry
            content = SkillRegistry.get_instance().load(self._skill_names)
            if content:
                parts.append(content)

        return "\n\n---\n\n".join(parts)

    @abstractmethod
    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
        history: list | None = None,
    ) -> tuple[str, str]:
        """
        调用具体 LLM，返回 (text, new_session_id)。
        session_id 空 → 新建 session；非空 → resume 已有 session。
        history: LangGraph state["messages"] 完整历史（HumanMessage/AIMessage 列表）。
          - Claude/Gemini: 忽略（server-side session 已含历史）
          - OllamaNode: 用于重建多轮对话（Ollama 无 server-side session）
        子类必须实现。
        """

    async def __call__(self, state: dict) -> dict:
        msgs = state.get("messages") or []
        latest_input = state.get("routing_context") or (msgs[-1].content if msgs else "")
        # project_root（!setproject）优先；退回 per-session workspace
        project_root = state.get("project_root", "") or state.get("workspace", "") or None

        # resume_prompt：当节点共享 session（session_key != node_id），
        # 且 state 中已有该 session 时，使用固定指令替代 msgs[-1]，
        # 避免将前一个共享节点的 AIMessage 输出原封不动回传导致重复。
        _resume_prompt = self._cfg.get("resume_prompt")
        ns_peek = state.get("node_sessions") or {}
        if _resume_prompt and ns_peek.get(self._session_key):
            latest_input = _resume_prompt

        # node_sessions UUID 路由（session_key 允许多节点共享 session）
        _ns = state.get("node_sessions") or {}
        session_id = _ns.get(self._session_key, "")

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

        # ── Debug thinking 拦截器 ──────────────────────────────────────────
        _thinking_chunks: list[str] = []
        _output_chunks: list[str] = []
        _original_cb = None
        if is_debug():
            _original_cb = _stream_cb.get()

            def _debug_intercept(text: str, is_thinking: bool = False) -> None:
                if is_thinking:
                    _thinking_chunks.append(text)
                else:
                    _output_chunks.append(text)
                if _original_cb:
                    _original_cb(text, is_thinking)

            _stream_cb.set(_debug_intercept)

        # ── Token 安全阀 ─────────────────────────────────────────────────
        # prompt 已包含 msgs[-1]（latest_input），history 只取前缀避免双重计数
        try:
            check_before_llm(
                prompt=prompt, history=list(msgs[:-1]),
                node_id=self._node_id, limit=self._token_limit,
            )
        except TokenLimitExceeded as exc:
            if is_debug() and _original_cb is not None:
                _stream_cb.set(_original_cb)
            logger.error(str(exc))
            return {
                "messages": [AIMessage(content=f"⛔ {exc}")],
                "routing_target": "__end__",
                "node_sessions": {self._session_key: session_id},
                "success": False,
                "abort_reason": str(exc),
            }

        # ── LLM 调用（持资源锁）────────────────────────────────────────────
        try:
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
                    history=list(msgs),
                )
        finally:
            # 恢复原始 callback
            if is_debug():
                _stream_cb.set(_original_cb)

        if is_debug():
            logger.debug(f"[{self._node_id}] raw_output_preview={raw_output[:200]!r}")
            # 记录思考内容到日志文件
            thinking_text = "".join(_thinking_chunks)
            log_node_thinking(
                node_id=self._node_id,
                thinking_text=thinking_text,
                output_text=raw_output,
            )

        # ── 路由信号检测（用注册的 SignalParser）────────────────────────────
        # 信号格式：{"route": "<node_id>", "context": "<question|background>"}
        signal = self._signal_parser.parse(raw_output)
        routing_target = signal.get("route", "") if signal else ""
        routing_context = signal.get("context", "") if signal else ""

        if routing_target:
            logger.info(f"[{self._node_id}] routing signal: target={routing_target!r}")
            result: dict = {
                "messages": [AIMessage(content=raw_output)],
                "routing_target": routing_target,
                "routing_context": routing_context,
                "node_sessions": {self._session_key: new_session_id},
            }
        else:
            result = {
                "messages": [AIMessage(content=raw_output)],
                "routing_target": "",
                "routing_context": "",
                "rollback_reason": "",
                # 注意：不在此处重置 retry_count！
                # retry_count 由 DETERMINISTIC validator 节点管理，
                # LLM 节点强制归零会破坏 validator 的重试逻辑（如 colony_coder 死循环 bug）。
                "node_sessions": {self._session_key: new_session_id},
            }

        # ── output_field 映射（子图末尾节点用）──────────────────────────────
        # 当节点配置了 output_field 时，把 LLM 输出自动写入指定 state 字段，
        # 使子图结论通过 LangGraph 原生 state 合并传回父图。
        if self._output_field and raw_output:
            result[self._output_field] = raw_output

        return result

    # ── 框架层内部方法 ──────────────────────────────────────────────────────────

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
        """用户消息前缀（含接口标识）+ @Gemini 强制咨询检测。"""
        prefix = self._cfg.get("user_msg_prefix", "")
        # 动态注入接口标识：「老板: 」→「老板(Discord): 」
        connector = state.get("connector", "")
        if connector and prefix:
            sep = prefix.find(": ")
            if sep >= 0:
                label = connector.upper()
                prefix = f"{prefix[:sep]}({label}): "
        if self._gemini_mention_re and self._gemini_mention_re.search(user_input):
            topic = self._gemini_mention_re.sub("", user_input).strip() or user_input
            logger.info(f"[{self._node_id}] @Gemini trigger: topic={topic!r}")
            return (
                f"{prefix}{user_input}\n\n"
                f"【系统指令·强制咨询】用户明确要求咨询 Gemini，关于：{topic}\n"
                f"请把这个问题组装成专业提问，"
                f"**第一行且只有第一行**输出以下 JSON，不要任何前缀或解释：\n"
                f'{{"route": "debate_brainstorm", "context": "<提炼后的问题>|<相关项目状态>"}}'
            )
        return f"{prefix}{user_input}" if prefix else user_input

    def _select_tools(self, user_input: str) -> list[str] | None:
        """tool_rules 关键词匹配后动态追加工具。node_config.tools 优先于顶层 config.tools。

        注意：node_config["tools"] = [] 表示"禁用所有工具"，
        不应 fallback 到 config.tools。仅当 key 不存在时才 fallback。
        """
        _MISSING = object()
        node_tools = self._cfg.get("tools", _MISSING)
        if node_tools is _MISSING:
            tools = list(self.config.tools)
        else:
            tools = list(node_tools or [])
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
        logger.debug(f"[llm_node] project_section sections={len(sections)}")

    return "\n\n".join(sections)


# 向后兼容别名
AgentNode = LlmNode
AgentClaudeNode = LlmNode
