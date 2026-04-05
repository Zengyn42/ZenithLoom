"""
框架级 Ollama LLM 节点 — framework/nodes/llm/ollama.py

OllamaNode 继承 LlmNode，实现 call_llm() 接口：
  call_llm(prompt, session_id, tools, cwd) → (text, session_id)

通过 Ollama OpenAI 兼容端点 /v1/chat/completions 调用本地模型。
该端点支持 thinking + tool_calls + streaming 同时工作。
keep_alive=-1 确保模型常驻 RAM（防止 5 分钟后自动卸载）。

entity.json 配置：
  node_config["model"]:    "qwen3.5:27b"               # 模型名
  node_config["endpoint"]: "http://localhost:11434"      # Ollama endpoint（默认）
  node_config["timeout"]:  120                           # 超时秒数（默认）
  node_config["max_iterations"]: 10                      # tool-call 循环上限（默认）
  node_config["options"]:  {}                            # 额外参数，如 temperature 等

permission_mode 实现：
  Ollama 无原生权限控制机制，通过两层软限制模拟：
    plan 模式：
      1. system_prompt 末尾注入禁写指令（_PLAN_MODE_SUFFIX）
      2. _call_with_tools() 中过滤掉 _WRITE_TOOLS
    其他模式（default / acceptEdits / bypassPermissions）：
      行为相同，正常调用。
"""

import json
import logging

import httpx

from framework.config import AgentConfig
from framework.debug import is_debug
from framework.nodes.llm.llm_node import LlmNode as AgentNode
from framework.token_guard import TokenLimitExceeded, check_before_llm

logger = logging.getLogger(__name__)


class OllamaNode(AgentNode):
    """
    Ollama LLM 节点，继承 AgentNode。

    通过 Ollama /v1/chat/completions 端点（OpenAI 兼容）调用本地模型。
    基类 AgentNode.__call__() 处理所有图协议逻辑。
    """

    def __init__(self, config: AgentConfig, node_config: dict):
        super().__init__(config, node_config)
        self._model = node_config.get("model", "llama3")
        self._endpoint = node_config.get("endpoint", "http://localhost:11434")
        self._timeout = node_config.get("timeout", 120)
        self._max_iterations = node_config.get("max_iterations", 10)
        base_prompt: str = node_config.get("system_prompt", "")
        skill_content = self._load_skill_content()
        self._system_prompt = f"{base_prompt}\n\n{skill_content}" if skill_content else base_prompt
        self._options = node_config.get("options", {})
        self._tools: list = node_config.get("tools", [])
        self._node_config = node_config
        logger.info(f"[ollama] model={self._model} endpoint={self._endpoint} "
                     f"max_iter={self._max_iterations} options={self._options}")

    async def __call__(self, state: dict) -> dict:
        if self._tools or self._has_dynamic_tools():
            return await self._call_with_tools(state)
        return await super().__call__(state)

    @staticmethod
    def _has_dynamic_tools() -> bool:
        """检查 TOOL_REGISTRY 中是否存在动态注册的工具（如 heartbeat_*）。"""
        from framework.nodes.llm.tools import TOOL_REGISTRY
        return any(k.startswith("heartbeat_") for k in TOOL_REGISTRY)

    # ── /v1/chat/completions 流式调用 ────────────────────────────────

    async def _chat_completions(
        self, messages: list, tools: list | None = None
    ) -> dict:
        """流式 POST to /v1/chat/completions（OpenAI 兼容端点）。

        返回格式：
          {"role": "assistant", "content": "...", "tool_calls": [...]}

        优势：thinking（reasoning）+ tool_calls + streaming 同时工作。
        """
        from framework.nodes.llm.llm_node import get_stream_callback

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "keep_alive": -1,
        }
        if tools:
            payload["tools"] = tools

        # 额外 options 映射到 OpenAI 格式的顶层字段
        if "temperature" in self._options:
            payload["temperature"] = self._options["temperature"]
        if "num_predict" in self._options:
            payload["max_tokens"] = self._options["num_predict"]

        stream_cb = get_stream_callback()
        full_content = ""
        tool_calls_map: dict[int, dict] = {}  # index → {id, function: {name, arguments}}

        timeout_cfg = httpx.Timeout(self._timeout, connect=30)
        async with httpx.AsyncClient(timeout=timeout_cfg) as client:
            async with client.stream(
                "POST",
                f"{self._endpoint}/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {body.decode()[:500]}",
                        request=response.request,
                        response=response,
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    # Reasoning (thinking) — stream to callback
                    reasoning = delta.get("reasoning", "")
                    if reasoning and stream_cb is not None:
                        stream_cb(reasoning, True)

                    # Content
                    token = delta.get("content", "")
                    if token:
                        full_content += token
                        if stream_cb is not None:
                            stream_cb(token, False)

                    # Tool calls — streamed incrementally by index
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.get("id", f"call_{idx}"),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_calls_map[idx]["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]

        # Assemble result
        result: dict = {"role": "assistant", "content": full_content}
        if tool_calls_map:
            result["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
        return result

    # ── Multi-turn tool-calling loop ─────────────────────────────────

    async def _call_with_tools(self, state: dict) -> dict:
        """Multi-turn Ollama tool-calling loop via /v1/chat/completions.

        Session management:
          - ollama_sessions: dict[uuid → messages list]
          - node_sessions:   dict[session_key → uuid]
          - Max 200 messages per session (prune oldest non-system messages).
        """
        import uuid as _uuid
        from framework.nodes.llm.tools import TOOL_REGISTRY, build_tool_schemas

        # tool_rules 关键词匹配（与基类 __call__ 中的 _select_tools 一致）
        lm = (state.get("messages") or [])
        last = lm[-1] if lm else None
        latest_input = (getattr(last, "content", None) or last.get("content", "") if last else "") or ""
        effective_tools = self._select_tools(latest_input) or []

        # 自动发现动态注册的工具（heartbeat_* 等）
        dynamic = [k for k in TOOL_REGISTRY if k.startswith("heartbeat_") and k not in effective_tools]
        if dynamic:
            effective_tools.extend(dynamic)
            if is_debug():
                logger.debug(f"[ollama] dynamic tools discovered: {dynamic}")
        tool_schemas = build_tool_schemas(effective_tools)
        session_key = self._node_config.get("session_key", self._node_config.get("id", ""))

        # Load or init session messages
        ollama_sessions: dict = dict(state.get("ollama_sessions") or {})
        session_uuid = (state.get("node_sessions") or {}).get(session_key)
        messages: list = list(ollama_sessions.get(session_uuid, [])) if session_uuid else []

        # System prompt（plan 模式追加禁写指令）
        if self._system_prompt and not any(m.get("role") == "system" for m in messages):
            sys_content = self._system_prompt
            if self.is_plan_mode:
                sys_content = sys_content + self._PLAN_MODE_SUFFIX
            messages.insert(0, {"role": "system", "content": sys_content})

        # Append current user message
        lm = (state.get("messages") or [])
        if lm:
            last = lm[-1]
            content = getattr(last, "content", None) or (last.get("content", "") if isinstance(last, dict) else "")
            if content:
                messages.append({"role": "user", "content": content})

        MAX_MESSAGES = 200
        terminal_result = None
        last_msg = {}

        # ── Token 安全阀（首轮检查）──
        try:
            check_before_llm(messages=messages, node_id=self._node_id, limit=self._token_limit)
        except TokenLimitExceeded as exc:
            logger.error(str(exc))
            from langchain_core.messages import AIMessage
            return {
                "messages": [AIMessage(content=f"⛔ {exc}")],
                "routing_target": "__end__",
                "success": False,
                "abort_reason": str(exc),
            }

        for iteration in range(self._max_iterations):
            if iteration > 0:
                try:
                    check_before_llm(messages=messages, node_id=self._node_id, limit=self._token_limit)
                except TokenLimitExceeded as exc:
                    logger.error(f"Tool loop iteration {iteration}: {exc}")
                    from langchain_core.messages import AIMessage
                    return {
                        "messages": [AIMessage(content=f"⛔ {exc}")],
                        "routing_target": "__end__",
                        "success": False,
                        "abort_reason": str(exc),
                    }

            logger.info(f"[ollama] iteration {iteration + 1}/{self._max_iterations} "
                        f"msgs={len(messages)}")

            # 始终传 tools — /v1/ 端点支持 tool result + tools 同传
            try:
                last_msg = await self._chat_completions(messages, tools=tool_schemas)
            except httpx.ConnectError as e:
                err = (f"[Ollama 连接失败] 无法连接到 {self._endpoint}，"
                       f"请确认 Ollama 正在运行。({e})")
                logger.error(err)
                from langchain_core.messages import AIMessage
                return {"messages": [AIMessage(content=err)]}
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                err = f"[Ollama 超时] 推理超过 {self._timeout}s 未返回，请重试或简化问题。"
                logger.error(f"{err} ({type(e).__name__}: {e})")
                from langchain_core.messages import AIMessage
                return {"messages": [AIMessage(content=err)]}
            messages.append(last_msg)

            tool_calls = last_msg.get("tool_calls") or []
            if not tool_calls:
                break  # text response — loop ends

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"].get("arguments", {})
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        fn_args = {"raw": fn_args}

                logger.info(f"[ollama] tool call: {fn_name}({list(fn_args.keys())})")
                tool_fn = TOOL_REGISTRY.get(fn_name)
                if tool_fn:
                    tool_result = await tool_fn(**fn_args)
                else:
                    tool_result = {"error": f"unknown tool: {fn_name}"}

                # /v1/ 端点需要 tool_call_id 来关联 tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": str(tool_result),
                })

                if tool_result.get("_terminal"):
                    terminal_result = {k: v for k, v in tool_result.items() if k != "_terminal"}
                    break

            if terminal_result:
                break

        # Prune session to MAX_MESSAGES
        if len(messages) > MAX_MESSAGES:
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            non_sys  = [m for m in messages if m.get("role") != "system"]
            messages = sys_msgs + non_sys[-(MAX_MESSAGES - len(sys_msgs)):]

        # Persist session
        if not session_uuid:
            session_uuid = str(_uuid.uuid4())
        ollama_sessions[session_uuid] = messages

        updates: dict = {
            "node_sessions": {session_key: session_uuid},
            "ollama_sessions": ollama_sessions,
        }
        if terminal_result:
            updates["validation_output"] = terminal_result
        else:
            text = last_msg.get("content", "")
            if text:
                from langchain_core.messages import AIMessage
                updates["messages"] = [AIMessage(content=text)]

        return updates

    # plan 模式下追加到 system_prompt 的禁写指令
    _PLAN_MODE_SUFFIX = (
        "\n\n⛔ 你当前处于「规划模式」。绝对禁止：\n"
        "- 创建、修改、删除任何文件或代码\n"
        "- 执行任何 shell 命令\n"
        "- 调用任何写入类工具（Write, Edit, Bash 等）\n"
        "你只能进行纯文本讨论、分析和规划。"
    )

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
        history: list | None = None,
    ) -> tuple[str, str]:
        """
        调用 Ollama /v1/chat/completions。返回 (text, session_id)。

        session_id 语义：Ollama 无持久 session，返回传入值不变。
        history: 完整 LangGraph 消息历史，用于重建多轮对话上下文。
        keep_alive=-1：模型常驻 RAM，防止 5 分钟后自动卸载。
        """
        if is_debug():
            logger.debug(f"[ollama] model={self._model} prompt_len={len(prompt)} "
                         f"history_len={len(history) if history else 0}")

        messages = []
        # plan 模式：system_prompt 追加禁写指令
        sys_prompt = self._system_prompt
        if self.is_plan_mode and sys_prompt:
            sys_prompt = sys_prompt + self._PLAN_MODE_SUFFIX
        elif self.is_plan_mode:
            sys_prompt = self._PLAN_MODE_SUFFIX.strip()
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # 多轮对话历史（跳过最后一条，即当前 prompt 的 HumanMessage）
        if history and len(history) > 1:
            for msg in history[:-1]:
                msg_type = getattr(msg, "type", "")
                role = "user" if msg_type == "human" else "assistant"
                content = msg.content if isinstance(msg.content, str) else ""
                if content.strip():
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})

        # ── Token 安全阀 ──
        check_before_llm(messages=messages, node_id=self._node_id, limit=self._token_limit)

        try:
            resp = await self._chat_completions(messages)
            full_text = resp.get("content", "")
        except httpx.ConnectError as e:
            full_text = (
                f"[Ollama 连接失败] 无法连接到 {self._endpoint}，"
                f"请确认 Ollama 正在运行。({e})"
            )
            logger.error(full_text)
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            full_text = f"[Ollama 错误] {e}"
            logger.error(full_text)

        return full_text, session_id

    def get_recent_history(self, session_id: str, limit: int = 10) -> list:
        """Ollama 无持久 session，返回空列表。"""
        return []


# Backward compatibility alias — builtins.py imports LlamaNode directly from this module
LlamaNode = OllamaNode
