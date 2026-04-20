"""
框架级 Grok LLM 节点 — framework/nodes/llm/grok.py

GrokNode 继承自 LlmNode，作为纯正的大模型节点。
在底层使用 ChromeExecutor 工具类进行解耦的 Playwright 通信交互。
"""

import logging
import os

from framework.config import AgentConfig
from framework.nodes.llm.llm_node import LlmNode, _stream_cb
from framework.token_guard import check_before_llm
from framework.utils.chrome_executor import ChromeExecutor

logger = logging.getLogger(__name__)


class GrokNode(LlmNode):
    """
    Grok LLM 节点，继承自 LlmNode。
    内部使用 ChromeExecutor 处理浏览器自动化逻辑。
    """

    _PLAN_MODE_SUFFIX = (
        "\n\n⛔ 你当前处于「规划模式」。绝对禁止：\n"
        "- 创建、修改、删除任何文件或代码\n"
        "- 执行任何 shell 命令\n"
        "- 调用任何写入类工具（Write, Edit, Bash 等）\n"
        "你只能进行纯文本讨论、分析和规划。"
    )

    def __init__(self, config: AgentConfig, node_config: dict, system_prompt: str = ""):
        super().__init__(config, node_config)
        # 解析配置：优先从嵌套的 node_config 取，其次从顶层取
        real_config = node_config.get("node_config", {})
        self._model = node_config.get("model") or real_config.get("model", "grok")
        self._timeout = real_config.get("timeout") or node_config.get("timeout", 180)
        
        base_prompt: str = node_config.get("system_prompt", system_prompt)
        skill_content = self._load_skill_content()
        self._system_prompt = f"{base_prompt}\n\n{skill_content}" if skill_content else base_prompt
        
        # 运行时数据隔离：从图配置读取 chrome profile
        self._chrome_profile_dir = real_config.get("chrome_profile_dir") or node_config.get("chrome_profile_dir")
        
        # 实例化执行器
        bridge_script = os.path.join(
            os.path.dirname(__file__),
            "../../utils/bridges/grok_playwright_bridge.py"
        )
        self._executor = ChromeExecutor(
            script_path=bridge_script,
            timeout=self._timeout,
            model_name=self._model,
            user_data_dir=self._chrome_profile_dir
        )

        logger.info(f"[grok] initialized via ChromeExecutor (profile={self._chrome_profile_dir})")

    async def call_llm(
        self,
        prompt: str,
        session_id: str = "",
        tools: list[str] | None = None,
        cwd: str | None = None,
        history: list | None = None,
        inherit_from: str = "",
    ) -> tuple[str, str]:
        
        sys_prompt = self._system_prompt
        if self.is_plan_mode and sys_prompt:
            sys_prompt = sys_prompt + self._PLAN_MODE_SUFFIX
        elif self.is_plan_mode:
            sys_prompt = self._PLAN_MODE_SUFFIX.strip()

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        if history and len(history) > 1:
            for msg in history[:-1]:
                msg_type = getattr(msg, "type", "")
                role = "user" if msg_type == "human" else "assistant"
                content = msg.content if isinstance(msg.content, str) else ""
                if content.strip():
                    messages.append({"role": role, "content": content})
                    
        messages.append({"role": "user", "content": prompt})

        print(f"DEBUG: GrokNode.call_llm started for node {self._node_id}", flush=True)
        # Token 安全检查
        check_before_llm(messages=messages, node_id=self._node_id, limit=self._token_limit)
        print("DEBUG: Token check passed", flush=True)

        # 只发送最新的消息，因为 Grok 网页端已经有历史记录了
        full_prompt = prompt
        
        # 如果有 System Prompt，则作为前缀注入（或者在首轮发送）
        if sys_prompt and not session_id:
            full_prompt = f"{sys_prompt}\n\n{prompt}"
        
        print(f"DEBUG: Prepared prompt (len={len(full_prompt)})", flush=True)

        # 交给 Executor 执行
        cb = _stream_cb.get()
        print("DEBUG: Calling executor.execute...", flush=True)
        return await self._executor.execute(
            prompt=full_prompt,
            session_id=session_id,
            cwd=cwd,
            stream_cb=cb
        )
