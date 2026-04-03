import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from framework.token_guard import check_before_llm, TokenLimitExceeded
from framework.registry import get_node_factory

logger = logging.getLogger(__name__)

class RetryableError(Exception):
    pass

class FatalError(Exception):
    pass

@dataclass
class ResilienceTrace:
    attempted_providers: List[str]
    final_provider: str
    degraded: bool
    circuit_states: Dict[str, str]
    latency_ms: int
    error_chain: List[str]
    
    def to_dict(self):
        return {
            "attempted_providers": self.attempted_providers,
            "final_provider": self.final_provider,
            "degraded": self.degraded,
            "circuit_states": self.circuit_states,
            "latency_ms": self.latency_ms,
            "error_chain": self.error_chain
        }

def is_retryable(e: Exception) -> bool:
    import httpx
    if isinstance(e, httpx.TimeoutException): return True
    if isinstance(e, httpx.HTTPStatusError):
        # 429 Too Many Requests, 5xx server errors
        if e.response.status_code in (429, 500, 502, 503, 504): return True
        return False
    
    err_str = str(e).lower()
    keywords = ["capacity", "429", "quota", "exhausted", "overloaded", "rate limit", "timeout", "bad gateway"]
    if any(k in err_str for k in keywords):
        return True
    return False

async def invoke_with_resilience(
    primary_node,
    prompt: str,
    session_id: str,
    tools: list[str] | None,
    cwd: str | None,
    history: list | None
) -> tuple[str, str, ResilienceTrace | None]:
    start_time = time.monotonic()
    
    trace = ResilienceTrace(
        attempted_providers=[primary_node._node_id],
        final_provider=primary_node._node_id,
        degraded=False,
        circuit_states={},
        latency_ms=0,
        error_chain=[]
    )
    
    try:
        reply, new_sid = await primary_node.call_llm(
            prompt, session_id=session_id, tools=tools, cwd=cwd, history=history
        )
        trace.latency_ms = int((time.monotonic() - start_time) * 1000)
        return reply, new_sid, trace
    except Exception as e:
        if not is_retryable(e):
            logger.error(f"[resilience] 致命错误 (Fatal): {e}")
            raise e
        
        trace.error_chain.append(f"{primary_node._node_id}: {str(e)}")
        logger.warning(f"[resilience] {primary_node._node_id} 发生可重试错误: {e}")

    # Fallback loop
    fallback_configs = primary_node._node_config.get("fallback_providers", [])
    if not fallback_configs:
        raise RuntimeError(f"主节点 {primary_node._node_id} 失败且未配置降级链: {trace.error_chain[0]}")

    from framework.nodes.llm.llm_node import get_stream_callback
    stream_cb = get_stream_callback()
    
    for fb_conf in fallback_configs:
        fb_type = fb_conf.get("type")
        if not fb_type:
            continue
            
        fb_id = fb_conf.get("id", f"fallback_{fb_type.lower()}")
        trace.attempted_providers.append(fb_id)
        trace.degraded = True
        
        # Instantiate fallback node
        try:
            factory = get_node_factory(fb_type)
            # Merge primary config with fallback specific config
            merged_config = {**primary_node._node_config, **fb_conf}
            # Remove output_field and fallback_providers to avoid recursion
            merged_config.pop("fallback_providers", None)
            merged_config.pop("output_field", None)
            fb_node = factory(primary_node._cfg, merged_config)
        except Exception as e:
            trace.error_chain.append(f"{fb_id} (init failed): {e}")
            continue

        # 1. Token Guard Skip
        try:
            # Base class passes `history[:-1]` for guard
            _guard_history = list(history) if history else []
            if _guard_history: _guard_history = _guard_history[:-1]
            check_before_llm(prompt=prompt, history=_guard_history, node_id=fb_id, limit=fb_node._token_limit)
        except TokenLimitExceeded as exc:
            trace.error_chain.append(f"{fb_id} (Token Skip): {exc}")
            logger.warning(f"[resilience] 跳过 {fb_id}: {exc}")
            continue

        # 2. Capability Check
        current_tools = tools
        current_prompt = prompt
        if tools:
            # Simple heuristic for tool capability
            fb_supports_tools = fb_type in ("CLAUDE_SDK", "CLAUDE_CLI", "GEMINI_CLI") or (fb_type == "OLLAMA" and getattr(fb_node, "_tools", None))
            if not fb_supports_tools:
                current_tools = None
                warning_msg = f"[⚠️ 工具能力受限，已降级为纯文本模式 (目标: {fb_id})]\\n\\n"
                if stream_cb:
                    stream_cb(warning_msg, False)
                # Append warning to prompt so LLM knows
                current_prompt = warning_msg + current_prompt

        # Push UI warning for fallback
        warning_msg = f"\\n[⏳ 模型响应超时/错误，正在为您无缝切换至 {fb_id}...]\\n"
        if stream_cb:
            stream_cb(warning_msg, False)
            
        try:
            reply, new_sid = await fb_node.call_llm(
                current_prompt, session_id=session_id, tools=current_tools, cwd=cwd, history=history
            )
            trace.final_provider = fb_id
            trace.latency_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(f"[resilience] 降级成功，最终使用: {fb_id}")
            return reply, new_sid, trace
        except Exception as e:
            if not is_retryable(e):
                logger.error(f"[resilience] 降级节点 {fb_id} 发生致命错误: {e}")
                raise e
            trace.error_chain.append(f"{fb_id}: {str(e)}")
            logger.warning(f"[resilience] 降级节点 {fb_id} 失败: {e}")
            continue

    trace.latency_ms = int((time.monotonic() - start_time) * 1000)
    raise RuntimeError(f"所有可用 LLM 节点均失败: {trace.error_chain}")
