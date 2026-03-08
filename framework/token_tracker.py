"""
框架级 Token 统计器

进程级累计统计，供所有 Agent 共享。
"""

import logging

logger = logging.getLogger(__name__)

_token_stats = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "calls": 0,
}


def get_token_stats() -> dict:
    """返回当前进程的累计 token 使用统计。"""
    return dict(_token_stats)


def reset_token_stats() -> None:
    for k in _token_stats:
        _token_stats[k] = 0


def update_token_stats(usage) -> None:
    """
    从 SDK ResultMessage.usage 或 dict 更新统计。
    兼容 dict 和对象两种形式。
    """
    _token_stats["calls"] += 1

    if isinstance(usage, dict):
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            _token_stats[key] += usage.get(key, 0)
    else:
        # 对象形式（SDK TaskUsage）
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            _token_stats[key] += getattr(usage, key, 0)

    inp = _token_stats["input_tokens"]
    out = _token_stats["output_tokens"]
    calls = _token_stats["calls"]
    logger.info(f"[tokens] in={inp} out={out} calls={calls}")
