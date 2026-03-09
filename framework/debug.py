"""
框架级 debug 工具 — framework/debug.py

统一 DEBUG 标志读取，避免各模块重复 os.getenv("DEBUG", "").lower() in ("1", "true")。
"""

import os


def is_debug() -> bool:
    """返回当前进程是否处于 debug 模式（DEBUG=1 或 DEBUG=true）。"""
    return os.getenv("DEBUG", "").lower() in ("1", "true")
