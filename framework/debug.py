"""
框架级 debug 工具 — framework/debug.py

统一 DEBUG 标志，由 main.py 通过 set_debug() 在启动时设置。
"""

_DEBUG: bool = False


def set_debug(value: bool) -> None:
    """由 main.py 在解析 --debug 参数后调用。"""
    global _DEBUG
    _DEBUG = bool(value)


def is_debug() -> bool:
    """返回当前进程是否处于 debug 模式。"""
    return _DEBUG
