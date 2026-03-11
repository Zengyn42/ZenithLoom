"""
框架级资源锁 — framework/resource_lock.py

基于 asyncio.Semaphore 的资源互斥锁，防止多节点同时占用同一物理资源（GPU/CPU）。

用法：
    async with acquire_resource("GPU_0_VRAM_22GB", holder="worker_llama"):
        text, sid = await self.call_llm(prompt, ...)

node_config 配置：
    "resource_lock": "GPU_0_VRAM_22GB"   # 需要持锁的资源名
    "resource_timeout": 600              # 超时秒数（默认 300）

超时后抛 RuntimeError，图可通过 on_error 条件边处理，不会无限等待。
!resources 命令调用 get_resource_status() 展示当前状态。
"""

import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# resource_name → asyncio.Semaphore
_SEMAPHORES: dict[str, asyncio.Semaphore] = {}

# resource_name → 当前持有者 node_id（acquire 成功后写入，finally 清除）
_LOCK_HOLDERS: dict[str, str] = {}

_DEFAULT_TIMEOUT = 300.0


def register_resource(name: str, concurrency: int = 1) -> None:
    """注册资源及其最大并发数（已注册的资源不会被覆盖）。"""
    if name not in _SEMAPHORES:
        _SEMAPHORES[name] = asyncio.Semaphore(concurrency)
        logger.debug(f"[resource_lock] registered {name!r} (concurrency={concurrency})")


# 内置资源（模块加载时注册）
register_resource("GPU_0_VRAM_22GB", concurrency=1)
register_resource("GPU_1_VRAM_22GB", concurrency=1)
register_resource("SYSTEM_CPU", concurrency=4)


@asynccontextmanager
async def acquire_resource(
    name: str | None,
    timeout: float = _DEFAULT_TIMEOUT,
    holder: str = "unknown",
):
    """
    异步上下文管理器：持有命名资源锁期间执行代码块。

    - name=None 或 name 未注册：直接执行，不加锁
    - 超时（timeout 秒内未能获取锁）：抛 RuntimeError，包含当前持有者信息
    - asyncio 单线程安全：_LOCK_HOLDERS 写入在 acquire 成功后，读取无竞争
    """
    if name is None or name not in _SEMAPHORES:
        yield
        return

    sem = _SEMAPHORES[name]
    try:
        await asyncio.wait_for(sem.acquire(), timeout=timeout)
    except asyncio.TimeoutError:
        current_holder = _LOCK_HOLDERS.get(name)
        raise RuntimeError(
            f"Resource lock timeout ({timeout}s): {name!r} "
            f"held by {current_holder!r}"
        )

    # acquire 成功后安全写入 holder
    _LOCK_HOLDERS[name] = holder
    logger.debug(f"[resource_lock] {name!r} acquired by {holder!r}")
    try:
        yield
    finally:
        _LOCK_HOLDERS.pop(name, None)
        sem.release()
        logger.debug(f"[resource_lock] {name!r} released by {holder!r}")


def get_resource_status() -> dict[str, dict]:
    """
    返回所有注册资源的当前状态，供 !resources 命令展示。

    返回格式：
    {
        "GPU_0_VRAM_22GB": {"holder": "worker_llama", "available": 0, "capacity": 1},
        "GPU_1_VRAM_22GB": {"holder": None,           "available": 1, "capacity": 1},
        ...
    }
    """
    return {
        name: {
            "holder": _LOCK_HOLDERS.get(name),
            "available": sem._value,
            "capacity": sem._value + (1 if name in _LOCK_HOLDERS else 0),
        }
        for name, sem in _SEMAPHORES.items()
    }


def format_resource_status() -> str:
    """格式化资源状态为可读字符串，供 !resources 命令直接输出。"""
    status = get_resource_status()
    if not status:
        return "No resources registered."
    lines = []
    for name, info in status.items():
        holder = info["holder"]
        capacity = info["capacity"]
        if holder:
            lines.append(f"  {name:<24} BUSY  → {holder}  (capacity={capacity})")
        else:
            lines.append(f"  {name:<24} FREE               (capacity={capacity})")
    return "\n".join(lines)
