"""
SYSTEM_STATS 节点 — 系统资源采集节点，不继承 AgentNode。

采集 CPU 负载、内存、Swap、磁盘、GPU 等指标，
将结果以 AIMessage 写入 state["messages"]。

支持阈值告警：当指标超过阈值时，在返回内容中标注 level=warning。
节点本身不负责推送——推送由 HeartbeatManager 的 on_complete 回调统一处理。

node_config 字段：
  interval_hours   float  采集频率（由 HeartbeatManager 管理）
  thresholds       dict   阈值配置（可选），默认：
    mem_percent      float  内存使用率告警阈值（默认 90）
    swap_percent     float  Swap 使用率告警阈值（默认 80）
    disk_percent     float  磁盘使用率告警阈值（默认 90）
    gpu_temp         int    GPU 温度告警阈值（默认 85）
    gpu_mem_percent  float  GPU 显存使用率告警阈值（默认 95）
"""

import asyncio
import logging
import os
import shutil

from langchain_core.messages import AIMessage

from framework.debug import is_debug

logger = logging.getLogger(__name__)

# 默认阈值
_DEFAULT_THRESHOLDS = {
    "mem_percent": 90.0,
    "swap_percent": 80.0,
    "disk_percent": 90.0,
    "gpu_temp": 85,
    "gpu_mem_percent": 95.0,
}


class SystemStatsNode:
    """系统资源采集节点，实现标准 LangGraph node 接口 __call__(state) -> dict。"""

    def __init__(self, node_config: dict):
        self._thresholds = {**_DEFAULT_THRESHOLDS, **node_config.get("thresholds", {})}

    async def __call__(self, state: dict) -> dict:
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, self._collect)

        if is_debug():
            logger.debug(f"[system_stats] collected: {stats}")

        # 判定告警级别
        warnings = self._check_thresholds(stats)
        level = "warning" if warnings else "info"

        # 格式化输出
        report = self._format_report(stats, warnings, level)
        logger.info(f"[system_stats] level={level} warnings={len(warnings)}")

        return {"messages": [AIMessage(content=report)]}

    # ── 采集 ──────────────────────────────────────────────────────────────

    def _collect(self) -> dict:
        """同步采集所有系统指标，返回结构化 dict。"""
        stats = {}

        # CPU
        stats["cpu"] = self._collect_cpu()

        # Memory
        stats["memory"] = self._collect_memory()

        # Disk
        stats["disk"] = self._collect_disk()

        # GPU
        stats["gpu"] = self._collect_gpu()

        # Uptime
        stats["uptime"] = self._collect_uptime()

        return stats

    def _collect_cpu(self) -> dict:
        """采集 CPU 信息：核心数 + 负载。"""
        result = {"cores": os.cpu_count() or 0, "load_1m": 0, "load_5m": 0, "load_15m": 0}
        try:
            load = os.getloadavg()
            result["load_1m"] = round(load[0], 2)
            result["load_5m"] = round(load[1], 2)
            result["load_15m"] = round(load[2], 2)
        except (OSError, AttributeError):
            pass
        return result

    def _collect_memory(self) -> dict:
        """从 /proc/meminfo 采集内存和 Swap。"""
        result = {
            "total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0,
            "swap_total_gb": 0, "swap_used_gb": 0, "swap_percent": 0,
        }
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        info[key] = int(parts[1])  # kB

            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            used = total - available

            result["total_gb"] = round(total / 1048576, 1)
            result["available_gb"] = round(available / 1048576, 1)
            result["used_gb"] = round(used / 1048576, 1)
            result["percent"] = round(used / total * 100, 1) if total else 0

            swap_total = info.get("SwapTotal", 0)
            swap_free = info.get("SwapFree", 0)
            swap_used = swap_total - swap_free

            result["swap_total_gb"] = round(swap_total / 1048576, 1)
            result["swap_used_gb"] = round(swap_used / 1048576, 1)
            result["swap_percent"] = round(swap_used / swap_total * 100, 1) if swap_total else 0

        except Exception as e:
            logger.warning(f"[system_stats] memory collection failed: {e}")
        return result

    def _collect_disk(self) -> dict:
        """采集根分区磁盘使用率。"""
        result = {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0}
        try:
            usage = shutil.disk_usage("/")
            result["total_gb"] = round(usage.total / (1024 ** 3), 1)
            result["used_gb"] = round(usage.used / (1024 ** 3), 1)
            result["free_gb"] = round(usage.free / (1024 ** 3), 1)
            result["percent"] = round(usage.used / usage.total * 100, 1) if usage.total else 0
        except Exception as e:
            logger.warning(f"[system_stats] disk collection failed: {e}")
        return result

    def _collect_gpu(self) -> list[dict]:
        """通过 nvidia-smi 采集 GPU 指标。返回 GPU 列表。"""
        import subprocess

        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return []

            gpus = []
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    mem_used = int(parts[2])
                    mem_total = int(parts[3])
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "mem_used_mib": mem_used,
                        "mem_total_mib": mem_total,
                        "mem_percent": round(mem_used / mem_total * 100, 1) if mem_total else 0,
                        "utilization": int(parts[4]),
                        "temperature": int(parts[5]),
                    })
            return gpus
        except Exception:
            return []

    def _collect_uptime(self) -> str:
        """从 /proc/uptime 读取运行时间。"""
        try:
            with open("/proc/uptime") as f:
                seconds = float(f.read().split()[0])
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            minutes = int((seconds % 3600) // 60)
            if days > 0:
                return f"{days}d {hours}h {minutes}m"
            return f"{hours}h {minutes}m"
        except Exception:
            return "unknown"

    # ── 阈值检查 ──────────────────────────────────────────────────────────

    def _check_thresholds(self, stats: dict) -> list[str]:
        """检查阈值，返回告警描述列表。"""
        warnings = []
        th = self._thresholds

        mem = stats.get("memory", {})
        if mem.get("percent", 0) >= th["mem_percent"]:
            warnings.append(f"内存使用 {mem['percent']}% ≥ {th['mem_percent']}%")

        if mem.get("swap_percent", 0) >= th["swap_percent"]:
            warnings.append(f"Swap 使用 {mem['swap_percent']}% ≥ {th['swap_percent']}%")

        disk = stats.get("disk", {})
        if disk.get("percent", 0) >= th["disk_percent"]:
            warnings.append(f"磁盘使用 {disk['percent']}% ≥ {th['disk_percent']}%")

        for gpu in stats.get("gpu", []):
            if gpu.get("temperature", 0) >= th["gpu_temp"]:
                warnings.append(f"GPU{gpu['index']} 温度 {gpu['temperature']}°C ≥ {th['gpu_temp']}°C")
            if gpu.get("mem_percent", 0) >= th["gpu_mem_percent"]:
                warnings.append(
                    f"GPU{gpu['index']} 显存 {gpu['mem_percent']}% ≥ {th['gpu_mem_percent']}%"
                )

        return warnings

    # ── 格式化 ────────────────────────────────────────────────────────────

    def _format_report(self, stats: dict, warnings: list[str], level: str) -> str:
        """格式化为结构化文本报告。"""
        lines = [f"[SYSTEM_STATS] level={level}"]

        # CPU
        cpu = stats.get("cpu", {})
        lines.append(
            f"CPU: {cpu.get('cores', '?')} cores | "
            f"load: {cpu.get('load_1m', '?')}/{cpu.get('load_5m', '?')}/{cpu.get('load_15m', '?')}"
        )

        # Memory
        mem = stats.get("memory", {})
        lines.append(
            f"MEM: {mem.get('used_gb', '?')}G / {mem.get('total_gb', '?')}G "
            f"({mem.get('percent', '?')}%) | "
            f"avail: {mem.get('available_gb', '?')}G"
        )
        lines.append(
            f"SWAP: {mem.get('swap_used_gb', '?')}G / {mem.get('swap_total_gb', '?')}G "
            f"({mem.get('swap_percent', '?')}%)"
        )

        # Disk
        disk = stats.get("disk", {})
        lines.append(
            f"DISK(/): {disk.get('used_gb', '?')}G / {disk.get('total_gb', '?')}G "
            f"({disk.get('percent', '?')}%) | "
            f"free: {disk.get('free_gb', '?')}G"
        )

        # GPU
        for gpu in stats.get("gpu", []):
            lines.append(
                f"GPU{gpu['index']}({gpu['name']}): "
                f"mem {gpu['mem_used_mib']}M/{gpu['mem_total_mib']}M ({gpu['mem_percent']}%) | "
                f"util {gpu['utilization']}% | temp {gpu['temperature']}°C"
            )

        # Uptime
        lines.append(f"UPTIME: {stats.get('uptime', 'unknown')}")

        # Warnings
        if warnings:
            lines.append("--- WARNINGS ---")
            for w in warnings:
                lines.append(f"⚠ {w}")

        return "\n".join(lines)
