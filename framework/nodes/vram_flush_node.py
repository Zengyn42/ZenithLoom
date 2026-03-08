"""
框架级 GPU 清洗节点 — VramFlushNode

可插入任何 agent 图的末尾或错误处理路径。
无 GPU 环境时静默跳过。

前提：宿主机需配置 sudo 免密（仅限 fuser 命令）。
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


class VramFlushNode:
    """扫描并强制清理所有占用 GPU 的残留进程。"""

    def __call__(self, state: dict) -> dict:
        logger.info("[vram_flush] 开始扫描 GPU 占用...")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug("[vram_flush] nvidia-smi 不可用，跳过")
                return {}

            pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not pids:
                logger.info("[vram_flush] GPU 干净，无残留进程")
                return {}

            logger.warning(
                f"[vram_flush] 检测到 {len(pids)} 个 GPU 残留进程: {pids}"
            )

            kill_result = subprocess.run(
                ["sudo", "fuser", "-k", "-9", "/dev/nvidia*"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if kill_result.returncode == 0:
                logger.info("[vram_flush] GPU 大清洗完成")
            else:
                logger.error(
                    f"[vram_flush] 大清洗失败: {kill_result.stderr.strip()}"
                )

        except FileNotFoundError:
            logger.debug("[vram_flush] nvidia-smi / fuser 不存在，跳过")
        except subprocess.TimeoutExpired:
            logger.error("[vram_flush] nvidia-smi 超时，跳过本次清洗")

        return {}
