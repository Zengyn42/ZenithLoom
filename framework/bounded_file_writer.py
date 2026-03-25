"""
BoundedFileWriter — 带截断保护的文件写入器

防止后台子进程死循环输出撑满磁盘。
超过 max_bytes 后静默丢弃后续写入，并在截断点写入标记。
截断不 kill 进程——kill 由 hard_timeout 负责。
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_TRUNCATION_MARKER = b"\n\n=== OUTPUT TRUNCATED (BoundedFileWriter limit reached) ===\n"


class BoundedFileWriter:
    """带 max_bytes 截断保护的文件写入器。

    超限后静默丢弃后续数据，并在截断点写入一次标记。
    线程安全：单个写入线程场景（tee 线程），无需额外加锁。
    """

    def __init__(self, path: str | Path, max_bytes: int = 50_000_000) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._written: int = 0
        self._truncated: bool = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._path, "wb")

    @property
    def path(self) -> str:
        """输出文件的绝对路径。"""
        return str(self._path)

    @property
    def truncated(self) -> bool:
        """是否已触发截断。"""
        return self._truncated

    def write(self, data: bytes) -> int:
        """写入数据。超限后静默丢弃并写一次截断标记。

        Returns:
            实际写入的字节数（截断后返回 0）。
        """
        if self._truncated:
            return 0

        remaining = self._max_bytes - self._written
        if len(data) > remaining:
            # 写入剩余可用空间
            if remaining > 0:
                self._fp.write(data[:remaining])
                self._written += remaining
            # 写截断标记
            self._fp.write(_TRUNCATION_MARKER)
            self._fp.flush()
            self._truncated = True
            logger.warning(
                f"[bounded_file_writer] output truncated at {self._written} bytes: {self._path}"
            )
            return remaining

        self._fp.write(data)
        self._written += len(data)
        return len(data)

    def close(self) -> None:
        """关闭文件句柄。"""
        if self._fp and not self._fp.closed:
            self._fp.flush()
            self._fp.close()

    def __enter__(self) -> "BoundedFileWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
