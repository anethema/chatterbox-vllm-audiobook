"""Small Linux memory-pressure helpers used by long audiobook jobs."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import gc
from pathlib import Path


@dataclass(frozen=True)
class MemoryStatus:
    available_bytes: int
    swap_free_bytes: int

    @property
    def headroom_bytes(self) -> int:
        return self.available_bytes + self.swap_free_bytes


def read_memory_status(path: str | Path = "/proc/meminfo") -> MemoryStatus:
    """Read available RAM and unused swap in bytes from Linux meminfo."""

    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name, separator, remainder = line.partition(":")
        if separator and name in {"MemAvailable", "SwapFree"}:
            values[name] = int(remainder.split()[0]) * 1024
    if "MemAvailable" not in values or "SwapFree" not in values:
        raise RuntimeError("Could not read Linux memory availability")
    return MemoryStatus(values["MemAvailable"], values["SwapFree"])


def release_unused_memory() -> None:
    """Collect Python objects and ask glibc to return free heap pages."""

    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass
