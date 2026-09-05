"""Progress calculations shared by long-running audiobook jobs."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event


@dataclass(frozen=True)
class ProgressEstimate:
    realtime_speed: float
    eta_seconds: float


class GenerationControl:
    """Thread-safe cooperative stop state for the single queued EPUB job."""

    def __init__(self) -> None:
        self._active = Event()
        self._stop_requested = Event()

    def begin(self) -> None:
        """Start a new job with any previous stop request cleared."""

        self._stop_requested.clear()
        self._active.set()

    def finish(self) -> None:
        """Clear active and stop state after job cleanup."""

        self._active.clear()
        self._stop_requested.clear()

    def request_stop(self) -> bool:
        """Request cancellation only while a job is active."""

        if not self._active.is_set():
            return False
        self._stop_requested.set()
        return True

    def stop_requested(self) -> bool:
        """Read the cooperative stop flag without blocking."""

        return self._stop_requested.is_set()


def estimate_progress(
    generated_audio_seconds: float,
    elapsed_seconds: float,
    completed_characters: int,
    total_characters: int,
) -> ProgressEstimate:
    """Estimate generation speed and remaining wall time from completed text."""

    elapsed = max(float(elapsed_seconds), 1e-9)
    completed = max(0, int(completed_characters))
    total = max(completed, int(total_characters))
    realtime_speed = max(0.0, float(generated_audio_seconds)) / elapsed
    if completed == 0:
        eta_seconds = 0.0
    else:
        eta_seconds = elapsed * (total - completed) / completed
    return ProgressEstimate(realtime_speed, max(0.0, eta_seconds))


def format_duration(seconds: float) -> str:
    """Format nonnegative elapsed seconds as a compact human-readable duration."""

    total_seconds = max(0, round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"
