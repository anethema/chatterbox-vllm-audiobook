"""Persisted, process-wide state for the active audiobook job.

This is deliberately independent of Gradio's queue event. A browser can come
and go without affecting the authoritative server-side job state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from html import escape
import json
import math
import os
from pathlib import Path
import tempfile
from threading import RLock
import time
from typing import Any
import warnings


ACTIVE_JOB_STATUS_NAME = "active-job.json"
ACTIVE_STATES = frozenset({"running", "stopping"})
TERMINAL_STATES = frozenset({"idle", "completed", "stopped", "failed", "interrupted"})


@dataclass(frozen=True)
class JobSnapshot:
    """A serializable view of the one audiobook job the server can run."""

    state: str = "idle"
    phase: str = "idle"
    message: str = "No audiobook job is running."
    fraction: float = 0.0
    completed_chunks: int = 0
    total_chunks: int = 0
    realtime_speed: float | None = None
    eta_seconds: float | None = None
    project_id: str | None = None
    started_at: float | None = None
    updated_at: float = 0.0

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "JobSnapshot":
        if not isinstance(value, dict):
            raise ValueError("job status must be an object")
        allowed = set(cls.__dataclass_fields__)
        fields = {key: value[key] for key in allowed if key in value}
        snapshot = cls(**fields)
        if snapshot.state not in ACTIVE_STATES | TERMINAL_STATES:
            raise ValueError("unknown job state")
        if not isinstance(snapshot.phase, str) or not isinstance(snapshot.message, str):
            raise ValueError("job text fields must be strings")
        if snapshot.project_id is not None and not isinstance(snapshot.project_id, str):
            raise ValueError("project id must be a string or null")
        for name in ("fraction", "updated_at"):
            number = getattr(snapshot, name)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(snapshot.fraction) <= 1.0:
            raise ValueError("fraction is outside its allowed range")
        for name in ("started_at", "realtime_speed", "eta_seconds"):
            number = getattr(snapshot, name)
            if number is not None and (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number < 0
            ):
                raise ValueError(f"{name} must be a non-negative finite number or null")
        for name in ("completed_chunks", "total_chunks"):
            number = getattr(snapshot, name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        return snapshot


class JobStatusStore:
    """Thread-safe status with an atomic output-root checkpoint.

    If the process dies while a job is active, the next app process changes that
    checkpoint to ``interrupted`` rather than pretending it can resume work.
    """

    def __init__(self, output_root: str | Path, *, clock=time.time) -> None:
        self._root = Path(output_root)
        self._path = self._root / ACTIVE_JOB_STATUS_NAME
        self._clock = clock
        self._lock = RLock()
        self._snapshot = self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> JobSnapshot:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            snapshot = JobSnapshot.from_dict(data)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return JobSnapshot(updated_at=self._clock())
        if snapshot.active:
            snapshot = replace(
                snapshot,
                state="interrupted",
                phase="interrupted",
                message=(
                    "The previous app session ended while this job was running. "
                    "Its saved project can be resumed."
                ),
                eta_seconds=None,
                updated_at=self._clock(),
            )
            try:
                self._write(snapshot)
            except OSError as error:
                warnings.warn(
                    f"Could not update active job status checkpoint: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return snapshot

    def _write(self, snapshot: JobSnapshot) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._root,
                prefix=f".{ACTIVE_JOB_STATUS_NAME}-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(snapshot.to_dict(), temporary, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return self._snapshot

    def try_start(self, *, project_id: str | None = None) -> tuple[bool, JobSnapshot]:
        """Acquire the one active-job slot without racing another browser."""

        with self._lock:
            if self._snapshot.active:
                return False, self._snapshot
            snapshot = self.update(
                state="running",
                phase="starting",
                message="Starting audiobook job…",
                fraction=0.0,
                completed_chunks=0,
                total_chunks=0,
                realtime_speed=None,
                eta_seconds=None,
                project_id=project_id,
                started_at=self._clock(),
            )
            return True, snapshot

    def update(self, **changes: Any) -> JobSnapshot:
        """Atomically publish a partial update and return the new snapshot."""

        with self._lock:
            if "fraction" in changes and changes["fraction"] is not None:
                changes["fraction"] = max(0.0, min(1.0, float(changes["fraction"])))
            for key in ("completed_chunks", "total_chunks"):
                if key in changes and changes[key] is not None:
                    changes[key] = max(0, int(changes[key]))
            changes["updated_at"] = self._clock()
            self._snapshot = replace(self._snapshot, **changes)
            try:
                self._write(self._snapshot)
            except OSError as error:
                warnings.warn(
                    f"Could not update active job status checkpoint: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self._snapshot

    def request_stop(self) -> JobSnapshot:
        with self._lock:
            if not self._snapshot.active:
                return self._snapshot
            return self.update(
                state="stopping",
                phase="stopping",
                message="Stop requested; waiting for the current operation to finish…",
                eta_seconds=None,
            )

    def finish(self, state: str, message: str, **changes: Any) -> JobSnapshot:
        if state not in TERMINAL_STATES:
            raise ValueError(f"terminal state required, got {state!r}")
        changes.update(
            {"state": state, "phase": state, "message": message, "eta_seconds": None}
        )
        return self.update(**changes)


def render_job_status(snapshot: JobSnapshot, *, format_duration) -> str:
    """Render a compact, escaped HTML progress panel for the active job."""

    state_label = {
        "running": "🟢 Running",
        "stopping": "🟡 Stopping",
        "completed": "✅ Complete",
        "stopped": "⏹️ Stopped",
        "failed": "❌ Failed",
        "interrupted": "⚠️ Interrupted",
        "idle": "⚪ Idle",
    }.get(snapshot.state, snapshot.state.title())
    fraction = max(0.0, min(1.0, float(snapshot.fraction)))
    percentage = 100.0 * fraction
    metrics = [f"{percentage:.1f}%"]
    if snapshot.total_chunks:
        metrics.append(f"{snapshot.completed_chunks:,}/{snapshot.total_chunks:,} chunks")
    if snapshot.realtime_speed is not None:
        metrics.append(f"{snapshot.realtime_speed:.2f}× realtime")
    if snapshot.eta_seconds is not None:
        metrics.append(f"ETA {format_duration(snapshot.eta_seconds)}")
    if snapshot.project_id:
        metrics.append(f"Project: {snapshot.project_id}")

    return (
        '<section class="active-job-monitor" role="status" aria-live="polite">'
        '<div class="active-job-monitor__heading">'
        f'<strong>Job monitor — {escape(state_label)}</strong>'
        f'<span>{escape(snapshot.phase.replace("_", " ").title())}</span>'
        '</div>'
        '<div class="active-job-monitor__track" role="progressbar" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percentage:.1f}" '
        f'aria-label="Audiobook job progress: {percentage:.1f}%">'
        f'<div class="active-job-monitor__fill" style="width: {percentage:.1f}%"></div>'
        '</div>'
        f'<div class="active-job-monitor__metrics">{escape(" • ".join(metrics))}</div>'
        f'<div class="active-job-monitor__message">{escape(snapshot.message)}</div>'
        '</section>'
    )
