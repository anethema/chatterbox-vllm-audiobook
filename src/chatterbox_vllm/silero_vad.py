"""Conservative Silero VAD no-speech-gap detection.

This module intentionally does not decide how a detected range is repaired.
Keeping it independent from ``audio.py`` avoids coupling a model-backed VAD to
the existing deterministic waveform checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torchaudio.functional as ta_functional


SILERO_SAMPLE_RATE = 16_000
MIN_NO_SPEECH_SECONDS = 1.0
MIN_NO_SPEECH_RMS_DBFS = -38.0


@dataclass(frozen=True)
class NoSpeechRange:
    """A loud VAD-confirmed no-speech interval, expressed in seconds."""

    start_seconds: float
    end_seconds: float


SpeechTimestampGetter = Callable[
    [torch.Tensor, Any, int], Sequence[Mapping[str, int | float]]
]
ModelLoader = Callable[[], Any]


def _load_silero_cpu_model() -> Any:
    """Load Silero's bundled TorchScript model only when first requested."""

    from silero_vad import load_silero_vad

    # The bundled TorchScript model uses the project's existing Torch runtime;
    # avoiding ONNX keeps this feature from adding another large dependency.
    return load_silero_vad()


def _get_silero_speech_timestamps(
    waveform: torch.Tensor,
    model: Any,
    sample_rate: int,
) -> Sequence[Mapping[str, int | float]]:
    from silero_vad import get_speech_timestamps

    return get_speech_timestamps(
        waveform,
        model,
        threshold=0.35,
        sampling_rate=sample_rate,
        min_speech_duration_ms=180,
        min_silence_duration_ms=200,
        speech_pad_ms=40,
    )


def _mono_float_cpu(waveform: torch.Tensor | Sequence[float]) -> torch.Tensor:
    samples = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
    if samples.ndim == 2 and samples.shape[0] == 1:
        samples = samples.squeeze(0)
    if samples.ndim != 1:
        raise ValueError("Silero VAD expects a mono waveform shaped [samples] or [1, samples]")
    if not torch.isfinite(samples).all():
        raise ValueError("Silero VAD waveform contains NaN or infinite samples")
    return samples.contiguous()


def resample_for_silero(
    waveform: torch.Tensor | Sequence[float],
    sample_rate: int,
) -> torch.Tensor:
    """Return a CPU mono float waveform at Silero's required 16 kHz rate."""

    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    samples = _mono_float_cpu(waveform)
    if int(sample_rate) == SILERO_SAMPLE_RATE:
        return samples
    return ta_functional.resample(samples, int(sample_rate), SILERO_SAMPLE_RATE)


def _speech_ranges(
    timestamps: Sequence[Mapping[str, int | float]],
    sample_count: int,
) -> list[tuple[int, int]]:
    ranges = []
    for timestamp in timestamps:
        try:
            start = max(0, min(sample_count, int(timestamp["start"])))
            end = max(0, min(sample_count, int(timestamp["end"])))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Silero VAD timestamps must contain numeric start/end samples") from error
        if end > start:
            ranges.append((start, end))
    ranges.sort()

    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _is_loud_enough(samples: torch.Tensor, minimum_rms_dbfs: float) -> bool:
    if samples.numel() == 0:
        return False
    rms = torch.sqrt(torch.mean(samples.to(torch.float64).square())).item()
    if rms == 0.0:
        return False
    rms_dbfs = 20.0 * math.log10(rms)
    return rms_dbfs >= float(minimum_rms_dbfs) - 1e-6


def find_loud_no_speech_ranges(
    waveform_16k: torch.Tensor | Sequence[float],
    speech_timestamps: Sequence[Mapping[str, int | float]],
    *,
    minimum_seconds: float = MIN_NO_SPEECH_SECONDS,
    minimum_rms_dbfs: float = MIN_NO_SPEECH_RMS_DBFS,
) -> tuple[NoSpeechRange, ...]:
    """Find loud internal gaps from injected Silero timestamp results.

    The helper is pure: callers and unit tests may supply timestamps without
    loading the Silero model.  Leading/trailing silence is ignored whenever
    speech exists; an all-no-speech file is considered as one whole interval.
    """

    if minimum_seconds <= 0:
        raise ValueError("minimum_seconds must be positive")
    samples = _mono_float_cpu(waveform_16k)
    speech_ranges = _speech_ranges(speech_timestamps, samples.numel())
    if not speech_ranges:
        gaps = [(0, samples.numel())]
    else:
        gaps = [
            (left_end, right_start)
            for (_, left_end), (right_start, _) in zip(speech_ranges, speech_ranges[1:])
            if right_start > left_end
        ]

    findings = []
    for start, end in gaps:
        if (end - start) / SILERO_SAMPLE_RATE < minimum_seconds:
            continue
        if not _is_loud_enough(samples[start:end], minimum_rms_dbfs):
            continue
        findings.append(
            NoSpeechRange(
                start_seconds=start / SILERO_SAMPLE_RATE,
                end_seconds=end / SILERO_SAMPLE_RATE,
            )
        )
    return tuple(findings)


class SileroVadDetector:
    """Lazily loaded, lock-protected CPU Silero VAD adapter."""

    def __init__(
        self,
        *,
        model_loader: ModelLoader = _load_silero_cpu_model,
        speech_timestamp_getter: SpeechTimestampGetter = _get_silero_speech_timestamps,
    ) -> None:
        self._model_loader = model_loader
        self._speech_timestamp_getter = speech_timestamp_getter
        self._model: Any | None = None
        # Silero's recurrent model resets state between calls, so loading and
        # timestamp extraction must be one critical section.
        self._lock = threading.Lock()

    def find_loud_no_speech_ranges(
        self,
        waveform: torch.Tensor | Sequence[float],
        sample_rate: int,
        *,
        minimum_seconds: float = MIN_NO_SPEECH_SECONDS,
        minimum_rms_dbfs: float = MIN_NO_SPEECH_RMS_DBFS,
    ) -> tuple[NoSpeechRange, ...]:
        samples = resample_for_silero(waveform, sample_rate)
        with self._lock:
            if self._model is None:
                self._model = self._model_loader()
            reset_states = getattr(self._model, "reset_states", None)
            if callable(reset_states):
                reset_states()
            try:
                timestamps = self._speech_timestamp_getter(
                    samples,
                    self._model,
                    SILERO_SAMPLE_RATE,
                )
            finally:
                if callable(reset_states):
                    reset_states()
        return find_loud_no_speech_ranges(
            samples,
            timestamps,
            minimum_seconds=minimum_seconds,
            minimum_rms_dbfs=minimum_rms_dbfs,
        )


default_silero_vad_detector = SileroVadDetector()
