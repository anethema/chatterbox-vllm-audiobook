from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterator, Mapping
from uuid import uuid4
import wave

import numpy as np


TARGET_LUFS = -18.0
TRUE_PEAK_DBTP = -2.0
REFERENCE_TARGET_LUFS = -20.0
REFERENCE_TRUE_PEAK_DBTP = -3.0
LOUDNESS_RANGE_LU = 7.0
MAX_INTERNAL_PAUSE_SECONDS = 0.5
INTERNAL_PAUSE_THRESHOLD_DBFS = -50.0
INTERNAL_PAUSE_FRAME_SECONDS = 0.01
ZERO_CROSSING_SEARCH_SECONDS = 0.005


def loudness_filter() -> str:
    """Return the EBU R128 normalization used for audiobook speech chunks."""

    return (
        f"loudnorm=I={TARGET_LUFS:g}:TP={TRUE_PEAK_DBTP:g}:"
        f"LRA={LOUDNESS_RANGE_LU:g}"
    )


def reference_loudness_filter(
    measurements: Mapping[str, str] | None = None,
) -> str:
    """Return the two-pass EBU R128 filter used for voice references."""

    base = (
        f"loudnorm=I={REFERENCE_TARGET_LUFS:g}:"
        f"TP={REFERENCE_TRUE_PEAK_DBTP:g}:LRA={LOUDNESS_RANGE_LU:g}"
    )
    if measurements is None:
        return f"aformat=channel_layouts=mono,{base}:print_format=json"

    fields = {
        "measured_I": measurements["input_i"],
        "measured_LRA": measurements["input_lra"],
        "measured_TP": measurements["input_tp"],
        "measured_thresh": measurements["input_thresh"],
        "offset": measurements["target_offset"],
    }
    if not all(math.isfinite(float(value)) for value in fields.values()):
        raise RuntimeError(
            "Reference audio is silent or too quiet for loudness normalization"
        )
    measured = ":".join(f"{key}={value}" for key, value in fields.items())
    return (
        f"aformat=channel_layouts=mono,{base}:{measured}:"
        "linear=true:print_format=json"
    )


def _parse_loudness_measurements(stderr: str) -> dict[str, str]:
    match = re.search(r'\{\s*"input_i".*?\}', stderr, flags=re.DOTALL)
    if not match:
        raise RuntimeError("FFmpeg did not report reference-audio loudness")
    try:
        measurements = json.loads(match.group(0))
        reference_loudness_filter(measurements)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("FFmpeg reported invalid reference-audio loudness") from error
    return measurements


@contextmanager
def normalized_reference_audio(
    path: str | Path,
    sample_rate: int,
    *,
    ffmpeg: str | None = None,
) -> Iterator[Path]:
    """Yield a normalized temporary WAV without modifying the voice sample."""

    source = Path(path)
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to normalize reference audio")

    print(f"[Reference normalization] Measuring: {source}", flush=True)
    measurement_command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-i",
        str(source),
        "-af",
        reference_loudness_filter(),
        "-f",
        "null",
        "-",
    ]
    measured = subprocess.run(
        measurement_command,
        capture_output=True,
        text=True,
    )
    if measured.returncode != 0:
        detail = measured.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg failed to measure reference audio: {detail}")
    measurements = _parse_loudness_measurements(measured.stderr)
    print(
        "[Reference normalization] "
        f"Input: {measurements['input_i']} LUFS, "
        f"{measurements['input_tp']} dBTP",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="chatterbox-reference-") as directory:
        normalized = Path(directory) / "normalized-reference.wav"
        normalization_command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-y",
            "-i",
            str(source),
            "-af",
            reference_loudness_filter(measurements),
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            str(normalized),
        ]
        result = subprocess.run(
            normalization_command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed to normalize reference audio: {detail}")
        if not normalized.is_file() or normalized.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce normalized reference audio")
        output = _parse_loudness_measurements(result.stderr)
        print(
            "[Reference normalization] "
            f"Output: {output.get('output_i', 'unknown')} LUFS, "
            f"{output.get('output_tp', 'unknown')} dBTP "
            f"({output.get('normalization_type', 'unknown')}, "
            f"{int(sample_rate)} Hz mono; source unchanged)",
            flush=True,
        )
        yield normalized


def normalize_speech_wav(
    path: str | Path,
    sample_rate: int,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Normalize a speech WAV in place without risking the original on failure."""

    source = Path(path)
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to normalize audiobook audio")

    temporary = source.with_name(
        f".{source.stem}.normalized-{uuid4().hex}{source.suffix}"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        loudness_filter(),
        "-ar",
        str(int(sample_rate)),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed to normalize speech audio: {detail}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce normalized speech audio")
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)
    return source


def _read_pcm16_mono(path: Path, expected_sample_rate: int) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
            content = audio.readframes(frames)
    except (EOFError, OSError, wave.Error) as error:
        raise RuntimeError(f"Could not read normalized speech audio: {path}") from error
    if (
        channels != 1
        or sample_width != 2
        or sample_rate != int(expected_sample_rate)
        or frames <= 0
    ):
        raise RuntimeError(
            "Internal-pause limiting requires nonempty mono 16-bit PCM audio "
            f"at {int(expected_sample_rate)} Hz"
        )
    return np.frombuffer(content, dtype="<i2").copy()


def _internal_silence_ranges(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float,
) -> list[tuple[int, int]]:
    """Return sample ranges that are silent and surrounded by speech."""

    frame_size = max(1, round(sample_rate * INTERNAL_PAUSE_FRAME_SECONDS))
    usable = len(samples) // frame_size * frame_size
    if usable < frame_size:
        return []
    floating = samples[:usable].astype(np.float32) / 32768.0
    frames = floating.reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    active = rms >= 10 ** (float(threshold_dbfs) / 20)
    active_indices = np.flatnonzero(active)
    if active_indices.size < 2:
        return []

    first_active = int(active_indices[0])
    last_active = int(active_indices[-1])
    ranges: list[tuple[int, int]] = []
    start_frame: int | None = None
    for frame in range(first_active + 1, last_active + 1):
        if not active[frame] and start_frame is None:
            start_frame = frame
        elif active[frame] and start_frame is not None:
            ranges.append((start_frame * frame_size, frame * frame_size))
            start_frame = None
    return ranges


def _quietest_index(samples: np.ndarray, lower: int, upper: int) -> int:
    lower = max(0, int(lower))
    upper = min(len(samples), int(upper))
    if upper <= lower:
        return lower
    offset = int(np.argmin(np.abs(samples[lower:upper].astype(np.int32))))
    return lower + offset


def limit_internal_pauses_wav(
    path: str | Path,
    sample_rate: int,
    *,
    maximum_seconds: float = MAX_INTERNAL_PAUSE_SECONDS,
    threshold_dbfs: float = INTERNAL_PAUSE_THRESHOLD_DBFS,
) -> Path:
    """Cap only silence surrounded by speech, replacing the WAV atomically.

    Leading and trailing silence are deliberately excluded. Cuts are joined at
    quiet samples inside the detected pause and never retain more than the
    configured maximum, subject to the 10 ms detector resolution.
    """

    source = Path(path)
    maximum_samples = round(float(maximum_seconds) * int(sample_rate))
    if maximum_samples < 1:
        raise ValueError("maximum_seconds must be positive")
    samples = _read_pcm16_mono(source, sample_rate)
    radius = max(1, round(sample_rate * ZERO_CROSSING_SEARCH_SECONDS))
    cuts: list[tuple[int, int]] = []
    for start, end in _internal_silence_ranges(
        samples,
        sample_rate,
        threshold_dbfs=threshold_dbfs,
    ):
        length = end - start
        if length <= maximum_samples:
            continue
        retained_left = maximum_samples // 2
        retained_right = maximum_samples - retained_left
        ideal_start = start + retained_left
        ideal_end = end - retained_right
        # Search outward from the ideal cut. This finds near-zero join samples
        # while guaranteeing the retained pause cannot exceed the requested cap.
        cut_start = _quietest_index(
            samples,
            max(start, ideal_start - radius),
            ideal_start + 1,
        )
        cut_end = _quietest_index(
            samples,
            ideal_end,
            min(end, ideal_end + radius) + 1,
        )
        if cut_end > cut_start:
            cuts.append((cut_start, cut_end))
    if not cuts:
        return source

    pieces: list[np.ndarray] = []
    cursor = 0
    for cut_start, cut_end in cuts:
        pieces.append(samples[cursor:cut_start])
        cursor = cut_end
    pieces.append(samples[cursor:])
    processed = np.concatenate(pieces)

    temporary = source.with_name(
        f".{source.stem}.pauses-{uuid4().hex}{source.suffix}"
    )
    try:
        with wave.open(str(temporary), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(int(sample_rate))
            audio.writeframes(np.asarray(processed, dtype="<i2").tobytes())
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("Internal-pause limiting did not produce valid audio")
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)
    return source
