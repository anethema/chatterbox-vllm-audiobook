from contextlib import contextmanager
from dataclasses import dataclass
import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
import time
from typing import Iterator, Mapping
from uuid import uuid4
import wave

import numpy as np

from .silero_vad import default_silero_vad_detector


TARGET_LUFS = -18.0
TRUE_PEAK_DBTP = -2.0
REFERENCE_TARGET_LUFS = -20.0
REFERENCE_TRUE_PEAK_DBTP = -3.0
REFERENCE_DENOISE_SAMPLE_RATE = 48_000
MAX_REFERENCE_BYTES = 64 * 1024 * 1024
MAX_REFERENCE_SECONDS = 300
REFERENCE_PROCESS_TIMEOUT = 120
# Do not permit playlists/concat or other formats that can open secondary files.
REFERENCE_FORMATS = "wav,mp3,flac,ogg,mov,matroska,webm,aac,aiff,au"
REFERENCE_CACHE_BYTES = 128 * 1024 * 1024
_reference_cache_lock = threading.RLock()
_reference_cache_directory = None
_reference_cache: OrderedDict[tuple, Path] = OrderedDict()
LOUDNESS_RANGE_LU = 7.0
MAX_INTERNAL_PAUSE_SECONDS = 0.5
INTERNAL_PAUSE_THRESHOLD_DBFS = -50.0
INTERNAL_PAUSE_FRAME_SECONDS = 0.01
ZERO_CROSSING_SEARCH_SECONDS = 0.005
AUDIO_QUALITY_FRAME_SECONDS = 0.25
AUDIO_QUALITY_MINIMUM_DBFS = -38.0
BROADBAND_NOISE_MINIMUM_SECONDS = 1.0
BROADBAND_NOISE_MINIMUM_ZERO_CROSSING_RATE = 0.22
BROADBAND_NOISE_MINIMUM_CENTROID_HZ = 2200.0
BROADBAND_NOISE_MINIMUM_FLATNESS = 0.07
TONAL_NOISE_MINIMUM_SECONDS = 2.0
TONAL_NOISE_MINIMUM_FREQUENCY_HZ = 500.0
TONAL_NOISE_MAXIMUM_FREQUENCY_MAD_HZ = 60.0
TONAL_NOISE_MAXIMUM_CENTROID_MAD_HZ = 250.0
TONAL_NOISE_MAXIMUM_CENTROID_OFFSET_HZ = 400.0
TONAL_NOISE_MAXIMUM_LEVEL_RANGE_DB = 12.0
TONAL_NOISE_MAXIMUM_FLATNESS = 0.04
LOW_FREQUENCY_COLLAPSE_MINIMUM_SECONDS = 1.0
LOW_FREQUENCY_COLLAPSE_MAXIMUM_HZ = 120.0
LOW_FREQUENCY_COLLAPSE_MINIMUM_POWER_FRACTION = 0.80
LOW_FREQUENCY_COLLAPSE_MAXIMUM_DOMINANT_HZ = 100.0
LOW_FREQUENCY_COLLAPSE_MINIMUM_DBFS = -35.0
# Generated speech normally sits far above this level even before the final
# loudness-normalization pass. This remains quieter than normal soft speech,
# while also treating low-level residual noise as silence.
GENERATED_AUDIO_NEAR_SILENCE_MAXIMUM_DBFS = -50.0
# Allow conventional sentence/chapter-edge padding while rejecting the long
# silent tails or delayed starts produced by synthesis failures.
GENERATED_AUDIO_EDGE_SILENCE_MINIMUM_SECONDS = 3.0


@dataclass(frozen=True)
class AudioQualityIssue:
    kind: str
    start_seconds: float
    end_seconds: float


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


def _reference_input_options(path: Path) -> list[str]:
    """Bound file size and limit demuxers before FFmpeg can inspect an upload."""

    if not path.is_file() or not 0 < path.stat().st_size <= MAX_REFERENCE_BYTES:
        raise ValueError("Reference audio must be a nonempty file of at most 64 MiB")
    return [
        "-protocol_whitelist", "file,pipe",
        "-format_whitelist", REFERENCE_FORMATS,
        "-i", str(path.resolve()),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-t", str(MAX_REFERENCE_SECONDS + 1),
    ]


def _check_reference_duration(path: Path) -> None:
    """Reject empty or overlong decoded references before feature extraction."""

    import soundfile as sf

    info = sf.info(str(path))
    if info.frames < 1 or info.duration > MAX_REFERENCE_SECONDS:
        raise ValueError("Reference audio must be nonempty and at most 5 minutes long")


def clear_reference_cache() -> None:
    """Release process-owned preparation artifacts when the model shuts down."""

    global _reference_cache_directory
    with _reference_cache_lock:
        _reference_cache.clear()
        if _reference_cache_directory is not None:
            _reference_cache_directory.cleanup()
            _reference_cache_directory = None


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
        *_reference_input_options(source),
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
        timeout=REFERENCE_PROCESS_TIMEOUT,
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
            *_reference_input_options(source),
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
            timeout=REFERENCE_PROCESS_TIMEOUT,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed to normalize reference audio: {detail}")
        if not normalized.is_file() or normalized.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce normalized reference audio")
        _check_reference_duration(normalized)
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


def denoise_reference_audio(
    path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Denoise a temporary 48 kHz mono copy with RNNoise."""

    source = Path(path)
    destination = Path(output_path)
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to prepare reference audio for denoising")

    converted = destination.with_name(f".{destination.stem}-rnnoise-input.wav")
    conversion_command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *_reference_input_options(source),
        "-ar",
        str(REFERENCE_DENOISE_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(converted),
    ]
    started = time.perf_counter()
    print(f"[Reference denoise] Preparing RNNoise input: {source}", flush=True)
    try:
        converted_result = subprocess.run(
            conversion_command,
            capture_output=True,
            text=True,
            timeout=REFERENCE_PROCESS_TIMEOUT,
        )
        if converted_result.returncode != 0:
            detail = converted_result.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed to prepare RNNoise input: {detail}")

        from pyrnnoise import RNNoise

        # pyrnnoise's file helper currently calls an obsolete audiolab Reader
        # attribute (``rate`` rather than ``sample_rate``). Feed its supported
        # array API directly so this path is independent of that adapter bug.
        _check_reference_duration(converted)
        denoiser = RNNoise(REFERENCE_DENOISE_SAMPLE_RATE)
        frames_written = 0
        with wave.open(str(converted), "rb") as reader, wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(REFERENCE_DENOISE_SAMPLE_RATE)
            # Keep only one second of input and the current RNNoise frame live.
            while content := reader.readframes(REFERENCE_DENOISE_SAMPLE_RATE):
                samples = np.frombuffer(content, dtype="<i2").copy()
                final = reader.tell() == reader.getnframes()
                for _, frame in denoiser.denoise_chunk(samples[np.newaxis, :], partial=final):
                    pcm = np.clip(frame, -32768, 32767).astype("<i2")
                    writer.writeframes(pcm.tobytes())
                    frames_written += pcm.size
        if not frames_written:
            raise RuntimeError("RNNoise did not return any denoised audio frames")
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("RNNoise did not produce denoised reference audio")
        print(
            "[Reference denoise] Completed in "
            f"{time.perf_counter() - started:.2f}s: {destination}",
            flush=True,
        )
        return destination
    finally:
        converted.unlink(missing_ok=True)


@contextmanager
def _prepare_reference_audio(
    path: str | Path,
    sample_rate: int,
    *,
    denoise: bool = False,
    ffmpeg: str | None = None,
) -> Iterator[Path]:
    """Yield the exact denoised/normalized reference used for conditioning.

    The source is never modified. If optional RNNoise processing fails, emit a
    visible warning and safely fall back to normalizing the original reference.
    """

    source = Path(path)
    normalization_source = source
    with tempfile.TemporaryDirectory(prefix="chatterbox-reference-prep-") as directory:
        if denoise:
            denoised = Path(directory) / "denoised-reference.wav"
            try:
                normalization_source = denoise_reference_audio(
                    source,
                    denoised,
                    ffmpeg=ffmpeg,
                )
            except Exception as error:
                print(
                    "[Reference denoise] WARNING: RNNoise failed; using the "
                    f"original reference instead: {error}",
                    flush=True,
                )
                normalization_source = source

        with normalized_reference_audio(
            normalization_source,
            sample_rate,
            ffmpeg=ffmpeg,
        ) as normalized:
            yield normalized


@contextmanager
def prepared_reference_audio(
    path: str | Path,
    sample_rate: int,
    *,
    denoise: bool = False,
    ffmpeg: str | None = None,
) -> Iterator[Path]:
    """Reuse bounded immutable preparation artifacts, including across copies.

    Content keys detect replacement at the same filename and let a saved project
    reuse exactly the processing heard in the upload preview. Hold the lock while
    an artifact is borrowed so another request cannot evict an active reader.
    """

    global _reference_cache_directory
    source = Path(path)
    _reference_input_options(source)
    with source.open("rb") as stream:
        digest = hashlib.sha256()
        while block := stream.read(1024 * 1024):
            digest.update(block)
    key = (digest.hexdigest(), int(sample_rate), bool(denoise), ffmpeg)
    with _reference_cache_lock:
        if _reference_cache_directory is None:
            _reference_cache_directory = tempfile.TemporaryDirectory(prefix="chatterbox-prepared-")
        cached = _reference_cache.get(key)
        if cached is None or not cached.is_file():
            with _prepare_reference_audio(source, sample_rate, denoise=denoise, ffmpeg=ffmpeg) as prepared:
                cached = Path(_reference_cache_directory.name) / f"{uuid4().hex}.wav"
                shutil.copyfile(prepared, cached)
            _reference_cache[key] = cached
        else:
            print("[Reference preparation] Reusing verified prepared audio", flush=True)
        _reference_cache.move_to_end(key)
        try:
            yield cached
        finally:
            total_bytes = sum(item.stat().st_size for item in _reference_cache.values() if item.is_file())
            while len(_reference_cache) > 16 or total_bytes > REFERENCE_CACHE_BYTES:
                _, expired = _reference_cache.popitem(last=False)
                if expired.is_file():
                    total_bytes -= expired.stat().st_size
                    expired.unlink()


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


def _quality_ranges(mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    """Return half-open runs of flagged frames that meet the minimum length."""

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, flagged in enumerate(mask):
        if flagged and start is None:
            start = index
        elif not flagged and start is not None:
            if index - start >= minimum_frames:
                ranges.append((start, index))
            start = None
    if start is not None and len(mask) - start >= minimum_frames:
        ranges.append((start, len(mask)))
    return ranges


def _expand_quality_ranges(
    ranges: list[tuple[int, int]],
    support: np.ndarray,
) -> list[tuple[int, int]]:
    """Expand high-confidence detections across their adjacent artifact frames."""

    expanded: list[tuple[int, int]] = []
    for start, end in ranges:
        while start > 0 and support[start - 1]:
            start -= 1
        while end < len(support) and support[end]:
            end += 1
        if expanded and start <= expanded[-1][1]:
            expanded[-1] = (expanded[-1][0], max(expanded[-1][1], end))
        else:
            expanded.append((start, end))
    return expanded


def _is_near_silent(samples: np.ndarray) -> bool:
    """Whether a sample range is quiet enough to be treated as silence."""

    if samples.size < 1:
        return True
    centered = samples - np.mean(samples)
    rms = np.sqrt(np.mean(centered * centered) + 1e-12)
    rms_dbfs = 20.0 * np.log10(rms + 1e-12)
    return rms_dbfs <= GENERATED_AUDIO_NEAR_SILENCE_MAXIMUM_DBFS


def _append_vad_issues(
    issues: list[AudioQualityIssue],
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    """Append model-confirmed non-speech regions to waveform findings."""

    for no_speech in default_silero_vad_detector.find_loud_no_speech_ranges(
        samples,
        sample_rate,
    ):
        issues.append(
            AudioQualityIssue(
                "audible non-speech synthesis blob",
                no_speech.start_seconds,
                no_speech.end_seconds,
            )
        )


def find_generated_audio_issues(
    samples,
    sample_rate: int,
    *,
    include_vad: bool = False,
) -> list[AudioQualityIssue]:
    """Find known Chatterbox silence and synthesis-collapse patterns.

    Whole-clip and edge-silence checks use the waveform's actual sample count.
    Spectral collapse checks analyze every complete 250 ms frame so a bad
    middle section is not hidden merely because the final seconds sound normal.
    """

    rate = int(sample_rate)
    if rate <= 0:
        return []
    frame_size = max(1, round(rate * AUDIO_QUALITY_FRAME_SECONDS))
    flattened = np.asarray(samples, dtype=np.float32).reshape(-1)
    if flattened.size < 1:
        return []
    frame_count = flattened.size // frame_size
    if frame_count < 1:
        if _is_near_silent(flattened):
            return [
                AudioQualityIssue(
                    "near-silent generated audio",
                    0.0,
                    flattened.size / rate,
                )
            ]
        return []

    frames = flattened[:frame_count * frame_size].reshape(frame_count, frame_size)
    frames = frames - np.mean(frames, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_dbfs = 20.0 * np.log10(rms + 1e-12)
    zero_crossings = np.mean(
        np.signbit(frames[:, :-1]) != np.signbit(frames[:, 1:]), axis=1
    )

    # All spectral artifact detectors require audible frame levels, so an
    # entirely near-silent clip cannot gain any spectral findings. Preserve
    # the separately requested VAD pass while avoiding FFT allocations.
    near_silent = rms_dbfs <= GENERATED_AUDIO_NEAR_SILENCE_MAXIMUM_DBFS
    remainder = flattened[frame_count * frame_size :]
    if np.all(near_silent) and _is_near_silent(remainder):
        issues = [
            AudioQualityIssue(
                "near-silent generated audio",
                0.0,
                flattened.size / rate,
            )
        ]
        if include_vad:
            _append_vad_issues(issues, flattened, rate)
        return sorted(issues, key=lambda issue: (issue.start_seconds, issue.end_seconds))

    window = np.hanning(frame_size).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    power[:, 0] = 0
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / rate)
    totals = np.sum(power, axis=1) + 1e-30
    dominant = frequencies[np.argmax(power, axis=1)]
    centroids = np.sum(power * frequencies, axis=1) / totals
    flatness = np.exp(np.mean(np.log(power + 1e-30), axis=1)) / (
        np.mean(power, axis=1) + 1e-30
    )

    issues: list[AudioQualityIssue] = []

    minimum_edge_samples = max(
        1,
        math.ceil(GENERATED_AUDIO_EDGE_SILENCE_MINIMUM_SECONDS * rate),
    )
    leading_end = 0
    while leading_end < frame_count and near_silent[leading_end]:
        leading_end += 1
    leading_samples = leading_end * frame_size
    if (
        leading_samples < minimum_edge_samples
        and flattened.size >= minimum_edge_samples
        and _is_near_silent(flattened[:minimum_edge_samples])
    ):
        leading_samples = minimum_edge_samples
    if leading_samples >= minimum_edge_samples:
        issues.append(
            AudioQualityIssue(
                "excessive leading near-silence",
                0.0,
                leading_samples / rate,
            )
        )

    # Reframe from the end so an otherwise-valid 3-second tail is not
    # shortened by a partial speech frame at its leading boundary.
    trailing_frames = flattened[-frame_count * frame_size :].reshape(
        frame_count,
        frame_size,
    )
    trailing_frames = trailing_frames - np.mean(
        trailing_frames,
        axis=1,
        keepdims=True,
    )
    trailing_rms = np.sqrt(
        np.mean(trailing_frames * trailing_frames, axis=1) + 1e-12
    )
    trailing_near_silent = (
        20.0 * np.log10(trailing_rms + 1e-12)
        <= GENERATED_AUDIO_NEAR_SILENCE_MAXIMUM_DBFS
    )
    trailing_count = 0
    while trailing_count < frame_count and trailing_near_silent[-trailing_count - 1]:
        trailing_count += 1
    trailing_samples = trailing_count * frame_size
    if (
        trailing_samples < minimum_edge_samples
        and flattened.size >= minimum_edge_samples
        and _is_near_silent(flattened[-minimum_edge_samples:])
    ):
        trailing_samples = minimum_edge_samples
    if trailing_samples >= minimum_edge_samples:
        issues.append(
            AudioQualityIssue(
                "excessive trailing near-silence",
                (flattened.size - trailing_samples) / rate,
                flattened.size / rate,
            )
        )

    low_frequency_power = np.sum(
        power[:, frequencies <= LOW_FREQUENCY_COLLAPSE_MAXIMUM_HZ],
        axis=1,
    ) / totals
    low_frequency_core = (
        (rms_dbfs >= LOW_FREQUENCY_COLLAPSE_MINIMUM_DBFS)
        & (
            low_frequency_power
            >= LOW_FREQUENCY_COLLAPSE_MINIMUM_POWER_FRACTION
        )
    )
    low_frequency_support = (
        (rms_dbfs >= LOW_FREQUENCY_COLLAPSE_MINIMUM_DBFS)
        & (dominant <= LOW_FREQUENCY_COLLAPSE_MAXIMUM_DOMINANT_HZ)
    )
    low_frequency_frames = max(
        1,
        round(
            LOW_FREQUENCY_COLLAPSE_MINIMUM_SECONDS
            / AUDIO_QUALITY_FRAME_SECONDS
        ),
    )
    low_frequency_ranges = _quality_ranges(
        low_frequency_core,
        low_frequency_frames,
    )
    for start, end in _expand_quality_ranges(
        low_frequency_ranges,
        low_frequency_support,
    ):
        issues.append(
            AudioQualityIssue(
                "low-frequency synthesis collapse",
                start * AUDIO_QUALITY_FRAME_SECONDS,
                end * AUDIO_QUALITY_FRAME_SECONDS,
            )
        )

    broadband = (
        (rms_dbfs >= AUDIO_QUALITY_MINIMUM_DBFS)
        & (zero_crossings >= BROADBAND_NOISE_MINIMUM_ZERO_CROSSING_RATE)
        & (centroids >= BROADBAND_NOISE_MINIMUM_CENTROID_HZ)
        & (flatness >= BROADBAND_NOISE_MINIMUM_FLATNESS)
    )
    broadband_frames = max(
        1, round(BROADBAND_NOISE_MINIMUM_SECONDS / AUDIO_QUALITY_FRAME_SECONDS)
    )
    for start, end in _quality_ranges(broadband, broadband_frames):
        issues.append(
            AudioQualityIssue(
                "broadband digital noise",
                start * AUDIO_QUALITY_FRAME_SECONDS,
                end * AUDIO_QUALITY_FRAME_SECONDS,
            )
        )

    tonal_frames = max(
        1, round(TONAL_NOISE_MINIMUM_SECONDS / AUDIO_QUALITY_FRAME_SECONDS)
    )
    tonal = np.zeros(frame_count, dtype=bool)
    for start in range(0, frame_count - tonal_frames + 1):
        end = start + tonal_frames
        levels = rms_dbfs[start:end]
        tones = dominant[start:end]
        centers = centroids[start:end]
        texture = flatness[start:end]
        tone_median = float(np.median(tones))
        center_median = float(np.median(centers))
        if (
            float(np.min(levels)) >= AUDIO_QUALITY_MINIMUM_DBFS
            and float(np.max(levels) - np.min(levels))
            <= TONAL_NOISE_MAXIMUM_LEVEL_RANGE_DB
            and tone_median >= TONAL_NOISE_MINIMUM_FREQUENCY_HZ
            and float(np.median(np.abs(tones - tone_median)))
            <= TONAL_NOISE_MAXIMUM_FREQUENCY_MAD_HZ
            and float(np.median(np.abs(centers - center_median)))
            <= TONAL_NOISE_MAXIMUM_CENTROID_MAD_HZ
            and float(np.median(np.abs(centers - tones)))
            <= TONAL_NOISE_MAXIMUM_CENTROID_OFFSET_HZ
            and float(np.median(texture)) <= TONAL_NOISE_MAXIMUM_FLATNESS
        ):
            tonal[start:end] |= (
                (levels >= AUDIO_QUALITY_MINIMUM_DBFS)
                & (
                    np.abs(tones - tone_median)
                    <= 2 * TONAL_NOISE_MAXIMUM_FREQUENCY_MAD_HZ
                )
                & (
                    np.abs(centers - center_median)
                    <= 2 * TONAL_NOISE_MAXIMUM_CENTROID_MAD_HZ
                )
                & (texture <= TONAL_NOISE_MAXIMUM_FLATNESS)
            )
    for start, end in _quality_ranges(tonal, tonal_frames):
        issues.append(
            AudioQualityIssue(
                "sustained synthetic tone",
                start * AUDIO_QUALITY_FRAME_SECONDS,
                end * AUDIO_QUALITY_FRAME_SECONDS,
            )
        )

    if include_vad:
        _append_vad_issues(issues, flattened, rate)

    return sorted(issues, key=lambda issue: (issue.start_seconds, issue.end_seconds))


def format_audio_quality_issues(issues: list[AudioQualityIssue]) -> str:
    """Format detector labels with second-based ranges for logs and errors."""

    return "; ".join(
        f"{issue.kind} at {issue.start_seconds:.2f}-{issue.end_seconds:.2f}s"
        for issue in issues
    )


def wav_generated_audio_issues(
    path: str | Path,
    sample_rate: int,
) -> list[AudioQualityIssue]:
    """Scan a normalized mono PCM16 chunk for generated-audio corruption."""

    samples = _read_pcm16_mono(Path(path), sample_rate).astype(np.float32) / 32768.0
    return find_generated_audio_issues(samples, sample_rate, include_vad=True)


def _read_pcm16_mono(path: Path, expected_sample_rate: int) -> np.ndarray:
    """Read signed PCM16 samples only after validating the complete WAV payload."""

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
    if len(content) != frames * channels * sample_width:
        raise RuntimeError(f"Could not read normalized speech audio: {path}")
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
