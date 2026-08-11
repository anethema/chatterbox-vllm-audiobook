"""Parallel FFmpeg assembly and metadata helpers for M4B audiobooks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Callable, Sequence
from uuid import uuid4
import wave

from chatterbox_vllm.epub import EpubBook, TextChunk


@dataclass(frozen=True)
class ChapterMarker:
    title: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class AudioEntry:
    path: Path
    duration_ms: int
    starts_chapter: bool = False
    is_pause: bool = False


@dataclass(frozen=True)
class EncodingSegment:
    index: int
    entries: tuple[AudioEntry, ...]
    duration_ms: int


@dataclass(frozen=True)
class AssemblyProgress:
    phase: str
    fraction: float
    processed_seconds: float
    total_seconds: float
    elapsed_seconds: float
    eta_seconds: float
    realtime_speed: float
    workers: int


class AssemblyStopped(Exception):
    """Raised after a requested stop has terminated all FFmpeg children."""


@dataclass
class _RunningEncoder:
    segment: EncodingSegment
    process: subprocess.Popen
    progress_path: Path
    error_path: Path
    progress_handle: object
    error_handle: object


def delete_intermediate_chunks(project_dir: str | Path) -> None:
    """Delete only the verified project's direct ``chunks`` directory."""

    project = Path(project_dir).resolve()
    chunks = (project / "chunks").resolve()
    if chunks.parent != project or chunks.name != "chunks":
        raise RuntimeError(f"Refusing to delete unexpected chunk path: {chunks}")
    if not chunks.is_dir():
        raise RuntimeError(f"Intermediate chunk directory is missing: {chunks}")
    shutil.rmtree(chunks)


def verify_m4b(
    path: str | Path,
    *,
    ffprobe: str | None = None,
    expected_duration_seconds: float | None = None,
    expected_chapters: int | None = None,
    duration_tolerance_seconds: float = 2.0,
) -> float:
    """Verify M4B audio, duration, and optional chapter count."""

    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError("The completed M4B is missing or empty")
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe is required to verify the completed M4B")

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,duration:format=duration:chapter=start_time,end_time",
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown FFprobe error"
        raise RuntimeError(f"FFprobe could not read the completed M4B: {detail}")
    try:
        probe = json.loads(result.stdout)
        stream = probe["streams"][0]
        codec_name = stream["codec_name"]
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("The completed M4B does not contain valid audio") from error
    duration = None
    for candidate in (stream.get("duration"), probe.get("format", {}).get("duration")):
        try:
            parsed_duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed_duration) and parsed_duration > 0:
            duration = parsed_duration
            break
    if codec_name != "aac" or duration is None:
        raise RuntimeError("The completed M4B does not contain positive-duration AAC audio")

    if expected_duration_seconds is not None:
        tolerance = max(0.0, float(duration_tolerance_seconds))
        if abs(duration - expected_duration_seconds) > tolerance:
            raise RuntimeError(
                "The completed M4B duration differs from its source audio "
                f"({duration:.2f}s instead of {expected_duration_seconds:.2f}s)"
            )
    if expected_chapters is not None:
        chapters = probe.get("chapters", [])
        if len(chapters) != expected_chapters:
            raise RuntimeError(
                "The completed M4B has the wrong chapter count "
                f"({len(chapters)} instead of {expected_chapters})"
            )
        previous_start = -1.0
        for chapter in chapters:
            try:
                start = float(chapter["start_time"])
                end = float(chapter["end_time"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("The completed M4B has invalid chapter timestamps") from error
            if not math.isfinite(start) or not math.isfinite(end) or start < previous_start or end <= start:
                raise RuntimeError("The completed M4B has invalid chapter timestamps")
            previous_start = start
    return duration


def _escape(value: str) -> str:
    value = " ".join(str(value).split())
    for character in ("\\", "=", ";", "#"):
        value = value.replace(character, "\\" + character)
    return value


def build_ffmetadata(book: EpubBook, chapters: list[ChapterMarker]) -> str:
    lines = [";FFMETADATA1", f"title={_escape(book.title)}"]
    if book.authors:
        lines.extend(
            [
                f"artist={_escape(', '.join(book.authors))}",
                f"album_artist={_escape(', '.join(book.authors))}",
            ]
        )
    lines.append(f"album={_escape(book.title)}")
    lines.append("genre=Audiobook")
    for key, value in (
        ("language", book.language),
        ("publisher", book.publisher),
        ("comment", book.description),
        ("date", book.date),
        ("copyright", book.identifier),
    ):
        if value:
            lines.append(f"{key}={_escape(value)}")

    for chapter in chapters:
        lines.extend(
            [
                "",
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={chapter.start_ms}",
                f"END={chapter.end_ms}",
                f"title={_escape(chapter.title)}",
            ]
        )
    return "\n".join(lines) + "\n"


def default_m4b_workers(cpu_count: int | None = None) -> int:
    override = os.environ.get("CHATTERBOX_M4B_WORKERS")
    if override:
        try:
            workers = int(override)
        except ValueError as error:
            raise RuntimeError("CHATTERBOX_M4B_WORKERS must be an integer") from error
        if workers < 1:
            raise RuntimeError("CHATTERBOX_M4B_WORKERS must be at least 1")
        return workers
    available = os.cpu_count() if cpu_count is None else cpu_count
    return min(16, max(1, (available or 2) // 2))


def _wav_duration_ms(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnframes() <= 0 or audio.getframerate() <= 0:
                raise RuntimeError(f"Audio chunk is empty: {path}")
            return round(audio.getnframes() * 1000 / audio.getframerate())
    except (OSError, EOFError, wave.Error) as error:
        raise RuntimeError(f"Could not read audio chunk: {path}") from error


def _write_silence(path: Path, sample_rate: int, duration_seconds: float) -> int:
    frames = max(1, round(sample_rate * duration_seconds))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * frames)
    return round(frames * 1000 / sample_rate)


def build_audio_timeline(
    project_dir: str | Path,
    assembly_dir: str | Path,
    chunks: Sequence[TextChunk],
    sample_rate: int,
) -> tuple[list[AudioEntry], list[ChapterMarker], int]:
    """Build the exact chunk/pause timeline used for encoding and chapters."""

    if not chunks:
        raise RuntimeError("Cannot assemble an audiobook without speech chunks")
    project = Path(project_dir)
    assembly = Path(assembly_dir)
    short_pause = assembly / "pause-between-chunks.wav"
    chapter_pause = assembly / "pause-between-chapters.wav"
    short_pause_ms = _write_silence(short_pause, sample_rate, 0.18)
    chapter_pause_ms = _write_silence(chapter_pause, sample_rate, 0.9)

    entries: list[AudioEntry] = []
    chapter_starts: list[tuple[str, int]] = []
    timeline_ms = 0
    for index, chunk in enumerate(chunks):
        chapter_changed = index == 0 or chunk.chapter_index != chunks[index - 1].chapter_index
        if chapter_changed:
            chapter_starts.append((chunk.chapter_title, timeline_ms))
        if index:
            if chapter_changed:
                entries.append(AudioEntry(chapter_pause, chapter_pause_ms, True, True))
                timeline_ms += chapter_pause_ms
            else:
                entries.append(AudioEntry(short_pause, short_pause_ms, False, True))
                timeline_ms += short_pause_ms
        chunk_path = project / "chunks" / f"{index:06d}.wav"
        duration_ms = _wav_duration_ms(chunk_path)
        entries.append(AudioEntry(chunk_path, duration_ms))
        timeline_ms += duration_ms

    markers = [
        ChapterMarker(
            title,
            start,
            chapter_starts[index + 1][1] if index + 1 < len(chapter_starts) else timeline_ms,
        )
        for index, (title, start) in enumerate(chapter_starts)
    ]
    return entries, markers, timeline_ms


def plan_encoding_segments(
    entries: Sequence[AudioEntry],
    workers: int,
) -> list[EncodingSegment]:
    """Split a timeline into duration-balanced, preferably chapter-aligned ranges."""

    if not entries:
        raise ValueError("At least one audio entry is required")
    segment_count = min(max(1, int(workers)), len(entries))
    prefix = [0]
    for entry in entries:
        if entry.duration_ms <= 0:
            raise ValueError("Audio entry durations must be positive")
        prefix.append(prefix[-1] + entry.duration_ms)

    boundaries = [0]
    for split_number in range(1, segment_count):
        remaining_splits = segment_count - split_number
        lower = boundaries[-1] + 1
        upper = len(entries) - remaining_splits
        target = prefix[-1] * split_number / segment_count
        possible = range(lower, upper + 1)
        chapter_boundaries = [index for index in possible if entries[index].starts_chapter]
        pause_boundaries = [index for index in possible if entries[index].is_pause]
        candidates = chapter_boundaries or pause_boundaries or list(possible)
        boundaries.append(min(candidates, key=lambda index: abs(prefix[index] - target)))
    boundaries.append(len(entries))

    return [
        EncodingSegment(
            index,
            tuple(entries[start:end]),
            prefix[end] - prefix[start],
        )
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]


def _concat_line(path: Path) -> str:
    escaped = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'\n"


def _write_concat(path: Path, entries: Sequence[AudioEntry | Path]) -> None:
    paths = [entry.path if isinstance(entry, AudioEntry) else entry for entry in entries]
    path.write_text("".join(_concat_line(entry) for entry in paths), encoding="utf-8")


def _latest_out_time_seconds(path: Path) -> float:
    try:
        with path.open("rb") as progress:
            progress.seek(0, os.SEEK_END)
            size = progress.tell()
            progress.seek(max(0, size - 65536))
            tail = progress.read()
    except OSError:
        return 0.0
    matches = re.findall(rb"(?:out_time_us|out_time_ms)=(\d+)", tail)
    return int(matches[-1]) / 1_000_000 if matches else 0.0


def _terminate_encoders(encoders: Sequence[_RunningEncoder]) -> None:
    for encoder in encoders:
        if encoder.process.poll() is None:
            encoder.process.terminate()
    deadline = time.monotonic() + 5.0
    for encoder in encoders:
        if encoder.process.poll() is None:
            try:
                encoder.process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                encoder.process.kill()
    for encoder in encoders:
        if encoder.process.poll() is None:
            encoder.process.wait()


def _close_encoder_logs(encoders: Sequence[_RunningEncoder]) -> None:
    for encoder in encoders:
        encoder.progress_handle.close()
        encoder.error_handle.close()


def _report_progress(
    callback: Callable[[AssemblyProgress], None] | None,
    phase: str,
    processed_seconds: float,
    total_seconds: float,
    started: float,
    workers: int,
) -> None:
    if callback is None:
        return
    elapsed = max(time.monotonic() - started, 1e-9)
    processed = min(max(processed_seconds, 0.0), total_seconds)
    fraction = processed / total_seconds if total_seconds else 1.0
    speed = processed / elapsed
    eta = (total_seconds - processed) / speed if speed > 0 else 0.0
    callback(
        AssemblyProgress(
            phase,
            fraction,
            processed,
            total_seconds,
            elapsed,
            eta,
            speed,
            workers,
        )
    )


def _run_parallel_encoders(
    ffmpeg: str,
    assembly_dir: Path,
    segments: Sequence[EncodingSegment],
    stop_requested: Callable[[], bool],
    progress_callback: Callable[[AssemblyProgress], None] | None,
) -> list[Path]:
    running: list[_RunningEncoder] = []
    outputs: list[Path] = []
    total_seconds = sum(segment.duration_ms for segment in segments) / 1000
    started = time.monotonic()
    try:
        for segment in segments:
            concat_path = assembly_dir / f"segment-{segment.index:03d}.concat.txt"
            output_path = assembly_dir / f"segment-{segment.index:03d}.m4a"
            progress_path = assembly_dir / f"segment-{segment.index:03d}.progress"
            error_path = assembly_dir / f"segment-{segment.index:03d}.stderr.log"
            _write_concat(concat_path, segment.entries)
            progress_handle = progress_path.open("wb")
            error_handle = error_path.open("wb")
            try:
                process = subprocess.Popen(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_path),
                        "-vn",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "96k",
                        "-progress",
                        "pipe:1",
                        "-nostats",
                        "-f",
                        "mp4",
                        str(output_path),
                    ],
                    stdout=progress_handle,
                    stderr=error_handle,
                )
            except Exception:
                progress_handle.close()
                error_handle.close()
                raise
            running.append(
                _RunningEncoder(
                    segment,
                    process,
                    progress_path,
                    error_path,
                    progress_handle,
                    error_handle,
                )
            )
            outputs.append(output_path)

        while True:
            if stop_requested():
                _terminate_encoders(running)
                raise AssemblyStopped
            failed = next(
                (encoder for encoder in running if encoder.process.poll() not in (None, 0)),
                None,
            )
            if failed is not None:
                _terminate_encoders(running)
                failed.error_handle.flush()
                detail = failed.error_path.read_text(encoding="utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"FFmpeg failed while encoding M4B segment {failed.segment.index + 1}: "
                    f"{detail or 'unknown FFmpeg error'}"
                )
            processed = sum(
                min(
                    encoder.segment.duration_ms / 1000,
                    _latest_out_time_seconds(encoder.progress_path),
                )
                for encoder in running
            )
            _report_progress(
                progress_callback,
                "encoding",
                processed,
                total_seconds,
                started,
                len(segments),
            )
            if all(encoder.process.poll() == 0 for encoder in running):
                break
            time.sleep(0.25)
        _report_progress(
            progress_callback,
            "encoding",
            total_seconds,
            total_seconds,
            started,
            len(segments),
        )
        return outputs
    finally:
        _terminate_encoders(running)
        _close_encoder_logs(running)


def _run_mux(
    command: Sequence[str],
    progress_path: Path,
    error_path: Path,
    total_seconds: float,
    stop_requested: Callable[[], bool],
    progress_callback: Callable[[AssemblyProgress], None] | None,
) -> None:
    started = time.monotonic()
    with progress_path.open("wb") as progress_handle, error_path.open("wb") as error_handle:
        process = subprocess.Popen(command, stdout=progress_handle, stderr=error_handle)
        try:
            while process.poll() is None:
                if stop_requested():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise AssemblyStopped
                _report_progress(
                    progress_callback,
                    "muxing",
                    _latest_out_time_seconds(progress_path),
                    total_seconds,
                    started,
                    1,
                )
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if process.returncode != 0:
        detail = error_path.read_text(encoding="utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed while finalizing the M4B: {detail or 'unknown error'}")
    _report_progress(
        progress_callback,
        "muxing",
        total_seconds,
        total_seconds,
        started,
        1,
    )


def _cover_suffix(media_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
    }.get(media_type.lower(), "")


def assemble_audiobook(
    project_dir: str | Path,
    book: EpubBook,
    chunks: Sequence[TextChunk],
    sample_rate: int,
    *,
    workers: int | None = None,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[AssemblyProgress], None] | None = None,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> Path:
    """Encode balanced ranges concurrently, then stream-copy them into one M4B."""

    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe are required to assemble the audiobook M4B")
    stop_requested = stop_requested or (lambda: False)
    project = Path(project_dir)
    assembly_dir = project / ".assembly"
    if assembly_dir.exists():
        shutil.rmtree(assembly_dir)
    assembly_dir.mkdir()

    try:
        entries, markers, timeline_ms = build_audio_timeline(
            project, assembly_dir, chunks, sample_rate,
        )
        segment_plan = plan_encoding_segments(
            entries, workers if workers is not None else default_m4b_workers(),
        )
        metadata_path = project / "audiobook.ffmetadata"
        metadata_path.write_text(build_ffmetadata(book, markers), encoding="utf-8")

        cover_path = None
        cover_suffix = _cover_suffix(book.cover_media_type)
        if book.cover_image and cover_suffix:
            cover_path = project / f"cover{cover_suffix}"
            cover_path.write_bytes(book.cover_image)

        segment_outputs = _run_parallel_encoders(
            ffmpeg,
            assembly_dir,
            segment_plan,
            stop_requested,
            progress_callback,
        )
        segment_concat = assembly_dir / "encoded-segments.concat.txt"
        _write_concat(segment_concat, segment_outputs)
        partial_output = assembly_dir / f"audiobook-{uuid4().hex}.partial.m4b"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(segment_concat),
            "-f",
            "ffmetadata",
            "-i",
            str(metadata_path),
        ]
        if cover_path:
            command.extend(["-i", str(cover_path)])
        command.extend(["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"])
        if cover_path:
            command.extend(
                [
                    "-map",
                    "2:v:0",
                    "-c:v",
                    "copy",
                    "-disposition:v:0",
                    "attached_pic",
                    "-metadata:s:v:0",
                    "title=Cover",
                ]
            )
        command.extend(
            [
                "-c:a",
                "copy",
                "-progress",
                "pipe:1",
                "-nostats",
                "-f",
                "mp4",
                str(partial_output),
            ]
        )
        _run_mux(
            command,
            assembly_dir / "mux.progress",
            assembly_dir / "mux.stderr.log",
            timeline_ms / 1000,
            stop_requested,
            progress_callback,
        )
        verify_m4b(
            partial_output,
            ffprobe=ffprobe,
            expected_duration_seconds=timeline_ms / 1000,
            expected_chapters=len(markers),
            duration_tolerance_seconds=max(2.0, len(segment_plan) * 0.1),
        )
        output_path = project / "audiobook.m4b"
        os.replace(partial_output, output_path)
        return output_path
    finally:
        shutil.rmtree(assembly_dir, ignore_errors=True)
