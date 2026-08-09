"""Metadata helpers for FFmpeg M4B audiobook assembly."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess

from chatterbox_vllm.epub import EpubBook


@dataclass(frozen=True)
class ChapterMarker:
    title: str
    start_ms: int
    end_ms: int


def delete_intermediate_chunks(project_dir: str | Path) -> None:
    """Delete only the verified project's direct ``chunks`` directory."""

    project = Path(project_dir).resolve()
    chunks = (project / "chunks").resolve()
    if chunks.parent != project or chunks.name != "chunks":
        raise RuntimeError(f"Refusing to delete unexpected chunk path: {chunks}")
    if not chunks.is_dir():
        raise RuntimeError(f"Intermediate chunk directory is missing: {chunks}")
    shutil.rmtree(chunks)


def verify_m4b(path: str | Path, *, ffprobe: str | None = None) -> float:
    """Verify that an M4B is nonempty and has positive-duration audio."""

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
            "stream=codec_name,duration:format=duration",
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
    if not codec_name or duration is None:
        raise RuntimeError("The completed M4B does not contain positive-duration audio")
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
