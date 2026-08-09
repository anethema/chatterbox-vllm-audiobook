"""Metadata helpers for FFmpeg M4B audiobook assembly."""

from __future__ import annotations

from dataclasses import dataclass

from chatterbox_vllm.epub import EpubBook


@dataclass(frozen=True)
class ChapterMarker:
    title: str
    start_ms: int
    end_ms: int


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
