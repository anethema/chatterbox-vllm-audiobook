"""Discovery and validation helpers for resumable audiobook projects."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import wave

from chatterbox_vllm.epub import EpubBook, TextChunk, chunk_book


class ResumeProjectError(ValueError):
    """Raised when an incomplete project cannot be resumed safely."""


@dataclass(frozen=True)
class ResumePlan:
    project_dir: Path
    metadata: dict
    chunks: tuple[TextChunk, ...]
    durable_chunks: int
    resume_index: int


def _project_directory(output_root: str | Path, project_name: str) -> Path:
    root = Path(output_root).resolve()
    if not project_name or Path(project_name).name != project_name:
        raise ResumeProjectError("Select a project directly below audiobook_outputs")
    project = (root / project_name).resolve()
    if project.parent != root:
        raise ResumeProjectError("Refusing to access a project outside audiobook_outputs")
    return project


def load_project_metadata(output_root: str | Path, project_name: str) -> tuple[Path, dict]:
    project = _project_directory(output_root, project_name)
    metadata_path = project / "metadata.json"
    chunks_dir = project / "chunks"
    if not metadata_path.is_file() or not chunks_dir.is_dir():
        raise ResumeProjectError("The selected project has no resumable chunks")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeProjectError("The selected project's metadata is unreadable") from error
    if metadata.get("output_file"):
        raise ResumeProjectError("The selected project is already complete")
    return project, metadata


def incomplete_project_choices(output_root: str | Path) -> list[tuple[str, str]]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    choices = []
    for project in root.iterdir():
        if not project.is_dir():
            continue
        try:
            _, metadata = load_project_metadata(root, project.name)
            completed = int(metadata.get("completed_chunks", 0))
            total = int(metadata["total_chunks"])
        except (KeyError, TypeError, ValueError, OSError, ResumeProjectError):
            continue
        label = f"{project.name} — recorded {completed:,}/{total:,}"
        choices.append((project.stat().st_mtime, label, project.name))
    choices.sort(reverse=True)
    return [(label, name) for _, label, name in choices]


def _valid_wav(path: Path, sample_rate: int) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            return (
                audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and audio.getframerate() == sample_rate
                and audio.getnframes() > 0
            )
    except (OSError, EOFError, wave.Error):
        return False


def contiguous_chunk_count(
    chunks_dir: str | Path,
    total_chunks: int,
    sample_rate: int,
) -> int:
    """Return the durable WAV prefix, stopping before interrupted temp output."""

    directory = Path(chunks_dir)
    durable = 0
    for index in range(total_chunks):
        if not _valid_wav(directory / f"{index:06d}.wav", sample_rate):
            break
        durable = index + 1

    temp_pattern = re.compile(r"^\.(\d{6})\.normalized-[^.]+\.wav$")
    for path in directory.glob(".*.normalized-*.wav"):
        match = temp_pattern.match(path.name)
        if match:
            durable = min(durable, int(match.group(1)))
    return durable


def build_resume_plan(
    output_root: str | Path,
    project_name: str,
    book: EpubBook,
    sample_rate: int,
) -> ResumePlan:
    project, metadata = load_project_metadata(output_root, project_name)
    try:
        settings = metadata["settings"]
        max_chars = int(settings["max_chars"])
        batch_size = int(settings["batch_size"])
        saved_chunks = metadata["chunks"]
        total_chunks = int(metadata["total_chunks"])
    except (KeyError, TypeError, ValueError) as error:
        raise ResumeProjectError("The selected project is missing generation settings") from error
    if batch_size < 1:
        raise ResumeProjectError("The selected project has an invalid batch size")

    chunks = tuple(chunk_book(book, max_chars=max_chars))
    if total_chunks != len(chunks) or len(saved_chunks) != len(chunks):
        raise ResumeProjectError(
            "The uploaded EPUB does not produce the same chunk count as this project"
        )
    for index, (saved, chunk) in enumerate(zip(saved_chunks, chunks)):
        if (
            saved.get("index") != index
            or saved.get("chapter_index") != chunk.chapter_index
            or saved.get("chapter_title") != chunk.chapter_title
            or saved.get("text") != chunk.text
        ):
            raise ResumeProjectError(
                f"The uploaded EPUB differs from the saved project at chunk {index:,}"
            )

    durable = contiguous_chunk_count(project / "chunks", len(chunks), sample_rate)
    resume_index = len(chunks) if durable == len(chunks) else (durable // batch_size) * batch_size
    return ResumePlan(project, metadata, chunks, durable, resume_index)
