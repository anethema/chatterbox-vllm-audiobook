"""Discovery and validation helpers for resumable audiobook projects."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4
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


PROJECT_INPUTS_DIRECTORY = "inputs"
PROJECT_EPUB_NAME = "source.epub"
REFERENCE_AUDIO_PREFIX = "reference-audio"
PROJECT_PROGRESS_NAME = "progress.json"


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
    progress_path = project / PROJECT_PROGRESS_NAME
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            for key in ("completed_chunks", "scheduled_chunks"):
                if key in progress:
                    metadata[key] = int(progress[key])
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return project, metadata


def write_project_progress(
    project_dir: str | Path,
    completed_chunks: int,
    scheduled_chunks: int,
) -> Path:
    project = Path(project_dir)
    progress_path = project / PROJECT_PROGRESS_NAME
    temporary = project / f".{PROJECT_PROGRESS_NAME}-{uuid4().hex}.tmp"
    data = {
        "completed_chunks": int(completed_chunks),
        "scheduled_chunks": int(scheduled_chunks),
    }
    try:
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, progress_path)
    finally:
        temporary.unlink(missing_ok=True)
    return progress_path


def saved_project_inputs(project_dir: str | Path) -> tuple[Path | None, Path | None]:
    inputs = Path(project_dir) / PROJECT_INPUTS_DIRECTORY
    epub = inputs / PROJECT_EPUB_NAME
    references = sorted(
        (path for path in inputs.glob(f"{REFERENCE_AUDIO_PREFIX}.*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if inputs.is_dir() else []
    return (epub if epub.is_file() else None, references[0] if references else None)


def _copy_atomically(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    temporary = destination.with_name(f".{destination.name}-{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def persist_project_inputs(
    project_dir: str | Path,
    epub_path: str | Path,
    reference_audio_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically retain the source EPUB and voice reference with a project."""

    project = Path(project_dir)
    if not project.is_dir():
        raise ResumeProjectError("The audiobook project directory is missing")
    source_epub = Path(epub_path)
    source_reference = Path(reference_audio_path)
    if not source_epub.is_file() or source_epub.suffix.lower() != ".epub":
        raise ResumeProjectError("The source EPUB is missing or invalid")
    if not source_reference.is_file():
        raise ResumeProjectError("The reference audio file is missing")

    inputs = project / PROJECT_INPUTS_DIRECTORY
    inputs.mkdir(exist_ok=True)
    saved_epub = inputs / PROJECT_EPUB_NAME
    reference_suffix = source_reference.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", reference_suffix):
        reference_suffix = ".audio"
    saved_reference = inputs / f"{REFERENCE_AUDIO_PREFIX}{reference_suffix}"
    _copy_atomically(source_epub, saved_epub)
    _copy_atomically(source_reference, saved_reference)
    for old_reference in inputs.glob(f"{REFERENCE_AUDIO_PREFIX}.*"):
        if old_reference != saved_reference and old_reference.is_file():
            old_reference.unlink()
    return saved_epub, saved_reference


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
