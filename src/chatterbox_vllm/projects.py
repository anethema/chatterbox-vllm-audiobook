"""Discovery and validation helpers for resumable audiobook projects."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import stat
from uuid import uuid4
import wave

from chatterbox_vllm.epub import EpubBook, TextChunk, chunk_book
from chatterbox_vllm.model_variants import (
    LEGACY_PROJECT_MODEL_ID,
    model_label,
    resolve_model_id,
)


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
PROJECT_QUALITY_SCAN_NAME = "quality-scan.json"
QUALITY_SCAN_CHECKPOINT_SCHEMA_VERSION = 1
QUALITY_SCAN_DETECTOR_VERSION = 2


def wav_file_identity(path: str | Path) -> dict[str, int] | None:
    """Return the inexpensive identity used to validate a scanned WAV later."""

    try:
        status = Path(path).stat()
    except OSError:
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    return {"size": int(status.st_size), "mtime_ns": int(status.st_mtime_ns)}


def _write_json_atomically(path: Path, data: object, *, indent: int | None = None) -> None:
    """Replace ``path`` only after its complete JSON payload reaches a temp file."""

    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=indent) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _parse_quality_scan_checkpoint(data: object, processing_signature=None) -> dict[int, dict[str, int]]:
    """Read a strict quality-scan cache, failing safely on malformed data."""

    if not isinstance(data, dict):
        return {}
    expected_keys = {
        "schema_version",
        "detector_version",
        "verified_clean_chunks",
    }
    if processing_signature is not None:
        expected_keys.add("processing_signature")
        if data.get("processing_signature") != processing_signature:
            return {}
    if set(data) != expected_keys:
        return {}
    if data.get("schema_version") != QUALITY_SCAN_CHECKPOINT_SCHEMA_VERSION:
        return {}
    if data.get("detector_version") != QUALITY_SCAN_DETECTOR_VERSION:
        return {}
    raw_verified = data.get("verified_clean_chunks")
    if not isinstance(raw_verified, dict):
        return {}

    verified: dict[int, dict[str, int]] = {}
    for raw_index, raw_identity in raw_verified.items():
        if (
            not isinstance(raw_index, str)
            or not raw_index.isdecimal()
            or not isinstance(raw_identity, dict)
            or set(raw_identity) != {"size", "mtime_ns"}
        ):
            return {}
        size = raw_identity["size"]
        mtime_ns = raw_identity["mtime_ns"]
        if (
            isinstance(size, bool)
            or isinstance(mtime_ns, bool)
            or not isinstance(size, int)
            or not isinstance(mtime_ns, int)
            or size <= 0
            or mtime_ns < 0
        ):
            return {}
        verified[int(raw_index)] = {"size": size, "mtime_ns": mtime_ns}
    return verified


def load_quality_scan_checkpoint(project_dir: str | Path, *, processing_signature=None) -> dict[int, dict[str, int]]:
    """Load verified-clean chunk identities, or an empty cache if unsafe to use."""

    progress_path = Path(project_dir) / PROJECT_QUALITY_SCAN_NAME
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return _parse_quality_scan_checkpoint(data, processing_signature)


def quality_scan_checkpoint_entry_matches(
    checkpoint: dict[int, dict[str, int]],
    chunk_index: int,
    path: str | Path,
) -> bool:
    """Whether a saved clean result still belongs to this exact WAV file."""

    identity = wav_file_identity(path)
    return identity is not None and checkpoint.get(int(chunk_index)) == identity


def write_quality_scan_checkpoint(
    project_dir: str | Path,
    verified_clean_chunks: dict[int, dict[str, int]],
    *,
    processing_signature=None,
) -> Path:
    """Atomically persist clean WAV identities verified by the current detector."""

    project = Path(project_dir)
    checkpoint_path = project / PROJECT_QUALITY_SCAN_NAME
    clean_entries = {
        f"{index:06d}": identity
        for index, identity in sorted(verified_clean_chunks.items())
        if (
            isinstance(index, int)
            and index >= 0
            and isinstance(identity, dict)
            and set(identity) == {"size", "mtime_ns"}
            and isinstance(identity["size"], int)
            and not isinstance(identity["size"], bool)
            and identity["size"] > 0
            and isinstance(identity["mtime_ns"], int)
            and not isinstance(identity["mtime_ns"], bool)
            and identity["mtime_ns"] >= 0
        )
    }
    data = {
        "schema_version": QUALITY_SCAN_CHECKPOINT_SCHEMA_VERSION,
        "detector_version": QUALITY_SCAN_DETECTOR_VERSION,
        "verified_clean_chunks": clean_entries,
    }
    if processing_signature is not None:
        data["processing_signature"] = processing_signature
    _write_json_atomically(checkpoint_path, data, indent=2)
    return checkpoint_path


def delete_quality_scan_checkpoint(project_dir: str | Path) -> None:
    """Remove resumable scan metadata after a project is fully assembled."""

    (Path(project_dir) / PROJECT_QUALITY_SCAN_NAME).unlink(missing_ok=True)


def load_quality_summary(project_dir: Path, total_chunks: int) -> tuple[set[int], set[int], set[int]]:
    """Restore whole-project repair counts without trusting malformed metadata."""
    try:
        data = json.loads((project_dir / "quality-summary.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1 or data.get("total_chunks") != total_chunks:
            raise ValueError("incompatible quality summary")
        values = []
        for name in ("detected", "fixed", "retained"):
            indices = data[name]
            if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < total_chunks for i in indices):
                raise ValueError("invalid quality indices")
            values.append(set(indices))
        detected, fixed, retained = values
        if not fixed <= detected or not retained <= detected or fixed & retained:
            raise ValueError("inconsistent quality summary")
        return detected, fixed, retained
    except (OSError, ValueError, KeyError, TypeError):
        return set(), set(), set()


def write_quality_summary(project_dir: Path, total_chunks: int, detected: set[int], fixed: set[int], retained: set[int]) -> None:
    path = project_dir / "quality-summary.json"
    _write_json_atomically(
        path,
        {
            "version": 1,
            "total_chunks": total_chunks,
            "detected": sorted(detected),
            "fixed": sorted(fixed),
            "retained": sorted(retained),
        },
    )


def project_model_id(metadata: dict) -> str:
    """Return a project's model, treating pre-versioning projects as English V1."""

    raw_model_id = metadata.get("model_id", LEGACY_PROJECT_MODEL_ID)
    try:
        return resolve_model_id(raw_model_id)
    except ValueError as error:
        raise ResumeProjectError(
            f"The selected project names an unsupported model: {raw_model_id!r}"
        ) from error


def _project_directory(output_root: str | Path, project_name: str) -> Path:
    root = Path(output_root).resolve()
    if not isinstance(project_name, str) or not project_name or Path(project_name).name != project_name:
        raise ResumeProjectError("Select a project directly below audiobook_outputs")
    project = (root / project_name).resolve()
    if project.parent != root:
        raise ResumeProjectError("Refusing to access a project outside audiobook_outputs")
    return project


def load_project_metadata(output_root: str | Path, project_name: str) -> tuple[Path, dict]:
    """Load an incomplete project's metadata after validating its layout."""

    project = _project_directory(output_root, project_name)
    metadata_path = project / "metadata.json"
    chunks_dir = project / "chunks"
    if not metadata_path.is_file() or not chunks_dir.is_dir():
        raise ResumeProjectError("The selected project has no resumable chunks")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResumeProjectError("The selected project's metadata is unreadable") from error
    if not isinstance(metadata, dict):
        raise ResumeProjectError("The selected project's metadata must be an object")
    if "settings" in metadata and not isinstance(metadata["settings"], dict):
        raise ResumeProjectError("The selected project's settings must be an object")
    if metadata.get("output_file"):
        raise ResumeProjectError("The selected project is already complete")
    progress_path = project / PROJECT_PROGRESS_NAME
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if not isinstance(progress, dict):
                raise ValueError("invalid progress record")
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
    """Atomically save the small counters used to refresh resume choices."""

    progress_path = Path(project_dir) / PROJECT_PROGRESS_NAME
    _write_json_atomically(
        progress_path,
        {
            "completed_chunks": int(completed_chunks),
            "scheduled_chunks": int(scheduled_chunks),
        },
        indent=2,
    )
    return progress_path


def saved_project_inputs(project_dir: str | Path) -> tuple[Path | None, Path | None]:
    """Return the retained EPUB and most recently modified voice reference."""

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
    """List resumable projects with a display label, newest project first."""

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
            saved_model_label = model_label(project_model_id(metadata))
        except (KeyError, TypeError, ValueError, OSError, ResumeProjectError):
            continue
        label = (
            f"{project.name} — recorded {completed:,}/{total:,} • "
            f"{saved_model_label}"
        )
        choices.append((project.stat().st_mtime, label, project.name))
    choices.sort(reverse=True)
    return [(label, name) for _, label, name in choices]


def _valid_wav(path: Path, sample_rate: int) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            frames = audio.getnframes()
            if (
                channels != 1
                or sample_width != 2
                or audio.getframerate() != sample_rate
                or frames <= 0
            ):
                return False
            audio.setpos(frames - 1)
            return len(audio.readframes(1)) == channels * sample_width
    except (OSError, EOFError, wave.Error):
        return False


def contiguous_chunk_count(
    chunks_dir: str | Path,
    total_chunks: int,
    sample_rate: int,
) -> int:
    """Return the durable WAV prefix, stopping before interrupted temp output.

    A normalization temp file newer than its final WAV means the process may
    have died before the atomic replacement. An older temp file is debris from
    an earlier interruption whose final WAV has since been regenerated.
    """

    directory = Path(chunks_dir)
    durable = 0
    for index in range(total_chunks):
        if not _valid_wav(directory / f"{index:06d}.wav", sample_rate):
            break
        durable = index + 1

    temp_pattern = re.compile(r"^\.(\d{6})\.normalized-[^.]+\.wav$")
    for path in directory.glob(".*.normalized-*.wav"):
        match = temp_pattern.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        final_path = directory / f"{index:06d}.wav"
        try:
            temp_mtime = path.stat().st_mtime_ns
            final_mtime = final_path.stat().st_mtime_ns
        except OSError:
            durable = min(durable, index)
            continue
        if final_mtime > temp_mtime and _valid_wav(final_path, sample_rate):
            path.unlink(missing_ok=True)
            continue
        durable = min(durable, index)
    return durable


def build_resume_plan(
    output_root: str | Path,
    project_name: str,
    book: EpubBook,
    sample_rate: int,
    expected_model_id: str | None = None,
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
    if not isinstance(saved_chunks, list) or any(not isinstance(item, dict) for item in saved_chunks):
        raise ResumeProjectError("The selected project has an invalid chunk plan")

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
    saved_model_id = project_model_id(metadata)
    if expected_model_id is not None:
        expected_model_id = resolve_model_id(expected_model_id)
        if resume_index < len(chunks) and saved_model_id != expected_model_id:
            raise ResumeProjectError(
                f"This project uses {model_label(saved_model_id)}, but the app is "
                f"running {model_label(expected_model_id)}. Restart with "
                f"CHATTERBOX_MODEL_VARIANT={saved_model_id} to resume its remaining "
                "speech chunks"
            )
    return ResumePlan(project, metadata, chunks, durable, resume_index)
