"""Cache-aware download support for Chatterbox checkpoint variants."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from time import monotonic

from huggingface_hub import (
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
)
from huggingface_hub.constants import HF_HUB_CACHE

from .model_variants import (
    ENGLISH_V1_MODEL_ID,
    MULTILINGUAL_CHECKPOINTS,
    resolve_model_id,
)


DEFAULT_REPO_ID = "ResembleAI/chatterbox"
ENGLISH_V1_REVISION = "1b475dffa71fb191cb6d5901215eb6f55635a9b6"

# ChatterboxTTS imports these central definitions when loading models.
ENGLISH_V1_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
)
MULTILINGUAL_FILES = (
    "ve.safetensors",
    "s3gen.safetensors",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)

ProgressCallback = Callable[[float, str], None]
_PROGRESS_POLL_SECONDS = 1.0


def _download_plan(model_id: str, revision: str | None) -> tuple[str, str, tuple[str, ...]]:
    """Resolve a stable model ID to its checkpoint revision and required files."""

    resolved_model_id = resolve_model_id(model_id)
    if resolved_model_id == ENGLISH_V1_MODEL_ID:
        return resolved_model_id, revision or ENGLISH_V1_REVISION, ENGLISH_V1_FILES

    t3_filename, default_revision = MULTILINGUAL_CHECKPOINTS[resolved_model_id]
    return (
        resolved_model_id,
        revision or default_revision,
        (t3_filename, *MULTILINGUAL_FILES),
    )


def _report(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    """Send a well-formed progress event when the caller requested one."""

    if progress is not None:
        progress(min(1.0, max(0.0, fraction)), message)


def _format_bytes(size: int) -> str:
    """Format a byte count compactly for a progress message."""

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _incomplete_blob_path(
    repo_id: str, filename: str, revision: str
) -> tuple[Path | None, int | None]:
    """Find the hub-owned temporary blob and expected size, if metadata is available."""

    try:
        metadata = get_hf_file_metadata(
            hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
        )
    except Exception:
        # Metadata is optional. hf_hub_download remains the authority for
        # actual network and authentication failures.
        return None, None

    expected_size = metadata.size
    if (
        metadata.xet_file_data is not None
        or not metadata.etag
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        return None, None

    repository_cache = f"models--{repo_id.replace('/', '--')}"
    incomplete_path = (
        Path(HF_HUB_CACHE) / repository_cache / "blobs" / f"{metadata.etag}.incomplete"
    )
    return incomplete_path, expected_size


def _download_file(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    index: int,
    total: int,
    progress: ProgressCallback | None,
) -> Path:
    """Download one file while reporting elapsed time and cache blob byte progress."""

    initial_fraction = (index - 1) / total
    _report(
        progress,
        initial_fraction,
        f"Downloading checkpoint file {index}/{total}: {filename} (0s elapsed)",
    )
    incomplete_path, expected_size = (
        _incomplete_blob_path(repo_id, filename, revision)
        if progress is not None
        else (None, None)
    )
    started_at = monotonic()

    # huggingface_hub 0.35 does not offer a public tqdm hook on
    # hf_hub_download. A separate worker lets us poll only its standard
    # incomplete blob, so there is no global monkeypatch or custom transfer.
    # The hub still owns resume behavior and moves the finished blob itself.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            hf_hub_download,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
        while True:
            try:
                downloaded_path = future.result(timeout=_PROGRESS_POLL_SECONDS)
            except TimeoutError:
                # A downloader can itself raise TimeoutError. Only treat the
                # exception as a polling timeout while the worker still runs.
                if future.done():
                    return Path(future.result())

                elapsed_seconds = int(monotonic() - started_at)
                fraction = initial_fraction
                byte_progress = ""
                if incomplete_path is not None and expected_size is not None:
                    try:
                        downloaded_bytes = incomplete_path.stat().st_size
                    except OSError:
                        downloaded_bytes = 0
                    downloaded_bytes = min(downloaded_bytes, expected_size)
                    file_fraction = downloaded_bytes / expected_size
                    fraction = (index - 1 + file_fraction) / total
                    byte_progress = (
                        f": {_format_bytes(downloaded_bytes)} / "
                        f"{_format_bytes(expected_size)} ({file_fraction:.0%})"
                    )
                _report(
                    progress,
                    fraction,
                    (
                        f"Downloading checkpoint file {index}/{total}: {filename}"
                        f"{byte_progress} ({elapsed_seconds}s elapsed)"
                    ),
                )
            else:
                return Path(downloaded_path)


def ensure_model_downloaded(
    model_id: str,
    progress: ProgressCallback | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
) -> Path:
    """Return the standard Hugging Face snapshot directory for a model variant.

    Cached files are discovered without network access. Missing files use
    hf_hub_download with its default cache behavior, including interrupted
    download resume support. Progress reports cached files, current file count,
    elapsed time, and hub blob bytes when metadata is available.
    """

    resolved_model_id, selected_revision, filenames = _download_plan(model_id, revision)
    total = len(filenames)
    _report(progress, 0.0, f"Preparing {resolved_model_id} checkpoint ({total} files)")

    cached_paths = {
        filename: try_to_load_from_cache(
            repo_id=repo_id,
            filename=filename,
            revision=selected_revision,
        )
        for filename in filenames
    }

    paths: list[Path] = []
    for index, filename in enumerate(filenames, start=1):
        cached_path = cached_paths[filename]
        if isinstance(cached_path, str):
            path = Path(cached_path)
            _report(
                progress,
                index / total,
                f"Using cached checkpoint file {index}/{total}: {filename}",
            )
        else:
            path = _download_file(
                repo_id=repo_id,
                filename=filename,
                revision=selected_revision,
                index=index,
                total=total,
                progress=progress,
            )
            _report(
                progress,
                index / total,
                f"Downloaded checkpoint file {index}/{total}: {filename}",
            )
        paths.append(path)

    snapshot_directory = paths[0].parent
    _report(progress, 1.0, f"{resolved_model_id} checkpoint is ready")
    return snapshot_directory

