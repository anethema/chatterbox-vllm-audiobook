import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import sys
import tempfile
import time
import traceback
from uuid import uuid4

import gradio as gr
import numpy as np
import torch
import torchaudio as ta

from chatterbox_vllm.audio import (
    AudioQualityIssue,
    LOUDNESS_RANGE_LU,
    MAX_INTERNAL_PAUSE_SECONDS,
    TARGET_LUFS,
    TRUE_PEAK_DBTP,
    find_generated_audio_issues,
    format_audio_quality_issues,
    limit_internal_pauses_wav,
    normalize_speech_wav,
    prepared_reference_audio,
    wav_generated_audio_issues,
)
from chatterbox_vllm.background import BackgroundTaskPool
from chatterbox_vllm.downloads import register_completed_audiobook
from chatterbox_vllm.epub import (
    EpubBook,
    EpubError,
    TextChunk,
    chunk_book,
    load_epub,
    split_sentences,
    split_text_for_recovery,
)
from chatterbox_vllm.m4b import (
    AssemblyProgress,
    AssemblyStopped,
    assemble_audiobook,
    delete_intermediate_chunks,
    verify_m4b,
)
from chatterbox_vllm.memory import read_memory_status, release_unused_memory
from chatterbox_vllm.model_variants import (
    DEFAULT_MODEL_ID,
    MULTILINGUAL_V3_MODEL_ID,
    model_label,
    resolve_model_id,
)
from chatterbox_vllm.projects import (
    ResumeProjectError,
    build_resume_plan,
    contiguous_chunk_count,
    delete_quality_scan_checkpoint,
    incomplete_project_choices,
    load_quality_scan_checkpoint,
    load_project_metadata,
    persist_project_inputs,
    project_model_id,
    quality_scan_checkpoint_entry_matches,
    saved_project_inputs,
    wav_file_identity,
    write_quality_scan_checkpoint,
    write_project_progress,
)
from chatterbox_vllm.job_status import JobStatusStore, render_job_status
from chatterbox_vllm.progress import GenerationControl, estimate_progress, format_duration
from chatterbox_vllm.tts import ChatterboxTTS


DEVICE = "cuda"
OUTPUT_ROOT = Path(__file__).resolve().parent / "audiobook_outputs"
ACTIVE_MODEL_ID = resolve_model_id(
    os.environ.get("CHATTERBOX_MODEL_VARIANT", DEFAULT_MODEL_ID)
)
AUDIO_WORKERS = min(4, os.cpu_count() or 1)
MAX_PENDING_AUDIO_TASKS = AUDIO_WORKERS * 16
MEMORY_CLEANUP_BATCHES = 64
MEMORY_CLEANUP_HEADROOM = 4 * 1024 ** 3
MINIMUM_MEMORY_HEADROOM = 2 * 1024 ** 3
DEFAULT_BATCH_SIZE = 64
MAX_BATCH_SIZE = 64
MAX_WHOLE_CHUNK_ATTEMPTS = 2
MAX_SPLIT_PART_ATTEMPTS = 2
MAX_RECOVERY_SPLIT_DEPTH = 4
SPLIT_JOIN_SILENCE_SECONDS = 0.12
RESUME_SCAN_PROGRESS_UPDATE_SECONDS = 1.0
RESUME_SCAN_PROGRESS_UPDATE_CHUNKS = 100
QUALITY_SCAN_CHECKPOINT_UPDATE_SECONDS = 30.0
QUALITY_SCAN_CHECKPOINT_UPDATE_CHUNKS = 1000

JOB_MONITOR_CSS = """
#active-job-monitor { margin-top: 0.6rem; margin-bottom: 0.6rem; }
#active-job-monitor .active-job-monitor {
    border: 1px solid var(--border-color-primary, #9ca3af);
    border-radius: 8px;
    background: var(--block-background-fill, #ffffff);
    padding: 0.65rem 0.8rem;
}
#active-job-monitor .active-job-monitor__heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.45rem;
}
#active-job-monitor .active-job-monitor__heading span,
#active-job-monitor .active-job-monitor__metrics {
    color: var(--body-text-color-subdued, #6b7280);
    font-size: 0.9em;
}
#active-job-monitor .active-job-monitor__track {
    height: 0.55rem;
    overflow: hidden;
    border-radius: 999px;
    background: var(--neutral-300, #d1d5db);
}
#active-job-monitor .active-job-monitor__fill {
    height: 100%;
    min-width: 0;
    border-radius: inherit;
    background: var(--primary-500, #2563eb);
    transition: width 0.25s ease-out;
}
#active-job-monitor .active-job-monitor__metrics { margin-top: 0.4rem; }
#active-job-monitor .active-job-monitor__message { margin-top: 0.2rem; }
"""

config_seed = None
global_model = None
generation_control = GenerationControl()
job_status = JobStatusStore(OUTPUT_ROOT)
reference_preview_directory: tempfile.TemporaryDirectory | None = None


class GenerationStopped(Exception):
    pass


class MemoryPressureError(RuntimeError):
    pass


class GeneratedAudioValidationError(ValueError):
    """The model returned a waveform that cannot safely be saved."""


class GeneratedAudioQualityError(GeneratedAudioValidationError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__(format_audio_quality_issues(list(self.issues)))


@dataclass(frozen=True)
class RecoveredWaveform:
    waveform: torch.Tensor
    detected_quality_issues: bool
    retained_quality_issues: tuple[AudioQualityIssue, ...]
    retained_with_warning: bool = False


@dataclass(frozen=True)
class RecoveredSplitPart:
    waveform: torch.Tensor
    retained_quality_issues: tuple[AudioQualityIssue, ...]
    retained_with_warning: bool = False


@dataclass(frozen=True)
class SavedChunkQuality:
    """The final on-disk result after deterministic audio transforms."""

    path: Path
    verified_clean: bool


def _quality_log(message: str, *, color: str | None = None) -> None:
    colors = {"red": "\033[1;31m", "green": "\033[1;32m", "yellow": "\033[1;33m"}
    if color in colors and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        message = f"{colors[color]}{message}\033[0m"
    print(message, flush=True)


def _resume_scan_progress_message(
    scanned_chunks: int,
    total_chunks: int,
    elapsed_seconds: float,
    *,
    cached_verified_chunks: int = 0,
) -> str:
    """Describe resume validation progress and an estimate of its completion."""

    total = max(0, int(total_chunks))
    scanned = max(0, min(int(scanned_chunks), total))
    percent = 100.0 * scanned / total if total else 100.0
    if total == 0:
        eta = "ETA 0s"
    elif scanned:
        remaining_seconds = (
            max(0.0, float(elapsed_seconds)) * (total - scanned) / scanned
        )
        eta = f"ETA {format_duration(remaining_seconds)}"
    else:
        eta = "ETA calculating…"
    message = (
        "[Audio quality scan] Scanning existing chunks: "
        f"{scanned:,}/{total:,} ({percent:.1f}%) — {eta}"
    )
    if cached_verified_chunks:
        message += f" • skipped {cached_verified_chunks:,} cached verified"
    return message


def _report_resume_scan_progress(
    progress,
    scanned_chunks: int,
    total_chunks: int,
    started_at: float,
    *,
    log_stdout: bool = True,
    cached_verified_chunks: int = 0,
) -> None:
    message = _resume_scan_progress_message(
        scanned_chunks,
        total_chunks,
        time.perf_counter() - started_at,
        cached_verified_chunks=cached_verified_chunks,
    )
    progress(scanned_chunks / total_chunks if total_chunks else 1.0, desc=message)
    if log_stdout:
        _quality_log(message, color="green")


def _log_batch_quality_summary(
    start: int,
    count: int,
    retained_quality_chunks: list[int],
) -> None:
    quality_ok = count - len(retained_quality_chunks)
    label = f"Batch {start:06d}-{start + count - 1:06d}"
    if retained_quality_chunks:
        _quality_log(
            f"[Audio quality scan] {label}: {quality_ok}/{count} OK after "
            "recovery; retained with warnings: "
            f"{', '.join(f'{index:06d}' for index in retained_quality_chunks)}",
            color="red",
        )
    else:
        _quality_log(
            f"[Audio quality scan] {label}: {count}/{count} OK",
            color="green",
        )


def _log_project_quality_summary(
    total_chunks: int,
    detected_chunks: set[int],
    fixed_chunks: set[int],
    retained_chunks: set[int],
) -> None:
    retained_labels = (
        ", ".join(f"{index:06d}" for index in sorted(retained_chunks))
        if retained_chunks
        else "none"
    )
    _quality_log(
        f"[Audio quality summary] Project complete: {total_chunks:,} chunks "
        f"scanned; bad chunks detected: {len(detected_chunks):,}; fixed: "
        f"{len(fixed_chunks):,}; retained with warnings: "
        f"{len(retained_chunks):,} ({retained_labels})",
        color="red" if retained_chunks else "green",
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    global config_seed
    config_seed = seed


def selected_seed(seed_num) -> int | None:
    seed = int(seed_num or 0)
    if seed == 0:
        return None
    set_seed(seed)
    return seed


def prepare_reference_preview(
    audio_prompt_path: str | None,
    denoise_reference: bool = False,
) -> str | None:
    """Return the persistent processed reference preview used for conditioning."""

    if not audio_prompt_path:
        return None
    global reference_preview_directory
    if reference_preview_directory is None:
        reference_preview_directory = tempfile.TemporaryDirectory(
            prefix="chatterbox-reference-previews-"
        )
    destination = (
        Path(reference_preview_directory.name)
        / f"normalized-reference-{uuid4().hex}.wav"
    )
    with prepared_reference_audio(
        audio_prompt_path,
        24000,
        denoise=bool(denoise_reference),
    ) as prepared:
        shutil.copyfile(prepared, destination)
    print(
        "[Reference preparation] Gradio player and conditioning now use: "
        f"{destination}",
        flush=True,
    )
    return str(destination)


def prepare_uploaded_reference(
    audio_prompt_path: str | None,
    denoise_reference: bool,
) -> tuple[str | None, str | None]:
    """Process a user upload while retaining its untouched source path."""

    return (
        prepare_reference_preview(audio_prompt_path, denoise_reference),
        audio_prompt_path,
    )


def load_model():
    print(f"Loading {model_label(ACTIVE_MODEL_ID)} ({ACTIVE_MODEL_ID})...")
    global global_model
    global_model = ChatterboxTTS.from_model_id(
        ACTIVE_MODEL_ID,
        gpu_memory_utilization=0.6,
        max_model_len=1000,

        # Disable CUDA graphs - it's causing tensors to get corrupted right now.
        enforce_eager=True,
    )
    return global_model


def generation_arguments(exaggeration, cfg_weight, temperature, diffusion_steps,
                         min_p, top_p, repetition_penalty, seed):
    return {
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "temperature": float(temperature),
        "diffusion_steps": int(diffusion_steps),
        "min_p": float(min_p),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "seed": seed,
    }


def generate_sample(text, audio_prompt_path, denoise_reference, exaggeration,
                    cfg_weight, temperature, seed_num, diffusion_steps, min_p, top_p,
                    repetition_penalty):
    if not text or not text.strip():
        raise gr.Error("Enter some text to synthesize.")

    seed = selected_seed(seed_num)
    args = generation_arguments(
        exaggeration, cfg_weight, temperature, diffusion_steps, min_p, top_p,
        repetition_penalty, seed,
    )
    print(f"Using text: {text}")
    print(f"Using audio_prompt_path: {audio_prompt_path}")
    print(f"Using settings: {args}")

    retry_seeds: set[int] = set()
    wav = global_model.generate(
        [text.strip()],
        audio_prompt_path=audio_prompt_path,
        denoise_reference=bool(denoise_reference),
        **args,
    )
    recovery = _recover_generated_waveform(
        wav[0],
        text.strip(),
        global_model.sr,
        "text sample",
        lambda: global_model.generate(
            [text.strip()],
            audio_prompt_path=audio_prompt_path,
            denoise_reference=bool(denoise_reference),
            **_retry_generation_args(args, retry_seeds),
        )[0],
        lambda part: global_model.generate(
            [part],
            audio_prompt_path=audio_prompt_path,
            denoise_reference=bool(denoise_reference),
            **_retry_generation_args(args, retry_seeds),
        )[0],
    )
    waveform = recovery.waveform
    with tempfile.TemporaryDirectory(prefix="chatterbox-preview-") as directory:
        output_path = Path(directory) / "preview.wav"
        ta.save(
            str(output_path),
            waveform,
            global_model.sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        normalize_speech_wav(output_path, global_model.sr)
        if global_model.model_id == MULTILINGUAL_V3_MODEL_ID:
            limit_internal_pauses_wav(output_path, global_model.sr)
        normalized, sample_rate = ta.load(str(output_path))
    return (sample_rate, normalized.squeeze(0).numpy())


def inspect_epub_file(epub_path, max_chars):
    if not epub_path:
        return "Upload a DRM-free EPUB to inspect its chapters before generation."
    try:
        book = load_epub(epub_path)
        chunks = chunk_book(book, max_chars=int(max_chars))
        characters = sum(len(chapter.text) for chapter in book.chapters)
        return (
            f"**{book.title}** — {len(book.chapters)} readable spine documents, "
            f"{characters:,} characters, and **{len(chunks):,} speech chunks** at "
            f"the current {int(max_chars)}-character limit."
        )
    except EpubError as error:
        return f"❌ {error}"


def _job_monitor_render():
    """Return durable job state for any browser session polling the app."""

    snapshot = job_status.snapshot()
    return (
        render_job_status(snapshot, format_duration=format_duration),
        gr.Button(interactive=snapshot.state == "running"),
        gr.Button(interactive=not snapshot.active),
        gr.Button(interactive=not snapshot.active),
    )


def request_generation_stop():
    if not generation_control.request_stop():
        return "No EPUB generation is currently running."
    job_status.request_stop()
    return (
        "⏹️ Stop requested. The current vLLM batch will finish, or active M4B "
        "encoding processes will be stopped."
    )


def refresh_resume_projects():
    choices = incomplete_project_choices(OUTPUT_ROOT)
    return gr.Dropdown(choices=choices, value=None)


def inspect_resume_project(project_name):
    if not project_name:
        return (
            "Select an incomplete project. Saved inputs will load automatically.",
            None,
            None,
            False,
        )
    try:
        for label, name in incomplete_project_choices(OUTPUT_ROOT):
            if name == project_name:
                project_dir = OUTPUT_ROOT / name
                _, metadata = load_project_metadata(OUTPUT_ROOT, name)
                saved_epub, saved_reference = saved_project_inputs(project_dir)
                denoise_reference = bool(
                    metadata.get("settings", {}).get("denoise_reference", False)
                )
                epub_status = "saved EPUB loaded" if saved_epub else "upload the original EPUB once"
                reference_status = (
                    "saved reference audio loaded"
                    if saved_reference
                    else "upload the reference audio once"
                )
                return (
                    f"Selected **{label}** — {epub_status}; {reference_status}. "
                    "Resume validates the EPUB before writing.",
                    str(saved_epub) if saved_epub else None,
                    str(saved_reference) if saved_reference else None,
                    denoise_reference,
                )
    except (OSError, ResumeProjectError) as error:
        return f"❌ Could not inspect incomplete projects: {error}", None, None, False
    return (
        "❌ The selected incomplete project is no longer available.",
        None,
        None,
        False,
    )


def _safe_project_name(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-._")
    return (slug[:80] or "audiobook") + "-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]


def _waveform_for_save(
    waveform,
    text: str,
    sample_rate: int,
    *,
    allow_quality_issues: bool = False,
) -> torch.Tensor:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    try:
        tensor = torch.as_tensor(waveform).to(dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise GeneratedAudioValidationError(
            "The model returned a waveform that could not be converted to audio"
        ) from error
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor.transpose(0, 1)
    if tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] == 0:
        raise GeneratedAudioValidationError(
            "The model returned an empty or non-mono waveform"
        )
    if not torch.isfinite(tensor).all():
        raise GeneratedAudioValidationError("The model returned NaN or infinite audio")

    duration = tensor.shape[1] / sample_rate
    words = max(1, len(text.split()))
    # Keep a modest floor for short utterances, while requiring enough time for
    # every word in longer chunks. A fixed cap lets a long response be accepted
    # even when the model only returned its opening fragment.
    if duration < max(0.25, 0.04 * words):
        raise GeneratedAudioValidationError(
            f"Generated audio is truncated ({duration:.2f}s for {words} words)"
        )
    if duration > 15.0 + 5.0 * words:
        raise GeneratedAudioValidationError(
            f"Generated audio is implausibly long ({duration:.1f}s for {words} words)"
        )
    issues = find_generated_audio_issues(
        tensor.numpy(),
        sample_rate,
        include_vad=True,
    )
    if issues and not allow_quality_issues:
        raise GeneratedAudioQualityError(issues)
    return tensor


def _retry_generation_args(
    settings: dict,
    used_seeds: set[int] | None = None,
) -> dict:
    retry_args = dict(settings)
    previous_seed = retry_args.get("seed")
    used_seeds = used_seeds if used_seeds is not None else set()
    if previous_seed is not None:
        used_seeds.add(int(previous_seed))
    fresh_seed = secrets.randbelow(2**31 - 1) + 1
    while fresh_seed in used_seeds:
        fresh_seed = secrets.randbelow(2**31 - 1) + 1
    used_seeds.add(fresh_seed)
    retry_args["seed"] = fresh_seed
    _quality_log(
        f"[Audio quality repair] Using fresh random seed {fresh_seed}",
        color="yellow",
    )
    return retry_args


def _join_split_waveforms(
    waveforms: list[torch.Tensor],
    sample_rate: int,
) -> torch.Tensor:
    silence = torch.zeros(
        (1, max(1, round(sample_rate * SPLIT_JOIN_SILENCE_SECONDS))),
        dtype=torch.float32,
    )
    joined: list[torch.Tensor] = []
    for index, waveform in enumerate(waveforms):
        if index:
            joined.append(silence)
        joined.append(waveform)
    return torch.cat(joined, dim=1)


def _quality_issues_from_error(
    error: GeneratedAudioValidationError | None,
) -> tuple[AudioQualityIssue, ...]:
    if isinstance(error, GeneratedAudioQualityError):
        return tuple(error.issues)
    return ()


def _silence_waveform(text: str, sample_rate: int) -> torch.Tensor:
    """Return a short, valid placeholder when every model result is unusable."""

    words = max(1, len(text.split()))
    duration = min(3.0, max(0.25, 0.16 * words))
    return torch.zeros(
        (1, max(1, round(sample_rate * duration))),
        dtype=torch.float32,
    )


def _best_effort_waveform(
    audio,
    text: str,
    sample_rate: int,
    label: str,
) -> tuple[torch.Tensor, bool]:
    """Retain scan-only issues, but never retain a structurally invalid waveform."""

    try:
        return _waveform_for_save(
            audio,
            text,
            sample_rate,
            allow_quality_issues=True,
        ), False
    except GeneratedAudioValidationError as error:
        _quality_log(
            f"[Audio quality warning] {label}: {error}; using a short silent "
            "fallback so generation can continue",
            color="red",
        )
        return _silence_waveform(text, sample_rate), True


def _recover_split_part(
    text: str,
    sample_rate: int,
    label: str,
    generate_part,
    depth: int,
) -> RecoveredSplitPart:
    last_error: GeneratedAudioValidationError | None = None
    last_audio = None
    for attempt in range(MAX_SPLIT_PART_ATTEMPTS):
        audio = generate_part(text)
        last_audio = audio
        try:
            waveform = _waveform_for_save(audio, text, sample_rate)
            _quality_log(
                f"[Audio quality repair] {label}: shorter part passed the "
                "full-waveform scan",
                color="green",
            )
            return RecoveredSplitPart(waveform, ())
        except GeneratedAudioValidationError as error:
            last_error = error
            _quality_log(
                f"[Audio quality scan] {label}: found {error}",
                color="red",
            )
            if attempt + 1 < MAX_SPLIT_PART_ATTEMPTS:
                _quality_log(
                    f"[Audio quality repair] {label}: regenerating shorter part "
                    f"with another seed (retry {attempt + 1}/"
                    f"{MAX_SPLIT_PART_ATTEMPTS - 1})",
                    color="yellow",
                )

    if len(split_sentences(text)) <= 1:
        _quality_log(
            f"[Audio quality warning] {label}: single-sentence audio still has "
            f"{last_error}; using bounded best-effort recovery; included anyway "
            "so generation can continue",
            color="red",
        )
        waveform, _ = _best_effort_waveform(
            last_audio,
            text,
            sample_rate,
            label,
        )
        return RecoveredSplitPart(
            waveform,
            _quality_issues_from_error(last_error),
            True,
        )

    nested_parts = split_text_for_recovery(text)
    can_split = (
        depth < MAX_RECOVERY_SPLIT_DEPTH
        and len(nested_parts) >= 2
        and all(len(part) < len(text) for part in nested_parts)
    )
    if not can_split:
        detail = str(last_error) if last_error is not None else "unknown audio issue"
        _quality_log(
            f"[Audio quality warning] {label}: reached the bounded split limit "
            f"with {detail}; using bounded best-effort recovery; included anyway "
            "so generation can continue",
            color="red",
        )
        waveform, _ = _best_effort_waveform(
            last_audio,
            text,
            sample_rate,
            label,
        )
        return RecoveredSplitPart(
            waveform,
            _quality_issues_from_error(last_error),
            True,
        )

    _quality_log(
        f"[Audio quality repair] {label}: shorter part still failed; recursively "
        f"splitting it into {len(nested_parts)} smaller parts at depth {depth + 1}/"
        f"{MAX_RECOVERY_SPLIT_DEPTH}",
        color="yellow",
    )
    recovered = [
        _recover_split_part(
            nested,
            sample_rate,
            f"{label}.{index + 1}",
            generate_part,
            depth + 1,
        )
        for index, nested in enumerate(nested_parts)
    ]
    combined = _join_split_waveforms(
        [part.waveform for part in recovered],
        sample_rate,
    )
    issues = find_generated_audio_issues(combined.numpy(), sample_rate)
    retained_quality_issues = tuple(
        issue
        for part in recovered
        for issue in part.retained_quality_issues
    )
    retained_with_warning = any(
        part.retained_with_warning for part in recovered
    )
    if issues:
        retained_quality_issues += tuple(issues)
        retained_with_warning = True
        _quality_log(
            f"[Audio quality warning] {label}: recursively split output retains "
            f"{format_audio_quality_issues(issues)}; included anyway so generation "
            "can continue",
            color="red",
        )
    return RecoveredSplitPart(
        combined,
        retained_quality_issues,
        retained_with_warning,
    )


def _recover_generated_waveform(
    initial_audio,
    text: str,
    sample_rate: int,
    label: str,
    regenerate_whole,
    generate_part,
) -> RecoveredWaveform:
    try:
        return RecoveredWaveform(
            _waveform_for_save(initial_audio, text, sample_rate),
            False,
            (),
        )
    except GeneratedAudioValidationError as error:
        _quality_log(
            f"[Audio quality scan] {label}: found {error}",
            color="red",
        )

    _quality_log(
        f"[Audio quality repair] {label}: regenerating the whole chunk "
        f"(retry 1/{MAX_WHOLE_CHUNK_ATTEMPTS - 1})",
        color="yellow",
    )
    retried_audio = regenerate_whole()
    retry_failure: GeneratedAudioValidationError | None = None
    try:
        waveform = _waveform_for_save(retried_audio, text, sample_rate)
        _quality_log(
            f"[Audio quality repair] {label}: whole-chunk replacement passed "
            "the full-waveform scan",
            color="green",
        )
        return RecoveredWaveform(waveform, True, ())
    except GeneratedAudioValidationError as retry_error:
        retry_failure = retry_error
        _quality_log(
            f"[Audio quality scan] {label}: whole-chunk replacement also "
            f"failed: {retry_error}",
            color="red",
        )

    parts = split_text_for_recovery(text)
    if len(split_sentences(text)) <= 1 or len(parts) < 2:
        _quality_log(
            f"[Audio quality warning] {label}: single-sentence audio still has "
            f"{retry_failure}; using bounded best-effort recovery; included anyway "
            "so generation can continue",
            color="red",
        )
        waveform, _ = _best_effort_waveform(
            retried_audio,
            text,
            sample_rate,
            label,
        )
        return RecoveredWaveform(
            waveform,
            True,
            _quality_issues_from_error(retry_failure),
            True,
        )
    _quality_log(
        f"[Audio quality repair] {label}: splitting repeatedly failed text "
        f"into {len(parts)} shorter parts ({', '.join(str(len(part)) for part in parts)} "
        "characters)",
        color="yellow",
    )
    part_waveforms = [
        _recover_split_part(
            part,
            sample_rate,
            f"{label} split {part_index + 1}/{len(parts)}",
            generate_part,
            1,
        )
        for part_index, part in enumerate(parts)
    ]

    combined = _join_split_waveforms(
        [part.waveform for part in part_waveforms],
        sample_rate,
    )
    retained_quality_issues = tuple(
        issue
        for part in part_waveforms
        for issue in part.retained_quality_issues
    )
    retained_with_warning = any(
        part.retained_with_warning for part in part_waveforms
    )
    try:
        result = _waveform_for_save(combined, text, sample_rate)
    except GeneratedAudioValidationError as combined_error:
        retained_quality_issues += _quality_issues_from_error(combined_error)
        retained_with_warning = True
        if isinstance(combined_error, GeneratedAudioQualityError):
            _quality_log(
                f"[Audio quality warning] {label}: combined sentence-sized output "
                f"retains {combined_error}; included anyway so generation can "
                "continue",
                color="red",
            )
        else:
            _quality_log(
                f"[Audio quality warning] {label}: combined sentence-sized output "
                f"failed validation ({combined_error}); using bounded best-effort "
                "recovery so generation can continue",
                color="red",
            )
        result, used_silence_fallback = _best_effort_waveform(
            combined,
            text,
            sample_rate,
            label,
        )
        retained_with_warning = retained_with_warning or used_silence_fallback
    if not retained_with_warning:
        _quality_log(
            f"[Audio quality repair] {label}: fixed with {len(parts)} shorter "
            "parts; combined waveform passed the full scan",
            color="green",
        )
    return RecoveredWaveform(
        result,
        True,
        retained_quality_issues,
        retained_with_warning,
    )


def _save_and_normalize_chunk(
    path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
    ffmpeg: str,
    maximum_internal_pause_seconds: float | None,
) -> Path:
    """Atomically replace ``path`` only after all deterministic transforms pass."""

    temporary = path.with_name(f".{path.stem}-{uuid4().hex}.pending{path.suffix}")
    try:
        ta.save(
            str(temporary),
            waveform,
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        normalize_speech_wav(temporary, sample_rate, ffmpeg=ffmpeg)
        if maximum_internal_pause_seconds is not None:
            limit_internal_pauses_wav(
                temporary,
                sample_rate,
                maximum_seconds=maximum_internal_pause_seconds,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _save_normalize_and_record_chunk(
    path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
    ffmpeg: str,
    maximum_internal_pause_seconds: float | None,
    *,
    cacheable: bool,
) -> SavedChunkQuality:
    """Persist a transformed chunk and verify the final bytes before caching."""

    _save_and_normalize_chunk(
        path,
        waveform,
        sample_rate,
        ffmpeg,
        maximum_internal_pause_seconds,
    )
    final_issues = wav_generated_audio_issues(path, sample_rate)
    return SavedChunkQuality(
        path=path,
        verified_clean=cacheable and not final_issues,
    )


def _limit_pauses_and_record_chunk(
    path: Path,
    sample_rate: int,
    *,
    maximum_seconds: float,
) -> SavedChunkQuality:
    """Limit pauses in an existing chunk and revalidate its final bytes."""

    limit_internal_pauses_wav(
        path,
        sample_rate,
        maximum_seconds=maximum_seconds,
    )
    return SavedChunkQuality(
        path=path,
        verified_clean=not wav_generated_audio_issues(path, sample_rate),
    )


def _write_metadata(path: Path, book: EpubBook, source_path: str, chunks: list[TextChunk],
                    settings: dict, completed: int, model_id: str,
                    output_path: str | None = None,
                    scheduled: int | None = None,
                    chunks_available: bool = True):
    saved_epub, saved_reference = saved_project_inputs(path.parent)
    data = {
        "title": book.title,
        "authors": list(book.authors),
        "language": book.language,
        "publisher": book.publisher,
        "description": book.description,
        "date": book.date,
        "identifier": book.identifier,
        "model_id": resolve_model_id(model_id),
        "source_epub": Path(source_path).name,
        "project_epub_file": (
            saved_epub.relative_to(path.parent).as_posix() if saved_epub else None
        ),
        "reference_audio_file": (
            saved_reference.relative_to(path.parent).as_posix()
            if saved_reference
            else None
        ),
        "settings": settings,
        "completed_chunks": completed,
        "scheduled_chunks": completed if scheduled is None else scheduled,
        "total_chunks": len(chunks),
        "output_file": output_path,
        "chunks": [
            {
                "index": index,
                "chapter_index": chunk.chapter_index,
                "chapter_title": chunk.chapter_title,
                "text": chunk.text,
                "audio_file": (
                    f"chunks/{index:06d}.wav"
                    if chunks_available and index < completed
                    else None
                ),
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mark_verified_project_complete(
    metadata_path: Path,
    project_dir: Path,
    book: EpubBook,
    source_path: str,
    chunks: list[TextChunk],
    settings: dict,
    model_id: str,
    output_path: str,
) -> None:
    """Durably mark a verified audiobook complete before cleanup begins."""

    _write_metadata(
        metadata_path,
        book,
        source_path,
        chunks,
        settings,
        len(chunks),
        model_id,
        output_path,
        chunks_available=False,
    )
    for description, cleanup in (
        (
            "progress record",
            lambda: (project_dir / "progress.json").unlink(missing_ok=True),
        ),
        (
            "quality scan checkpoint",
            lambda: delete_quality_scan_checkpoint(project_dir),
        ),
        ("intermediate chunks", lambda: delete_intermediate_chunks(project_dir)),
    ):
        try:
            cleanup()
        except (OSError, RuntimeError) as error:
            _quality_log(
                f"[Cleanup warning] Could not remove {description}: {error}. "
                "The verified audiobook remains complete.",
                color="yellow",
            )



def _record_durable_results(
    audio_tasks,
    durable_indices: set[int],
    durable_chunks: int,
    verified_clean_chunks: dict[int, dict[str, int]],
) -> tuple[int, int]:
    """Record completed background output and cache only verified final WAVs."""

    checkpoint_change_count = 0
    audio_tasks.check()
    for result in audio_tasks.take_results():
        if not isinstance(result, SavedChunkQuality):
            raise RuntimeError("Background audio task returned an unexpected result")
        path = result.path
        chunk_index = int(path.stem)
        durable_indices.add(int(Path(path).stem))
        identity = wav_file_identity(path)
        if result.verified_clean and identity is not None:
            if verified_clean_chunks.get(chunk_index) != identity:
                verified_clean_chunks[chunk_index] = identity
                checkpoint_change_count += 1
        else:
            if chunk_index in verified_clean_chunks:
                verified_clean_chunks.pop(chunk_index, None)
                checkpoint_change_count += 1
    while durable_chunks in durable_indices:
        durable_indices.remove(durable_chunks)
        durable_chunks += 1
    return durable_chunks, checkpoint_change_count


def _protect_system_memory(batch_number: int) -> None:
    status = read_memory_status()
    if (
        batch_number % MEMORY_CLEANUP_BATCHES == 0
        or status.headroom_bytes < MEMORY_CLEANUP_HEADROOM
    ):
        release_unused_memory()
        status = read_memory_status()
        print(
            "[MEMORY] available "
            f"{status.available_bytes / 1024 ** 3:.2f} GiB + swap free "
            f"{status.swap_free_bytes / 1024 ** 3:.2f} GiB"
        )
    if status.headroom_bytes < MINIMUM_MEMORY_HEADROOM:
        raise MemoryPressureError(
            "System memory is critically low. Generation paused before Linux could "
            "kill the process; restart the app and resume this project"
        )


def _monitored_resume_project_name(selected_project_name: str | None) -> str | None:
    """Prefer an explicit choice; otherwise expose only a safely resumable job."""

    if selected_project_name:
        return selected_project_name
    snapshot = job_status.snapshot()
    if snapshot.state in {"stopped", "interrupted", "failed"}:
        return snapshot.project_id
    return None


def resume_epub_audiobook(epub_path, audio_prompt_path, exaggeration, cfg_weight,
                           temperature, seed_num, diffusion_steps, min_p,
                           top_p, repetition_penalty, max_chars, batch_size,
                           denoise_reference, resume_project_name,
                           progress=gr.Progress()):
    """Resume an explicit project or the latest safely resumable monitored job."""

    project_name = _monitored_resume_project_name(resume_project_name)
    if not project_name:
        return None, "⚠️ Select an incomplete project, or wait for an active job to stop."
    return generate_epub_audiobook(
        epub_path, audio_prompt_path, exaggeration, cfg_weight, temperature,
        seed_num, diffusion_steps, min_p, top_p, repetition_penalty, max_chars,
        batch_size, denoise_reference, project_name, progress,
    )


def generate_epub_audiobook(epub_path, audio_prompt_path, exaggeration, cfg_weight,
                            temperature, seed_num, diffusion_steps, min_p,
                            top_p, repetition_penalty, max_chars, batch_size,
                            denoise_reference, resume_project_name,
                            progress=gr.Progress()):
    """Coordinate durable project setup, repair, generation, and M4B assembly."""

    resuming = bool(resume_project_name)
    job_started, _ = job_status.try_start(project_id=resume_project_name or None)
    if not job_started:
        return None, "⚠️ Another audiobook job is already running."
    generation_control.begin()
    if resuming:
        try:
            resume_dir, _ = load_project_metadata(OUTPUT_ROOT, resume_project_name)
            saved_epub, saved_reference = saved_project_inputs(resume_dir)
            epub_path = epub_path or (str(saved_epub) if saved_epub else None)
            audio_prompt_path = (
                audio_prompt_path
                or (str(saved_reference) if saved_reference else None)
            )
        except ResumeProjectError as error:
            job_status.finish("failed", f"Could not resume selected project: {error}")
            generation_control.finish()
            return None, f"❌ {error}."
    if not epub_path:
        job_status.finish("failed", "No EPUB was available for this job.")
        generation_control.finish()
        return None, "❌ Upload the original EPUB once so it can be saved with this project."
    if not audio_prompt_path:
        job_status.finish("failed", "No reference audio was available for this job.")
        generation_control.finish()
        return None, "❌ Upload or record reference audio once so it can be saved with this project."

    project_dir = None
    chunks = []
    completed_chunks = 0
    durable_chunks = 0
    durable_indices = set()
    audio_tasks = None
    persist_quality_scan_checkpoint = None
    active_project_model_id = global_model.model_id
    try:
        # Phase 1: restore or create the immutable text plan and saved inputs.
        book = load_epub(epub_path)
        source_epub_name = Path(epub_path).name
        if resuming:
            resume_plan = build_resume_plan(
                OUTPUT_ROOT, resume_project_name, book, global_model.sr,
                expected_model_id=global_model.model_id,
            )
            project_dir = resume_plan.project_dir
            chunks = list(resume_plan.chunks)
            settings = dict(resume_plan.metadata["settings"])
            denoise_reference = bool(settings.get("denoise_reference", False))
            active_project_model_id = project_model_id(resume_plan.metadata)
            source_epub_name = resume_plan.metadata.get(
                "source_epub", source_epub_name,
            )
            seed = settings.get("seed")
            if seed is not None:
                set_seed(int(seed))
            batch_size = int(settings["batch_size"])
            durable_chunks = resume_plan.resume_index
            completed_chunks = durable_chunks
            print(
                f"Resuming {project_dir.name}: {resume_plan.durable_chunks:,} valid "
                f"chunks found; restarting at {durable_chunks:,}"
            )
        else:
            chunks = chunk_book(book, max_chars=int(max_chars))
            if not chunks:
                raise EpubError("EPUB did not produce any speech chunks")
            seed = selected_seed(seed_num)
            settings = generation_arguments(
                exaggeration, cfg_weight, temperature, diffusion_steps, min_p,
                top_p,
                repetition_penalty, seed,
            )
            settings.update(
                {
                    "max_chars": int(max_chars),
                    "batch_size": int(batch_size),
                    "loudness_target_lufs": TARGET_LUFS,
                    "true_peak_dbtp": TRUE_PEAK_DBTP,
                    "loudness_range_lu": LOUDNESS_RANGE_LU,
                    "maximum_internal_pause_seconds": (
                        MAX_INTERNAL_PAUSE_SECONDS
                        if active_project_model_id == MULTILINGUAL_V3_MODEL_ID
                        else None
                    ),
                    "denoise_reference": bool(denoise_reference),
                }
            )

        if active_project_model_id == MULTILINGUAL_V3_MODEL_ID:
            settings.setdefault(
                "maximum_internal_pause_seconds",
                MAX_INTERNAL_PAUSE_SECONDS,
            )
        else:
            settings.setdefault("maximum_internal_pause_seconds", None)

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required to normalize and assemble audiobook audio")

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        project_dir = project_dir or OUTPUT_ROOT / _safe_project_name(book.title)
        chunks_dir = project_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=resuming)
        saved_epub, saved_reference = persist_project_inputs(
            project_dir, epub_path, audio_prompt_path,
        )
        epub_path = str(saved_epub)
        audio_prompt_path = str(saved_reference)
        metadata_path = project_dir / "metadata.json"
        _write_metadata(
            metadata_path, book, source_epub_name, chunks, settings,
            durable_chunks, active_project_model_id,
            scheduled=durable_chunks,
        )
        write_project_progress(project_dir, durable_chunks, durable_chunks)
        job_status.update(
            phase="preparing",
            message="Preparing audiobook generation…",
            fraction=durable_chunks / len(chunks),
            completed_chunks=durable_chunks,
            total_chunks=len(chunks),
            project_id=project_dir.name,
        )

        batch_size = int(batch_size)
        remaining_chunks = chunks[durable_chunks:]
        maximum_internal_pause_seconds = settings.get(
            "maximum_internal_pause_seconds"
        )
        damaged_chunks = {}
        quality_detected_chunks: set[int] = set()
        quality_fixed_chunks: set[int] = set()
        quality_retained_chunks: set[int] = set()
        verified_clean_chunks = load_quality_scan_checkpoint(project_dir)
        stale_checkpoint_indices = [
            chunk_index
            for chunk_index in verified_clean_chunks
            if chunk_index >= durable_chunks
        ]
        for chunk_index in stale_checkpoint_indices:
            if chunk_index >= durable_chunks:
                verified_clean_chunks.pop(chunk_index)
        checkpoint_dirty_chunks = len(stale_checkpoint_indices)
        checkpoint_last_saved_at = time.monotonic()

        def persist_quality_scan_checkpoint(*, force: bool = False) -> None:
            """Avoid frequent large metadata rewrites while preserving scan work."""

            nonlocal checkpoint_dirty_chunks, checkpoint_last_saved_at
            if not checkpoint_dirty_chunks:
                return
            now = time.monotonic()
            if (
                not force
                and checkpoint_dirty_chunks < QUALITY_SCAN_CHECKPOINT_UPDATE_CHUNKS
                and now - checkpoint_last_saved_at
                < QUALITY_SCAN_CHECKPOINT_UPDATE_SECONDS
            ):
                return
            write_quality_scan_checkpoint(project_dir, verified_clean_chunks)
            checkpoint_dirty_chunks = 0
            checkpoint_last_saved_at = now

        # Phase 2: validate only existing WAVs not already cached by this exact
        # detector version and file identity.
        if resuming:
            scan_indices = [
                chunk_index
                for chunk_index in range(durable_chunks)
                if not quality_scan_checkpoint_entry_matches(
                    verified_clean_chunks,
                    chunk_index,
                    chunks_dir / f"{chunk_index:06d}.wav",
                )
            ]
            cached_verified_chunks = durable_chunks - len(scan_indices)
            scan_started = time.perf_counter()
            last_scan_report_at = scan_started
            last_logged_scan_count = 0
            _report_resume_scan_progress(
                progress,
                0,
                len(scan_indices),
                scan_started,
                cached_verified_chunks=cached_verified_chunks,
            )
            job_status.update(
                phase="scanning",
                message="Validating existing audio chunks…",
                fraction=0.0,
                completed_chunks=0,
                total_chunks=len(scan_indices),
                realtime_speed=None,
                eta_seconds=None,
            )
            for scanned_chunks, chunk_index in enumerate(scan_indices, start=1):
                if generation_control.stop_requested():
                    raise GenerationStopped
                chunk_path = chunks_dir / f"{chunk_index:06d}.wav"
                issues = wav_generated_audio_issues(chunk_path, global_model.sr)
                if issues:
                    damaged_chunks[chunk_index] = issues
                    if chunk_index in verified_clean_chunks:
                        verified_clean_chunks.pop(chunk_index)
                        checkpoint_dirty_chunks += 1
                else:
                    identity = wav_file_identity(chunk_path)
                    if identity is not None and verified_clean_chunks.get(chunk_index) != identity:
                        verified_clean_chunks[chunk_index] = identity
                        checkpoint_dirty_chunks += 1
                persist_quality_scan_checkpoint()
                now = time.perf_counter()
                should_update_progress = (
                    scanned_chunks == len(scan_indices)
                    or now - last_scan_report_at
                    >= RESUME_SCAN_PROGRESS_UPDATE_SECONDS
                )
                should_log = (
                    scanned_chunks == len(scan_indices)
                    or scanned_chunks - last_logged_scan_count
                    >= RESUME_SCAN_PROGRESS_UPDATE_CHUNKS
                )
                if should_update_progress or should_log:
                    _report_resume_scan_progress(
                        progress,
                        scanned_chunks,
                        len(scan_indices),
                        scan_started,
                        log_stdout=should_log,
                        cached_verified_chunks=cached_verified_chunks,
                    )
                    last_scan_report_at = now
                    elapsed_scan = now - scan_started
                    eta_scan = (
                        elapsed_scan * (len(scan_indices) - scanned_chunks) / scanned_chunks
                        if scanned_chunks else None
                    )
                    job_status.update(
                        phase="scanning",
                        message="Validating existing audio chunks…",
                        fraction=scanned_chunks / len(scan_indices) if scan_indices else 1.0,
                        completed_chunks=scanned_chunks,
                        total_chunks=len(scan_indices),
                        realtime_speed=None,
                        eta_seconds=eta_scan,
                    )
                    if should_log:
                        last_logged_scan_count = scanned_chunks
            persist_quality_scan_checkpoint(force=True)
            quality_detected_chunks.update(damaged_chunks)
            labels = (
                ", ".join(f"{index:06d}.wav" for index in damaged_chunks)
                if damaged_chunks
                else "none"
            )
            _quality_log(
                f"[Audio quality scan] Scanned {len(scan_indices):,} existing "
                f"chunk(s); skipped {cached_verified_chunks:,} cached verified; "
                f"found {len(damaged_chunks):,} damaged: {labels}",
                color="red" if damaged_chunks else None,
            )
            for chunk_index, issues in damaged_chunks.items():
                _quality_log(
                    f"[Audio quality scan] {chunk_index:06d}.wav: "
                    f"{format_audio_quality_issues(issues)}",
                    color="red",
                )
            if damaged_chunks:
                _quality_log(
                    "[Audio quality repair] Flagged chunks will be regenerated "
                    "before normal resume generation",
                    color="yellow",
                )
        total_characters = sum(len(chunk.text) for chunk in remaining_chunks)
        completed_characters = 0
        generated_audio_seconds = 0.0
        generation_started = time.perf_counter()
        if remaining_chunks or damaged_chunks or (
            durable_chunks and maximum_internal_pause_seconds is not None
        ):
            audio_tasks = BackgroundTaskPool(
                max_workers=AUDIO_WORKERS,
                max_pending=MAX_PENDING_AUDIO_TASKS,
            )
        if remaining_chunks or damaged_chunks:
            pending_description = (
                f"{len(remaining_chunks):,} remaining chunks"
                if remaining_chunks
                else f"{len(damaged_chunks):,} damaged chunks"
            )
            progress(0, desc=f"Preparing voice for {pending_description}")
            job_status.update(
                phase="preparing",
                message=f"Preparing voice for {pending_description}",
                completed_chunks=durable_chunks,
                total_chunks=len(chunks),
                fraction=durable_chunks / len(chunks),
            )
            s3gen_ref, cond_emb = global_model.get_audio_conditionals(
                audio_prompt_path,
                denoise_reference=bool(denoise_reference),
            )
        elif audio_tasks is not None:
            progress(
                0.975,
                desc=f"Applying V3 pause limit to {durable_chunks:,} existing chunks",
            )
        else:
            progress(0.975, desc="All speech chunks found; preparing M4B assembly")
        repaired_chunks = []
        retained_noisy_chunks = []
        # Phase 3: replace damaged durable chunks before generating new ones.
        if damaged_chunks:
            job_status.update(
                phase="repairing",
                message="Regenerating damaged audio chunks…",
                completed_chunks=0,
                total_chunks=len(damaged_chunks),
                fraction=0.0,
                realtime_speed=None,
                eta_seconds=None,
            )
        for repair_number, chunk_index in enumerate(damaged_chunks, start=1):
            if generation_control.stop_requested():
                raise GenerationStopped
            chunk = chunks[chunk_index]
            repair_args = dict(settings)
            retry_seeds: set[int] = set()
            for metadata_key in (
                "max_chars",
                "batch_size",
                "loudness_target_lufs",
                "true_peak_dbtp",
                "loudness_range_lu",
                "maximum_internal_pause_seconds",
                "denoise_reference",
            ):
                repair_args.pop(metadata_key)
            repaired = global_model.generate_with_conds(
                [chunk.text],
                s3gen_ref=s3gen_ref,
                cond_emb=cond_emb,
                **_retry_generation_args(repair_args, retry_seeds),
            )[0]
            recovery = _recover_generated_waveform(
                repaired,
                chunk.text,
                global_model.sr,
                f"existing chunk {chunk_index:06d}",
                lambda chunk=chunk, chunk_index=chunk_index: (
                    global_model.generate_with_conds(
                        [chunk.text],
                        s3gen_ref=s3gen_ref,
                        cond_emb=cond_emb,
                        **_retry_generation_args(repair_args, retry_seeds),
                    )[0]
                ),
                lambda part: (
                    global_model.generate_with_conds(
                        [part],
                        s3gen_ref=s3gen_ref,
                        cond_emb=cond_emb,
                        **_retry_generation_args(repair_args, retry_seeds),
                    )[0]
                ),
            )
            waveform = recovery.waveform
            _save_and_normalize_chunk(
                chunks_dir / f"{chunk_index:06d}.wav",
                waveform,
                global_model.sr,
                ffmpeg,
                maximum_internal_pause_seconds,
            )
            repaired_path = chunks_dir / f"{chunk_index:06d}.wav"
            saved_issues = wav_generated_audio_issues(repaired_path, global_model.sr)
            if saved_issues or recovery.retained_with_warning:
                if chunk_index in verified_clean_chunks:
                    verified_clean_chunks.pop(chunk_index)
                    checkpoint_dirty_chunks += 1
                retained_noisy_chunks.append(chunk_index)
                quality_retained_chunks.add(chunk_index)
                quality_fixed_chunks.discard(chunk_index)
                detail = (
                    format_audio_quality_issues(saved_issues)
                    if saved_issues
                    else "bounded recovery retained an invalid-output fallback"
                )
                _quality_log(
                    f"[Audio quality warning] {repaired_path.name}: {detail}; "
                    "included anyway so generation can continue",
                    color="red",
                )
            repaired_chunks.append(chunk_index)
            job_status.update(
                phase="repairing",
                message="Regenerating damaged audio chunks…",
                completed_chunks=repair_number,
                total_chunks=len(damaged_chunks),
                fraction=repair_number / len(damaged_chunks),
            )
            if not saved_issues and not recovery.retained_with_warning:
                identity = wav_file_identity(repaired_path)
                if identity is not None and verified_clean_chunks.get(chunk_index) != identity:
                    verified_clean_chunks[chunk_index] = identity
                    checkpoint_dirty_chunks += 1
                quality_fixed_chunks.add(chunk_index)
                quality_retained_chunks.discard(chunk_index)
                _quality_log(
                    f"[Audio quality repair] Fixed and verified {chunk_index:06d}.wav",
                    color="green",
                )
        persist_quality_scan_checkpoint(force=True)
        if resuming:
            fixed_chunks = [
                index for index in repaired_chunks
                if index not in retained_noisy_chunks
            ]
            fixed_labels = (
                ", ".join(
                    f"{index:06d}.wav" for index in fixed_chunks
                )
                if fixed_chunks
                else "none"
            )
            retained_labels = (
                ", ".join(
                    f"{index:06d}.wav" for index in retained_noisy_chunks
                )
                if retained_noisy_chunks
                else "none"
            )
            _quality_log(
                f"[Audio quality repair] Scan result: found "
                f"{len(damaged_chunks):,} damaged chunk(s); fixed "
                f"{len(fixed_chunks):,}: {fixed_labels}; retained with noise "
                f"after bounded recovery {len(retained_noisy_chunks):,}: "
                f"{retained_labels}",
                color=(
                    "red" if retained_noisy_chunks
                    else "green" if repaired_chunks
                    else None
                ),
            )
        if durable_chunks and maximum_internal_pause_seconds is not None:
            job_status.update(
                phase="pause_processing",
                message="Applying internal pause limit to existing audio…",
                completed_chunks=0,
                total_chunks=durable_chunks,
                fraction=0.0,
                realtime_speed=None,
                eta_seconds=None,
            )
            for chunk_index in range(durable_chunks):
                if generation_control.stop_requested():
                    raise GenerationStopped
                if chunk_index in damaged_chunks:
                    continue
                audio_tasks.submit(
                    _limit_pauses_and_record_chunk,
                    chunks_dir / f"{chunk_index:06d}.wav",
                    global_model.sr,
                maximum_seconds=maximum_internal_pause_seconds,
            )
        # Phase 4: generate remaining text in GPU batches while the bounded
        # background pool writes and validates final WAVs.
        for batch_number, start in enumerate(
            range(durable_chunks, len(chunks), batch_size)
        ):
            if generation_control.stop_requested():
                raise GenerationStopped
            _protect_system_memory(batch_number)
            batch = chunks[start:start + batch_size]
            # Before the first measurement, show which batch is starting. After
            # that, leave the latest speed/ETA visible while the next batch runs;
            # replacing it here made the useful metrics flash too briefly to see.
            if start == durable_chunks:
                job_status.update(
                    phase="generating",
                    message="Generating speech chunks…",
                    completed_chunks=completed_chunks,
                    total_chunks=len(chunks),
                    fraction=completed_chunks / len(chunks),
                    realtime_speed=None,
                    eta_seconds=None,
                )
                progress(
                    0,
                    desc=(
                        f"Generating chunks {start + 1}–{start + len(batch)} "
                        f"of {len(chunks)}"
                    ),
                )
            batch_args = dict(settings)
            for metadata_key in (
                "max_chars",
                "batch_size",
                "loudness_target_lufs",
                "true_peak_dbtp",
                "loudness_range_lu",
                "maximum_internal_pause_seconds",
                "denoise_reference",
            ):
                batch_args.pop(metadata_key)
            if seed is not None:
                batch_args["seed"] = seed + start
            audios = global_model.generate_with_conds(
                [chunk.text for chunk in batch],
                s3gen_ref=s3gen_ref,
                cond_emb=cond_emb,
                **batch_args,
            )
            durable_chunks, checkpoint_change_count = _record_durable_results(
                audio_tasks,
                durable_indices,
                durable_chunks,
                verified_clean_chunks,
            )
            if checkpoint_change_count:
                checkpoint_dirty_chunks += checkpoint_change_count
                persist_quality_scan_checkpoint()
            if len(audios) != len(batch):
                raise RuntimeError(f"Model returned {len(audios)} outputs for a batch of {len(batch)}")
            retained_quality_chunks = []
            for offset, (chunk, audio) in enumerate(zip(batch, audios)):
                chunk_index = start + offset
                retry_seeds: set[int] = set()
                recovery = _recover_generated_waveform(
                    audio,
                    chunk.text,
                    global_model.sr,
                    f"chunk {chunk_index:06d}",
                    lambda chunk=chunk, chunk_index=chunk_index: (
                        global_model.generate_with_conds(
                            [chunk.text],
                            s3gen_ref=s3gen_ref,
                            cond_emb=cond_emb,
                            **_retry_generation_args(batch_args, retry_seeds),
                        )[0]
                    ),
                    lambda part: (
                        global_model.generate_with_conds(
                            [part],
                            s3gen_ref=s3gen_ref,
                            cond_emb=cond_emb,
                            **_retry_generation_args(batch_args, retry_seeds),
                        )[0]
                    ),
                )
                waveform = recovery.waveform
                initial_quality_issues = recovery.detected_quality_issues
                if initial_quality_issues:
                    quality_detected_chunks.add(chunk_index)
                if recovery.retained_with_warning:
                    retained_quality_chunks.append(chunk_index)
                    quality_detected_chunks.add(chunk_index)
                    quality_retained_chunks.add(chunk_index)
                    quality_fixed_chunks.discard(chunk_index)
                elif initial_quality_issues:
                    quality_fixed_chunks.add(chunk_index)
                    quality_retained_chunks.discard(chunk_index)
                generated_audio_seconds += waveform.shape[1] / global_model.sr
                completed_characters += len(chunk.text)
                chunk_path = chunks_dir / f"{chunk_index:06d}.wav"
                audio_tasks.submit(
                    _save_normalize_and_record_chunk,
                    chunk_path,
                    waveform,
                    global_model.sr,
                    ffmpeg,
                    maximum_internal_pause_seconds,
                    cacheable=not recovery.retained_with_warning,
                )
            _log_batch_quality_summary(
                start,
                len(batch),
                retained_quality_chunks,
            )
            completed_chunks = start + len(batch)
            durable_chunks, checkpoint_change_count = _record_durable_results(
                audio_tasks,
                durable_indices,
                durable_chunks,
                verified_clean_chunks,
            )
            if checkpoint_change_count:
                checkpoint_dirty_chunks += checkpoint_change_count
                persist_quality_scan_checkpoint()
            write_project_progress(project_dir, durable_chunks, completed_chunks)
            elapsed = time.perf_counter() - generation_started
            estimate = estimate_progress(
                generated_audio_seconds,
                elapsed,
                completed_characters,
                total_characters,
            )
            progress(
                min(0.97, completed_chunks / len(chunks)),
                desc=(
                    f"{completed_chunks:,}/{len(chunks):,} chunks • "
                    f"{estimate.realtime_speed:.2f}× realtime • "
                    f"ETA {format_duration(estimate.eta_seconds)}"
                ),
            )
            job_status.update(
                phase="generating",
                message="Generating speech chunks…",
                completed_chunks=completed_chunks,
                total_chunks=len(chunks),
                fraction=completed_chunks / len(chunks),
                realtime_speed=estimate.realtime_speed,
                eta_seconds=estimate.eta_seconds,
            )
            if generation_control.stop_requested():
                raise GenerationStopped

        if audio_tasks is not None:
            progress(0.975, desc="Finishing background audio processing")
            job_status.update(
                phase="pause_processing",
                message="Finishing background audio processing…",
                completed_chunks=0,
                total_chunks=durable_chunks,
                fraction=0.975,
                realtime_speed=None,
                eta_seconds=None,
            )
            audio_tasks.finish()
            durable_chunks, checkpoint_change_count = _record_durable_results(
                audio_tasks,
                durable_indices,
                durable_chunks,
                verified_clean_chunks,
            )
            if checkpoint_change_count:
                checkpoint_dirty_chunks += checkpoint_change_count
            audio_tasks = None
            job_status.update(
                phase="pause_processing",
                message="Background audio processing complete.",
                completed_chunks=durable_chunks,
                total_chunks=durable_chunks,
                fraction=0.975,
            )
        persist_quality_scan_checkpoint(force=True)
        _log_project_quality_summary(
            len(chunks),
            quality_detected_chunks,
            quality_fixed_chunks,
            quality_retained_chunks,
        )
        _write_metadata(
            metadata_path, book, source_epub_name, chunks, settings,
            len(chunks), active_project_model_id, scheduled=len(chunks),
        )
        write_project_progress(project_dir, len(chunks), len(chunks))
        # Phase 5: assemble, verify, durably mark complete, then clean up.
        progress(0.98, desc="Preparing parallel M4B encoding")
        job_status.update(
            phase="encoding",
            message="Preparing M4B encoding…",
            completed_chunks=len(chunks),
            total_chunks=len(chunks),
            fraction=0.98,
            realtime_speed=None,
            eta_seconds=None,
        )

        def report_assembly(update: AssemblyProgress) -> None:
            if update.phase == "encoding":
                position = 0.98 + 0.015 * update.fraction
                operation = f"Encoding M4B with {update.workers} workers"
            else:
                position = 0.995 + 0.004 * update.fraction
                operation = "Finalizing M4B"
            progress(
                min(0.999, position),
                desc=(
                    f"{operation}: {update.fraction:.1%} • "
                    f"{update.realtime_speed:.2f}× realtime • "
                    f"ETA {format_duration(update.eta_seconds)}"
                ),
            )
            job_status.update(
                phase=update.phase,
                message=operation,
                completed_chunks=len(chunks),
                total_chunks=len(chunks),
                fraction=min(0.999, position),
                realtime_speed=update.realtime_speed,
                eta_seconds=update.eta_seconds,
            )

        output_path = assemble_audiobook(
            project_dir,
            book,
            chunks,
            global_model.sr,
            stop_requested=generation_control.stop_requested,
            progress_callback=report_assembly,
            ffmpeg=ffmpeg,
        )
        progress(0.999, desc="Verifying completed M4B")
        job_status.update(
            phase="finalizing",
            message="Verifying completed M4B…",
            completed_chunks=len(chunks),
            total_chunks=len(chunks),
            fraction=0.999,
            realtime_speed=None,
            eta_seconds=None,
        )
        final_audio_seconds = verify_m4b(output_path)
        relative_output = output_path.relative_to(
            Path(__file__).resolve().parent
        ).as_posix()
        _mark_verified_project_complete(
            metadata_path,
            project_dir,
            book,
            source_epub_name,
            chunks,
            settings,
            active_project_model_id,
            relative_output,
        )
        generation_elapsed = time.perf_counter() - generation_started
        final_speed = final_audio_seconds / max(generation_elapsed, 1e-9)
        progress(1, desc="Audiobook complete")
        output_path = register_completed_audiobook(
            output_path,
            OUTPUT_ROOT,
            gr.set_static_paths,
        )
        job_status.finish(
            "completed",
            "Audiobook complete.",
            completed_chunks=len(chunks),
            total_chunks=len(chunks),
            fraction=1.0,
            realtime_speed=final_speed,
        )
        return str(output_path), (
            f"✅ **{book.title}** complete: {len(book.chapters)} chapters and "
            f"{len(chunks):,} speech chunks. Generated "
            f"{format_duration(final_audio_seconds)} of audio in "
            f"{format_duration(generation_elapsed)} ({final_speed:.2f}× realtime). "
            f"Files were saved under `{project_dir}`; intermediate chunks were "
            "cleaned up when possible."
        )
    except (GenerationStopped, AssemblyStopped):
        if audio_tasks is not None:
            audio_tasks.cancel_and_wait()
            audio_tasks = None
        try:
            durable_chunks = contiguous_chunk_count(
                project_dir / "chunks",
                len(chunks),
                global_model.sr,
            )
            write_project_progress(project_dir, durable_chunks, completed_chunks)
            print(
                f"EPUB generation stopped with {durable_chunks} of "
                f"{len(chunks)} chunks safely recorded"
            )
            job_status.finish(
                "stopped",
                "Generation stopped; the saved project can be resumed.",
                completed_chunks=durable_chunks,
                total_chunks=len(chunks),
                fraction=durable_chunks / len(chunks) if chunks else 0.0,
            )
            return None, (
                f"⏹️ Generation stopped with {durable_chunks:,} of "
                f"{len(chunks):,} chunks safely recorded. The project, saved inputs, "
                "and chunks were preserved and can be resumed."
            )
        except Exception as progress_error:
            traceback.print_exc()
            job_status.finish(
                "stopped",
                "Generation stopped; the saved project can be resumed.",
                completed_chunks=completed_chunks,
                total_chunks=len(chunks),
                fraction=completed_chunks / len(chunks) if chunks else 0.0,
            )
            return None, (
                f"⚠️ Generation stopped and project files were preserved, but its "
                f"progress record could not be updated: {progress_error}"
            )
    except Exception as error:
        if audio_tasks is not None:
            audio_tasks.cancel_and_wait()
            audio_tasks = None
        traceback.print_exc()
        job_status.finish(
            "failed",
            f"Generation failed: {error}",
            completed_chunks=completed_chunks,
            total_chunks=len(chunks),
            fraction=completed_chunks / len(chunks) if chunks else 0.0,
        )
        if isinstance(error, MemoryPressureError):
            return None, f"⚠️ {error}. Partial files remain in `{project_dir}`."
        location = f" Partial files remain in `{project_dir}`." if project_dir else ""
        return None, f"❌ {error}.{location}"
    finally:
        if audio_tasks is not None:
            audio_tasks.cancel_and_wait()
        if persist_quality_scan_checkpoint is not None:
            try:
                persist_quality_scan_checkpoint(force=True)
            except Exception:
                traceback.print_exc()
        generation_control.finish()


with gr.Blocks(title="Chatterbox vLLM Audiobook", css=JOB_MONITOR_CSS) as demo:
    gr.Markdown(
        "# Chatterbox vLLM\n"
        "Quick voice tests and batched EPUB audiobook generation.  \n"
        f"**Active model:** {model_label(ACTIVE_MODEL_ID)} (`{ACTIVE_MODEL_ID}`)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Voice and generation settings")
            reference_source = gr.State(None)
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio (the processed conditioning audio)",
                value=None,
            )
            denoise_reference = gr.Checkbox(
                value=False,
                label="Denoise reference before normalization (RNNoise)",
                info=(
                    "Optional. The player above will update to the exact processing "
                    "used by both Text Sample and EPUB generation; the source file "
                    "is never modified."
                ),
            )
            exaggeration = gr.Slider(
                0.25, 2, step=.05,
                label="Exaggeration (Neutral = 0.5; extreme values can be unstable)",
                value=.5,
            )
            cfg_weight = gr.Slider(
                0.0, 1.0, step=.05,
                label="CFG/Pace (lower = looser and often slower; neutral = 0.5)",
                value=.5,
            )
            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                diffusion_steps = gr.Slider(
                    1, 15, step=1,
                    label="Diffusion Steps (more = slower and higher quality)",
                    value=15,
                )
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)
                min_p = gr.Slider(
                    0.00, 1.00, step=0.01,
                    label="Min-P (0 disables)",
                    value=0.05,
                )
                top_p = gr.Slider(
                    0.00, 1.00, step=0.01,
                    label="Top-P (1.0 disables)",
                    value=1.00,
                )
                repetition_penalty = gr.Slider(
                    1.00, 2.00, step=0.1,
                    label="Repetition Penalty",
                    value=1.2,
                )

        with gr.Column(scale=2):
            with gr.Tab("EPUB Audiobook"):
                epub_file = gr.File(
                    label="DRM-free EPUB",
                    file_types=[".epub"],
                    type="filepath",
                )
                with gr.Row():
                    max_chars = gr.Slider(
                        120, 300, step=10, value=200,
                        label="Maximum characters per speech chunk",
                    )
                    batch_size = gr.Slider(
                        1, MAX_BATCH_SIZE, step=1, value=DEFAULT_BATCH_SIZE,
                        label="vLLM batch size",
                    )
                epub_info = gr.Markdown(
                    "Upload a DRM-free EPUB to inspect its chapters before generation."
                )
                resume_choices = incomplete_project_choices(OUTPUT_ROOT)
                with gr.Accordion("Resume an incomplete project", open=False):
                    resume_project = gr.Dropdown(
                        choices=resume_choices,
                        value=None,
                        label="Incomplete project",
                    )
                    refresh_resume_btn = gr.Button("Refresh Incomplete Projects")
                    resume_info = gr.Markdown(
                        "Select an incomplete project. Saved inputs will load automatically; "
                        "older projects require each input once."
                    )
                with gr.Row():
                    epub_btn = gr.Button("Generate EPUB Audiobook", variant="primary")
                    resume_epub_btn = gr.Button("Resume Selected / Monitored Project")
                    stop_epub_btn = gr.Button("Stop Generation", variant="stop")
                epub_status = gr.Markdown("")
                job_monitor = gr.HTML(
                    render_job_status(job_status.snapshot(), format_duration=format_duration),
                    elem_id="active-job-monitor",
                )
                epub_result_status = gr.State("")
                new_project_state = gr.State("")
                epub_audio_output = gr.Audio(label="Completed Audiobook (M4B)")

            with gr.Tab("Text Sample"):
                text = gr.Textbox(
                    value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse.",
                    label="Text to synthesize",
                    max_lines=6,
                )
                run_btn = gr.Button("Generate Sample", variant="primary")
                audio_output = gr.Audio(label="Output Audio")

    job_monitor_timer = gr.Timer(1.5)
    demo.load(
        fn=_job_monitor_render,
        outputs=[job_monitor, stop_epub_btn, epub_btn, resume_epub_btn],
        queue=False,
    )
    job_monitor_timer.tick(
        fn=_job_monitor_render,
        outputs=[job_monitor, stop_epub_btn, epub_btn, resume_epub_btn],
        queue=False,
    )

    ref_wav.input(
        fn=prepare_uploaded_reference,
        inputs=[ref_wav, denoise_reference],
        outputs=[ref_wav, reference_source],
    )
    denoise_reference.input(
        fn=prepare_reference_preview,
        inputs=[reference_source, denoise_reference],
        outputs=ref_wav,
    )
    run_btn.click(
        fn=generate_sample,
        inputs=[
            text, reference_source, denoise_reference, exaggeration, cfg_weight,
            temp, seed_num,
            diffusion_steps, min_p, top_p, repetition_penalty,
        ],
        outputs=audio_output,
    )
    epub_file.change(
        fn=inspect_epub_file,
        inputs=[epub_file, max_chars],
        outputs=epub_info,
    )
    max_chars.release(
        fn=inspect_epub_file,
        inputs=[epub_file, max_chars],
        outputs=epub_info,
    )
    epub_generation_event = epub_btn.click(
        fn=generate_epub_audiobook,
        inputs=[
            epub_file, reference_source, exaggeration, cfg_weight, temp, seed_num,
            diffusion_steps, min_p, top_p, repetition_penalty, max_chars,
            batch_size, denoise_reference, new_project_state,
        ],
        # Keep the textual result hidden while this event is running. Making a
        # visible Markdown component a live output causes Gradio to render the
        # same queue/progress message both there and in its normal progress UI.
        outputs=[epub_audio_output, epub_result_status],
    )
    resume_generation_event = resume_epub_btn.click(
        fn=resume_epub_audiobook,
        inputs=[
            epub_file, reference_source, exaggeration, cfg_weight, temp, seed_num,
            diffusion_steps, min_p, top_p, repetition_penalty, max_chars,
            batch_size, denoise_reference, resume_project,
        ],
        outputs=[epub_audio_output, epub_result_status],
    )
    resume_generation_event.then(
        fn=lambda status: status,
        inputs=epub_result_status,
        outputs=epub_status,
        queue=False,
    )
    refresh_resume_btn.click(
        fn=refresh_resume_projects,
        outputs=resume_project,
        queue=False,
    )
    resume_project_event = resume_project.change(
        fn=inspect_resume_project,
        inputs=resume_project,
        outputs=[resume_info, epub_file, reference_source, denoise_reference],
        queue=False,
    )
    resume_project_event.then(
        fn=prepare_reference_preview,
        inputs=[reference_source, denoise_reference],
        outputs=ref_wav,
    )
    epub_generation_event.then(
        fn=lambda status: status,
        inputs=epub_result_status,
        outputs=epub_status,
        queue=False,
    )
    stop_epub_btn.click(
        fn=request_generation_stop,
        outputs=epub_status,
        queue=False,
    )


def parse_command_line():
    parser = argparse.ArgumentParser(description="Launch the Chatterbox vLLM web UI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="create a public Gradio share link (no authentication is configured)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    command_line = parse_command_line()
    # Don't let Gradio manage model loading; it causes issues with vLLM workers.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    load_model()

    print("Starting Gradio app...")
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=command_line.share,
        show_api=False,
    )
