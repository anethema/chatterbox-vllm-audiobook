import argparse
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
    LOUDNESS_RANGE_LU,
    MAX_INTERNAL_PAUSE_SECONDS,
    TARGET_LUFS,
    TRUE_PEAK_DBTP,
    find_generated_audio_issues,
    format_audio_quality_issues,
    limit_internal_pauses_wav,
    normalized_reference_audio,
    normalize_speech_wav,
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
    incomplete_project_choices,
    load_project_metadata,
    persist_project_inputs,
    project_model_id,
    saved_project_inputs,
    write_project_progress,
)
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

config_seed = None
global_model = None
generation_control = GenerationControl()
reference_preview_directory: tempfile.TemporaryDirectory | None = None


class GenerationStopped(Exception):
    pass


class MemoryPressureError(RuntimeError):
    pass


class GeneratedAudioQualityError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__(format_audio_quality_issues(list(self.issues)))


def _quality_log(message: str, *, color: str | None = None) -> None:
    colors = {"red": "\033[1;31m", "green": "\033[1;32m", "yellow": "\033[1;33m"}
    if color in colors and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        message = f"{colors[color]}{message}\033[0m"
    print(message, flush=True)


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


def prepare_reference_preview(audio_prompt_path: str | None) -> str | None:
    """Return a persistent normalized copy for Gradio playback and generation."""

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
    with normalized_reference_audio(audio_prompt_path, 24000) as normalized:
        shutil.copyfile(normalized, destination)
    print(
        f"[Reference normalization] Gradio player now uses: {destination}",
        flush=True,
    )
    return str(destination)


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


def generate_sample(text, audio_prompt_path, exaggeration, cfg_weight, temperature,
                    seed_num, diffusion_steps, min_p, top_p,
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
        **args,
    )
    waveform = _recover_generated_waveform(
        wav[0],
        text.strip(),
        global_model.sr,
        "text sample",
        lambda: global_model.generate(
            [text.strip()],
            audio_prompt_path=audio_prompt_path,
            **_retry_generation_args(args, retry_seeds),
        )[0],
        lambda part: global_model.generate(
            [part],
            audio_prompt_path=audio_prompt_path,
            **_retry_generation_args(args, retry_seeds),
        )[0],
    )
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


def request_generation_stop():
    if not generation_control.request_stop():
        return "No EPUB generation is currently running."
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
        )
    try:
        for label, name in incomplete_project_choices(OUTPUT_ROOT):
            if name == project_name:
                project_dir = OUTPUT_ROOT / name
                saved_epub, saved_reference = saved_project_inputs(project_dir)
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
                )
    except OSError as error:
        return f"❌ Could not inspect incomplete projects: {error}", None, None
    return "❌ The selected incomplete project is no longer available.", None, None


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
    tensor = torch.as_tensor(waveform).to(dtype=torch.float32, device="cpu")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor.transpose(0, 1)
    if tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] == 0:
        raise ValueError("The model returned an empty or non-mono waveform")
    if not torch.isfinite(tensor).all():
        raise ValueError("The model returned NaN or infinite audio")

    duration = tensor.shape[1] / sample_rate
    words = max(1, len(text.split()))
    if duration < min(0.25, 0.04 * words):
        raise ValueError(f"Generated audio is truncated ({duration:.2f}s for {words} words)")
    if duration > 15.0 + 5.0 * words:
        raise ValueError(f"Generated audio is implausibly long ({duration:.1f}s for {words} words)")
    issues = find_generated_audio_issues(tensor.numpy(), sample_rate)
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


def _recover_split_part(
    text: str,
    sample_rate: int,
    label: str,
    generate_part,
    depth: int,
) -> torch.Tensor:
    last_error = None
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
            return waveform
        except GeneratedAudioQualityError as error:
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
            f"{last_error}; included anyway so generation can continue",
            color="red",
        )
        return _waveform_for_save(
            last_audio,
            text,
            sample_rate,
            allow_quality_issues=True,
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
            f"with {detail}; included anyway so generation can continue",
            color="red",
        )
        return _waveform_for_save(
            last_audio,
            text,
            sample_rate,
            allow_quality_issues=True,
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
    combined = _join_split_waveforms(recovered, sample_rate)
    issues = find_generated_audio_issues(combined.numpy(), sample_rate)
    if issues:
        _quality_log(
            f"[Audio quality warning] {label}: recursively split output retains "
            f"{format_audio_quality_issues(issues)}; included anyway so generation "
            "can continue",
            color="red",
        )
    return combined


def _recover_generated_waveform(
    initial_audio,
    text: str,
    sample_rate: int,
    label: str,
    regenerate_whole,
    generate_part,
) -> torch.Tensor:
    try:
        return _waveform_for_save(initial_audio, text, sample_rate)
    except GeneratedAudioQualityError as error:
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
    retry_failure = None
    try:
        waveform = _waveform_for_save(retried_audio, text, sample_rate)
        _quality_log(
            f"[Audio quality repair] {label}: whole-chunk replacement passed "
            "the full-waveform scan",
            color="green",
        )
        return waveform
    except GeneratedAudioQualityError as retry_error:
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
            f"{retry_failure}; included anyway so generation can continue",
            color="red",
        )
        return _waveform_for_save(
            retried_audio,
            text,
            sample_rate,
            allow_quality_issues=True,
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

    combined = _join_split_waveforms(part_waveforms, sample_rate)
    retained_quality_issues = False
    try:
        result = _waveform_for_save(combined, text, sample_rate)
    except GeneratedAudioQualityError as combined_error:
        retained_quality_issues = True
        _quality_log(
            f"[Audio quality warning] {label}: combined sentence-sized output "
            f"retains {combined_error}; included anyway so generation can continue",
            color="red",
        )
        result = _waveform_for_save(
            combined,
            text,
            sample_rate,
            allow_quality_issues=True,
        )
    if not retained_quality_issues:
        _quality_log(
            f"[Audio quality repair] {label}: fixed with {len(parts)} shorter "
            "parts; combined waveform passed the full scan",
            color="green",
        )
    return result


def _save_and_normalize_chunk(
    path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
    ffmpeg: str,
    maximum_internal_pause_seconds: float | None,
) -> Path:
    ta.save(
        str(path),
        waveform,
        sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    normalize_speech_wav(path, sample_rate, ffmpeg=ffmpeg)
    if maximum_internal_pause_seconds is not None:
        limit_internal_pauses_wav(
            path,
            sample_rate,
            maximum_seconds=maximum_internal_pause_seconds,
        )
    return path


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


def _record_durable_results(audio_tasks, durable_indices: set[int], durable_chunks: int) -> int:
    audio_tasks.check()
    for path in audio_tasks.take_results():
        durable_indices.add(int(Path(path).stem))
    while durable_chunks in durable_indices:
        durable_indices.remove(durable_chunks)
        durable_chunks += 1
    return durable_chunks


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


def generate_epub_audiobook(epub_path, audio_prompt_path, exaggeration, cfg_weight,
                            temperature, seed_num, diffusion_steps, min_p,
                            top_p, repetition_penalty, max_chars, batch_size,
                            resume_project_name,
                            progress=gr.Progress()):
    resuming = bool(resume_project_name)
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
            return None, f"❌ {error}."
    if not epub_path:
        return None, "❌ Upload the original EPUB once so it can be saved with this project."
    if not audio_prompt_path:
        return None, "❌ Upload or record reference audio once so it can be saved with this project."

    project_dir = None
    chunks = []
    completed_chunks = 0
    durable_chunks = 0
    durable_indices = set()
    audio_tasks = None
    active_project_model_id = global_model.model_id
    generation_control.begin()
    try:
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

        batch_size = int(batch_size)
        remaining_chunks = chunks[durable_chunks:]
        maximum_internal_pause_seconds = settings.get(
            "maximum_internal_pause_seconds"
        )
        damaged_chunks = {}
        quality_detected_chunks: set[int] = set()
        quality_fixed_chunks: set[int] = set()
        quality_retained_chunks: set[int] = set()
        if resuming:
            for chunk_index in range(durable_chunks):
                chunk_path = chunks_dir / f"{chunk_index:06d}.wav"
                issues = wav_generated_audio_issues(chunk_path, global_model.sr)
                if issues:
                    damaged_chunks[chunk_index] = issues
            quality_detected_chunks.update(damaged_chunks)
            labels = (
                ", ".join(f"{index:06d}.wav" for index in damaged_chunks)
                if damaged_chunks
                else "none"
            )
            _quality_log(
                f"[Audio quality scan] Scanned {durable_chunks:,} existing "
                f"chunk(s); found {len(damaged_chunks):,} damaged: {labels}",
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
            s3gen_ref, cond_emb = global_model.get_audio_conditionals(audio_prompt_path)
        elif audio_tasks is not None:
            progress(
                0.975,
                desc=f"Applying V3 pause limit to {durable_chunks:,} existing chunks",
            )
        else:
            progress(0.975, desc="All speech chunks found; preparing M4B assembly")
        repaired_chunks = []
        retained_noisy_chunks = []
        for chunk_index in damaged_chunks:
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
            ):
                repair_args.pop(metadata_key)
            repaired = global_model.generate_with_conds(
                [chunk.text],
                s3gen_ref=s3gen_ref,
                cond_emb=cond_emb,
                **_retry_generation_args(repair_args, retry_seeds),
            )[0]
            waveform = _recover_generated_waveform(
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
            _save_and_normalize_chunk(
                chunks_dir / f"{chunk_index:06d}.wav",
                waveform,
                global_model.sr,
                ffmpeg,
                maximum_internal_pause_seconds,
            )
            repaired_path = chunks_dir / f"{chunk_index:06d}.wav"
            saved_issues = wav_generated_audio_issues(repaired_path, global_model.sr)
            if saved_issues:
                retained_noisy_chunks.append(chunk_index)
                quality_retained_chunks.add(chunk_index)
                quality_fixed_chunks.discard(chunk_index)
                _quality_log(
                    f"[Audio quality warning] {repaired_path.name}: post-save scan "
                    f"still found {format_audio_quality_issues(saved_issues)}; "
                    "included anyway so generation can continue",
                    color="red",
                )
            repaired_chunks.append(chunk_index)
            if not saved_issues:
                quality_fixed_chunks.add(chunk_index)
                quality_retained_chunks.discard(chunk_index)
                _quality_log(
                    f"[Audio quality repair] Fixed and verified {chunk_index:06d}.wav",
                    color="green",
                )
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
            for chunk_index in range(durable_chunks):
                if chunk_index in damaged_chunks:
                    continue
                audio_tasks.submit(
                    limit_internal_pauses_wav,
                    chunks_dir / f"{chunk_index:06d}.wav",
                    global_model.sr,
                    maximum_seconds=maximum_internal_pause_seconds,
                )
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
            durable_chunks = _record_durable_results(
                audio_tasks, durable_indices, durable_chunks,
            )
            if len(audios) != len(batch):
                raise RuntimeError(f"Model returned {len(audios)} outputs for a batch of {len(batch)}")
            retained_quality_chunks = []
            for offset, (chunk, audio) in enumerate(zip(batch, audios)):
                chunk_index = start + offset
                retry_seeds: set[int] = set()
                initial_quality_issues = find_generated_audio_issues(
                    audio.detach().cpu().numpy()
                    if hasattr(audio, "detach")
                    else audio,
                    global_model.sr,
                )
                if initial_quality_issues:
                    quality_detected_chunks.add(chunk_index)
                waveform = _recover_generated_waveform(
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
                final_quality_issues = find_generated_audio_issues(
                    waveform.numpy(),
                    global_model.sr,
                )
                if final_quality_issues:
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
                    _save_and_normalize_chunk,
                    chunk_path,
                    waveform,
                    global_model.sr,
                    ffmpeg,
                    maximum_internal_pause_seconds,
                )
            _log_batch_quality_summary(
                start,
                len(batch),
                retained_quality_chunks,
            )
            completed_chunks = start + len(batch)
            durable_chunks = _record_durable_results(
                audio_tasks, durable_indices, durable_chunks,
            )
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
            if generation_control.stop_requested():
                raise GenerationStopped

        if audio_tasks is not None:
            progress(0.975, desc="Finishing background audio processing")
            audio_tasks.finish()
            for path in audio_tasks.take_results():
                durable_indices.add(int(Path(path).stem))
            while durable_chunks in durable_indices:
                durable_chunks += 1
            audio_tasks = None
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
        progress(0.98, desc="Preparing parallel M4B encoding")

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
        final_audio_seconds = verify_m4b(output_path)
        delete_intermediate_chunks(project_dir)
        relative_output = output_path.relative_to(Path(__file__).resolve().parent).as_posix()
        _write_metadata(
            metadata_path,
            book,
            source_epub_name,
            chunks,
            settings,
            len(chunks),
            active_project_model_id,
            relative_output,
            chunks_available=False,
        )
        (project_dir / "progress.json").unlink(missing_ok=True)
        generation_elapsed = time.perf_counter() - generation_started
        final_speed = final_audio_seconds / max(generation_elapsed, 1e-9)
        progress(1, desc="Audiobook complete")
        output_path = register_completed_audiobook(
            output_path,
            OUTPUT_ROOT,
            gr.set_static_paths,
        )
        return str(output_path), (
            f"✅ **{book.title}** complete: {len(book.chapters)} chapters and "
            f"{len(chunks):,} speech chunks. Generated "
            f"{format_duration(final_audio_seconds)} of audio in "
            f"{format_duration(generation_elapsed)} ({final_speed:.2f}× realtime). "
            f"Files were saved under `{project_dir}` and intermediate chunks were removed."
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
            return None, (
                f"⏹️ Generation stopped with {durable_chunks:,} of "
                f"{len(chunks):,} chunks safely recorded. The project, saved inputs, "
                "and chunks were preserved and can be resumed."
            )
        except Exception as progress_error:
            traceback.print_exc()
            return None, (
                f"⚠️ Generation stopped and project files were preserved, but its "
                f"progress record could not be updated: {progress_error}"
            )
    except Exception as error:
        if audio_tasks is not None:
            audio_tasks.cancel_and_wait()
            audio_tasks = None
        traceback.print_exc()
        if isinstance(error, MemoryPressureError):
            return None, f"⚠️ {error}. Partial files remain in `{project_dir}`."
        location = f" Partial files remain in `{project_dir}`." if project_dir else ""
        return None, f"❌ {error}.{location}"
    finally:
        if audio_tasks is not None:
            audio_tasks.cancel_and_wait()
        generation_control.finish()


with gr.Blocks(title="Chatterbox vLLM Audiobook") as demo:
    gr.Markdown(
        "# Chatterbox vLLM\n"
        "Quick voice tests and batched EPUB audiobook generation.  \n"
        f"**Active model:** {model_label(ACTIVE_MODEL_ID)} (`{ACTIVE_MODEL_ID}`)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Voice and generation settings")
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio (normalized to -20 LUFS after upload)",
                value=None,
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
                    resume_epub_btn = gr.Button("Resume Selected Project")
                    stop_epub_btn = gr.Button("Stop Generation", variant="stop")
                epub_status = gr.Markdown("")
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

    ref_wav.input(
        fn=prepare_reference_preview,
        inputs=ref_wav,
        outputs=ref_wav,
    )
    run_btn.click(
        fn=generate_sample,
        inputs=[
            text, ref_wav, exaggeration, cfg_weight, temp, seed_num,
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
            epub_file, ref_wav, exaggeration, cfg_weight, temp, seed_num,
            diffusion_steps, min_p, top_p, repetition_penalty, max_chars,
            batch_size, new_project_state,
        ],
        # Keep the textual result hidden while this event is running. Making a
        # visible Markdown component a live output causes Gradio to render the
        # same queue/progress message both there and in its normal progress UI.
        outputs=[epub_audio_output, epub_result_status],
    )
    resume_generation_event = resume_epub_btn.click(
        fn=generate_epub_audiobook,
        inputs=[
            epub_file, ref_wav, exaggeration, cfg_weight, temp, seed_num,
            diffusion_steps, min_p, top_p, repetition_penalty, max_chars,
            batch_size, resume_project,
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
        outputs=[resume_info, epub_file, ref_wav],
        queue=False,
    )
    resume_project_event.then(
        fn=prepare_reference_preview,
        inputs=ref_wav,
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
