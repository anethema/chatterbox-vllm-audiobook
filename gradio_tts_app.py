import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
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
    TARGET_LUFS,
    TRUE_PEAK_DBTP,
    normalize_speech_wav,
)
from chatterbox_vllm.background import BackgroundTaskPool
from chatterbox_vllm.epub import EpubBook, EpubError, TextChunk, chunk_book, load_epub
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
    ENGLISH_V1_MODEL_ID,
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
from chatterbox_vllm.ab_samples import AB_SAMPLE_PROMPTS


DEVICE = "cuda"
OUTPUT_ROOT = Path(__file__).resolve().parent / "audiobook_outputs"
AB_SAMPLE_ROOT = OUTPUT_ROOT / "_v3_ab_samples"
ACTIVE_MODEL_ID = resolve_model_id(
    os.environ.get("CHATTERBOX_MODEL_VARIANT", DEFAULT_MODEL_ID)
)
AUDIO_WORKERS = min(4, os.cpu_count() or 1)
MAX_PENDING_AUDIO_TASKS = AUDIO_WORKERS * 16
MEMORY_CLEANUP_BATCHES = 64
MEMORY_CLEANUP_HEADROOM = 4 * 1024 ** 3
MINIMUM_MEMORY_HEADROOM = 2 * 1024 ** 3

config_seed = None
global_model = None
generation_control = GenerationControl()


def ab_sample_value(model_id: str, sample_name: str) -> str | None:
    path = AB_SAMPLE_ROOT / f"{model_id}-{sample_name}.wav"
    return str(path) if path.is_file() else None


class GenerationStopped(Exception):
    pass


class MemoryPressureError(RuntimeError):
    pass


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


def generation_arguments(exaggeration, temperature, diffusion_steps, min_p,
                         top_p, repetition_penalty, seed):
    return {
        "exaggeration": float(exaggeration),
        "temperature": float(temperature),
        "diffusion_steps": int(diffusion_steps),
        "min_p": float(min_p),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "seed": seed,
    }


def generate_sample(text, audio_prompt_path, exaggeration, temperature, seed_num,
                    diffusion_steps, min_p, top_p, repetition_penalty):
    if not text or not text.strip():
        raise gr.Error("Enter some text to synthesize.")

    seed = selected_seed(seed_num)
    args = generation_arguments(
        exaggeration, temperature, diffusion_steps, min_p, top_p,
        repetition_penalty, seed,
    )
    print(f"Using text: {text}")
    print(f"Using audio_prompt_path: {audio_prompt_path}")
    print(f"Using settings: {args}")

    wav = global_model.generate(
        [text.strip()],
        audio_prompt_path=audio_prompt_path,
        **args,
    )
    waveform = _waveform_for_save(wav[0], text.strip(), global_model.sr)
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


def _waveform_for_save(waveform, text: str, sample_rate: int) -> torch.Tensor:
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
    return tensor


def _save_and_normalize_chunk(
    path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
    ffmpeg: str,
) -> Path:
    ta.save(
        str(path),
        waveform,
        sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    return normalize_speech_wav(path, sample_rate, ffmpeg=ffmpeg)


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


def generate_epub_audiobook(epub_path, audio_prompt_path, exaggeration, temperature,
                            seed_num, diffusion_steps, min_p, top_p,
                            repetition_penalty, max_chars, batch_size,
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
                exaggeration, temperature, diffusion_steps, min_p, top_p,
                repetition_penalty, seed,
            )
            settings.update(
                {
                    "max_chars": int(max_chars),
                    "batch_size": int(batch_size),
                    "loudness_target_lufs": TARGET_LUFS,
                    "true_peak_dbtp": TRUE_PEAK_DBTP,
                    "loudness_range_lu": LOUDNESS_RANGE_LU,
                }
            )

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
        total_characters = sum(len(chunk.text) for chunk in remaining_chunks)
        completed_characters = 0
        generated_audio_seconds = 0.0
        generation_started = time.perf_counter()
        if remaining_chunks:
            audio_tasks = BackgroundTaskPool(
                max_workers=AUDIO_WORKERS,
                max_pending=MAX_PENDING_AUDIO_TASKS,
            )
            progress(0, desc=f"Preparing voice for {len(remaining_chunks):,} remaining chunks")
            s3gen_ref, cond_emb = global_model.get_audio_conditionals(audio_prompt_path)
        else:
            progress(0.975, desc="All speech chunks found; preparing M4B assembly")
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
            for offset, (chunk, audio) in enumerate(zip(batch, audios)):
                chunk_index = start + offset
                waveform = _waveform_for_save(audio, chunk.text, global_model.sr)
                generated_audio_seconds += waveform.shape[1] / global_model.sr
                completed_characters += len(chunk.text)
                chunk_path = chunks_dir / f"{chunk_index:06d}.wav"
                audio_tasks.submit(
                    _save_and_normalize_chunk,
                    chunk_path,
                    waveform,
                    global_model.sr,
                    ffmpeg,
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
                label="Reference Audio File",
                value=None,
            )
            exaggeration = gr.Slider(
                0.25, 2, step=.05,
                label="Exaggeration (Neutral = 0.5; extreme values can be unstable)",
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
                        120, 300, step=10, value=280,
                        label="Maximum characters per speech chunk",
                    )
                    batch_size = gr.Slider(
                        1, 32, step=1, value=16,
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

    with gr.Accordion("Temporary V3 A/B listening samples", open=True):
        gr.Markdown(
            "These samples use the same Jessica reference, prompt, seed, and "
            "generation settings. **A** is the original English model and **B** "
            "is Multilingual V3 in English mode. Both are normalized to -18 LUFS."
        )
        for sample_number, (sample_name, sample_text) in enumerate(
            AB_SAMPLE_PROMPTS,
            start=1,
        ):
            gr.Markdown(f"**Sample {sample_number}:** {sample_text}")
            with gr.Row():
                gr.Audio(
                    value=ab_sample_value(ENGLISH_V1_MODEL_ID, sample_name),
                    label="A — Original English",
                    interactive=False,
                )
                gr.Audio(
                    value=ab_sample_value(MULTILINGUAL_V3_MODEL_ID, sample_name),
                    label="B — Multilingual V3 (English)",
                    interactive=False,
                )

    run_btn.click(
        fn=generate_sample,
        inputs=[
            text, ref_wav, exaggeration, temp, seed_num, diffusion_steps,
            min_p, top_p, repetition_penalty,
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
            epub_file, ref_wav, exaggeration, temp, seed_num, diffusion_steps,
            min_p, top_p, repetition_penalty, max_chars, batch_size,
            new_project_state,
        ],
        # Keep the textual result hidden while this event is running. Making a
        # visible Markdown component a live output causes Gradio to render the
        # same queue/progress message both there and in its normal progress UI.
        outputs=[epub_audio_output, epub_result_status],
    )
    resume_generation_event = resume_epub_btn.click(
        fn=generate_epub_audiobook,
        inputs=[
            epub_file, ref_wav, exaggeration, temp, seed_num, diffusion_steps,
            min_p, top_p, repetition_penalty, max_chars, batch_size,
            resume_project,
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
    resume_project.change(
        fn=inspect_resume_project,
        inputs=resume_project,
        outputs=[resume_info, epub_file, ref_wav],
        queue=False,
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
    ).launch(server_name="0.0.0.0", server_port=7860, share=command_line.share)
