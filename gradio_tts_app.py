import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
import traceback
from uuid import uuid4
import wave

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
from chatterbox_vllm.epub import EpubBook, EpubError, TextChunk, chunk_book, load_epub
from chatterbox_vllm.m4b import ChapterMarker, build_ffmetadata
from chatterbox_vllm.progress import GenerationControl, estimate_progress, format_duration
from chatterbox_vllm.tts import ChatterboxTTS


DEVICE = "cuda"
OUTPUT_ROOT = Path(__file__).resolve().parent / "audiobook_outputs"

config_seed = None
global_model = None
generation_control = GenerationControl()


class GenerationStopped(Exception):
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
    print("Loading model...")
    global global_model
    global_model = ChatterboxTTS.from_pretrained(
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
    return (global_model.sr, wav[0].squeeze(0).numpy())


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
    return "⏹️ Stop requested. The current vLLM batch will finish, then generation will stop."


def _safe_project_name(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-._")
    return (slug[:80] or "audiobook") + "-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]


def _delete_stopped_project(project_dir: Path) -> None:
    """Delete only a generated project that is directly below OUTPUT_ROOT."""

    output_root = OUTPUT_ROOT.resolve()
    target = project_dir.resolve()
    if target == output_root or target.parent != output_root:
        raise RuntimeError(f"Refusing to delete unexpected project path: {target}")
    shutil.rmtree(target)


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


def _write_metadata(path: Path, book: EpubBook, source_path: str, chunks: list[TextChunk],
                    settings: dict, completed: int, output_path: str | None = None):
    data = {
        "title": book.title,
        "authors": list(book.authors),
        "language": book.language,
        "publisher": book.publisher,
        "description": book.description,
        "date": book.date,
        "identifier": book.identifier,
        "source_epub": Path(source_path).name,
        "settings": settings,
        "completed_chunks": completed,
        "total_chunks": len(chunks),
        "output_file": output_path,
        "chunks": [
            {
                "index": index,
                "chapter_index": chunk.chapter_index,
                "chapter_title": chunk.chapter_title,
                "text": chunk.text,
                "audio_file": f"chunks/{index:06d}.wav" if index < completed else None,
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_silence(path: Path, sample_rate: int, duration: float):
    samples = torch.zeros((1, max(1, int(sample_rate * duration))), dtype=torch.float32)
    ta.save(str(path), samples, sample_rate, encoding="PCM_S", bits_per_sample=16)


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return round(audio.getnframes() * 1000 / audio.getframerate())


def _cover_suffix(media_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
    }.get(media_type.lower(), "")


def _assemble_audiobook(
    project_dir: Path,
    book: EpubBook,
    chunks: list[TextChunk],
    sample_rate: int,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to assemble the audiobook M4B")

    short_pause = project_dir / "pause-between-chunks.wav"
    chapter_pause = project_dir / "pause-between-chapters.wav"
    _write_silence(short_pause, sample_rate, 0.18)
    _write_silence(chapter_pause, sample_rate, 0.9)

    concat_path = project_dir / "concat.txt"
    entries: list[Path] = []
    chapter_starts: list[tuple[str, int]] = []
    timeline_ms = 0
    for index, chunk in enumerate(chunks):
        chapter_changed = index == 0 or chunk.chapter_index != chunks[index - 1].chapter_index
        if chapter_changed:
            chapter_starts.append((chunk.chapter_title, timeline_ms))
        if index:
            previous = chunks[index - 1]
            pause = chapter_pause if chunk.chapter_index != previous.chapter_index else short_pause
            entries.append(pause)
            timeline_ms += _wav_duration_ms(pause)
        chunk_path = project_dir / "chunks" / f"{index:06d}.wav"
        entries.append(chunk_path)
        timeline_ms += _wav_duration_ms(chunk_path)
    concat_path.write_text(
        "".join(f"file '{entry.as_posix()}'\n" for entry in entries),
        encoding="utf-8",
    )

    markers = [
        ChapterMarker(
            title,
            start,
            chapter_starts[index + 1][1] if index + 1 < len(chapter_starts) else timeline_ms,
        )
        for index, (title, start) in enumerate(chapter_starts)
    ]
    ffmetadata_path = project_dir / "audiobook.ffmetadata"
    ffmetadata_path.write_text(build_ffmetadata(book, markers), encoding="utf-8")

    cover_path = None
    cover_suffix = _cover_suffix(book.cover_media_type)
    if book.cover_image and cover_suffix:
        cover_path = project_dir / f"cover{cover_suffix}"
        cover_path.write_bytes(book.cover_image)

    output_path = project_dir / "audiobook.m4b"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-f", "ffmetadata", "-i", str(ffmetadata_path),
    ]
    if cover_path:
        command.extend(["-i", str(cover_path)])
    command.extend(["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"])
    if cover_path:
        command.extend(
            [
                "-map", "2:v:0", "-c:v", "copy", "-disposition:v:0", "attached_pic",
                "-metadata:s:v:0", "title=Cover",
            ]
        )
    command.extend(
        [
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
            "-f", "mp4", str(output_path),
        ]
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to assemble the M4B: {result.stderr.strip()}")
    return output_path


def generate_epub_audiobook(epub_path, audio_prompt_path, exaggeration, temperature,
                            seed_num, diffusion_steps, min_p, top_p,
                            repetition_penalty, max_chars, batch_size,
                            progress=gr.Progress()):
    if not epub_path:
        return None, "❌ Upload an EPUB first."
    if not audio_prompt_path:
        return None, "❌ Upload or record reference audio first."

    project_dir = None
    chunks = []
    completed_chunks = 0
    generation_control.begin()
    try:
        book = load_epub(epub_path)
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
        project_dir = OUTPUT_ROOT / _safe_project_name(book.title)
        chunks_dir = project_dir / "chunks"
        chunks_dir.mkdir(parents=True)
        metadata_path = project_dir / "metadata.json"
        _write_metadata(metadata_path, book, epub_path, chunks, settings, 0)

        progress(0, desc=f"Preparing voice for {len(chunks):,} chunks")
        s3gen_ref, cond_emb = global_model.get_audio_conditionals(audio_prompt_path)
        batch_size = int(batch_size)
        total_characters = sum(len(chunk.text) for chunk in chunks)
        completed_characters = 0
        generated_audio_seconds = 0.0
        generation_started = time.perf_counter()
        for start in range(0, len(chunks), batch_size):
            if generation_control.stop_requested():
                raise GenerationStopped
            batch = chunks[start:start + batch_size]
            # Before the first measurement, show which batch is starting. After
            # that, leave the latest speed/ETA visible while the next batch runs;
            # replacing it here made the useful metrics flash too briefly to see.
            if start == 0:
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
            if len(audios) != len(batch):
                raise RuntimeError(f"Model returned {len(audios)} outputs for a batch of {len(batch)}")
            for offset, (chunk, audio) in enumerate(zip(batch, audios)):
                chunk_index = start + offset
                waveform = _waveform_for_save(audio, chunk.text, global_model.sr)
                generated_audio_seconds += waveform.shape[1] / global_model.sr
                completed_characters += len(chunk.text)
                chunk_path = chunks_dir / f"{chunk_index:06d}.wav"
                ta.save(
                    str(chunk_path),
                    waveform,
                    global_model.sr,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
                normalize_speech_wav(
                    chunk_path,
                    global_model.sr,
                    ffmpeg=ffmpeg,
                )
            completed_chunks = start + len(batch)
            _write_metadata(
                metadata_path, book, epub_path, chunks, settings,
                completed_chunks,
            )
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

        progress(0.98, desc="Assembling chaptered M4B")
        output_path = _assemble_audiobook(project_dir, book, chunks, global_model.sr)
        relative_output = output_path.relative_to(Path(__file__).resolve().parent).as_posix()
        _write_metadata(metadata_path, book, epub_path, chunks, settings, len(chunks), relative_output)
        generation_elapsed = time.perf_counter() - generation_started
        final_speed = generated_audio_seconds / max(generation_elapsed, 1e-9)
        progress(1, desc="Audiobook complete")
        return str(output_path), (
            f"✅ **{book.title}** complete: {len(book.chapters)} chapters and "
            f"{len(chunks):,} speech chunks. Generated "
            f"{format_duration(generated_audio_seconds)} of audio in "
            f"{format_duration(generation_elapsed)} ({final_speed:.2f}× realtime). "
            f"Files were saved under `{project_dir}`."
        )
    except GenerationStopped:
        print(f"EPUB generation stopped after {completed_chunks} of {len(chunks)} chunks")
        try:
            _delete_stopped_project(project_dir)
            return None, (
                f"⏹️ Generation stopped after {completed_chunks:,} of "
                f"{len(chunks):,} chunks. The incomplete project and its chunk "
                "files were deleted."
            )
        except Exception as cleanup_error:
            traceback.print_exc()
            return None, (
                f"⚠️ Generation stopped, but the incomplete project could not be "
                f"deleted: {cleanup_error}"
            )
    except Exception as error:
        traceback.print_exc()
        location = f" Partial files remain in `{project_dir}`." if project_dir else ""
        return None, f"❌ {error}.{location}"
    finally:
        generation_control.finish()


with gr.Blocks(title="Chatterbox vLLM Audiobook") as demo:
    gr.Markdown("# Chatterbox vLLM\nQuick voice tests and batched EPUB audiobook generation.")

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
                    value=10,
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
            with gr.Tab("Text Sample"):
                text = gr.Textbox(
                    value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse.",
                    label="Text to synthesize",
                    max_lines=6,
                )
                run_btn = gr.Button("Generate Sample", variant="primary")
                audio_output = gr.Audio(label="Output Audio")

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
                        1, 32, step=1, value=8,
                        label="vLLM batch size",
                    )
                epub_info = gr.Markdown(
                    "Upload a DRM-free EPUB to inspect its chapters before generation."
                )
                with gr.Row():
                    epub_btn = gr.Button("Generate EPUB Audiobook", variant="primary")
                    stop_epub_btn = gr.Button("Stop Generation", variant="stop")
                epub_status = gr.Markdown("")
                epub_result_status = gr.State("")
                epub_audio_output = gr.Audio(label="Completed Audiobook (M4B)")

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
        ],
        # Keep the textual result hidden while this event is running. Making a
        # visible Markdown component a live output causes Gradio to render the
        # same queue/progress message both there and in its normal progress UI.
        outputs=[epub_audio_output, epub_result_status],
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


if __name__ == "__main__":
    # Don't let Gradio manage model loading; it causes issues with vLLM workers.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    load_model()

    print("Starting Gradio app...")
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(server_name="0.0.0.0", server_port=7860, share=False)
