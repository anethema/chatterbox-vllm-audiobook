#!/usr/bin/env python3
"""Generate one side of the English V1 versus Multilingual V3 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torchaudio as ta

from chatterbox_vllm.ab_samples import AB_SAMPLE_PROMPTS
from chatterbox_vllm.audio import normalize_speech_wav
from chatterbox_vllm.model_variants import model_label, resolve_model_id
from chatterbox_vllm.tts import ChatterboxTTS


SETTINGS = {
    "exaggeration": 0.5,
    "temperature": 0.8,
    "diffusion_steps": 15,
    "min_p": 0.05,
    "top_p": 1.0,
    "repetition_penalty": 1.2,
    "seed": 314159,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def waveform_for_save(audio) -> torch.Tensor:
    waveform = torch.as_tensor(audio).detach().to(dtype=torch.float32, device="cpu")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] == 0:
        raise RuntimeError(f"Invalid generated waveform shape: {tuple(waveform.shape)}")
    if not torch.isfinite(waveform).all():
        raise RuntimeError("Generated waveform contains NaN or infinite samples")
    return waveform


def main() -> None:
    args = parse_args()
    model_id = resolve_model_id(args.model)
    reference = args.reference.resolve()
    if not reference.is_file():
        raise SystemExit(f"Reference audio does not exist: {reference}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SETTINGS["seed"])
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model = ChatterboxTTS.from_model_id(
        model_id,
        gpu_memory_utilization=0.6,
        max_model_len=1000,
        max_batch_size=len(AB_SAMPLE_PROMPTS),
        enforce_eager=True,
    )
    load_seconds = time.perf_counter() - load_started

    generation_started = time.perf_counter()
    audios = model.generate(
        [text for _, text in AB_SAMPLE_PROMPTS],
        audio_prompt_path=str(reference),
        language_id="en",
        **SETTINGS,
    )
    generation_seconds = time.perf_counter() - generation_started
    if len(audios) != len(AB_SAMPLE_PROMPTS):
        raise RuntimeError(
            f"Expected {len(AB_SAMPLE_PROMPTS)} outputs, received {len(audios)}"
        )

    samples = []
    for (sample_name, text), audio in zip(AB_SAMPLE_PROMPTS, audios):
        waveform = waveform_for_save(audio)
        output_path = args.output_dir / f"{model_id}-{sample_name}.wav"
        ta.save(
            str(output_path),
            waveform,
            model.sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        normalize_speech_wav(output_path, model.sr, ffmpeg=ffmpeg)
        normalized, sample_rate = ta.load(str(output_path))
        if not torch.isfinite(normalized).all() or normalized.numel() == 0:
            raise RuntimeError(f"Normalized output is invalid: {output_path}")
        samples.append(
            {
                "name": sample_name,
                "text": text,
                "path": output_path.name,
                "duration_seconds": normalized.shape[-1] / sample_rate,
            }
        )

    metrics = {
        "model_id": model_id,
        "model_label": model_label(model_id),
        "reference": reference.name,
        "sample_rate": model.sr,
        "settings": SETTINGS,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "total_audio_seconds": sum(item["duration_seconds"] for item in samples),
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_vram_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "samples": samples,
    }
    metrics_path = args.output_dir / f"{model_id}-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    model.shutdown()


if __name__ == "__main__":
    main()
