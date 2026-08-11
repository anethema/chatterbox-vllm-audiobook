#!/usr/bin/env python3
"""Run a no-output GPU batch smoke test for a selected model variant."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch

from chatterbox_vllm.ab_samples import AB_SAMPLE_PROMPTS
from chatterbox_vllm.model_variants import resolve_model_id
from chatterbox_vllm.tts import ChatterboxTTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if not args.reference.is_file():
        raise SystemExit(f"Reference audio does not exist: {args.reference}")
    model_id = resolve_model_id(args.model)
    seed = 271828
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = ChatterboxTTS.from_model_id(
        model_id,
        gpu_memory_utilization=0.6,
        max_model_len=1000,
        max_batch_size=args.batch_size,
        enforce_eager=True,
    )
    source_prompts = [text for _, text in AB_SAMPLE_PROMPTS]
    prompts = [
        f"{source_prompts[index % len(source_prompts)]}"
        for index in range(args.batch_size)
    ]
    started = time.perf_counter()
    audios = model.generate(
        prompts,
        audio_prompt_path=str(args.reference.resolve()),
        language_id="en",
        exaggeration=0.5,
        temperature=0.8,
        diffusion_steps=15,
        min_p=0.05,
        top_p=1.0,
        repetition_penalty=1.2,
        seed=seed,
    )
    elapsed = time.perf_counter() - started
    if len(audios) != args.batch_size:
        raise RuntimeError(f"Expected {args.batch_size} outputs, received {len(audios)}")
    durations = []
    for index, audio in enumerate(audios):
        waveform = torch.as_tensor(audio).detach().to(device="cpu")
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise RuntimeError(f"Batch output {index} is invalid")
        durations.append(waveform.shape[-1] / model.sr)
    total_audio = sum(durations)
    print(
        f"BATCH_SMOKE model={model_id} batch={args.batch_size} "
        f"generation_seconds={elapsed:.3f} audio_seconds={total_audio:.3f} "
        f"realtime={total_audio / elapsed:.3f}x "
        f"min_duration={min(durations):.3f} max_duration={max(durations):.3f}"
    )
    model.shutdown()


if __name__ == "__main__":
    main()
