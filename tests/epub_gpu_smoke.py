"""Opt-in RTX smoke test for EPUB generation; not run by unit discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from test_epub import make_epub  # noqa: E402
import gradio_tts_app as app  # noqa: E402


def main() -> None:
    source = Path("/tmp/chatterbox-epub-smoke.epub")
    make_epub(source)
    started = time.perf_counter()
    app.load_model()
    loaded = time.perf_counter()
    try:
        output, status = app.generate_epub_audiobook(
            str(source),
            "docs/audio-sample-01.mp3",
            0.5,
            0.8,
            1234,
            10,
            0.05,
            1.0,
            1.2,
            120,
            2,
            progress=lambda value, desc=None: print(
                "PROGRESS", f"{value:.2f}", desc or ""
            ),
        )
        finished = time.perf_counter()
        print("SMOKE_OUTPUT", output)
        print("SMOKE_STATUS", status)
        print("MODEL_LOAD_SECONDS", round(loaded - started, 3))
        print("GENERATION_SECONDS", round(finished - loaded, 3))
        if not output or not Path(output).is_file():
            raise RuntimeError("Smoke test did not create an MP3")
        metadata = json.loads((Path(output).parent / "metadata.json").read_text())
        print("COMPLETED_CHUNKS", metadata["completed_chunks"])
        print("TOTAL_CHUNKS", metadata["total_chunks"])
        print("OUTPUT_BYTES", Path(output).stat().st_size)
    finally:
        if app.global_model is not None:
            app.global_model.shutdown()


if __name__ == "__main__":
    main()
