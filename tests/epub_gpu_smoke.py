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
        progress_descriptions = []

        def report_progress(value, desc=None):
            description = desc or ""
            progress_descriptions.append(description)
            print("PROGRESS", f"{value:.2f}", description)

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
            progress=report_progress,
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
        if not any("× realtime" in desc and "ETA" in desc for desc in progress_descriptions):
            raise RuntimeError("Smoke test did not report realtime speed and ETA")

        existing_projects = set(app.OUTPUT_ROOT.iterdir())

        def stop_after_first_batch(value, desc=None):
            description = desc or ""
            print("STOP_PROGRESS", f"{value:.2f}", description)
            if description.startswith("1/2 chunks"):
                print("STOP_REQUEST", app.request_generation_stop())

        stopped_output, stopped_status = app.generate_epub_audiobook(
            str(source),
            "docs/audio-sample-01.mp3",
            0.5,
            0.8,
            5678,
            10,
            0.05,
            1.0,
            1.2,
            120,
            1,
            progress=stop_after_first_batch,
        )
        print("STOPPED_OUTPUT", stopped_output)
        print("STOPPED_STATUS", stopped_status)
        if stopped_output is not None or "stopped after 1 of 2 chunks" not in stopped_status:
            raise RuntimeError("Cooperative stop did not stop after the first batch")
        stopped_projects = set(app.OUTPUT_ROOT.iterdir()) - existing_projects
        if len(stopped_projects) != 1:
            raise RuntimeError("Stop smoke test did not create exactly one partial project")
        stopped_project = stopped_projects.pop()
        stopped_metadata = json.loads((stopped_project / "metadata.json").read_text())
        if stopped_metadata["completed_chunks"] != 1 or (stopped_project / "audiobook.mp3").exists():
            raise RuntimeError("Stopped project metadata or output state is invalid")
    finally:
        if app.global_model is not None:
            app.global_model.shutdown()


if __name__ == "__main__":
    main()
