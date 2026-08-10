"""Opt-in RTX smoke test for EPUB generation; not run by unit discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
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
            1,
            None,
            progress=report_progress,
        )
        finished = time.perf_counter()
        print("SMOKE_OUTPUT", output)
        print("SMOKE_STATUS", status)
        print("MODEL_LOAD_SECONDS", round(loaded - started, 3))
        print("GENERATION_SECONDS", round(finished - loaded, 3))
        if not output or not Path(output).is_file():
            raise RuntimeError("Smoke test did not create an M4B")
        if Path(output).suffix.lower() != ".m4b":
            raise RuntimeError("Smoke test output does not use the .m4b extension")
        metadata = json.loads((Path(output).parent / "metadata.json").read_text())
        print("COMPLETED_CHUNKS", metadata["completed_chunks"])
        print("TOTAL_CHUNKS", metadata["total_chunks"])
        print("OUTPUT_BYTES", Path(output).stat().st_size)
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format_tags:stream=codec_name:chapter=start_time,end_time:chapter_tags",
                    "-of", "json", output,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        print("M4B_PROBE", json.dumps(probe, sort_keys=True))
        if probe["streams"][0]["codec_name"] != "aac":
            raise RuntimeError("M4B audio stream is not AAC")
        if not any(stream.get("codec_name") == "png" for stream in probe["streams"]):
            raise RuntimeError("M4B does not contain the EPUB PNG cover")
        if len(probe.get("chapters", [])) != 2:
            raise RuntimeError("M4B does not contain two chapter markers")
        tags = probe.get("format", {}).get("tags", {})
        if tags.get("title") != "Test Book" or tags.get("artist") != "Test Author":
            raise RuntimeError("M4B did not preserve EPUB title and author metadata")
        if not any("× realtime" in desc and "ETA" in desc for desc in progress_descriptions):
            raise RuntimeError("Smoke test did not report realtime speed and ETA")
        first_metrics = next(
            index for index, desc in enumerate(progress_descriptions)
            if "× realtime" in desc
        )
        if any(
            desc.startswith("Generating chunks")
            for desc in progress_descriptions[first_metrics + 1:]
        ):
            raise RuntimeError("A generic batch message replaced the visible speed and ETA")

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
            None,
            progress=stop_after_first_batch,
        )
        print("STOPPED_OUTPUT", stopped_output)
        print("STOPPED_STATUS", stopped_status)
        if stopped_output is not None or "chunk files were deleted" not in stopped_status:
            raise RuntimeError("Cooperative stop did not stop after the first batch")
        stopped_projects = set(app.OUTPUT_ROOT.iterdir()) - existing_projects
        if stopped_projects:
            raise RuntimeError("Stopped project directory was not deleted")
    finally:
        if app.global_model is not None:
            app.global_model.shutdown()


if __name__ == "__main__":
    main()
