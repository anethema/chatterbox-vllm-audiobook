"""Opt-in RTX smoke test for EPUB generation; not run by unit discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
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
    app.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    projects_before_test = set(app.OUTPUT_ROOT.iterdir())
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
            0.5,
            0.8,
            1234,
            10,
            0.05,
            1.0,
            1.2,
            120,
            1,
            True,
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
        saved_epub, saved_reference = app.saved_project_inputs(Path(output).parent)
        if not saved_epub or not saved_epub.is_file():
            raise RuntimeError("Completed project did not retain its source EPUB")
        if not saved_reference or not saved_reference.is_file():
            raise RuntimeError("Completed project did not retain its reference audio")
        if saved_reference.read_bytes() != Path("docs/audio-sample-01.mp3").read_bytes():
            raise RuntimeError("Reference preparation modified the persisted source audio")
        metadata = json.loads((Path(output).parent / "metadata.json").read_text())
        if metadata["settings"].get("denoise_reference") is not True:
            raise RuntimeError("Project metadata did not retain reference denoising")
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
            0.5,
            0.8,
            5678,
            10,
            0.05,
            1.0,
            1.2,
            120,
            1,
            False,
            None,
            progress=stop_after_first_batch,
        )
        print("STOPPED_OUTPUT", stopped_output)
        print("STOPPED_STATUS", stopped_status)
        if stopped_output is not None or "chunks were preserved" not in stopped_status:
            raise RuntimeError("Cooperative stop did not stop after the first batch")
        stopped_projects = set(app.OUTPUT_ROOT.iterdir()) - existing_projects
        if len(stopped_projects) != 1:
            raise RuntimeError("Stopped generation did not preserve one incomplete project")
        stopped_project = stopped_projects.pop()
        saved_epub, saved_reference = app.saved_project_inputs(stopped_project)
        if not saved_epub or not saved_reference:
            raise RuntimeError("Stopped project did not retain both generation inputs")
        if not list((stopped_project / "chunks").glob("*.wav")):
            raise RuntimeError("Stopped project did not retain its completed chunks")

        stop_resumed_output, stop_resumed_status = app.generate_epub_audiobook(
            None,
            None,
            0.5,
            0.5,
            0.8,
            0,
            15,
            0.05,
            1.0,
            1.2,
            280,
            16,
            False,
            stopped_project.name,
            progress=report_progress,
        )
        print("STOP_RESUMED_OUTPUT", stop_resumed_output)
        print("STOP_RESUMED_STATUS", stop_resumed_status)
        if not stop_resumed_output or not Path(stop_resumed_output).is_file():
            raise RuntimeError("Stopped project could not be resumed to a completed M4B")

        existing_projects = set(app.OUTPUT_ROOT.iterdir())
        protect_memory = app._protect_system_memory

        def pause_before_second_batch(batch_number):
            if batch_number == 1:
                raise app.MemoryPressureError("intentional smoke-test pause")
            protect_memory(batch_number)

        app._protect_system_memory = pause_before_second_batch
        try:
            paused_output, paused_status = app.generate_epub_audiobook(
                str(source),
                "docs/audio-sample-01.mp3",
                0.5,
                0.5,
                0.8,
                9012,
                10,
                0.05,
                1.0,
                1.2,
                120,
                1,
                False,
                None,
                progress=report_progress,
            )
        finally:
            app._protect_system_memory = protect_memory
        print("PAUSED_OUTPUT", paused_output)
        print("PAUSED_STATUS", paused_status)
        if paused_output is not None or "intentional smoke-test pause" not in paused_status:
            raise RuntimeError("Memory-pressure pause did not preserve the project")
        paused_projects = set(app.OUTPUT_ROOT.iterdir()) - existing_projects
        if len(paused_projects) != 1:
            raise RuntimeError("Memory-pressure pause did not leave one incomplete project")
        paused_project = paused_projects.pop()
        saved_epub, saved_reference = app.saved_project_inputs(paused_project)
        if not saved_epub or not saved_reference:
            raise RuntimeError("Paused project did not retain both generation inputs")
        resume_info, loaded_epub, loaded_reference, loaded_denoise = (
            app.inspect_resume_project(paused_project.name)
        )
        print("RESUME_INPUT_STATUS", resume_info)
        if loaded_epub != str(saved_epub) or loaded_reference != str(saved_reference):
            raise RuntimeError("Resume selection did not restore the saved inputs")
        if loaded_denoise:
            raise RuntimeError("Resume selection changed the saved denoise setting")

        resumed_output, resumed_status = app.generate_epub_audiobook(
            None,
            None,
            0.5,
            0.5,
            0.8,
            0,
            15,
            0.05,
            1.0,
            1.2,
            280,
            16,
            False,
            paused_project.name,
            progress=report_progress,
        )
        print("RESUMED_OUTPUT", resumed_output)
        print("RESUMED_STATUS", resumed_status)
        if not resumed_output or not Path(resumed_output).is_file():
            raise RuntimeError("Resumed smoke project did not create an M4B")
        if (paused_project / "chunks").exists():
            raise RuntimeError("Resumed project retained chunks after verified completion")
    finally:
        if app.global_model is not None:
            app.global_model.shutdown()
        for project in set(app.OUTPUT_ROOT.iterdir()) - projects_before_test:
            if project.is_dir() and project.name.startswith("Test-Book-"):
                shutil.rmtree(project)
        source.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
