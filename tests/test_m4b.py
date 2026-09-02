import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch
import wave

from chatterbox_vllm.epub import EpubBook, TextChunk
from chatterbox_vllm.m4b import (
    AssemblyStopped,
    AudioEntry,
    ChapterMarker,
    assemble_audiobook,
    build_ffmetadata,
    build_audio_timeline,
    default_m4b_workers,
    delete_intermediate_chunks,
    physical_core_cpu_ids,
    plan_encoding_segments,
    verify_m4b,
)


class M4BTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path, duration_seconds=0.25, sample_rate=24000):
        frames = round(duration_seconds * sample_rate)
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(b"\0\0" * frames)

    @staticmethod
    def _write_wav_frames(path, frames, sample_rate):
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(b"\0\0" * frames)

    def test_builds_global_metadata_and_chapter_markers(self):
        book = EpubBook(
            "A Book",
            (),
            authors=("One Author", "Two Author"),
            language="en",
            publisher="Example; Press",
        )
        metadata = build_ffmetadata(
            book,
            [ChapterMarker("Chapter #1", 0, 1250), ChapterMarker("Two", 1250, 2500)],
        )
        self.assertIn("artist=One Author, Two Author", metadata)
        self.assertIn("publisher=Example\\; Press", metadata)
        self.assertIn("START=1250", metadata)
        self.assertIn("title=Chapter \\#1", metadata)
        self.assertEqual(metadata.count("[CHAPTER]"), 2)

    def test_verifies_nonempty_positive_duration_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "book.m4b"
            output.write_bytes(b"m4b")
            probe = subprocess.CompletedProcess(
                ["ffprobe"],
                0,
                json.dumps(
                    {
                        "streams": [{"codec_name": "aac", "duration": "N/A"}],
                        "format": {"duration": "123.45"},
                    }
                ),
                "",
            )
            with patch("chatterbox_vllm.m4b.subprocess.run", return_value=probe):
                duration = verify_m4b(output, ffprobe="ffprobe")

        self.assertEqual(duration, 123.45)

    def test_rejects_an_empty_m4b_before_running_ffprobe(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "book.m4b"
            output.touch()
            with patch("chatterbox_vllm.m4b.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    verify_m4b(output, ffprobe="ffprobe")
            run.assert_not_called()

    def test_rejects_probe_output_without_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "book.m4b"
            output.write_bytes(b"m4b")
            probe = subprocess.CompletedProcess(
                ["ffprobe"], 0, json.dumps({"streams": [], "format": {}}), ""
            )
            with patch("chatterbox_vllm.m4b.subprocess.run", return_value=probe):
                with self.assertRaisesRegex(RuntimeError, "valid audio"):
                    verify_m4b(output, ffprobe="ffprobe")

    def test_verifies_expected_chapters_and_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "book.m4b"
            output.write_bytes(b"m4b")
            probe = subprocess.CompletedProcess(
                ["ffprobe"],
                0,
                json.dumps(
                    {
                        "streams": [{"codec_name": "aac", "duration": "10.1"}],
                        "format": {"duration": "10.1"},
                        "chapters": [
                            {"start_time": "0", "end_time": "5"},
                            {"start_time": "5", "end_time": "10.1"},
                        ],
                    }
                ),
                "",
            )
            with patch("chatterbox_vllm.m4b.subprocess.run", return_value=probe):
                duration = verify_m4b(
                    output,
                    ffprobe="ffprobe",
                    expected_duration_seconds=10,
                    expected_chapters=2,
                )

        self.assertEqual(duration, 10.1)

    def test_rejects_invalid_chapter_topology(self):
        invalid_chapters = (
            [{"start_time": "-0.1", "end_time": "1"}],
            [{"start_time": "0", "end_time": "1"}, {"start_time": "0.9", "end_time": "2"}],
            [{"start_time": "0", "end_time": "5"}, {"start_time": "1", "end_time": "4"}],
            [{"start_time": "0", "end_time": "10.01"}],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "book.m4b"
            output.write_bytes(b"m4b")
            for chapters in invalid_chapters:
                probe = subprocess.CompletedProcess(
                    ["ffprobe"],
                    0,
                    json.dumps(
                        {
                            "streams": [{"codec_name": "aac", "duration": "10"}],
                            "format": {"duration": "10"},
                            "chapters": chapters,
                        }
                    ),
                    "",
                )
                with self.subTest(chapters=chapters), patch(
                    "chatterbox_vllm.m4b.subprocess.run", return_value=probe
                ):
                    with self.assertRaisesRegex(RuntimeError, "invalid chapter timestamps"):
                        verify_m4b(output, ffprobe="ffprobe")

    def test_timeline_rounds_cumulative_sample_durations_only(self):
        sample_rate = 44100
        chunk_frames = 24
        chunk_count = 200
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "book"
            chunks_dir = project / "chunks"
            assembly = project / ".assembly"
            chunks_dir.mkdir(parents=True)
            assembly.mkdir()
            for index in range(chunk_count):
                self._write_wav_frames(chunks_dir / f"{index:06d}.wav", chunk_frames, sample_rate)
            chunks = [TextChunk(0, "One", "Chunk") for _ in range(chunk_count)]

            _, markers, timeline_ms = build_audio_timeline(project, assembly, chunks, sample_rate)

        expected_duration = (
            chunk_count * Fraction(chunk_frames, sample_rate)
            + (chunk_count - 1) * Fraction(round(sample_rate * 0.18), sample_rate)
        )
        self.assertEqual(timeline_ms, round(expected_duration * 1000))
        self.assertEqual(markers, [ChapterMarker("One", 0, round(expected_duration * 1000))])

    def test_balances_segments_on_chapter_boundaries(self):
        entries = [
            AudioEntry(Path(f"{index}.wav"), duration, starts_chapter=chapter)
            for index, (duration, chapter) in enumerate(
                [
                    (1000, False),
                    (100, True),
                    (900, False),
                    (100, True),
                    (900, False),
                    (100, True),
                    (900, False),
                ]
            )
        ]

        segments = plan_encoding_segments(entries, workers=4)

        self.assertEqual(len(segments), 4)
        self.assertEqual([segment.entries[0].path.name for segment in segments], [
            "0.wav", "1.wav", "3.wav", "5.wav",
        ])
        self.assertEqual(
            [entry.path for segment in segments for entry in segment.entries],
            [entry.path for entry in entries],
        )
        self.assertTrue(all(segment.entries for segment in segments))

    def test_uses_pause_or_entry_boundaries_when_chapters_are_sparse(self):
        entries = [
            AudioEntry(Path("0.wav"), 1000),
            AudioEntry(Path("pause.wav"), 100, is_pause=True),
            AudioEntry(Path("1.wav"), 1000),
            AudioEntry(Path("2.wav"), 1000),
        ]

        segments = plan_encoding_segments(entries, workers=3)

        self.assertEqual(len(segments), 3)
        self.assertEqual(sum(len(segment.entries) for segment in segments), len(entries))
        self.assertEqual(segments[1].entries[0].path.name, "pause.wav")

    def test_defaults_to_sixteen_m4b_workers_on_32_threads(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_m4b_workers(cpu_count=32), 16)
            self.assertEqual(default_m4b_workers(cpu_count=4), 2)

    def test_selects_one_allowed_logical_cpu_per_physical_core(self):
        with tempfile.TemporaryDirectory() as directory:
            topology_root = Path(directory)
            for cpu, core in ((0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)):
                topology = topology_root / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text("0")
                (topology / "core_id").write_text(str(core))

            selected = physical_core_cpu_ids(
                available_cpu_ids=(1, 2, 3, 4, 5),
                topology_root=topology_root,
            )

        self.assertEqual(selected, (1, 2, 4))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_parallel_assembly_creates_verified_chaptered_m4b(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "book"
            chunks_dir = project / "chunks"
            chunks_dir.mkdir(parents=True)
            for index in range(4):
                self._write_wav(chunks_dir / f"{index:06d}.wav")
            chunks = [
                TextChunk(0, "One", "First"),
                TextChunk(0, "One", "Second"),
                TextChunk(1, "Two", "Third"),
                TextChunk(1, "Two", "Fourth"),
            ]
            updates = []

            output = assemble_audiobook(
                project,
                EpubBook("Book", ()),
                chunks,
                24000,
                workers=2,
                progress_callback=updates.append,
            )

            self.assertEqual(output, project / "audiobook.m4b")
            self.assertTrue(output.is_file())
            self.assertGreater(verify_m4b(output, expected_chapters=2), 0)
            self.assertFalse((project / ".assembly").exists())
            self.assertTrue(any(update.phase == "encoding" for update in updates))
            self.assertTrue(any(update.phase == "muxing" for update in updates))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_stop_terminates_encoding_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "book"
            chunks_dir = project / "chunks"
            chunks_dir.mkdir(parents=True)
            self._write_wav(chunks_dir / "000000.wav", duration_seconds=2)

            with self.assertRaises(AssemblyStopped):
                assemble_audiobook(
                    project,
                    EpubBook("Book", ()),
                    [TextChunk(0, "One", "First")],
                    24000,
                    workers=1,
                    stop_requested=lambda: True,
                )

            self.assertFalse((project / ".assembly").exists())
            self.assertFalse((project / "audiobook.m4b").exists())
            self.assertTrue((chunks_dir / "000000.wav").is_file())

    def test_deletes_only_the_intermediate_chunk_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "book"
            chunks = project / "chunks"
            chunks.mkdir(parents=True)
            (chunks / "000001.wav").write_bytes(b"audio")
            output = project / "audiobook.m4b"
            output.write_bytes(b"m4b")

            delete_intermediate_chunks(project)

            self.assertFalse(chunks.exists())
            self.assertTrue(output.is_file())
            self.assertTrue(project.is_dir())


if __name__ == "__main__":
    unittest.main()
