import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatterbox_vllm.epub import EpubBook
from chatterbox_vllm.m4b import (
    ChapterMarker,
    build_ffmetadata,
    delete_intermediate_chunks,
    verify_m4b,
)


class M4BTests(unittest.TestCase):
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
