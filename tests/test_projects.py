import json
import os
import tempfile
import unittest
import wave
from pathlib import Path

from chatterbox_vllm.epub import EpubBook, EpubChapter, chunk_book
from chatterbox_vllm.projects import (
    ResumeProjectError,
    build_resume_plan,
    incomplete_project_choices,
    load_project_metadata,
    persist_project_inputs,
    saved_project_inputs,
    write_project_progress,
)


def write_wav(path: Path, sample_rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * 100)


class ResumeProjectTests(unittest.TestCase):
    def setUp(self):
        self.book = EpubBook(
            "Book",
            (EpubChapter("Chapter", "Sentence one. Sentence two. " * 20, "c.xhtml"),),
        )
        self.chunks = chunk_book(self.book, max_chars=80)

    def make_project(self, root: Path, name: str = "Book-incomplete") -> Path:
        project = root / name
        (project / "chunks").mkdir(parents=True)
        metadata = {
            "title": self.book.title,
            "settings": {"max_chars": 80, "batch_size": 4, "seed": None},
            "completed_chunks": 6,
            "total_chunks": len(self.chunks),
            "output_file": None,
            "chunks": [
                {
                    "index": index,
                    "chapter_index": chunk.chapter_index,
                    "chapter_title": chunk.chapter_title,
                    "text": chunk.text,
                    "audio_file": f"chunks/{index:06d}.wav",
                }
                for index, chunk in enumerate(self.chunks)
            ],
        }
        (project / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return project

    def test_lists_incomplete_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_project(root)
            choices = incomplete_project_choices(root)
        self.assertEqual(choices[0][1], "Book-incomplete")

    def test_resume_uses_validated_prefix_and_rolls_back_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            for index in range(7):
                write_wav(project / "chunks" / f"{index:06d}.wav")
            write_wav(project / "chunks" / ".000006.normalized-deadbeef.wav")
            plan = build_resume_plan(root, project.name, self.book, 24000)
        self.assertEqual(plan.durable_chunks, 6)
        self.assertEqual(plan.resume_index, 4)

    def test_resume_ignores_and_removes_temp_older_than_rebuilt_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            chunks_dir = project / "chunks"
            for index in range(len(self.chunks)):
                write_wav(chunks_dir / f"{index:06d}.wav")
            stale_temp = chunks_dir / ".000006.normalized-deadbeef.wav"
            write_wav(stale_temp)
            final_path = chunks_dir / "000006.wav"
            write_wav(final_path)
            os.utime(stale_temp, ns=(1_000_000_000, 1_000_000_000))
            os.utime(final_path, ns=(2_000_000_000, 2_000_000_000))

            plan = build_resume_plan(root, project.name, self.book, 24000)
            self.assertEqual(plan.durable_chunks, len(self.chunks))
            self.assertEqual(plan.resume_index, len(self.chunks))
            self.assertFalse(stale_temp.exists())

    def test_fully_generated_project_resumes_at_the_assembly_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            for index in range(len(self.chunks)):
                write_wav(project / "chunks" / f"{index:06d}.wav")

            plan = build_resume_plan(root, project.name, self.book, 24000)

        self.assertEqual(plan.durable_chunks, len(self.chunks))
        self.assertEqual(plan.resume_index, len(self.chunks))
        self.assertEqual(plan.chunks[plan.resume_index:], ())

    def test_rejects_a_different_epub(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            different = EpubBook(
                "Book",
                (EpubChapter("Chapter", "Different text.", "c.xhtml"),),
            )
            with self.assertRaisesRegex(ResumeProjectError, "chunk count"):
                build_resume_plan(root, project.name, different, 24000)

    def test_rejects_project_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ResumeProjectError, "directly below"):
                build_resume_plan(directory, "../elsewhere", self.book, 24000)

    def test_persists_and_discovers_project_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            epub = root / "original.epub"
            reference = root / "voice.MP3"
            epub.write_bytes(b"epub-data")
            reference.write_bytes(b"audio-data")
            saved_epub, saved_reference = persist_project_inputs(
                project, epub, reference,
            )
            discovered = saved_project_inputs(project)
            self.assertEqual(saved_epub.name, "source.epub")
            self.assertEqual(saved_reference.name, "reference-audio.mp3")
            self.assertEqual(saved_epub.read_bytes(), b"epub-data")
            self.assertEqual(saved_reference.read_bytes(), b"audio-data")
            self.assertEqual(discovered, (saved_epub, saved_reference))

    def test_replacing_reference_removes_the_previous_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            epub = root / "original.epub"
            first = root / "voice.wav"
            second = root / "voice.flac"
            epub.write_bytes(b"epub-data")
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            persist_project_inputs(project, epub, first)
            _, saved_reference = persist_project_inputs(project, epub, second)
            references = list((project / "inputs").glob("reference-audio.*"))
        self.assertEqual(saved_reference.name, "reference-audio.flac")
        self.assertEqual(references, [saved_reference])

    def test_small_progress_file_overrides_stale_metadata_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            write_project_progress(project, 7, 8)
            _, metadata = load_project_metadata(root, project.name)
            progress_size = (project / "progress.json").stat().st_size
        self.assertEqual(metadata["completed_chunks"], 7)
        self.assertEqual(metadata["scheduled_chunks"], 8)
        self.assertLess(progress_size, 200)


if __name__ == "__main__":
    unittest.main()
