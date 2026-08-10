import json
import tempfile
import unittest
import wave
from pathlib import Path

from chatterbox_vllm.epub import EpubBook, EpubChapter, chunk_book
from chatterbox_vllm.projects import (
    ResumeProjectError,
    build_resume_plan,
    incomplete_project_choices,
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


if __name__ == "__main__":
    unittest.main()
