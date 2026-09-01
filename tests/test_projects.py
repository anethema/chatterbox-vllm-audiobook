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
    delete_quality_scan_checkpoint,
    incomplete_project_choices,
    load_quality_scan_checkpoint,
    load_project_metadata,
    persist_project_inputs,
    saved_project_inputs,
    quality_scan_checkpoint_entry_matches,
    wav_file_identity,
    write_quality_scan_checkpoint,
    write_project_progress,
)
from chatterbox_vllm.model_variants import (
    ENGLISH_V1_MODEL_ID,
    MULTILINGUAL_V3_MODEL_ID,
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

    def test_skips_a_project_with_an_unknown_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            metadata_path = project / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["model_id"] = "future-model"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            choices = incomplete_project_choices(root)
        self.assertEqual(choices, [])

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

    def test_rejects_switching_models_with_speech_chunks_remaining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            for index in range(4):
                write_wav(project / "chunks" / f"{index:06d}.wav")

            with self.assertRaisesRegex(
                ResumeProjectError,
                "CHATTERBOX_MODEL_VARIANT=english-v1",
            ):
                build_resume_plan(
                    root,
                    project.name,
                    self.book,
                    24000,
                    expected_model_id=MULTILINGUAL_V3_MODEL_ID,
                )

    def test_allows_assembly_only_resume_under_a_different_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            metadata_path = project / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["model_id"] = ENGLISH_V1_MODEL_ID
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            for index in range(len(self.chunks)):
                write_wav(project / "chunks" / f"{index:06d}.wav")

            plan = build_resume_plan(
                root,
                project.name,
                self.book,
                24000,
                expected_model_id=MULTILINGUAL_V3_MODEL_ID,
            )

        self.assertEqual(plan.resume_index, len(self.chunks))

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

    def test_legacy_project_without_quality_checkpoint_rescans_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            chunk_path = project / "chunks" / "000000.wav"
            write_wav(chunk_path)

            checkpoint = load_quality_scan_checkpoint(project)

        self.assertEqual(checkpoint, {})
        self.assertFalse(
            quality_scan_checkpoint_entry_matches(checkpoint, 0, chunk_path)
        )

    def test_unchanged_verified_chunk_is_safe_to_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            chunk_path = project / "chunks" / "000000.wav"
            write_wav(chunk_path)
            identity = wav_file_identity(chunk_path)
            self.assertIsNotNone(identity)
            write_quality_scan_checkpoint(project, {0: identity})

            checkpoint = load_quality_scan_checkpoint(project)
            matches = quality_scan_checkpoint_entry_matches(checkpoint, 0, chunk_path)

        self.assertTrue(matches)

    def test_changed_verified_chunk_is_rescanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            chunk_path = project / "chunks" / "000000.wav"
            write_wav(chunk_path)
            identity = wav_file_identity(chunk_path)
            self.assertIsNotNone(identity)
            write_quality_scan_checkpoint(project, {0: identity})
            write_wav(chunk_path)
            os.utime(chunk_path, ns=(2_000_000_000, 2_000_000_000))

            checkpoint = load_quality_scan_checkpoint(project)
            matches = quality_scan_checkpoint_entry_matches(checkpoint, 0, chunk_path)

        self.assertFalse(matches)

    def test_missing_verified_chunk_is_rescanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            chunk_path = project / "chunks" / "000000.wav"
            write_wav(chunk_path)
            identity = wav_file_identity(chunk_path)
            self.assertIsNotNone(identity)
            write_quality_scan_checkpoint(project, {0: identity})
            chunk_path.unlink()

            checkpoint = load_quality_scan_checkpoint(project)
            matches = quality_scan_checkpoint_entry_matches(checkpoint, 0, chunk_path)

        self.assertFalse(matches)

    def test_only_verified_new_or_repaired_chunks_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            new_path = project / "chunks" / "000000.wav"
            repaired_path = project / "chunks" / "000001.wav"
            write_wav(new_path)
            write_wav(repaired_path)
            write_quality_scan_checkpoint(
                project,
                {
                    0: wav_file_identity(new_path),
                    1: wav_file_identity(repaired_path),
                    # A noisy/retained chunk is deliberately absent.
                },
            )
            checkpoint = load_quality_scan_checkpoint(project)

        self.assertEqual(set(checkpoint), {0, 1})
        self.assertNotIn(2, checkpoint)

    def test_malformed_or_stale_quality_checkpoint_fails_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            checkpoint_path = project / "quality-scan.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "detector_version": 999,
                        "verified_clean_chunks": {"000000": {"size": 1}},
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = load_quality_scan_checkpoint(project)

        self.assertEqual(checkpoint, {})

    def test_completed_project_cleanup_removes_quality_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root)
            chunk_path = project / "chunks" / "000000.wav"
            write_wav(chunk_path)
            identity = wav_file_identity(chunk_path)
            self.assertIsNotNone(identity)
            write_quality_scan_checkpoint(project, {0: identity})

            delete_quality_scan_checkpoint(project)
            checkpoint = load_quality_scan_checkpoint(project)

        self.assertEqual(checkpoint, {})


if __name__ == "__main__":
    unittest.main()
