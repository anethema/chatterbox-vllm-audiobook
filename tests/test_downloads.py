import tempfile
import unittest
from pathlib import Path

import gradio as gr
from gradio import processing_utils

from chatterbox_vllm.downloads import register_completed_audiobook


class StaticPathRecorder:
    def __init__(self) -> None:
        self.calls: list[list[Path]] = []

    def __call__(self, *, paths: list[Path]) -> None:
        self.calls.append(paths)


class CompletedAudiobookDownloadTests(unittest.TestCase):
    def test_registers_only_the_exact_completed_m4b(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "audiobook_outputs"
            project = output_root / "Book-123"
            project.mkdir(parents=True)
            completed = project / "audiobook.m4b"
            completed.write_bytes(b"verified-m4b")
            (project / "source.epub").write_bytes(b"private-book")
            recorder = StaticPathRecorder()

            result = register_completed_audiobook(
                completed,
                output_root,
                recorder,
            )

        self.assertEqual(result, completed.resolve())
        self.assertEqual(recorder.calls, [[completed.resolve()]])

    def test_gradio_keeps_registered_output_at_its_original_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "audiobook_outputs"
            project = output_root / "Book-123"
            project.mkdir(parents=True)
            completed = project / "audiobook.m4b"
            completed.write_bytes(b"verified-m4b")

            registered = register_completed_audiobook(
                completed,
                output_root,
                gr.set_static_paths,
            )
            component = gr.Audio()
            payload = component.postprocess(str(registered))
            cached_payload = processing_utils.move_files_to_cache(
                payload,
                component,
                postprocess=True,
            )

        self.assertEqual(Path(cached_payload["path"]), completed.resolve())
        self.assertNotIn("/tmp/gradio/", cached_payload["path"])

    def test_missing_file_is_not_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "audiobook_outputs"
            output_root.mkdir()
            recorder = StaticPathRecorder()

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                register_completed_audiobook(
                    output_root / "Book-123" / "audiobook.m4b",
                    output_root,
                    recorder,
                )

        self.assertEqual(recorder.calls, [])

    def test_non_m4b_file_is_not_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "audiobook_outputs"
            output_root.mkdir()
            unexpected = output_root / "source.epub"
            unexpected.write_bytes(b"private-book")
            recorder = StaticPathRecorder()

            with self.assertRaisesRegex(ValueError, "must be an M4B"):
                register_completed_audiobook(
                    unexpected,
                    output_root,
                    recorder,
                )

        self.assertEqual(recorder.calls, [])

    def test_file_outside_output_root_is_not_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "audiobook_outputs"
            output_root.mkdir()
            outside = root / "private.m4b"
            outside.write_bytes(b"not-an-output")
            recorder = StaticPathRecorder()

            with self.assertRaisesRegex(ValueError, "inside the output root"):
                register_completed_audiobook(
                    outside,
                    output_root,
                    recorder,
                )

        self.assertEqual(recorder.calls, [])


if __name__ == "__main__":
    unittest.main()
