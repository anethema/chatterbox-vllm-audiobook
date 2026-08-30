from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio_tts_app


class GradioReferencePreviewTests(unittest.TestCase):
    def tearDown(self):
        directory = gradio_tts_app.reference_preview_directory
        if directory is not None:
            directory.cleanup()
        gradio_tts_app.reference_preview_directory = None

    def test_replaces_player_source_with_normalized_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quiet.mp3"
            normalized = Path(directory) / "normalized.wav"
            source.write_bytes(b"original")
            normalized.write_bytes(b"normalized")
            seen = []

            @contextmanager
            def fake_normalization(path, sample_rate):
                seen.append((path, sample_rate))
                yield normalized

            with patch.object(
                gradio_tts_app,
                "normalized_reference_audio",
                side_effect=fake_normalization,
            ):
                preview = gradio_tts_app.prepare_reference_preview(str(source))

            self.assertEqual(Path(preview).read_bytes(), b"normalized")
            self.assertEqual(source.read_bytes(), b"original")
            self.assertEqual(seen, [(str(source), 24000)])
            self.assertIn("normalized-reference-", Path(preview).name)

    def test_clearing_reference_clears_player(self):
        self.assertIsNone(gradio_tts_app.prepare_reference_preview(None))

    def test_batch_slider_defaults_to_and_allows_64(self):
        self.assertEqual(gradio_tts_app.batch_size.value, 64)
        self.assertEqual(gradio_tts_app.batch_size.maximum, 64)


if __name__ == "__main__":
    unittest.main()
