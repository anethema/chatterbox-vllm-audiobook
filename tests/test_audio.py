import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatterbox_vllm.audio import loudness_filter, normalize_speech_wav


class AudioNormalizationTests(unittest.TestCase):
    def test_targets_minus_18_lufs_with_true_peak_ceiling(self):
        self.assertEqual(loudness_filter(), "loudnorm=I=-18:TP=-2:LRA=7")

    def test_replaces_source_only_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            source.write_bytes(b"original")
            seen_command = None

            def successful_run(command, **kwargs):
                nonlocal seen_command
                seen_command = command
                Path(command[-1]).write_bytes(b"normalized")
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("chatterbox_vllm.audio.subprocess.run", side_effect=successful_run):
                result = normalize_speech_wav(source, 24000, ffmpeg="ffmpeg")

            self.assertEqual(result, source)
            self.assertEqual(source.read_bytes(), b"normalized")
            self.assertIn("loudnorm=I=-18:TP=-2:LRA=7", seen_command)
            self.assertIn("24000", seen_command)
            self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_failed_normalization_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            source.write_bytes(b"original")

            failure = subprocess.CompletedProcess(
                ["ffmpeg"], 1, "", "filter failed"
            )
            with patch("chatterbox_vllm.audio.subprocess.run", return_value=failure):
                with self.assertRaisesRegex(RuntimeError, "filter failed"):
                    normalize_speech_wav(source, 24000, ffmpeg="ffmpeg")

            self.assertEqual(source.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).iterdir()), [source])


if __name__ == "__main__":
    unittest.main()
