import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import wave

import numpy as np

from chatterbox_vllm.audio import (
    limit_internal_pauses_wav,
    loudness_filter,
    normalize_speech_wav,
)


def write_pcm16_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def read_pcm16_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").copy()


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


class InternalPauseLimitingTests(unittest.TestCase):
    sample_rate = 1000

    def silence(self, seconds):
        return np.zeros(round(self.sample_rate * seconds), dtype=np.int16)

    def speech(self, seconds):
        return np.full(round(self.sample_rate * seconds), 8000, dtype=np.int16)

    def test_caps_only_silence_surrounded_by_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            samples = np.concatenate(
                [
                    self.silence(0.8),
                    self.speech(0.25),
                    self.silence(1.2),
                    self.speech(0.25),
                    self.silence(0.9),
                ]
            )
            write_pcm16_mono(source, samples, self.sample_rate)

            result = limit_internal_pauses_wav(
                source,
                self.sample_rate,
                maximum_seconds=0.5,
            )

            processed = read_pcm16_mono(source)
            active = np.flatnonzero(processed)
            first_speech_end = 800 + 250
            second_speech_start = int(active[250])
            internal_pause = (second_speech_start - first_speech_end) / self.sample_rate
            self.assertEqual(result, source)
            self.assertEqual(int(active[0]), 800)
            self.assertEqual(len(processed) - 1 - int(active[-1]), 900)
            self.assertGreaterEqual(internal_pause, 0.48)
            self.assertLessEqual(internal_pause, 0.5)
            self.assertEqual(np.count_nonzero(processed), 500)
            self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_leaves_short_internal_and_long_edge_silence_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            samples = np.concatenate(
                [
                    self.silence(1.1),
                    self.speech(0.2),
                    self.silence(0.5),
                    self.speech(0.2),
                    self.silence(1.3),
                ]
            )
            write_pcm16_mono(source, samples, self.sample_rate)
            original = source.read_bytes()

            limit_internal_pauses_wav(source, self.sample_rate)

            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_failed_atomic_replacement_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            samples = np.concatenate(
                [self.speech(0.2), self.silence(1.0), self.speech(0.2)]
            )
            write_pcm16_mono(source, samples, self.sample_rate)
            original = source.read_bytes()

            with patch(
                "chatterbox_vllm.audio.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    limit_internal_pauses_wav(source, self.sample_rate)

            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_rejects_a_nonpositive_pause_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            write_pcm16_mono(source, self.speech(0.2), self.sample_rate)

            with self.assertRaisesRegex(ValueError, "must be positive"):
                limit_internal_pauses_wav(
                    source,
                    self.sample_rate,
                    maximum_seconds=0,
                )


if __name__ == "__main__":
    unittest.main()
