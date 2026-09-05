import json
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import wave

import numpy as np

from chatterbox_vllm.audio import (
    AudioQualityIssue,
    clear_reference_cache,
    find_generated_audio_issues,
    format_audio_quality_issues,
    limit_internal_pauses_wav,
    loudness_filter,
    normalized_reference_audio,
    prepared_reference_audio,
    normalize_speech_wav,
    reference_loudness_filter,
    wav_generated_audio_issues,
)
from chatterbox_vllm.silero_vad import NoSpeechRange


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


class ReferenceAudioNormalizationTests(unittest.TestCase):
    measurements = {
        "input_i": "-30.83",
        "input_tp": "-17.97",
        "input_lra": "1.70",
        "input_thresh": "-40.96",
        "target_offset": "0.24",
    }

    def test_uses_quiet_reference_targets_and_linear_second_pass(self):
        first_pass = reference_loudness_filter()
        second_pass = reference_loudness_filter(self.measurements)

        self.assertIn("aformat=channel_layouts=mono", first_pass)
        self.assertIn("loudnorm=I=-20:TP=-3:LRA=7", first_pass)
        self.assertIn("measured_I=-30.83", second_pass)
        self.assertIn("measured_TP=-17.97", second_pass)
        self.assertIn("linear=true", second_pass)

    def test_normalizes_a_temporary_copy_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quiet.mp3"
            source.write_bytes(b"original")
            calls = []

            def successful_run(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    stderr = "analysis\n" + json.dumps(self.measurements)
                    return subprocess.CompletedProcess(command, 0, "", stderr)
                Path(command[-1]).write_bytes(b"normalized")
                output = dict(
                    self.measurements,
                    output_i="-20.00",
                    output_tp="-4.13",
                    normalization_type="linear",
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "",
                    "normalization\n" + json.dumps(output),
                )

            with (
                patch(
                    "chatterbox_vllm.audio.subprocess.run",
                    side_effect=successful_run,
                ),
                patch("builtins.print") as printed,
                patch("chatterbox_vllm.audio._check_reference_duration"),
            ):
                with normalized_reference_audio(
                    source,
                    24000,
                    ffmpeg="ffmpeg",
                ) as normalized:
                    self.assertEqual(normalized.read_bytes(), b"normalized")
                    self.assertNotEqual(normalized, source)
                    self.assertIn("pcm_f32le", calls[1])
                    self.assertIn("24000", calls[1])

            messages = [str(call.args[0]) for call in printed.call_args_list]
            self.assertTrue(any("Input: -30.83 LUFS" in text for text in messages))
            self.assertTrue(any("Output: -20.00 LUFS" in text for text in messages))

            self.assertEqual(source.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).iterdir()), [source])

    def test_rejects_silent_reference_measurement(self):
        silent = dict(self.measurements, input_i="-inf")
        with self.assertRaisesRegex(RuntimeError, "silent or too quiet"):
            reference_loudness_filter(silent)


class ReferenceAudioPreparationTests(unittest.TestCase):
    def setUp(self):
        clear_reference_cache()

    def tearDown(self):
        clear_reference_cache()

    def test_enabled_denoising_precedes_normalization_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quiet.mp3"
            denoised = Path(directory) / "denoised.wav"
            normalized = Path(directory) / "normalized.wav"
            source.write_bytes(b"original")
            denoised.write_bytes(b"denoised")
            normalized.write_bytes(b"normalized")
            events = []

            def fake_denoise(path, output_path, *, ffmpeg=None):
                events.append(("denoise", Path(path), Path(output_path), ffmpeg))
                Path(output_path).write_bytes(b"denoised")
                return Path(output_path)

            @contextmanager
            def fake_normalize(path, sample_rate, *, ffmpeg=None):
                events.append(("normalize", Path(path), sample_rate))
                yield normalized

            with (
                patch(
                    "chatterbox_vllm.audio.denoise_reference_audio",
                    side_effect=fake_denoise,
                ),
                patch(
                    "chatterbox_vllm.audio.normalized_reference_audio",
                    side_effect=fake_normalize,
                ),
            ):
                with prepared_reference_audio(source, 24000, denoise=True) as result:
                    self.assertEqual(result.read_bytes(), normalized.read_bytes())

            self.assertEqual(
                events,
                [
                    (
                        "denoise",
                        source,
                        unittest.mock.ANY,
                        None,
                    ),
                    ("normalize", unittest.mock.ANY, 24000),
                ],
            )
            self.assertEqual(events[0][2], events[1][1])
            self.assertEqual(source.read_bytes(), b"original")

    def test_disabled_denoising_skips_denoiser_and_normalizes_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quiet.mp3"
            normalized = Path(directory) / "normalized.wav"
            source.write_bytes(b"original")
            normalized.write_bytes(b"normalized")

            @contextmanager
            def fake_normalize(path, sample_rate, *, ffmpeg=None):
                self.assertEqual(Path(path), source)
                self.assertEqual(sample_rate, 24000)
                yield normalized

            with (
                patch(
                    "chatterbox_vllm.audio.denoise_reference_audio",
                    side_effect=AssertionError("denoiser must remain disabled"),
                ),
                patch(
                    "chatterbox_vllm.audio.normalized_reference_audio",
                    side_effect=fake_normalize,
                ),
            ):
                with prepared_reference_audio(source, 24000, denoise=False) as result:
                    self.assertEqual(result.read_bytes(), normalized.read_bytes())

            self.assertEqual(source.read_bytes(), b"original")

    def test_denoiser_failure_falls_back_to_normalizing_the_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quiet.mp3"
            normalized = Path(directory) / "normalized.wav"
            source.write_bytes(b"original")
            normalized.write_bytes(b"normalized")
            printed = []

            def failed_denoise(path, output_path, *, ffmpeg=None):
                raise RuntimeError("RNNoise unavailable")

            @contextmanager
            def fake_normalize(path, sample_rate, *, ffmpeg=None):
                self.assertEqual(Path(path), source)
                yield normalized

            with (
                patch(
                    "chatterbox_vllm.audio.denoise_reference_audio",
                    side_effect=failed_denoise,
                ),
                patch(
                    "chatterbox_vllm.audio.normalized_reference_audio",
                    side_effect=fake_normalize,
                ),
                patch(
                    "builtins.print",
                    side_effect=lambda *args, **kwargs: printed.append(args[0]),
                ),
            ):
                with prepared_reference_audio(source, 24000, denoise=True) as result:
                    self.assertEqual(result.read_bytes(), normalized.read_bytes())

            self.assertEqual(source.read_bytes(), b"original")
            self.assertTrue(any("Reference denoise" in message for message in printed))


class GeneratedAudioQualityTests(unittest.TestCase):
    sample_rate = 24000

    def speech_like(self, seconds: float) -> np.ndarray:
        samples = round(seconds * self.sample_rate)
        time_axis = np.arange(samples, dtype=np.float32) / self.sample_rate
        return 0.12 * np.sin(2 * np.pi * 180 * time_axis)

    def test_excluding_vad_never_calls_the_detector(self):
        waveform = self.speech_like(1.0)
        with patch(
            "chatterbox_vllm.audio.default_silero_vad_detector."
            "find_loud_no_speech_ranges",
            side_effect=AssertionError("VAD must remain disabled"),
        ):
            self.assertEqual(
                find_generated_audio_issues(
                    waveform,
                    self.sample_rate,
                    include_vad=False,
                ),
                [],
            )

    def test_including_vad_maps_no_speech_ranges_to_audio_issues(self):
        waveform = self.speech_like(3.0)
        with patch(
            "chatterbox_vllm.audio.default_silero_vad_detector."
            "find_loud_no_speech_ranges",
            return_value=(NoSpeechRange(0.5, 2.0),),
        ) as detector:
            issues = find_generated_audio_issues(
                waveform,
                self.sample_rate,
                include_vad=True,
            )

        detector.assert_called_once()
        self.assertEqual(
            issues,
            [
                AudioQualityIssue(
                    "audible non-speech synthesis blob",
                    0.5,
                    2.0,
                )
            ],
        )

    def test_wav_scanner_includes_vad(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            write_pcm16_mono(
                source,
                (self.speech_like(2.0) * 32767).astype(np.int16),
                self.sample_rate,
            )
            with patch(
                "chatterbox_vllm.audio.default_silero_vad_detector."
                "find_loud_no_speech_ranges",
                return_value=(NoSpeechRange(0.25, 1.5),),
            ) as detector:
                issues = wav_generated_audio_issues(source, self.sample_rate)

        detector.assert_called_once()
        self.assertEqual(issues[0].kind, "audible non-speech synthesis blob")
        self.assertEqual((issues[0].start_seconds, issues[0].end_seconds), (0.25, 1.5))

    def test_detects_a_wholly_near_silent_waveform(self):
        waveform = np.zeros(3 * self.sample_rate, dtype=np.float32)

        issues = find_generated_audio_issues(waveform, self.sample_rate)

        self.assertEqual(
            issues,
            [AudioQualityIssue("near-silent generated audio", 0.0, 3.0)],
        )
        self.assertEqual(
            format_audio_quality_issues(issues),
            "near-silent generated audio at 0.00-3.00s",
        )

    def test_detects_a_wholly_near_silent_waveform_shorter_than_a_frame(self):
        waveform = np.zeros(round(0.1 * self.sample_rate), dtype=np.float32)

        issues = find_generated_audio_issues(waveform, self.sample_rate)

        self.assertEqual(
            issues,
            [AudioQualityIssue("near-silent generated audio", 0.0, 0.1)],
        )

    def test_detects_excessive_leading_and_trailing_near_silence(self):
        waveform = np.concatenate(
            [
                np.zeros(3 * self.sample_rate, dtype=np.float32),
                self.speech_like(2.0),
                np.zeros(4 * self.sample_rate, dtype=np.float32),
            ]
        )

        issues = find_generated_audio_issues(waveform, self.sample_rate)

        self.assertEqual(
            issues,
            [
                AudioQualityIssue("excessive leading near-silence", 0.0, 3.0),
                AudioQualityIssue("excessive trailing near-silence", 5.0, 9.0),
            ],
        )

    def test_detects_a_three_second_tail_after_a_non_frame_aligned_segment(self):
        waveform = np.concatenate(
            [self.speech_like(1.1), np.zeros(3 * self.sample_rate, dtype=np.float32)]
        )

        issues = find_generated_audio_issues(waveform, self.sample_rate)

        self.assertEqual(
            issues,
            [AudioQualityIssue("excessive trailing near-silence", 1.1, 4.1)],
        )

    def test_detects_a_three_second_prefix_when_frames_do_not_divide_it(self):
        sample_rate = 22049
        speech = 0.12 * np.sin(
            2 * np.pi * 180 * np.arange(sample_rate, dtype=np.float32) / sample_rate
        )
        waveform = np.concatenate(
            [np.zeros(3 * sample_rate, dtype=np.float32), speech]
        )

        issues = find_generated_audio_issues(waveform, sample_rate)

        self.assertEqual(
            issues,
            [AudioQualityIssue("excessive leading near-silence", 0.0, 3.0)],
        )

    def test_allows_short_leading_and_trailing_silence(self):
        waveform = np.concatenate(
            [
                np.zeros(round(0.75 * self.sample_rate), dtype=np.float32),
                self.speech_like(2.0),
                np.zeros(round(0.75 * self.sample_rate), dtype=np.float32),
            ]
        )

        self.assertEqual(find_generated_audio_issues(waveform, self.sample_rate), [])

    def test_wav_scanner_detects_excessive_trailing_near_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            waveform = np.concatenate(
                [self.speech_like(1.0), np.zeros(3 * self.sample_rate)]
            )
            write_pcm16_mono(
                source,
                (waveform * 32767).astype(np.int16),
                self.sample_rate,
            )
            with patch(
                "chatterbox_vllm.audio.default_silero_vad_detector."
                "find_loud_no_speech_ranges",
                return_value=(),
            ):
                issues = wav_generated_audio_issues(source, self.sample_rate)

        self.assertEqual(
            issues,
            [AudioQualityIssue("excessive trailing near-silence", 1.0, 4.0)],
        )

    def test_wav_scanner_rejects_a_truncated_pcm_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "chunk.wav"
            write_pcm16_mono(
                source,
                (self.speech_like(1.0) * 32767).astype(np.int16),
                self.sample_rate,
            )
            source.write_bytes(source.read_bytes()[:-1])

            with self.assertRaisesRegex(RuntimeError, "Could not read"):
                wav_generated_audio_issues(source, self.sample_rate)

    def test_detects_a_stable_digital_tone_with_normal_audio_after_it(self):
        frame = self.sample_rate // 4
        time_axis = np.arange(frame, dtype=np.float32) / self.sample_rate
        speech_like = np.concatenate(
            [
                0.2 * np.sin(2 * np.pi * frequency * time_axis)
                for frequency in (180, 230, 140, 310) * 3
            ]
        )
        bad_section = 0.2 * np.sin(
            2 * np.pi * 2100 * np.arange(2 * self.sample_rate) / self.sample_rate
        )
        waveform = np.concatenate([speech_like, bad_section, speech_like])

        issues = find_generated_audio_issues(waveform, self.sample_rate)
        self.assertTrue(any(issue.kind == "sustained synthetic tone" for issue in issues))
        self.assertIn("3.00-5.00s", format_audio_quality_issues(issues))

    def test_detects_a_broadband_digital_noise_blob_in_the_middle(self):
        rng = np.random.default_rng(42)
        time_axis = np.arange(2 * self.sample_rate, dtype=np.float32) / self.sample_rate
        speech_like = 0.2 * np.sin(2 * np.pi * 180 * time_axis)
        digital_noise = rng.normal(0, 0.12, round(1.5 * self.sample_rate))
        waveform = np.concatenate([speech_like, digital_noise, speech_like])

        issues = find_generated_audio_issues(waveform, self.sample_rate)

        self.assertTrue(any(issue.kind == "broadband digital noise" for issue in issues))
        self.assertIn("2.00-3.50s", format_audio_quality_issues(issues))

    def test_detects_a_low_frequency_synthesis_collapse(self):
        frame = self.sample_rate // 4
        time_axis = np.arange(frame, dtype=np.float32) / self.sample_rate
        speech_like = np.concatenate(
            [
                0.12 * np.sin(2 * np.pi * fundamental * time_axis)
                + 0.10 * np.sin(2 * np.pi * 2 * fundamental * time_axis)
                + 0.08 * np.sin(2 * np.pi * 3 * fundamental * time_axis)
                for fundamental in (95, 110, 130, 105) * 3
            ]
        )
        collapse_time = (
            np.arange(2 * self.sample_rate, dtype=np.float32)
            / self.sample_rate
        )
        collapse = (
            0.22 * np.sin(2 * np.pi * 60 * collapse_time)
            + 0.03 * np.sin(2 * np.pi * 180 * collapse_time)
        )
        waveform = np.concatenate([speech_like, collapse, speech_like])

        issues = find_generated_audio_issues(waveform, self.sample_rate)

        self.assertTrue(
            any(
                issue.kind == "low-frequency synthesis collapse"
                for issue in issues
            )
        )
        self.assertIn(
            "low-frequency synthesis collapse at 3.00-",
            format_audio_quality_issues(issues),
        )

    def test_allows_a_varied_speech_like_tail(self):
        frame = self.sample_rate // 4
        time_axis = np.arange(frame, dtype=np.float32) / self.sample_rate
        waveform = np.concatenate(
            [
                0.2 * np.sin(2 * np.pi * frequency * time_axis)
                for frequency in (130, 210, 170, 280, 150, 240, 190, 320, 160)
            ]
        )

        self.assertEqual(find_generated_audio_issues(waveform, self.sample_rate), [])


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
