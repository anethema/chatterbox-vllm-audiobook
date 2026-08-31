from contextlib import contextmanager
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

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
            def fake_preparation(path, sample_rate, *, denoise=False):
                seen.append((path, sample_rate, denoise))
                yield normalized

            with patch.object(
                gradio_tts_app,
                "prepared_reference_audio",
                side_effect=fake_preparation,
            ):
                preview = gradio_tts_app.prepare_reference_preview(str(source))

            self.assertEqual(Path(preview).read_bytes(), b"normalized")
            self.assertEqual(source.read_bytes(), b"original")
            self.assertEqual(seen, [(str(source), 24000, False)])
            self.assertIn("normalized-reference-", Path(preview).name)

    def test_upload_keeps_source_while_player_uses_denoised_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "quiet.mp3"
            prepared = Path(directory) / "prepared.wav"
            source.write_bytes(b"original")
            prepared.write_bytes(b"denoised and normalized")

            @contextmanager
            def fake_preparation(path, sample_rate, *, denoise=False):
                self.assertEqual(path, str(source))
                self.assertEqual(sample_rate, 24000)
                self.assertTrue(denoise)
                yield prepared

            with patch.object(
                gradio_tts_app,
                "prepared_reference_audio",
                side_effect=fake_preparation,
            ):
                preview, retained_source = (
                    gradio_tts_app.prepare_uploaded_reference(str(source), True)
                )

            self.assertEqual(Path(preview).read_bytes(), b"denoised and normalized")
            self.assertEqual(retained_source, str(source))
            self.assertEqual(source.read_bytes(), b"original")

    def test_clearing_reference_clears_player(self):
        self.assertIsNone(gradio_tts_app.prepare_reference_preview(None))

    def test_batch_slider_defaults_to_and_allows_64(self):
        self.assertEqual(gradio_tts_app.batch_size.value, 64)
        self.assertEqual(gradio_tts_app.batch_size.maximum, 64)

    def test_multilingual_audiobook_defaults(self):
        self.assertEqual(gradio_tts_app.max_chars.value, 200)
        self.assertEqual(gradio_tts_app.min_p.value, 0.05)
        self.assertEqual(gradio_tts_app.top_p.value, 1.0)
        self.assertEqual(gradio_tts_app.repetition_penalty.value, 1.2)
        self.assertFalse(gradio_tts_app.denoise_reference.value)


class AudioRecoveryTests(unittest.TestCase):
    sample_rate = 24000

    def setUp(self):
        self.vad_patch = patch(
            "chatterbox_vllm.audio.default_silero_vad_detector."
            "find_loud_no_speech_ranges",
            return_value=(),
        )
        self.vad_patch.start()
        self.addCleanup(self.vad_patch.stop)

    def good_audio(self):
        frame = self.sample_rate // 4
        time_axis = np.arange(frame, dtype=np.float32) / self.sample_rate
        samples = np.concatenate(
            [
                0.15 * np.sin(2 * np.pi * frequency * time_axis)
                for frequency in (140, 230, 170, 310) * 3
            ]
        ).astype(np.float32)
        return torch.from_numpy(samples).unsqueeze(0)

    def bad_audio(self):
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 0.12, round(1.5 * self.sample_rate)).astype(
            np.float32
        )
        good = self.good_audio().squeeze(0).numpy()
        return torch.from_numpy(np.concatenate([good, noise, good])).unsqueeze(0)

    def truncated_audio(self):
        # This is the exact duration in the reported "asked Vess." failure.
        return torch.zeros((1, round(0.04 * self.sample_rate)), dtype=torch.float32)

    def test_two_word_truncation_retries_the_entire_chunk(self):
        retry_calls = []

        recovery = gradio_tts_app._recover_generated_waveform(
            self.truncated_audio(),
            "asked Vess.",
            self.sample_rate,
            "chunk 029107",
            lambda: (retry_calls.append("retry") or self.good_audio()),
            lambda part: self.good_audio(),
        )

        self.assertEqual(retry_calls, ["retry"])
        self.assertTrue(recovery.detected_quality_issues)
        self.assertFalse(recovery.retained_with_warning)
        self.assertEqual(recovery.retained_quality_issues, ())
        self.assertEqual(recovery.waveform.shape, self.good_audio().shape)

    def test_repeated_invalid_single_sentence_output_uses_safe_silence(self):
        output = io.StringIO()
        with redirect_stdout(output):
            recovery = gradio_tts_app._recover_generated_waveform(
                self.truncated_audio(),
                "asked Vess.",
                self.sample_rate,
                "chunk 029107",
                self.truncated_audio,
                self.truncated_audio,
            )

        self.assertTrue(recovery.detected_quality_issues)
        self.assertTrue(recovery.retained_with_warning)
        self.assertEqual(recovery.retained_quality_issues, ())
        self.assertEqual(recovery.waveform.ndim, 2)
        self.assertEqual(recovery.waveform.shape[0], 1)
        self.assertGreater(recovery.waveform.shape[1], 0)
        self.assertTrue(torch.isfinite(recovery.waveform).all())
        self.assertEqual(torch.count_nonzero(recovery.waveform), 0)
        self.assertIn("short silent fallback", output.getvalue())

    def test_empty_and_nonfinite_outputs_cannot_escape_recovery(self):
        invalid_outputs = {
            "empty": torch.empty((1, 0), dtype=torch.float32),
            "nonfinite": torch.full((1, 1000), float("nan")),
        }
        for label, invalid_audio in invalid_outputs.items():
            with self.subTest(label=label), redirect_stdout(io.StringIO()):
                recovery = gradio_tts_app._recover_generated_waveform(
                    invalid_audio,
                    "asked Vess.",
                    self.sample_rate,
                    "chunk 029107",
                    lambda audio=invalid_audio: audio,
                    lambda part, audio=invalid_audio: audio,
                )

            self.assertTrue(recovery.detected_quality_issues)
            self.assertTrue(recovery.retained_with_warning)
            self.assertEqual(recovery.waveform.shape[0], 1)
            self.assertGreater(recovery.waveform.shape[1], 0)
            self.assertTrue(torch.isfinite(recovery.waveform).all())

    def test_invalid_split_outputs_are_retained_without_aborting(self):
        text = "First short sentence. Second short sentence. Third short sentence."
        part_calls = []

        def generate_part(part):
            part_calls.append(part)
            return self.truncated_audio()

        recovery = gradio_tts_app._recover_generated_waveform(
            self.truncated_audio(),
            text,
            self.sample_rate,
            "chunk 029108",
            self.truncated_audio,
            generate_part,
        )

        self.assertGreaterEqual(len(part_calls), 2)
        self.assertTrue(recovery.detected_quality_issues)
        self.assertTrue(recovery.retained_with_warning)
        self.assertEqual(recovery.waveform.ndim, 2)
        self.assertEqual(recovery.waveform.shape[0], 1)
        self.assertGreater(recovery.waveform.shape[1], 0)
        self.assertTrue(torch.isfinite(recovery.waveform).all())

    def test_whole_chunk_retry_can_recover(self):
        split_calls = []

        recovery = gradio_tts_app._recover_generated_waveform(
            self.bad_audio(),
            "A sentence that contains enough words for an ordinary speech test.",
            self.sample_rate,
            "chunk 000123",
            self.good_audio,
            lambda *args: split_calls.append(args),
        )

        self.assertEqual(split_calls, [])
        self.assertTrue(recovery.detected_quality_issues)
        self.assertEqual(recovery.retained_quality_issues, ())
        self.assertEqual(
            gradio_tts_app.find_generated_audio_issues(
                recovery.waveform, self.sample_rate
            ),
            [],
        )

    def test_clean_waveform_is_not_marked_as_repaired_or_retained(self):
        recovery = gradio_tts_app._recover_generated_waveform(
            self.good_audio(),
            "A clean sentence for the recovery-status test.",
            self.sample_rate,
            "chunk 000122",
            self.good_audio,
            lambda part: self.good_audio(),
        )

        self.assertFalse(recovery.detected_quality_issues)
        self.assertEqual(recovery.retained_quality_issues, ())

    def test_second_failure_splits_text_and_combines_clean_parts(self):
        text = (
            "If the most talented among us are preoccupied with maintaining the "
            "barrier and making life inside more pleasant, then what about the "
            "threats outside? They will only grow worse with time.” Brynn took "
            "Samuel’s hand."
        )
        part_calls = []

        def generate_part(part):
            part_calls.append(part)
            return self.good_audio()

        recovery = gradio_tts_app._recover_generated_waveform(
            self.bad_audio(),
            text,
            self.sample_rate,
            "chunk 000028",
            self.bad_audio,
            generate_part,
        )

        self.assertEqual(len(part_calls), 2)
        self.assertEqual(
            gradio_tts_app.find_generated_audio_issues(
                recovery.waveform, self.sample_rate
            ),
            [],
        )
        expected_samples = (
            2 * self.good_audio().shape[1]
            + round(self.sample_rate * gradio_tts_app.SPLIT_JOIN_SILENCE_SECONDS)
        )
        self.assertEqual(recovery.waveform.shape, (1, expected_samples))

    def test_failed_multi_sentence_part_recursively_splits_to_sentences(self):
        text = (
            "First sentence has several ordinary words. "
            "Second sentence also has several ordinary words. "
            "Third sentence finishes the passage normally."
        )
        calls = []

        def generate_part(part):
            calls.append(part)
            if len(gradio_tts_app.split_sentences(part)) > 1:
                return self.bad_audio()
            return self.good_audio()

        recovery = gradio_tts_app._recover_generated_waveform(
            self.bad_audio(),
            text,
            self.sample_rate,
            "chunk 000456",
            self.bad_audio,
            generate_part,
        )

        self.assertGreater(len(calls), 2)
        self.assertEqual(
            gradio_tts_app.find_generated_audio_issues(
                recovery.waveform, self.sample_rate
            ),
            [],
        )

    def test_single_sentence_failure_is_included_instead_of_stopping(self):
        recovery = gradio_tts_app._recover_generated_waveform(
            self.bad_audio(),
            "One sentence that repeatedly produces a digital audio artifact.",
            self.sample_rate,
            "chunk 000789",
            self.bad_audio,
            lambda part: self.bad_audio(),
        )

        self.assertTrue(
            gradio_tts_app.find_generated_audio_issues(
                recovery.waveform, self.sample_rate
            )
        )
        self.assertTrue(recovery.detected_quality_issues)
        self.assertTrue(recovery.retained_quality_issues)

    def test_retry_arguments_always_use_a_fresh_seed(self):
        used_seeds = set()
        with patch.object(
            gradio_tts_app.secrets,
            "randbelow",
            side_effect=[122, 123, 123, 124],
        ):
            first = gradio_tts_app._retry_generation_args(
                {"seed": 123}, used_seeds
            )
            second = gradio_tts_app._retry_generation_args(
                {"seed": None}, used_seeds
            )

        self.assertEqual(first["seed"], 124)
        self.assertEqual(second["seed"], 125)
        self.assertEqual(used_seeds, {123, 124, 125})

    def test_retained_split_noise_does_not_print_a_false_success(self):
        output = io.StringIO()
        with redirect_stdout(output):
            recovery = gradio_tts_app._recover_generated_waveform(
                self.bad_audio(),
                "The first sentence fails. The second sentence also fails.",
                self.sample_rate,
                "chunk 000999",
                self.bad_audio,
                lambda part: self.bad_audio(),
            )
            waveform = recovery.waveform

        self.assertTrue(
            gradio_tts_app.find_generated_audio_issues(
                waveform, self.sample_rate
            )
        )
        self.assertIn("included anyway so generation can continue", output.getvalue())
        self.assertNotIn("combined waveform passed the full scan", output.getvalue())
        self.assertTrue(recovery.detected_quality_issues)
        self.assertTrue(recovery.retained_quality_issues)

    def test_quality_warnings_are_red_in_an_interactive_terminal(self):
        class InteractiveBuffer(io.StringIO):
            def isatty(self):
                return True

        output = InteractiveBuffer()
        with redirect_stdout(output):
            gradio_tts_app._quality_log("damaged audio", color="red")

        self.assertIn("\033[1;31mdamaged audio\033[0m", output.getvalue())

    def test_clean_batch_summary_is_one_green_ok_line(self):
        class InteractiveBuffer(io.StringIO):
            def isatty(self):
                return True

        output = InteractiveBuffer()
        with redirect_stdout(output):
            gradio_tts_app._log_batch_quality_summary(64, 64, [])

        self.assertEqual(output.getvalue().count("\n"), 1)
        self.assertIn(
            "\033[1;32m[Audio quality scan] Batch 000064-000127: "
            "64/64 OK\033[0m",
            output.getvalue(),
        )

    def test_batch_summary_is_red_when_noise_is_retained(self):
        class InteractiveBuffer(io.StringIO):
            def isatty(self):
                return True

        output = InteractiveBuffer()
        with redirect_stdout(output):
            gradio_tts_app._log_batch_quality_summary(0, 4, [1, 3])

        self.assertIn("\033[1;31m", output.getvalue())
        self.assertIn("2/4 OK", output.getvalue())
        self.assertIn("000001, 000003", output.getvalue())

    def test_project_summary_is_green_when_all_bad_chunks_were_fixed(self):
        class InteractiveBuffer(io.StringIO):
            def isatty(self):
                return True

        output = InteractiveBuffer()
        with redirect_stdout(output):
            gradio_tts_app._log_project_quality_summary(
                128,
                {25, 27},
                {25, 27},
                set(),
            )

        self.assertIn("\033[1;32m", output.getvalue())
        self.assertIn("bad chunks detected: 2", output.getvalue())
        self.assertIn("fixed: 2", output.getvalue())
        self.assertIn("retained with warnings: 0", output.getvalue())

    def test_project_summary_is_red_and_lists_retained_chunks(self):
        class InteractiveBuffer(io.StringIO):
            def isatty(self):
                return True

        output = InteractiveBuffer()
        with redirect_stdout(output):
            gradio_tts_app._log_project_quality_summary(
                128,
                {25, 27},
                {25},
                {27},
            )

        self.assertIn("\033[1;31m", output.getvalue())
        self.assertIn("fixed: 1", output.getvalue())
        self.assertIn("retained with warnings: 1 (000027)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
