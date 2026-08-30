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


class AudioRecoveryTests(unittest.TestCase):
    sample_rate = 24000

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

    def test_whole_chunk_retry_can_recover(self):
        split_calls = []

        waveform = gradio_tts_app._recover_generated_waveform(
            self.bad_audio(),
            "A sentence that contains enough words for an ordinary speech test.",
            self.sample_rate,
            "chunk 000123",
            self.good_audio,
            lambda *args: split_calls.append(args),
        )

        self.assertEqual(split_calls, [])
        self.assertEqual(
            gradio_tts_app.find_generated_audio_issues(waveform, self.sample_rate),
            [],
        )

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

        waveform = gradio_tts_app._recover_generated_waveform(
            self.bad_audio(),
            text,
            self.sample_rate,
            "chunk 000028",
            self.bad_audio,
            generate_part,
        )

        self.assertEqual(len(part_calls), 2)
        self.assertEqual(
            gradio_tts_app.find_generated_audio_issues(waveform, self.sample_rate),
            [],
        )
        expected_samples = (
            2 * self.good_audio().shape[1]
            + round(self.sample_rate * gradio_tts_app.SPLIT_JOIN_SILENCE_SECONDS)
        )
        self.assertEqual(waveform.shape, (1, expected_samples))

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

        waveform = gradio_tts_app._recover_generated_waveform(
            self.bad_audio(),
            text,
            self.sample_rate,
            "chunk 000456",
            self.bad_audio,
            generate_part,
        )

        self.assertGreater(len(calls), 2)
        self.assertEqual(
            gradio_tts_app.find_generated_audio_issues(waveform, self.sample_rate),
            [],
        )

    def test_single_sentence_failure_is_included_instead_of_stopping(self):
        waveform = gradio_tts_app._recover_generated_waveform(
            self.bad_audio(),
            "One sentence that repeatedly produces a digital audio artifact.",
            self.sample_rate,
            "chunk 000789",
            self.bad_audio,
            lambda part: self.bad_audio(),
        )

        self.assertTrue(
            gradio_tts_app.find_generated_audio_issues(waveform, self.sample_rate)
        )

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
            waveform = gradio_tts_app._recover_generated_waveform(
                self.bad_audio(),
                "The first sentence fails. The second sentence also fails.",
                self.sample_rate,
                "chunk 000999",
                self.bad_audio,
                lambda part: self.bad_audio(),
            )

        self.assertTrue(
            gradio_tts_app.find_generated_audio_issues(
                waveform, self.sample_rate
            )
        )
        self.assertIn("included anyway so generation can continue", output.getvalue())
        self.assertNotIn("combined waveform passed the full scan", output.getvalue())

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
