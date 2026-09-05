"""Regression checks for scan avoidance and durable save bookkeeping."""
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

import torch

import gradio_tts_app as app


class WaveformValidationTests(unittest.TestCase):
    def test_best_effort_skips_unused_quality_scan(self):
        waveform = torch.zeros(1, 24_000)
        with patch.object(app, "find_generated_audio_issues") as scan:
            result = app._waveform_for_save(
                waveform, "A short sentence.", 24_000, allow_quality_issues=True,
            )
        scan.assert_not_called()
        torch.testing.assert_close(result, waveform)

    def test_best_effort_still_rejects_invalid_audio(self):
        cases = [
            torch.empty(1, 0),
            torch.zeros(2, 24_000),
            torch.full((1, 24_000), float("nan")),
            torch.full((1, 24_000), float("inf")),
            torch.zeros(1, 100),
        ]
        with patch.object(app, "find_generated_audio_issues") as scan:
            for waveform in cases:
                with self.subTest(shape=waveform.shape):
                    with self.assertRaises(app.GeneratedAudioValidationError):
                        app._waveform_for_save(
                            waveform, "A short sentence.", 24_000,
                            allow_quality_issues=True,
                        )
        scan.assert_not_called()

    def test_normal_validation_still_rejects_quality_findings(self):
        issue = app.AudioQualityIssue("near_silence", 0.0, 1.0)
        with patch.object(app, "find_generated_audio_issues", return_value=[issue]) as scan:
            with self.assertRaises(app.GeneratedAudioQualityError) as raised:
                app._waveform_for_save(torch.zeros(1, 24_000), "Hello.", 24_000)
        scan.assert_called_once()
        self.assertTrue(scan.call_args.kwargs["include_vad"])
        self.assertEqual(raised.exception.issues, (issue,))


class DurableResultsTests(unittest.TestCase):
    def test_repairs_and_out_of_order_completions_preserve_durable_prefix(self):
        pending = set()
        verified = {}
        observed = []

        def collect(indices, durable):
            tasks = Mock()
            tasks.take_results.return_value = [
                app.SavedChunkQuality(Path(f"{index:06d}.wav"), True)
                for index in indices
            ]
            with patch.object(app, "wav_file_identity", return_value={"size": 48_044, "mtime_ns": 1}):
                return app._record_durable_results(
                    tasks, pending, durable, verified, on_result=observed.append,
                )[0]

        # Chunk 1 is a repaired old file; chunk 4 finishes before chunks 2 and 3.
        self.assertEqual(collect([1, 4], 2), 2)
        self.assertEqual(pending, {4})
        self.assertEqual(collect([2], 2), 3)
        self.assertEqual(pending, {4})
        self.assertEqual(collect([3], 3), 5)
        self.assertEqual(pending, set())
        self.assertEqual(set(verified), {1, 2, 3, 4})
        self.assertEqual(len(observed), 4)


if __name__ == "__main__":
    unittest.main()
