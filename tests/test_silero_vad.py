import unittest

import torch

from chatterbox_vllm.silero_vad import (
    MIN_NO_SPEECH_RMS_DBFS,
    NoSpeechRange,
    SILERO_SAMPLE_RATE,
    SileroVadDetector,
    find_loud_no_speech_ranges,
)


def samples(seconds: float, amplitude: float) -> torch.Tensor:
    return torch.full((round(seconds * SILERO_SAMPLE_RATE),), amplitude)


class SileroVadGapTests(unittest.TestCase):
    def test_detects_an_internal_loud_no_speech_gap(self):
        waveform = torch.cat([samples(1, 0.1), samples(2, 0.2), samples(1, 0.1)])

        ranges = find_loud_no_speech_ranges(
            waveform,
            [{"start": 0, "end": 16_000}, {"start": 48_000, "end": 64_000}],
        )

        self.assertEqual(ranges, (NoSpeechRange(1.0, 3.0),))

    def test_ignores_a_quiet_internal_pause(self):
        waveform = torch.cat([samples(1, 0.1), samples(2, 0.001), samples(1, 0.1)])

        ranges = find_loud_no_speech_ranges(
            waveform,
            [{"start": 0, "end": 16_000}, {"start": 48_000, "end": 64_000}],
        )

        self.assertEqual(ranges, ())

    def test_ignores_leading_and_trailing_no_speech(self):
        waveform = samples(3, 0.2)

        ranges = find_loud_no_speech_ranges(
            waveform,
            [{"start": 16_000, "end": 32_000}],
        )

        self.assertEqual(ranges, ())

    def test_detects_a_whole_noisy_file_when_no_speech_exists(self):
        ranges = find_loud_no_speech_ranges(samples(2, 0.2), [])

        self.assertEqual(ranges, (NoSpeechRange(0.0, 2.0),))

    def test_accepts_exact_duration_and_level_boundaries(self):
        exact_amplitude = 10 ** (MIN_NO_SPEECH_RMS_DBFS / 20)
        self.assertEqual(
            find_loud_no_speech_ranges(samples(1, exact_amplitude), []),
            (NoSpeechRange(0.0, 1.0),),
        )
        self.assertEqual(
            find_loud_no_speech_ranges(samples(1, exact_amplitude * 0.99), []),
            (),
        )

    def test_injected_detector_resamples_without_loading_silero(self):
        seen = {}

        class FakeModel:
            def __init__(self):
                self.resets = 0

            def reset_states(self):
                self.resets += 1

        model = FakeModel()
        detector = SileroVadDetector(
            model_loader=lambda: model,
            speech_timestamp_getter=lambda waveform, loaded_model, sample_rate: (
                seen.update(
                    samples=waveform.numel(),
                    model=loaded_model,
                    sample_rate=sample_rate,
                )
                or []
            ),
        )

        ranges = detector.find_loud_no_speech_ranges(
            torch.full((8_000,), 0.2),
            8_000,
        )

        self.assertEqual(seen, {"samples": 16_000, "model": model, "sample_rate": 16_000})
        self.assertEqual(model.resets, 2)
        self.assertEqual(ranges, (NoSpeechRange(0.0, 1.0),))


if __name__ == "__main__":
    unittest.main()
