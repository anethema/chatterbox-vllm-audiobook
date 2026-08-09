import unittest

from chatterbox_vllm.progress import GenerationControl, estimate_progress, format_duration


class ProgressTests(unittest.TestCase):
    def test_realtime_speed_is_audio_duration_divided_by_wall_time(self):
        estimate = estimate_progress(
            generated_audio_seconds=60,
            elapsed_seconds=30,
            completed_characters=500,
            total_characters=1000,
        )
        self.assertEqual(estimate.realtime_speed, 2.0)
        self.assertEqual(estimate.eta_seconds, 30.0)

    def test_eta_scales_by_remaining_text(self):
        estimate = estimate_progress(30, 20, completed_characters=250, total_characters=1000)
        self.assertEqual(estimate.realtime_speed, 1.5)
        self.assertEqual(estimate.eta_seconds, 60.0)

    def test_duration_formatting(self):
        self.assertEqual(format_duration(9.6), "10s")
        self.assertEqual(format_duration(125), "2m 05s")
        self.assertEqual(format_duration(3725), "1h 02m")

    def test_generation_control_only_stops_an_active_job(self):
        control = GenerationControl()
        self.assertFalse(control.request_stop())
        control.begin()
        self.assertTrue(control.request_stop())
        self.assertTrue(control.stop_requested())
        control.finish()
        self.assertFalse(control.stop_requested())
        self.assertFalse(control.request_stop())


if __name__ == "__main__":
    unittest.main()
