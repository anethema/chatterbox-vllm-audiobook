from contextlib import contextmanager
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

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

    def test_monitored_resume_prefers_selected_project(self):
        from chatterbox_vllm.job_status import JobStatusStore

        with tempfile.TemporaryDirectory() as directory:
            old_status = gradio_tts_app.job_status
            try:
                gradio_tts_app.job_status = JobStatusStore(directory)
                gradio_tts_app.job_status.try_start(project_id="monitored-project")
                gradio_tts_app.job_status.finish("stopped", "ready")
                self.assertEqual(
                    gradio_tts_app._monitored_resume_project_name("selected-project"),
                    "selected-project",
                )
            finally:
                gradio_tts_app.job_status = old_status

    def test_monitored_resume_uses_only_terminal_resumable_state(self):
        from chatterbox_vllm.job_status import JobStatusStore

        old_status = gradio_tts_app.job_status
        try:
            for state in ("stopped", "interrupted", "failed"):
                with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                    store = JobStatusStore(directory)
                    store.try_start(project_id="monitored-project")
                    store.finish(state, "ready")
                    gradio_tts_app.job_status = store
                    self.assertEqual(
                        gradio_tts_app._monitored_resume_project_name(None),
                        "monitored-project",
                    )
            for state in ("running", "stopping", "completed", "idle"):
                with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                    store = JobStatusStore(directory)
                    if state != "idle":
                        store.try_start(project_id="monitored-project")
                    if state == "stopping":
                        store.request_stop()
                    elif state == "completed":
                        store.finish("completed", "done")
                    gradio_tts_app.job_status = store
                    self.assertIsNone(gradio_tts_app._monitored_resume_project_name(None))
        finally:
            gradio_tts_app.job_status = old_status

    def test_missing_inputs_release_job_control_and_publish_failure(self):
        from chatterbox_vllm.job_status import JobStatusStore
        from chatterbox_vllm.progress import GenerationControl

        with tempfile.TemporaryDirectory() as directory:
            old_status = gradio_tts_app.job_status
            old_control = gradio_tts_app.generation_control
            try:
                gradio_tts_app.job_status = JobStatusStore(directory)
                gradio_tts_app.generation_control = GenerationControl()
                output, message = gradio_tts_app.generate_epub_audiobook(
                    None, None, 0.5, 0.5, 0.8, 0, 15, 0.05, 1.0, 1.2,
                    200, 64, False, None, progress=Mock(),
                )

                self.assertIsNone(output)
                self.assertIn("Upload the original EPUB", message)
                self.assertEqual(gradio_tts_app.job_status.snapshot().state, "failed")
                self.assertFalse(gradio_tts_app.generation_control.request_stop())
            finally:
                gradio_tts_app.job_status = old_status
                gradio_tts_app.generation_control = old_control

    def test_job_monitor_poll_renders_progress_and_control_state(self):
        from chatterbox_vllm.job_status import JobStatusStore
        from chatterbox_vllm.progress import GenerationControl

        with tempfile.TemporaryDirectory() as directory:
            old_status = gradio_tts_app.job_status
            old_control = gradio_tts_app.generation_control
            try:
                gradio_tts_app.job_status = JobStatusStore(directory)
                gradio_tts_app.generation_control = GenerationControl()
                gradio_tts_app.job_status.try_start(project_id="resume-token")
                gradio_tts_app.job_status.update(
                    fraction=0.25, completed_chunks=25, total_chunks=100,
                )
                monitor, stop, generate, resume = gradio_tts_app._job_monitor_render()
                self.assertIn("25/100 chunks", monitor)
                self.assertIn('style="width: 25.0%"', monitor)
                self.assertEqual(gradio_tts_app.job_monitor.elem_id, "active-job-monitor")
                self.assertFalse(hasattr(gradio_tts_app, "job_monitor_progress"))
                self.assertTrue(stop.interactive)
                self.assertFalse(generate.interactive)
                self.assertFalse(resume.interactive)

                gradio_tts_app.generation_control.begin()
                message = gradio_tts_app.request_generation_stop()
                self.assertIn("Stop requested", message)
                self.assertEqual(gradio_tts_app.job_status.snapshot().state, "stopping")
            finally:
                gradio_tts_app.job_status = old_status
                gradio_tts_app.generation_control = old_control


class ResumeScanProgressTests(unittest.TestCase):
    def test_progress_message_includes_count_percentage_and_eta(self):
        message = gradio_tts_app._resume_scan_progress_message(25, 100, 10)

        self.assertEqual(
            message,
            "[Audio quality scan] Scanning existing chunks: 25/100 (25.0%) "
            "— ETA 30s",
        )

    def test_initial_progress_uses_calculating_eta(self):
        message = gradio_tts_app._resume_scan_progress_message(0, 29_056, 0)

        self.assertIn("0/29,056 (0.0%)", message)
        self.assertTrue(message.endswith("ETA calculating…"))

    def test_fully_cached_scan_has_no_pending_eta(self):
        message = gradio_tts_app._resume_scan_progress_message(
            0, 0, 0, cached_verified_chunks=29_056,
        )

        self.assertIn("0/0 (100.0%) — ETA 0s", message)
        self.assertTrue(message.endswith("skipped 29,056 cached verified"))

    def test_reporting_updates_gradio_without_requiring_stdout(self):
        progress = Mock()
        with patch.object(gradio_tts_app.time, "perf_counter", return_value=18):
            with patch.object(gradio_tts_app, "_quality_log") as quality_log:
                gradio_tts_app._report_resume_scan_progress(
                    progress, 40, 100, started_at=10, log_stdout=False,
                )

        progress.assert_called_once_with(
            0.4,
            desc=(
                "[Audio quality scan] Scanning existing chunks: 40/100 (40.0%) "
                "— ETA 12s"
            ),
        )
        quality_log.assert_not_called()


class QualityScanCheckpointRecordingTests(unittest.TestCase):
    class CompletedTasks:
        def __init__(self, results):
            self.results = list(results)

        def check(self):
            return None

        def take_results(self):
            results = self.results
            self.results = []
            return results

    @staticmethod
    def _write_wav(path: Path) -> None:
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24_000)
            audio.writeframes(b"\0\0" * 100)

    def test_only_verified_final_background_output_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            chunk_path = Path(directory) / "000000.wav"
            self._write_wav(chunk_path)
            verified = {}
            durable, changes = gradio_tts_app._record_durable_results(
                self.CompletedTasks(
                    [gradio_tts_app.SavedChunkQuality(chunk_path, True)]
                ),
                set(),
                0,
                verified,
            )

            self.assertEqual((durable, changes), (1, 1))
            self.assertIn(0, verified)

            durable, changes = gradio_tts_app._record_durable_results(
                self.CompletedTasks(
                    [gradio_tts_app.SavedChunkQuality(chunk_path, False)]
                ),
                set(),
                0,
                verified,
            )

        self.assertEqual((durable, changes), (1, 1))
        self.assertEqual(verified, {})


class ChunkPersistenceTests(unittest.TestCase):
    sample_rate = 24_000

    def test_minimum_duration_scales_with_word_count(self):
        text = "one two three four five six seven eight nine ten"
        too_short = torch.zeros((1, round(0.25 * self.sample_rate)))
        with self.assertRaises(gradio_tts_app.GeneratedAudioValidationError):
            gradio_tts_app._waveform_for_save(
                too_short,
                text,
                self.sample_rate,
                allow_quality_issues=True,
            )

        adequate = torch.zeros((1, round(0.45 * self.sample_rate)))
        saved = gradio_tts_app._waveform_for_save(
            adequate,
            text,
            self.sample_rate,
            allow_quality_issues=True,
        )
        self.assertEqual(saved.shape, adequate.shape)

    def test_failed_transform_keeps_previous_chunk_and_removes_staging_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000000.wav"
            path.write_bytes(b"previous")

            def fake_save(temporary, *_args, **_kwargs):
                Path(temporary).write_bytes(b"raw")

            with patch.object(gradio_tts_app.ta, "save", side_effect=fake_save), patch.object(
                gradio_tts_app,
                "normalize_speech_wav",
                side_effect=RuntimeError("normalization failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "normalization failed"):
                    gradio_tts_app._save_and_normalize_chunk(
                        path,
                        torch.zeros((1, self.sample_rate)),
                        self.sample_rate,
                        "ffmpeg",
                        None,
                    )

            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_staged_chunk_replaces_final_only_after_all_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000000.wav"
            path.write_bytes(b"previous")
            staged_paths = []

            def fake_save(temporary, *_args, **_kwargs):
                staged = Path(temporary)
                staged_paths.append(staged)
                staged.write_bytes(b"raw")

            def fake_normalize(temporary, *_args, **_kwargs):
                self.assertEqual(path.read_bytes(), b"previous")
                with Path(temporary).open("ab") as audio:
                    audio.write(b"-normalized")

            def fake_limit(temporary, *_args, **_kwargs):
                self.assertEqual(path.read_bytes(), b"previous")
                with Path(temporary).open("ab") as audio:
                    audio.write(b"-paused")

            with patch.object(gradio_tts_app.ta, "save", side_effect=fake_save), patch.object(
                gradio_tts_app, "normalize_speech_wav", side_effect=fake_normalize
            ), patch.object(
                gradio_tts_app, "limit_internal_pauses_wav", side_effect=fake_limit
            ):
                gradio_tts_app._save_and_normalize_chunk(
                    path,
                    torch.zeros((1, self.sample_rate)),
                    self.sample_rate,
                    "ffmpeg",
                    0.5,
                )

            self.assertEqual(path.read_bytes(), b"raw-normalized-paused")
            self.assertEqual(len(staged_paths), 1)
            self.assertEqual(staged_paths[0].parent, path.parent)
            self.assertFalse(staged_paths[0].exists())

    def test_cacheability_requires_a_clean_final_wav_scan(self):
        path = Path("000000.wav")
        issue = gradio_tts_app.AudioQualityIssue("artifact", 0.0, 1.0)
        with patch.object(gradio_tts_app, "_save_and_normalize_chunk"), patch.object(
            gradio_tts_app, "wav_generated_audio_issues", return_value=[issue]
        ) as scan:
            result = gradio_tts_app._save_normalize_and_record_chunk(
                path,
                torch.zeros((1, self.sample_rate)),
                self.sample_rate,
                "ffmpeg",
                None,
                cacheable=True,
            )

        self.assertFalse(result.verified_clean)
        scan.assert_called_once_with(path, self.sample_rate)

    def test_completion_metadata_is_written_before_best_effort_chunk_cleanup(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            project_dir = Path(directory)
            metadata_path = project_dir / "metadata.json"
            with patch.object(
                gradio_tts_app,
                "_write_metadata",
                side_effect=lambda *_args, **_kwargs: events.append("metadata"),
            ), patch.object(
                gradio_tts_app,
                "delete_quality_scan_checkpoint",
                side_effect=lambda _path: events.append("checkpoint"),
            ), patch.object(
                gradio_tts_app,
                "delete_intermediate_chunks",
                side_effect=lambda _path: events.append("chunks"),
            ):
                gradio_tts_app._mark_verified_project_complete(
                    metadata_path,
                    project_dir,
                    Mock(),
                    "book.epub",
                    [],
                    {},
                    "model",
                    "book.m4b",
                )

        self.assertEqual(events, ["metadata", "checkpoint", "chunks"])

    def test_metadata_failure_prevents_all_project_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            project_dir = Path(directory)
            metadata_path = project_dir / "metadata.json"
            with patch.object(
                gradio_tts_app,
                "_write_metadata",
                side_effect=OSError("disk full"),
            ), patch.object(
                gradio_tts_app,
                "delete_quality_scan_checkpoint",
            ) as delete_checkpoint, patch.object(
                gradio_tts_app,
                "delete_intermediate_chunks",
            ) as delete_chunks:
                with self.assertRaisesRegex(OSError, "disk full"):
                    gradio_tts_app._mark_verified_project_complete(
                        metadata_path,
                        project_dir,
                        Mock(),
                        "book.epub",
                        [],
                        {},
                        "model",
                        "book.m4b",
                    )

            delete_checkpoint.assert_not_called()
            delete_chunks.assert_not_called()


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

    def excessive_silence_audio(self):
        return torch.zeros((1, 9 * self.sample_rate), dtype=torch.float32)

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

    def test_near_silent_chunk_retries_the_entire_chunk(self):
        retry_calls = []

        recovery = gradio_tts_app._recover_generated_waveform(
            self.excessive_silence_audio(),
            "This ordinary test sentence has enough words to permit a normal duration.",
            self.sample_rate,
            "chunk 000124",
            lambda: (retry_calls.append("retry") or self.good_audio()),
            lambda part: self.good_audio(),
        )

        self.assertEqual(retry_calls, ["retry"])
        self.assertTrue(recovery.detected_quality_issues)
        self.assertFalse(recovery.retained_with_warning)
        self.assertEqual(recovery.retained_quality_issues, ())

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
