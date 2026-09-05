"""Regression coverage for final-file accounting, resume safety, and model state."""
from collections import OrderedDict
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import gc
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

import gradio_tts_app as app
from chatterbox_vllm.epub import EpubBook, EpubChapter, chunk_book
from chatterbox_vllm.job_status import JobStatusStore
from chatterbox_vllm.progress import GenerationControl
from chatterbox_vllm.projects import (
    ResumePlan, load_quality_scan_checkpoint, load_quality_summary,
    wav_file_identity, write_quality_scan_checkpoint, write_quality_summary,
)
from chatterbox_vllm.models.s3tokenizer import drop_invalid_tokens, SOS, EOS
from chatterbox_vllm.models.t3.modules.cond_enc import T3Cond
from chatterbox_vllm.tts import ChatterboxTTS, Conditionals


class TokenAndConditioningTests(unittest.TestCase):
    def test_euler_final_state_matches_reference_without_retaining_steps(self):
        from chatterbox_vllm.models.s3gen.flow_matching import ConditionalCFM, CFM_PARAMS

        class Estimator(torch.nn.Module):
            def forward(self, x, mask, mu, t, spks, cond):
                return x * .17 + mu * .13 + cond * .07 + t[:, None, None]

        solver = ConditionalCFM(80, CFM_PARAMS, estimator=Estimator())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            initial, mu, cond = (torch.randn(1, 80, 12) for _ in range(3))
        mask, spks = torch.ones(1, 1, 12), torch.ones(1, 80)
        times = 1 - torch.cos(torch.linspace(0, 1, 16) * .5 * torch.pi)
        expected, t, dt = initial, times[0], times[1] - times[0]
        for step in range(1, len(times)):
            conditional = expected * .17 + mu * .13 + cond * .07 + t
            unconditional = expected * .17 + torch.zeros_like(mu) * .13 + torch.zeros_like(cond) * .07 + t
            derivative = (1 + solver.inference_cfg_rate) * conditional - solver.inference_cfg_rate * unconditional
            expected = expected + dt * derivative
            t = t + dt
            if step < len(times) - 1:
                dt = times[step + 1] - t
        actual = solver.solve_euler(initial, times, mu, mask, spks, cond)
        self.assertTrue(torch.equal(actual, expected))

    def test_empty_single_and_boundary_tokens(self):
        cases = [([], []), ([7], [7]), ([EOS], []), ([SOS], []),
                 ([SOS, 7, 8, EOS, 9], [7, 8]), ([7, EOS], [7])]
        for tokens, expected in cases:
            for shape in (None, (1, len(tokens))):
                with self.subTest(tokens=tokens, shape=shape):
                    tensor = torch.tensor(tokens, dtype=torch.long)
                    if shape:
                        tensor = tensor.reshape(shape)
                    self.assertEqual(drop_invalid_tokens(tensor).tolist(), expected)

    def tiny_model(self):
        class TinyCond(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emotion_adv_fc = torch.nn.Linear(1, 4)

            def forward(self, cond):
                return torch.cat((cond.cond_prompt_speech_emb, self.emotion_adv_fc(cond.emotion_adv)), dim=0)

        model = ChatterboxTTS.__new__(ChatterboxTTS)
        model.target_device = 'cpu'
        model._inference_lock = threading.RLock()
        model._conditioning_cache = OrderedDict()
        model._t3_generation_lock = threading.Lock()
        model._closed = False
        model.flush_cuda_cache = True
        model.variant = 'english'
        model.max_model_len = 1000
        model.t3_config = SimpleNamespace(stop_speech_token=EOS)
        model.t3 = Mock()
        model.t3_speech_emb = torch.nn.Embedding(10, 4)
        model.t3_speech_pos_emb = lambda tokens: torch.zeros(tokens.numel(), 4)
        model.t3_cond_enc = TinyCond()
        model.default_conds = Conditionals(T3Cond(cond_prompt_speech_tokens=torch.tensor([[1, 2]]), speaker_emb=torch.zeros(4)), {})
        return model

    def test_cached_conditionals_are_inference_only_and_do_not_own_model(self):
        model = self.tiny_model()
        _, cond = model.get_audio_conditionals()
        changed = model.update_exaggeration(cond, .6)
        self.assertFalse(cond.requires_grad)
        self.assertIsNone(cond.grad_fn)
        self.assertIsNone(changed.grad_fn)
        reference = weakref.ref(model)
        model.shutdown()
        model.shutdown()
        del model
        gc.collect()
        self.assertIsNone(reference())

    def test_reference_replacement_invalidates_conditioning(self):
        model = self.tiny_model()
        model._compute_audio_conditionals = Mock(side_effect=[('first', torch.zeros(1)), ('second', torch.ones(1))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'voice.wav'
            path.write_bytes(b'old')
            self.assertEqual(model.get_audio_conditionals(str(path))[0], 'first')
            self.assertEqual(model.get_audio_conditionals(str(path))[0], 'first')
            path.write_bytes(b'replacement voice')
            self.assertEqual(model.get_audio_conditionals(str(path))[0], 'second')
        self.assertEqual(model._compute_audio_conditionals.call_count, 2)

    def test_shutdown_stops_worker_even_when_llm_is_still_referenced(self):
        model = self.tiny_model()
        retained_llm = model.t3
        model.shutdown()
        model.shutdown()
        retained_llm.llm_engine.engine_core.shutdown.assert_called_once_with()
        self.assertIsNone(model.t3)
        self.assertFalse(model._conditioning_cache)

    def test_empty_token_output_is_per_chunk_recoverable(self):
        model = self.tiny_model()
        model.t3.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[])])]
        output = model.generate_with_conds(['A sentence.'], {}, torch.zeros(2, 4))
        self.assertEqual(len(output), 1)
        self.assertEqual(tuple(output[0].shape), (1, 0))

    def test_engine_failure_still_propagates(self):
        model = self.tiny_model()
        model.t3.generate.side_effect = RuntimeError('engine unavailable')
        with self.assertRaisesRegex(RuntimeError, 'engine unavailable'):
            model.generate_with_conds(['A sentence.'], {}, torch.zeros(2, 4))


class CoordinatorRegressionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.output = self.root / 'outputs'
        self.output.mkdir()
        self.stack.enter_context(patch.object(app, '__file__', str(self.root / 'app.py')))
        self.stack.enter_context(patch.object(app, 'OUTPUT_ROOT', self.output))
        self.stack.enter_context(patch.object(app, 'job_status', JobStatusStore(self.output)))
        self.stack.enter_context(patch.object(app, 'generation_control', GenerationControl()))
        self.epub = self.root / 'book.epub'
        self.epub.write_bytes(b'epub fixture')
        self.reference = self.root / 'voice.wav'
        self.reference.write_bytes(b'original voice')
        self.book = EpubBook('Regression fixture', (EpubChapter('Chapter', 'One sentence for local regression tests.', 'c.xhtml'),))
        self.chunks = chunk_book(self.book, 200)
        self.waveform = torch.sin(torch.arange(48000) * .05).reshape(1, -1) * .1
        self.model = SimpleNamespace(
            model_id=app.MULTILINGUAL_V3_MODEL_ID, sr=24000,
            get_audio_conditionals=Mock(return_value=({}, torch.zeros(2, 4))),
            generate_with_conds=Mock(return_value=[self.waveform]),
        )
        self.stack.enter_context(patch.object(app, 'global_model', self.model))
        self.stack.enter_context(patch.object(app, 'load_epub', return_value=self.book))
        self.stack.enter_context(patch.object(app, 'find_generated_audio_issues', return_value=[]))
        self.stack.enter_context(patch.object(app, 'normalize_speech_wav', side_effect=lambda path, *a, **k: path))
        self.stack.enter_context(patch.object(app, 'limit_internal_pauses_wav', side_effect=lambda path, *a, **k: path))
        self.stack.enter_context(patch.object(app, '_protect_system_memory'))
        self.stack.enter_context(patch.object(app, 'verify_m4b', return_value=2.0))
        self.stack.enter_context(patch.object(app, 'register_completed_audiobook', side_effect=lambda path, *a: path))

        def assembly(project, *args, **kwargs):
            path = project / 'audiobook.m4b'
            path.write_bytes(b'm4b fixture')
            return path
        self.assembly = self.stack.enter_context(patch.object(app, 'assemble_audiobook', side_effect=assembly))
        self.stdout = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.stdout))
        self.stack.enter_context(redirect_stderr(io.StringIO()))

    def generate(self, project=None, reference=None):
        return app.generate_epub_audiobook(
            str(self.epub), str(reference or self.reference), .5, .5, .8, 0,
            15, .05, 1., 1.2, 200, 64, False, project, progress=Mock(),
        )

    def make_project(self):
        project = self.output / 'saved-project'
        (project / 'chunks').mkdir(parents=True)
        app.persist_project_inputs(project, self.epub, self.reference)
        settings = app.generation_arguments(.5, .5, .8, 15, .05, 1., 1.2, None)
        settings.update(max_chars=200, batch_size=64, loudness_target_lufs=-18, true_peak_dbtp=-2, loudness_range_lu=7)
        app._write_metadata(project / 'metadata.json', self.book, str(self.epub), self.chunks, settings, 0, self.model.model_id)
        return project, json.loads((project / 'metadata.json').read_text())

    def test_final_only_issue_is_repaired_before_summary_and_assembly(self):
        issue = app.AudioQualityIssue('final-only artifact', 0, 1)
        def repaired(path, *args):
            self.assertFalse(self.assembly.called)
            self.assertNotIn('1/1 OK', self.stdout.getvalue())
            return app.SavedChunkQuality(path, True)
        with patch.object(app, 'wav_generated_audio_issues', return_value=[issue]), patch.object(app, '_repair_saved_chunk', side_effect=repaired) as repair:
            output, _ = self.generate()
        self.assertIsNotNone(output)
        self.assertEqual(repair.call_count, 1)
        summary = json.loads((Path(output).parent / 'quality-summary.json').read_text())
        self.assertEqual(summary['detected'], [0])
        self.assertEqual(summary['fixed'], [0])
        self.assertEqual(summary['retained'], [])
        self.assertIn('bad chunks detected: 1; fixed: 1;', self.stdout.getvalue())

    def test_failed_final_repair_is_retained_once_and_reported(self):
        issue = app.AudioQualityIssue('final-only artifact', 0, 1)
        with patch.object(app, 'wav_generated_audio_issues', return_value=[issue]), patch.object(app, '_repair_saved_chunk', side_effect=lambda path, *a: app.SavedChunkQuality(path, False, (issue,))) as repair:
            output, _ = self.generate()
        self.assertIsNotNone(output)
        self.assertEqual(repair.call_count, 1)
        self.assertNotIn('1/1 OK', self.stdout.getvalue())
        self.assertIn('retained with warnings: 1 (000000)', self.stdout.getvalue())
        self.assertEqual(load_quality_summary(Path(output).parent, 1), ({0}, set(), {0}))

    def test_resume_preserves_voice_and_accepts_legacy_settings(self):
        project, _ = self.make_project()
        other = self.root / 'other.wav'
        other.write_bytes(b'different voice')
        with patch.object(app, 'wav_generated_audio_issues', return_value=[]):
            output, message = self.generate(project.name, other)
        self.assertIsNotNone(output, message)
        self.assertEqual((project / 'inputs/reference-audio.wav').read_bytes(), b'original voice')
        self.assertEqual(self.model.get_audio_conditionals.call_args.args[0], str(project / 'inputs/reference-audio.wav'))

    def test_matching_final_checkpoint_skips_all_scans_and_transforms(self):
        project, metadata = self.make_project()
        chunk = project / 'chunks/000000.wav'
        app.ta.save(str(chunk), self.waveform, 24000, encoding='PCM_S', bits_per_sample=16)
        signature = {'version': 1, 'sample_rate': 24000, 'maximum_internal_pause_seconds': .5,
                     'loudness_target_lufs': app.TARGET_LUFS, 'true_peak_dbtp': app.TRUE_PEAK_DBTP,
                     'loudness_range_lu': app.LOUDNESS_RANGE_LU}
        write_quality_scan_checkpoint(project, {0: wav_file_identity(chunk)}, processing_signature=signature)
        write_quality_summary(project, 1, {0}, {0}, set())
        plan = ResumePlan(project, metadata, tuple(self.chunks), 1, 1)
        with patch.object(app, 'build_resume_plan', return_value=plan), patch.object(app, 'wav_generated_audio_issues') as scan, patch.object(app, 'limit_internal_pauses_wav') as limit:
            output, message = self.generate(project.name)
        self.assertIsNotNone(output, message)
        scan.assert_not_called()
        limit.assert_not_called()
        self.model.generate_with_conds.assert_not_called()
        self.assertEqual(load_quality_summary(project, 1), ({0}, {0}, set()))

    def test_malformed_resume_releases_job_slot(self):
        project, _ = self.make_project()
        (project / 'metadata.json').write_text('[]')
        output, _ = self.generate(project.name)
        self.assertIsNone(output)
        self.assertEqual(app.job_status.snapshot().state, 'failed')
        self.assertFalse(app.generation_control.request_stop())
        self.assertTrue(app.generation_task_lock.acquire(blocking=False))
        app.generation_task_lock.release()

    def test_unreadable_metadata_is_ignored_during_project_discovery(self):
        from chatterbox_vllm.projects import incomplete_project_choices

        project, _ = self.make_project()
        for content in (b'\xff\xfe', b'{"settings": []}'):
            (project / 'metadata.json').write_bytes(content)
            self.assertEqual(incomplete_project_choices(self.output), [])

    def test_all_generation_events_share_concurrency_group(self):
        functions = [f for f in app.demo.fns.values() if f.fn and f.fn.__name__ in ('generate_sample', 'generate_epub_audiobook', 'resume_epub_audiobook')]
        self.assertEqual(len(functions), 3)
        self.assertEqual({f.concurrency_id for f in functions}, {'model-generation'})

    def test_reference_preprocessor_defers_decoding_to_restricted_ffmpeg(self):
        from gradio.data_classes import FileData

        with patch('gradio.processing_utils.audio_from_file', side_effect=AssertionError('Unrestricted decoder called')):
            self.assertEqual(app.ref_wav.preprocess(FileData(path=str(self.reference))), str(self.reference))

    def test_patched_web_stack_upload_download_and_ranges(self):
        from fastapi.testclient import TestClient
        from gradio.routes import App

        with patch.object(app.demo, 'max_file_size', 1024, create=True), patch.object(app.demo, 'allowed_paths', [str(self.reference)], create=True), patch.object(app.demo, 'blocked_paths', [], create=True):
            api = App.create_app(app.demo)
            with TestClient(api) as client:
                uploaded = client.post('/gradio_api/upload', files={'files': ('voice.wav', b'small fixture', 'audio/wav')})
                self.assertEqual(uploaded.status_code, 200)
                oversized = client.post('/gradio_api/upload', files={'files': ('large.wav', b'x' * 2048, 'audio/wav')})
                self.assertEqual(oversized.status_code, 413)
                route = '/gradio_api/file=' + str(self.reference)
                response = client.get(route, headers={'Range': 'bytes=0-3'})
                self.assertEqual(response.status_code, 206)
                self.assertEqual(response.content, b'orig')
                response = client.get(route, headers={'Range': 'bytes=' + '0' * 4000 + 'a-'})
                self.assertEqual(response.status_code, 400)

    def test_preview_replacement_deletes_only_our_preview(self):
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.object(app, 'reference_preview_directory', SimpleNamespace(name=directory)))
        owned = Path(directory) / 'normalized-reference-old.wav'
        owned.write_bytes(b'old preview')
        app.delete_reference_preview(str(self.reference))
        self.assertTrue(self.reference.exists())
        app.delete_reference_preview(str(owned))
        self.assertFalse(owned.exists())

    def test_status_write_failure_cannot_strand_generation_slot(self):
        @app._audiobook_job
        def early_return():
            return None, 'early failure'
        with patch.object(app.job_status, 'finish', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                early_return()
        self.assertTrue(app.generation_task_lock.acquire(blocking=False))
        app.generation_task_lock.release()

    def test_final_repair_scans_normalized_candidate_before_acceptance(self):
        path = self.output / '000000.wav'
        app.ta.save(str(path), self.waveform, 24000, encoding='PCM_S', bits_per_sample=16)
        issue = app.AudioQualityIssue('artifact', 0, 1)
        with patch.object(app, 'find_generated_audio_issues', side_effect=[[issue], [], []]):
            with patch.object(app, 'wav_generated_audio_issues', return_value=[]), patch.object(app, '_save_and_normalize_chunk', wraps=app._save_and_normalize_chunk) as save:
                generated = Mock(return_value=self.waveform)
                result = app._repair_saved_chunk(path, 'One short sentence.', 24000, 'ffmpeg', .5, generated)
        self.assertTrue(result.verified_clean)
        generated.assert_called_once_with('One short sentence.')
        self.assertEqual(save.call_count, 2)

    def test_transform_signature_change_invalidates_checkpoint(self):
        project, _ = self.make_project()
        write_quality_scan_checkpoint(project, {0: {'size': 12, 'mtime_ns': 1}}, processing_signature={'pause': .5})
        self.assertEqual(load_quality_scan_checkpoint(project, processing_signature={'pause': .6}), {})


if __name__ == '__main__':
    unittest.main()
