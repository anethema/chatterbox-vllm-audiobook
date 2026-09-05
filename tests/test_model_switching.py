"""CPU-only regression coverage for model switching in the Gradio app."""
import inspect
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import gradio as gr

import gradio_tts_app as app
import chatterbox_vllm.tts as tts
from chatterbox_vllm.model_variants import ENGLISH_V1_MODEL_ID


class ModelSwitchingTests(unittest.TestCase):
    def setUp(self):
        self.original_model = app.global_model
        self.original_active_model_id = app.ACTIVE_MODEL_ID
        self.original_generation_lock = app.generation_task_lock
        self.original_status_lock = app.model_status_lock
        self.original_status = app.model_status

        # Tests share the imported application module, so give each test its own
        # synchronization and mutable status state.
        self.test_generation_lock = threading.Lock()
        app.generation_task_lock = self.test_generation_lock
        app.model_status_lock = threading.Lock()
        app.model_status = {
            "message": "Model has not been loaded yet.",
            "fraction": None,
            "busy": False,
        }
        app.global_model = None
        app.ACTIVE_MODEL_ID = app.DEFAULT_MODEL_ID

    def tearDown(self):
        if self.test_generation_lock.locked():
            self.test_generation_lock.release()
        app.global_model = self.original_model
        app.ACTIVE_MODEL_ID = self.original_active_model_id
        app.generation_task_lock = self.original_generation_lock
        app.model_status_lock = self.original_status_lock
        app.model_status = self.original_status

    def test_switch_to_current_model_is_a_cached_noop(self):
        target = app.MULTILINGUAL_V3_MODEL_ID
        existing = Mock(model_id=target)
        app.global_model = existing
        app.ACTIVE_MODEL_ID = target
        app._set_model_status("A previous download failed.")

        with patch.object(app, "ensure_model_downloaded") as download, \
             patch.object(app, "_unload_model") as unload, \
             patch.object(app, "load_model") as load:
            result = app.switch_model(target, progress=Mock())

        self.assertEqual(result, f"Ready: {app.model_label(target)}.")
        self.assertIs(app.global_model, existing)
        self.assertEqual(
            app.model_status,
            {
                "message": f"Ready: {app.model_label(target)}. Downloaded models stay cached for reuse.",
                "fraction": None,
                "busy": False,
            },
        )
        download.assert_not_called()
        unload.assert_not_called()
        load.assert_not_called()

    def test_initial_model_selection_reflects_the_server_model(self):
        app.ACTIVE_MODEL_ID = app.MULTILINGUAL_V3_MODEL_ID
        app.global_model = Mock(model_id=ENGLISH_V1_MODEL_ID)
        self.assertEqual(app._initial_model_selection(), ENGLISH_V1_MODEL_ID)

        app.global_model = None
        self.assertEqual(app._initial_model_selection(), app.MULTILINGUAL_V3_MODEL_ID)

    def test_download_failure_keeps_the_existing_model_loaded(self):
        existing = Mock(model_id=app.MULTILINGUAL_V3_MODEL_ID)
        app.global_model = existing
        target = ENGLISH_V1_MODEL_ID

        with patch.object(
            app, "ensure_model_downloaded", side_effect=RuntimeError("network unavailable")
        ) as download, patch.object(app, "_unload_model") as unload, patch.object(
            app, "load_model"
        ) as load:
            result = app.switch_model(target, progress=Mock())

        self.assertIs(app.global_model, existing)
        download.assert_called_once()
        unload.assert_not_called()
        load.assert_not_called()
        self.assertIn("network unavailable", result)
        self.assertIn("still active", result)
        self.assertIn("Load / Retry", result)
        self.assertFalse(app.model_status["busy"])

    def test_successful_download_shuts_down_then_loads_and_updates_active_id(self):
        old_model = Mock(model_id=app.MULTILINGUAL_V3_MODEL_ID)
        target = ENGLISH_V1_MODEL_ID
        new_model = Mock(model_id=target)
        events = []
        progress = Mock()
        old_model.shutdown.side_effect = lambda: events.append("shutdown")

        def download(model_id, report):
            self.assertEqual(model_id, target)
            events.append("download")
            report(0.5, "Downloading checkpoint")

        def construct(*args, **kwargs):
            events.append("load")
            return new_model

        app.global_model = old_model
        with patch.object(app, "ensure_model_downloaded", side_effect=download), \
             patch.object(app.ChatterboxTTS, "from_model_id", side_effect=construct), \
             patch.object(app, "release_unused_memory") as release_memory, \
             patch.object(app.torch.cuda, "empty_cache") as empty_cache:
            result = app.switch_model(target, progress=progress)

        self.assertEqual(events, ["download", "shutdown", "load"])
        self.assertIs(app.global_model, new_model)
        self.assertEqual(app.ACTIVE_MODEL_ID, target)
        self.assertEqual(result, f"Ready: {app.model_label(target)}.")
        release_memory.assert_called_once()
        empty_cache.assert_called_once()
        self.assertIn(call(None, desc=f"Loading {app.model_label(target)} into GPU memory…"), progress.call_args_list)

    def test_failed_load_clears_model_and_prompts_for_retry(self):
        old_model = Mock(model_id=app.MULTILINGUAL_V3_MODEL_ID)
        target = ENGLISH_V1_MODEL_ID
        old_active_model_id = app.ACTIVE_MODEL_ID
        app.global_model = old_model

        with patch.object(app, "ensure_model_downloaded"), patch.object(
            app.ChatterboxTTS, "from_model_id", side_effect=RuntimeError("out of memory")
        ), patch.object(app, "release_unused_memory"), patch.object(
            app.torch.cuda, "empty_cache"
        ):
            result = app.switch_model(target, progress=Mock())

        old_model.shutdown.assert_called_once()
        self.assertIsNone(app.global_model)
        self.assertEqual(app.ACTIVE_MODEL_ID, old_active_model_id)
        self.assertIn("out of memory", result)
        self.assertIn("No model is loaded", result)
        self.assertIn("Load / Retry", result)
        self.assertFalse(app.model_status["busy"])

    def test_switch_waits_for_the_current_generation_to_release_the_model_slot(self):
        app.global_model = Mock(model_id=app.MULTILINGUAL_V3_MODEL_ID)
        target = ENGLISH_V1_MODEL_ID
        entered = threading.Event()
        completed = threading.Event()
        result = []
        progress = Mock()

        def run_switch():
            entered.set()
            result.append(app.switch_model(target, progress=progress))
            completed.set()

        with patch.object(app, "ensure_model_downloaded") as download, \
             patch.object(app, "_unload_model") as unload, \
             patch.object(app, "load_model") as load:
            self.test_generation_lock.acquire()
            worker = threading.Thread(target=run_switch)
            worker.start()
            try:
                self.assertTrue(entered.wait(timeout=1))
                self.assertFalse(completed.wait(timeout=0.1))
                download.assert_not_called()
                unload.assert_not_called()
                load.assert_not_called()
            finally:
                self.test_generation_lock.release()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(completed.is_set())
        self.assertEqual(result, [f"Ready: {app.model_label(target)}."])
        download.assert_called_once()
        unload.assert_called_once()
        load.assert_called_once_with(target)

    def test_generation_rejects_missing_or_mismatched_selected_model(self):
        for function in (
            app.generate_sample,
            app.generate_epub_audiobook,
            app.resume_epub_audiobook,
        ):
            self.assertIn("selected_model", inspect.signature(function).parameters)

        arguments = (
            "A short sample.", None, False, 0.5, 0.5, 0.8, 1, 10, 0.05,
            1.0, 1.2, app.MULTILINGUAL_V3_MODEL_ID,
        )
        with self.assertRaises(gr.Error) as missing:
            app.generate_sample(*arguments)
        self.assertIn("No model is loaded", str(missing.exception))

        loaded = Mock(model_id=ENGLISH_V1_MODEL_ID)
        app.global_model = loaded
        with self.assertRaises(gr.Error) as mismatch:
            app.generate_sample(*arguments)
        self.assertIn("selected model is not active", str(mismatch.exception))
        loaded.generate.assert_not_called()

    def test_gradio_progress_injection_preserves_selected_model_input(self):
        for function in (app.generate_epub_audiobook, app.resume_epub_audiobook):
            with self.subTest(function=function.__name__):
                parameters = list(inspect.signature(function).parameters.values())
                progress_index = next(
                    index for index, parameter in enumerate(parameters)
                    if parameter.name == "progress"
                )
                selected_index = next(
                    index for index, parameter in enumerate(parameters)
                    if parameter.name == "selected_model"
                )
                selected_model = object()
                inputs = [object() for _ in range(len(parameters) - 1)]
                inputs[-1] = selected_model

                injected, detected_progress_index, _ = gr.helpers.special_args(
                    function, list(inputs)
                )

                self.assertEqual(detected_progress_index, progress_index)
                self.assertIs(injected[selected_index], selected_model)
                self.assertIs(
                    injected[progress_index], parameters[progress_index].default
                )

    def test_model_status_html_escapes_download_messages(self):
        message = '<img src=x onerror="alert(1)">'
        app._set_model_status(message, 1.5, busy=True)

        active, status = app._model_monitor_render()

        self.assertIn("Active model:** None", active)
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", status)
        self.assertNotIn(message, status)
        self.assertIn('<progress max="1" value="1.0"', status)

    def test_decoder_initialization_failure_shuts_down_the_new_llm_worker(self):
        """A failure after LLM construction must not strand its worker process."""
        config = SimpleNamespace(
            speech_tokens_dict_size=10,
            n_channels=4,
            max_speech_tokens=12,
        )
        cond_encoder = Mock()
        cond_encoder.to.return_value = cond_encoder
        cond_encoder.eval.return_value = cond_encoder
        speech_embedding = Mock()
        speech_embedding.to.return_value = speech_embedding
        speech_embedding.eval.return_value = speech_embedding
        position_embedding = Mock()
        position_embedding.to.return_value = position_embedding
        position_embedding.eval.return_value = position_embedding
        llm = Mock()

        with patch.object(tts, "T3Config", return_value=config), \
             patch.object(tts, "load_file", return_value={}), \
             patch.object(tts, "T3CondEnc", return_value=cond_encoder), \
             patch.object(tts.torch.nn, "Embedding", return_value=speech_embedding), \
             patch.object(
                 tts, "LearnedPositionEmbeddings", return_value=position_embedding
             ), patch.object(
                 tts.torch.cuda,
                 "get_device_properties",
                 return_value=SimpleNamespace(total_memory=32 * 1024 ** 3),
             ), patch.object(tts.torch.cuda, "memory_allocated", return_value=0), \
             patch.object(tts, "LLM", return_value=llm), \
             patch.object(
                 tts, "VoiceEncoder", side_effect=RuntimeError("voice encoder failed")
             ):
            with self.assertRaisesRegex(RuntimeError, "voice encoder failed"):
                tts.ChatterboxTTS.from_local(
                    "/mock-checkpoint",
                    target_device="cuda",
                    variant="english",
                    model_id=tts.ENGLISH_V1_MODEL_ID,
                )

        llm.llm_engine.engine_core.shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
