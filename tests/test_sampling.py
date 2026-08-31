from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import torch

from chatterbox_vllm.tts import ChatterboxTTS


class _FakeT3:
    def __init__(self):
        self.prompts = []

    def collective_rpc(self, method, args):
        return [args[0]]

    def generate(self, prompts, sampling_params):
        self.prompts = prompts
        return []


class SamplingTests(unittest.TestCase):
    def test_passes_min_p_to_vllm_sampling_params(self):
        model = ChatterboxTTS.__new__(ChatterboxTTS)
        model.variant = "multilingual"
        model.max_model_len = 1000
        model.t3_config = SimpleNamespace(stop_speech_token=6562)
        model.t3 = _FakeT3()
        model._t3_generation_lock = threading.Lock()
        model.update_exaggeration = lambda cond_emb, exaggeration: cond_emb

        captured = {}

        def sampling_params(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        with patch("chatterbox_vllm.tts.SamplingParams", side_effect=sampling_params):
            result = model.generate_with_conds(
                prompts=["Test sentence."],
                s3gen_ref={},
                cond_emb=torch.zeros(1),
                min_p=0.05,
            )

        self.assertEqual(result, [])
        self.assertEqual(captured["min_p"], 0.05)

    def test_cleans_only_transient_prompts_and_logs_one_batch_summary(self):
        model = ChatterboxTTS.__new__(ChatterboxTTS)
        model.variant = "multilingual"
        model.max_model_len = 1000
        model.t3_config = SimpleNamespace(stop_speech_token=6562)
        model.t3 = _FakeT3()
        model._t3_generation_lock = threading.Lock()
        model.update_exaggeration = lambda cond_emb, exaggeration: cond_emb
        source_prompts = [
            'Narration. “Bonjour, café!” [12]\u200b ★',
            "Plain multilingual text 123.",
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            model.generate_with_conds(
                prompts=source_prompts,
                s3gen_ref={},
                cond_emb=torch.zeros(1),
            )

        self.assertEqual(
            source_prompts,
            [
                'Narration. “Bonjour, café!” [12]\u200b ★',
                "Plain multilingual text 123.",
            ],
        )
        self.assertEqual(
            model.t3.prompts[0]["prompt"],
            "<en>[START]Narration. Bonjour, café![STOP]",
        )
        self.assertEqual(
            model.t3.prompts[1]["prompt"],
            "<en>[START]Plain multilingual text 123.[STOP]",
        )
        summary_lines = [
            line for line in output.getvalue().splitlines()
            if line.startswith("[Text cleanup]")
        ]
        self.assertEqual(len(summary_lines), 1)
        self.assertIn("1/2 prompts changed", summary_lines[0])
        self.assertIn("footnotes=1", summary_lines[0])
        self.assertIn("quotes=2", summary_lines[0])
        self.assertIn("controls=1", summary_lines[0])
        self.assertIn("symbols=1", summary_lines[0])

    def test_logs_existing_punctuation_normalization_changes(self):
        model = ChatterboxTTS.__new__(ChatterboxTTS)
        model.variant = "multilingual"
        model.max_model_len = 1000
        model.t3_config = SimpleNamespace(stop_speech_token=6562)
        model.t3 = _FakeT3()
        model._t3_generation_lock = threading.Lock()
        model.update_exaggeration = lambda cond_emb, exaggeration: cond_emb

        output = io.StringIO()
        with redirect_stdout(output):
            model.generate_with_conds(
                prompts=["hello   there"],
                s3gen_ref={},
                cond_emb=torch.zeros(1),
            )

        self.assertEqual(
            model.t3.prompts[0]["prompt"],
            "<en>[START]Hello there.[STOP]",
        )
        self.assertIn("1/1 prompts changed", output.getvalue())
        self.assertIn("normalization=1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
