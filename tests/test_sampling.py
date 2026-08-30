from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import torch

from chatterbox_vllm.tts import ChatterboxTTS


class _FakeT3:
    def collective_rpc(self, method, args):
        return [args[0]]

    def generate(self, prompts, sampling_params):
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


if __name__ == "__main__":
    unittest.main()
