from types import SimpleNamespace
import unittest

from chatterbox_vllm.tts import T3WorkerExtension


class CfgScaleTests(unittest.TestCase):
    def test_sets_cfg_on_the_worker_model(self):
        class Model:
            def __init__(self):
                self.cfg_scale = 0.5

            def set_cfg_scale(self, cfg_scale):
                self.cfg_scale = float(cfg_scale)
                return self.cfg_scale

        model = Model()
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))

        result = T3WorkerExtension.set_t3_cfg_scale(worker, 0.8)

        self.assertEqual(result, 0.8)
        self.assertEqual(model.cfg_scale, 0.8)


if __name__ == "__main__":
    unittest.main()
