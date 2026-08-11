import unittest

from chatterbox_vllm.model_variants import (
    DEFAULT_MODEL_ID,
    ENGLISH_V1_MODEL_ID,
    MULTILINGUAL_V2_MODEL_ID,
    MULTILINGUAL_V3_MODEL_ID,
    resolve_model_id,
)


class ModelVariantTests(unittest.TestCase):
    def test_defaults_to_multilingual_v3(self):
        self.assertEqual(resolve_model_id(None), DEFAULT_MODEL_ID)
        self.assertEqual(DEFAULT_MODEL_ID, MULTILINGUAL_V3_MODEL_ID)

    def test_resolves_short_aliases(self):
        self.assertEqual(resolve_model_id("english"), ENGLISH_V1_MODEL_ID)
        self.assertEqual(resolve_model_id("v2"), MULTILINGUAL_V2_MODEL_ID)
        self.assertEqual(resolve_model_id("v3"), MULTILINGUAL_V3_MODEL_ID)

    def test_rejects_an_unknown_variant(self):
        with self.assertRaisesRegex(ValueError, "Unknown Chatterbox model"):
            resolve_model_id("v4")


if __name__ == "__main__":
    unittest.main()
