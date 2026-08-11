import unittest
from pathlib import Path

from tokenizers import Tokenizer

from chatterbox_vllm.models.t3.mtltokenizer import MTLTokenizer


class _TokenizerHarness:
    preprocess_text = MTLTokenizer.preprocess_text
    tokenizer = Tokenizer.from_file(
        str(
            Path(__file__).resolve().parents[1]
            / "src"
            / "chatterbox_vllm"
            / "models"
            / "t3"
            / "grapheme_mtl_merged_expanded_v1.json"
        )
    )


class MultilingualTokenizerTests(unittest.TestCase):
    def test_preserves_text_boundary_tokens_while_lowercasing_content(self):
        tokens = MTLTokenizer._tokenize(
            _TokenizerHarness(),
            '<en>[START]Hello There[STOP]',
        )

        self.assertEqual(tokens[:2], ["[START]", "[en]"])
        self.assertEqual(tokens[-1], "[STOP]")
        self.assertNotIn("[stop]", tokens)
        self.assertIn("he", tokens)
        self.assertIn("there", tokens)


if __name__ == "__main__":
    unittest.main()
