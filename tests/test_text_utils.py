import unittest

from chatterbox_vllm.text_utils import clean_tts_text, prepare_tts_text


class TextCleanupTests(unittest.TestCase):
    def test_preserves_multilingual_letters_numbers_and_combining_marks(self):
        cleaned, changes = clean_tts_text(
            "Cafe\u0301 mañana नमस्ते العربية 中文，真的！ ١٢٣ 42."
        )

        self.assertEqual(
            cleaned,
            "Cafe\u0301 mañana नमस्ते العربية 中文，真的！ ١٢٣ 42.",
        )
        self.assertEqual(changes, {})

    def test_removes_quotes_footnotes_controls_and_decorative_symbols(self):
        prepared, changes = prepare_tts_text(
            '“Keep” this—please [12]\u200b ★ Reference¹.'
        )

        self.assertEqual(prepared, "Keep this-please Reference.")
        self.assertEqual(changes["quotes"], 2)
        self.assertEqual(changes["footnotes"], 2)
        self.assertEqual(changes["controls"], 1)
        self.assertEqual(changes["symbols"], 1)
        self.assertEqual(changes["punctuation"], 1)

    def test_numeric_parenthetical_content_survives(self):
        prepared, changes = prepare_tts_text("The years (1999) and (12) matter.")

        self.assertEqual(prepared, "The years 1999 and 12 matter.")
        self.assertNotIn("footnotes", changes)

    def test_unsupported_separators_do_not_join_words(self):
        prepared, changes = prepare_tts_text("alpha/beta • gamma.")

        self.assertEqual(prepared, "Alpha beta gamma.")
        self.assertEqual(changes["symbols"], 2)

    def test_existing_punctuation_normalization_is_reported(self):
        prepared, changes = prepare_tts_text("hello   there")

        self.assertEqual(prepared, "Hello there.")
        self.assertEqual(changes, {"normalization": 1})

    def test_expands_only_dotted_ascii_initialisms(self):
        prepared, changes = prepare_tts_text(
            "J.R.R. Tolkien visited the U.S. at 3.14 p.m. Dr. A.B. came too."
        )

        self.assertEqual(
            prepared,
            "J R R Tolkien visited the U S at 3.14 p.m. Dr. A B came too.",
        )
        self.assertEqual(changes["initialisms"], 3)

    def test_removes_numeric_citations_attached_to_sentence_endings(self):
        prepared, changes = prepare_tts_text(
            'The first claim.188 "The second claim!"4 The third? 1914.'
        )

        self.assertEqual(
            prepared,
            "The first claim. The second claim! The third? 1914.",
        )
        self.assertEqual(changes["footnotes"], 2)


if __name__ == "__main__":
    unittest.main()
