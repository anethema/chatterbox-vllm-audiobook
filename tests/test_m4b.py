import unittest

from chatterbox_vllm.epub import EpubBook
from chatterbox_vllm.m4b import ChapterMarker, build_ffmetadata


class M4BTests(unittest.TestCase):
    def test_builds_global_metadata_and_chapter_markers(self):
        book = EpubBook(
            "A Book",
            (),
            authors=("One Author", "Two Author"),
            language="en",
            publisher="Example; Press",
        )
        metadata = build_ffmetadata(
            book,
            [ChapterMarker("Chapter #1", 0, 1250), ChapterMarker("Two", 1250, 2500)],
        )
        self.assertIn("artist=One Author, Two Author", metadata)
        self.assertIn("publisher=Example\\; Press", metadata)
        self.assertIn("START=1250", metadata)
        self.assertIn("title=Chapter \\#1", metadata)
        self.assertEqual(metadata.count("[CHAPTER]"), 2)


if __name__ == "__main__":
    unittest.main()
