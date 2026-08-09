import base64
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from chatterbox_vllm.epub import EpubError, chunk_book, chunk_text, load_epub, split_sentences


CONTAINER = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles>
</container>
"""

PACKAGE = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title><dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language><dc:publisher>Test Publisher</dc:publisher>
    <dc:identifier>book-id</dc:identifier><meta name="cover" content="cover"/>
  </metadata>
  <manifest>
    <item id="second" href="second.xhtml" media-type="application/xhtml+xml"/>
    <item id="first" href="first.xhtml" media-type="application/xhtml+xml"/>
    <item id="cover" href="cover.png" media-type="image/png"/>
  </manifest>
  <spine><itemref idref="first"/><itemref idref="second"/></spine>
</package>
"""


def make_epub(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        archive.writestr("META-INF/container.xml", CONTAINER, compress_type=ZIP_DEFLATED)
        archive.writestr("OEBPS/content.opf", PACKAGE, compress_type=ZIP_DEFLATED)
        archive.writestr(
            "OEBPS/first.xhtml",
            "<html><head><title>Hidden</title><style>bad</style></head><body>"
            "<nav>Skip this</nav><h1>First Chapter</h1><p>Dr. Rivera arrived. She sat down.</p>"
            "<script>ignored()</script></body></html>",
            compress_type=ZIP_DEFLATED,
        )
        archive.writestr(
            "OEBPS/cover.png",
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
            compress_type=ZIP_DEFLATED,
        )
        archive.writestr(
            "OEBPS/second.xhtml",
            "<html><body><h2>Second Chapter</h2><p>Another sentence!</p></body></html>",
            compress_type=ZIP_DEFLATED,
        )


class EpubTests(unittest.TestCase):
    def test_loads_spine_order_and_ignores_non_spoken_elements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.epub"
            make_epub(path)
            book = load_epub(path)

        self.assertEqual(book.title, "Test Book")
        self.assertEqual(book.authors, ("Test Author",))
        self.assertEqual(book.language, "en")
        self.assertEqual(book.publisher, "Test Publisher")
        self.assertEqual(book.identifier, "book-id")
        self.assertEqual(book.cover_media_type, "image/png")
        self.assertTrue(book.cover_image.startswith(b"\x89PNG"))
        self.assertEqual([chapter.title for chapter in book.chapters], ["First Chapter", "Second Chapter"])
        self.assertIn("Dr. Rivera arrived.", book.chapters[0].text)
        self.assertNotIn("Skip this", book.chapters[0].text)
        self.assertNotIn("ignored", book.chapters[0].text)

    def test_sentence_splitter_preserves_abbreviations_and_quotes(self):
        sentences = split_sentences('Dr. Rivera arrived at 3.5 p.m. "Sit down!" Then she left.')
        self.assertEqual(
            sentences,
            ['Dr. Rivera arrived at 3.5 p.m. "Sit down!"', "Then she left."],
        )

    def test_chunks_group_sentences_without_exceeding_limit(self):
        text = "One short sentence. Two short sentences. " + "word " * 50 + "done."
        chunks = chunk_text(text, max_chars=100)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertTrue(chunks[0].startswith("One short sentence. Two short sentences."))

    def test_chunks_pathologically_long_word_without_exceeding_limit(self):
        chunks = chunk_text("A prefix " + ("x" * 205) + ".", max_chars=100)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_chunks_do_not_cross_chapter_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.epub"
            make_epub(path)
            chunks = chunk_book(load_epub(path), max_chars=100)
        self.assertEqual({chunk.chapter_index for chunk in chunks}, {0, 1})
        self.assertTrue(all(chunk.chapter_title for chunk in chunks))

    def test_rejects_non_epub_input(self):
        with self.assertRaises(EpubError):
            load_epub("book.txt")


if __name__ == "__main__":
    unittest.main()
