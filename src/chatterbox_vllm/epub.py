"""Small, dependency-free EPUB reader and sentence-aware text chunker."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import posixpath
import re
from urllib.parse import unquote
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile


MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_TOTAL_DOCUMENT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class EpubChapter:
    title: str
    text: str
    source_path: str


@dataclass(frozen=True)
class EpubBook:
    title: str
    chapters: tuple[EpubChapter, ...]
    authors: tuple[str, ...] = ()
    language: str = ""
    publisher: str = ""
    description: str = ""
    date: str = ""
    identifier: str = ""
    cover_image: bytes | None = None
    cover_media_type: str = ""


@dataclass(frozen=True)
class TextChunk:
    chapter_index: int
    chapter_title: str
    text: str


class EpubError(ValueError):
    """Raised when an EPUB cannot be safely read as ordinary text."""


class _VisibleTextParser(HTMLParser):
    _IGNORED = {"head", "script", "style", "svg", "nav", "noscript"}
    _BLOCKS = {
        "address", "article", "aside", "blockquote", "br", "div", "figcaption",
        "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "li",
        "main", "p", "pre", "section", "table", "td", "th", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._IGNORED:
            self._ignored_depth += 1
            return
        if not self._ignored_depth and tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._ignored_depth and tag.lower() in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._IGNORED:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if not self._ignored_depth and tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n\n".join(line for line in lines if line)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_element(root: ET.Element, name: str) -> ET.Element | None:
    return next((element for element in root.iter() if _local_name(element.tag) == name), None)


_PACKAGE_METADATA_NAMES = frozenset(
    {"title", "creator", "language", "publisher", "description", "date", "identifier"}
)


def _package_metadata(root: ET.Element) -> tuple[dict[str, tuple[str, ...]], str | None]:
    """Collect spoken-book metadata and the first legacy cover reference in one pass."""

    values = {name: [] for name in _PACKAGE_METADATA_NAMES}
    cover_id = None
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "meta" and cover_id is None and element.get("name") == "cover":
            cover_id = element.get("content", "")
        elif name in values:
            value = " ".join("".join(element.itertext()).split())
            if value:
                values[name].append(value)
    return {name: tuple(items) for name, items in values.items()}, cover_id


def _safe_member_path(opf_path: str, href: str) -> str:
    decoded = unquote(href.split("#", 1)[0])
    member = posixpath.normpath(posixpath.join(posixpath.dirname(opf_path), decoded))
    if member.startswith("../") or member.startswith("/"):
        raise EpubError(f"EPUB manifest contains an unsafe path: {href}")
    return str(PurePosixPath(member))


def _read_member(archive: ZipFile, member: str) -> bytes:
    try:
        info = archive.getinfo(member)
    except KeyError as error:
        raise EpubError(f"EPUB is missing the manifest item: {member}") from error
    if info.file_size > MAX_DOCUMENT_BYTES:
        raise EpubError(f"EPUB document is unexpectedly large: {member}")
    return archive.read(info)


def _html_to_text(content: bytes) -> str:
    parser = _VisibleTextParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    parser.close()
    return parser.text()


def load_epub(path: str | Path) -> EpubBook:
    """Extract readable XHTML documents in the EPUB spine's declared order."""

    epub_path = Path(path)
    if epub_path.suffix.lower() != ".epub":
        raise EpubError("Expected a .epub file")

    try:
        archive = ZipFile(epub_path)
    except (BadZipFile, OSError) as error:
        raise EpubError("The uploaded file is not a readable EPUB archive") from error

    with archive:
        try:
            container_root = ET.fromstring(_read_member(archive, "META-INF/container.xml"))
        except (ET.ParseError, UnicodeError) as error:
            raise EpubError("EPUB container metadata is invalid") from error

        rootfile = _first_element(container_root, "rootfile")
        opf_path = rootfile.get("full-path") if rootfile is not None else None
        if not opf_path:
            raise EpubError("EPUB does not identify its package document")

        try:
            package_root = ET.fromstring(_read_member(archive, opf_path))
        except ET.ParseError as error:
            raise EpubError("EPUB package metadata is invalid") from error

        metadata, cover_id = _package_metadata(package_root)
        titles = metadata["title"]
        book_title = titles[0] if titles else ""
        if not book_title:
            book_title = epub_path.stem

        manifest: dict[str, tuple[str, str, str]] = {}
        manifest_element = _first_element(package_root, "manifest")
        if manifest_element is not None:
            for item in manifest_element:
                if _local_name(item.tag) != "item":
                    continue
                item_id = item.get("id")
                href = item.get("href")
                if item_id and href:
                    manifest[item_id] = (
                        href,
                        item.get("media-type", ""),
                        item.get("properties", ""),
                    )

        spine_ids: list[str] = []
        spine_element = _first_element(package_root, "spine")
        if spine_element is not None:
            spine_ids = [
                item.get("idref", "")
                for item in spine_element
                if _local_name(item.tag) == "itemref" and item.get("idref")
            ]

        if not spine_ids:
            spine_ids = [
                item_id
                for item_id, (_, media_type, _) in manifest.items()
                if media_type in {"application/xhtml+xml", "text/html"}
            ]

        chapters: list[EpubChapter] = []
        total_bytes = 0
        for item_id in spine_ids:
            manifest_item = manifest.get(item_id)
            if not manifest_item:
                continue
            href, media_type, _ = manifest_item
            if media_type not in {"application/xhtml+xml", "text/html", ""}:
                continue
            member = _safe_member_path(opf_path, href)
            content = _read_member(archive, member)
            total_bytes += len(content)
            if total_bytes > MAX_TOTAL_DOCUMENT_BYTES:
                raise EpubError("EPUB contains too much uncompressed document data")
            text = _html_to_text(content)
            if not text:
                continue
            first_line = text.split("\n", 1)[0]
            chapter_title = first_line if len(first_line) <= 120 else f"Chapter {len(chapters) + 1}"
            chapters.append(EpubChapter(chapter_title, text, member))

        if not chapters:
            raise EpubError("EPUB contains no readable chapters in its spine")

        cover_item = manifest.get(cover_id) if cover_id else None
        if cover_item is None:
            cover_item = next(
                (
                    item for item in manifest.values()
                    if "cover-image" in item[2].split()
                ),
                None,
            )
        if cover_item is None:
            cover_item = next(
                (
                    item for item_id, item in manifest.items()
                    if "cover" in item_id.lower() and item[1].startswith("image/")
                ),
                None,
            )
        cover_image = None
        cover_media_type = ""
        if cover_item is not None:
            cover_href, cover_media_type, _ = cover_item
            cover_image = _read_member(
                archive, _safe_member_path(opf_path, cover_href)
            )

        def first_metadata(name: str) -> str:
            values = metadata[name]
            return values[0] if values else ""

        return EpubBook(
            book_title,
            tuple(chapters),
            authors=metadata["creator"],
            language=first_metadata("language"),
            publisher=first_metadata("publisher"),
            description=first_metadata("description"),
            date=first_metadata("date"),
            identifier=first_metadata("identifier"),
            cover_image=cover_image,
            cover_media_type=cover_media_type,
        )


_ABBREVIATIONS = {
    "capt", "col", "dr", "e.g", "etc", "fig", "gen", "i.e", "jr", "lt",
    "mr", "mrs", "ms", "no", "prof", "rev", "sen", "sgt", "sr", "st", "vs",
}
_CLOSING_PUNCTUATION = '"\'”’»)]}'
_DIALOGUE_DOUBLE_QUOTES = frozenset('"“”„‟«»')
_MIN_STANDALONE_DIALOGUE_CHARS = 80


def _is_nonterminal_period(text: str, period_index: int) -> bool:
    if period_index and period_index + 1 < len(text):
        if text[period_index - 1].isdigit() and text[period_index + 1].isdigit():
            return True
    prefix = text[: period_index + 1]
    token_match = re.search(r"([A-Za-z.]+)\.$", prefix)
    token = token_match.group(1).lower() if token_match else ""
    if token in _ABBREVIATIONS:
        return True
    if re.fullmatch(r"(?:[a-z]\.){1,}[a-z]?", token + ".", flags=re.IGNORECASE):
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split prose without requiring a language model or downloaded NLP data."""

    sentences: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        start = 0
        index = 0
        while index < len(paragraph):
            if paragraph[index] not in ".!?":
                index += 1
                continue
            if paragraph[index] == "." and _is_nonterminal_period(paragraph, index):
                index += 1
                continue
            end = index + 1
            while end < len(paragraph) and paragraph[end] in ".!?":
                end += 1
            while end < len(paragraph) and paragraph[end] in _CLOSING_PUNCTUATION:
                end += 1
            if end < len(paragraph) and not paragraph[end].isspace():
                index = end
                continue
            sentence = paragraph[start:end].strip()
            if sentence:
                sentences.append(sentence)
            while end < len(paragraph) and paragraph[end].isspace():
                end += 1
            start = end
            index = end
        remainder = paragraph[start:].strip()
        if remainder:
            sentences.append(remainder)
    return sentences


def _split_oversized(text: str, max_chars: int) -> list[str]:
    clauses = re.split(r"(?<=[,;:—–])\s+", text)
    pieces: list[str] = []
    current = ""
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        candidate = f"{current} {clause}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(clause) <= max_chars:
            current = clause
            continue
        words = clause.split()
        word_group = ""
        for word in words:
            if len(word) > max_chars:
                if word_group:
                    pieces.append(word_group)
                    word_group = ""
                pieces.extend(
                    word[offset : offset + max_chars]
                    for offset in range(0, len(word), max_chars)
                )
                continue
            candidate = f"{word_group} {word}".strip()
            if word_group and len(candidate) > max_chars:
                pieces.append(word_group)
                word_group = word
            else:
                word_group = candidate
        if word_group:
            current = word_group
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int = 280) -> list[str]:
    """Group complete sentences into natural chunks below ``max_chars``."""

    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        sentence_parts = [sentence] if len(sentence) <= max_chars else _split_oversized(sentence, max_chars)
        is_dialogue_turn = (
            len(sentence) >= _MIN_STANDALONE_DIALOGUE_CHARS
            and any(mark in sentence for mark in _DIALOGUE_DOUBLE_QUOTES)
        )
        if is_dialogue_turn and current:
            chunks.append(current)
            current = ""
        for part in sentence_parts:
            if is_dialogue_turn:
                chunks.append(part)
                continue
            candidate = f"{current} {part}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_text_for_recovery(text: str) -> list[str]:
    """Split one failed speech chunk into two natural, balanced pieces."""

    normalized = " ".join(text.split())
    sentences = split_sentences(normalized)
    if len(sentences) >= 2:
        candidates = []
        for index in range(1, len(sentences)):
            left = " ".join(sentences[:index])
            right = " ".join(sentences[index:])
            candidates.append((abs(len(left) - len(right)), left, right))
        _, left, right = min(candidates, key=lambda candidate: candidate[0])
        return [left, right]

    midpoint = len(normalized) / 2
    boundaries = [
        match.end()
        for match in re.finditer(r"[,;:—–]\s+", normalized)
    ]
    if not boundaries:
        boundaries = [
            match.start()
            for match in re.finditer(r"\s+", normalized)
        ]
    if not boundaries:
        return [normalized]
    boundary = min(boundaries, key=lambda index: abs(index - midpoint))
    left = normalized[:boundary].strip()
    right = normalized[boundary:].strip()
    return [part for part in (left, right) if part]


def chunk_book(book: EpubBook, max_chars: int = 280) -> list[TextChunk]:
    """Chunk each chapter independently so chapter markers remain accurate."""

    chunks: list[TextChunk] = []
    for chapter_index, chapter in enumerate(book.chapters):
        chunks.extend(
            TextChunk(chapter_index, chapter.title, text)
            for text in chunk_text(chapter.text, max_chars=max_chars)
        )
    return chunks
