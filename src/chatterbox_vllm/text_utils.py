from collections import Counter
import re
import unicodedata


_DOUBLE_QUOTES = frozenset('"“”„‟«»')
_PUNCTUATION_REPLACEMENTS = {
    "…": ",",
    ":": ",",
    ";": ",",
    "—": "-",
    "–": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "−": "-",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "؟": "?",
}
_ALLOWED_PUNCTUATION = frozenset(".,!?'-、，。｡！？")
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_NUMERIC_FOOTNOTE = re.compile(
    rf"\[\s*[0-9{_SUPERSCRIPT_DIGITS}]+\s*\]"
)
_TRAILING_SUPERSCRIPT_FOOTNOTE = re.compile(
    rf"(?<=\w)[{_SUPERSCRIPT_DIGITS}]+(?=(?:\s|$|[.,!?]))"
)
_DOTTED_INITIALISM = re.compile(r"(?<![A-Za-z])(?:[A-Z]\.){2,}")
_ATTACHED_NUMERIC_REFERENCE = re.compile(
    r"(?<!\d)([.!?](?:[\"'”’»\)\]])?)[0-9]+(?=\s|$)"
)


def clean_tts_text(text: str) -> tuple[str, Counter[str]]:
    """Return an inference-only, tokenizer-safe form of ``text`` and changes.

    Letters, numbers, and combining marks are kept for multilingual narration.
    Everything else is reduced to ordinary sentence punctuation or removed.  The
    caller owns the source text; this function never mutates project metadata.
    """

    changes: Counter[str] = Counter()

    def remove_numeric_footnote(match: re.Match[str]) -> str:
        changes["footnotes"] += 1
        return " "

    text = _NUMERIC_FOOTNOTE.sub(remove_numeric_footnote, text)
    text = _TRAILING_SUPERSCRIPT_FOOTNOTE.sub(remove_numeric_footnote, text)

    def remove_attached_reference(match: re.Match[str]) -> str:
        changes["footnotes"] += 1
        return match.group(1)

    text = _ATTACHED_NUMERIC_REFERENCE.sub(remove_attached_reference, text)

    def expand_initialism(match: re.Match[str]) -> str:
        changes["initialisms"] += 1
        return " ".join(match.group(0).replace(".", "").strip())

    text = _DOTTED_INITIALISM.sub(expand_initialism, text)

    cleaned: list[str] = []
    for character in text:
        if character in _DOUBLE_QUOTES:
            changes["quotes"] += 1
            continue
        replacement = _PUNCTUATION_REPLACEMENTS.get(character)
        if replacement is not None:
            cleaned.append(replacement)
            changes["punctuation"] += 1
            continue
        if character in _ALLOWED_PUNCTUATION:
            cleaned.append(character)
            continue
        if character.isspace():
            cleaned.append(" ")
            if character != " ":
                changes["whitespace"] += 1
            continue

        category = unicodedata.category(character)
        if category[0] in {"L", "N", "M"}:
            cleaned.append(character)
        elif category[0] == "C":
            changes["controls"] += 1
        else:
            # Symbols and unsupported punctuation often separate words (for
            # example, slashes and bullets).  Keep that boundary audible.
            cleaned.append(" ")
            changes["symbols"] += 1

    result = "".join(cleaned)
    without_space_before_punctuation = re.sub(r"\s+([.,!?])", r"\1", result)
    if without_space_before_punctuation != result:
        changes["whitespace"] += 1
    return without_space_before_punctuation, changes


def prepare_tts_text(text: str) -> tuple[str, Counter[str]]:
    """Clean text immediately before inference, retaining existing punc cleanup."""

    cleaned, changes = clean_tts_text(text)
    normalized = punc_norm(cleaned)
    if normalized != cleaned:
        changes["normalization"] += 1
    return normalized, changes


def punc_norm(text: str) -> str:
    """Normalize whitespace and uncommon punctuation for the speech model."""
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}
