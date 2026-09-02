import logging
import os
from typing import List
import json
from unicodedata import category, normalize

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)


# Model repository
REPO_ID = "ResembleAI/chatterbox"

# Global instances for optional dependencies
# _kakasi = None
_dicta = None
_russian_stresser = None


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta
    
    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            _dicta = Dicta()
        
        return _dicta.add_diacritics(text)
        
    except ImportError:
        logger.warning("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        logger.warning(f"Hebrew diacritization failed: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""
    
    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        
        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        
        return initial + medial + final
    
    # Decompose syllables and normalize punctuation
    result = ''.join(decompose_hangul(char) for char in text)    
    return result.strip()


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""
    
    def __init__(self):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping()
        self._init_segmenter()
    
    def _load_cangjie_mapping(self):
        """Load Cangjie mapping from HuggingFace model repository."""        
        try:
            cangjie_file = hf_hub_download(
                repo_id=REPO_ID,
                filename="Cangjie5_TC.json",
            )
            
            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)
                    
        except Exception as e:
            logger.warning(f"Could not load Cangjie mapping: {e}")
    
    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from spacy_pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            logger.warning("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None
    
    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)
    
    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text
        
        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = []
                for c in cangjie:
                    code.append(f"[cj_{c}]")
                code.append("[cj_.]")
                code = "".join(code)
                output.append(code)
            else:
                output.append(t)
        return "".join(output)


def add_russian_stress(text: str) -> str:
    """Russian text normalization: adds stress marks to Russian text."""
    global _russian_stresser
    try:
        if _russian_stresser is None:
            from russian_text_stresser.text_stresser import RussianTextStresser
            _russian_stresser = RussianTextStresser()
        return _russian_stresser.stress_text(text)
    except ImportError:
        logger.warning("russian_text_stresser not available - Russian stress labeling skipped")
        return text
    except Exception as e:
        logger.warning(f"Russian stress labeling failed: {e}")
        return text


class MTLTokenizer(PreTrainedTokenizer):
    """
    A VLLM-compatible tokenizer that wraps the original MTLTokenizer implementation.
    """
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file_path: str,
        unk_token: str = UNK,
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        self.cangjie_converter = ChineseCangjieConverter()
        self.check_vocabset_sot_eot()

    @classmethod
    def from_pretrained(cls, **kwargs):
        """
        Instantiate a tokenizer from a pretrained model or path.
        
        Args:
            pretrained_model_name_or_path: Path to the tokenizer file or model name
            **kwargs: Additional arguments to pass to the tokenizer
        """
        # Prefer the tokenizer downloaded alongside the selected multilingual
        # checkpoint. The packaged file remains as a backward-compatible
        # fallback for callers that construct this tokenizer directly.
        vocab_file = os.environ.get("CHATTERBOX_MTL_TOKENIZER_FILE")
        if not vocab_file:
            vocab_file = os.path.join(
                os.path.dirname(__file__),
                "grapheme_mtl_merged_expanded_v1.json",
            )
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"Multilingual tokenizer is missing: {vocab_file}")
        return cls(vocab_file_path=vocab_file, **kwargs)

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def preprocess_text(self, raw_text: str, language_id: str = None, lowercase: bool = True, nfkd_normalize: bool = True):
        """
        Text preprocessor that handles lowercase conversion and NFKD normalization.
        """
        preprocessed_text = raw_text
        if lowercase:
            preprocessed_text = preprocessed_text.lower()
        if nfkd_normalize:
            preprocessed_text = normalize("NFKD", preprocessed_text)
        
        return preprocessed_text

    def _tokenize(self, text: str, **kwargs) -> List[str]:        
        # Parse out language token if it exists
        # This is injected by the ChatterboxTTS.generate_with_conds method
        language_id = None
        if text.startswith('<'):
            language_id = text.split('<')[1].split('>')[0]
            text = text.split('>')[1]

        # The vLLM adapter places the text boundary markers in the prompt
        # string. Preserve them while normalizing the spoken content: changing
        # `[STOP]` to `[stop]` makes it ordinary grapheme text and can cause the
        # model to literally say "stop" at the end of an utterance.
        has_start_token = text.startswith(SOT)
        has_stop_token = text.endswith(EOT)
        if has_start_token:
            text = text[len(SOT):]
        if has_stop_token:
            text = text[:-len(EOT)]

        text = self.preprocess_text(text, language_id)
        
        # Language-specific text processing
        if language_id == 'zh':
            text = self.cangjie_converter(text)
        elif language_id == 'ja':
            text = hiragana_normalize(text)
        elif language_id == 'he':
            text = add_hebrew_diacritics(text)
        elif language_id == 'ko':
            text = korean_normalize(text)
        elif language_id == 'ru':
            text = add_russian_stress(text)
        
        # Prepend language token again
        if language_id:
            text = f"[{language_id.lower()}]{text}"
        if has_start_token:
            text = SOT + text
        if has_stop_token:
            text += EOT

        text = text.replace(' ', SPACE)
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = text.replace(' ', '')
        text = text.replace(SPACE, ' ')
        text = text.replace(EOT, '')
        text = text.replace(UNK, '')
        return text

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    @property
    def max_token_id(self) -> int:
        return max(self.tokenizer.get_vocab().values())
