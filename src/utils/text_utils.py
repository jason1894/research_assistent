"""General text utility functions."""

import re
import unicodedata
from typing import List, Tuple


def clean_text(text: str) -> str:
    """Normalise and clean raw text.

    - Normalises Unicode to NFC.
    - Removes control characters (except newline/tab).
    - Collapses multiple spaces and blank lines.
    - Strips leading/trailing whitespace.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text string.
    """
    # Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # Remove control characters except \\n and \\t
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")

    # Collapse horizontal whitespace (preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_keywords(
    text: str,
    top_n: int = 20,
    min_length: int = 4,
    stopwords: bool = True,
) -> List[str]:
    """Extract the most frequent non-stopword tokens from text.

    This is a simple frequency-based extractor that does not require any
    external NLP libraries.

    Args:
        text: Input text.
        top_n: Maximum number of keywords to return.
        min_length: Minimum token length.
        stopwords: If True, filter common English stopwords.

    Returns:
        List of keyword strings sorted by frequency (descending).
    """
    _STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "that", "this", "these", "those",
        "it", "its", "we", "our", "they", "their", "you", "your", "i", "my",
        "which", "who", "what", "when", "where", "how", "not", "no", "also",
        "such", "can", "as", "if", "than", "then", "so", "more", "most",
    }

    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    freq: dict = {}
    for tok in tokens:
        if len(tok) < min_length:
            continue
        if stopwords and tok in _STOPWORDS:
            continue
        freq[tok] = freq.get(tok, 0) + 1

    sorted_tokens = sorted(freq, key=freq.get, reverse=True)  # type: ignore[arg-type]
    return sorted_tokens[:top_n]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """Split text into overlapping chunks.

    Tries to split on sentence boundaries (". ") first, then on
    whitespace, to avoid cutting words.

    Args:
        text: Input text.
        chunk_size: Target character size per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Prefer to break at a sentence boundary
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1
            else:
                # Fall back to word boundary
                space = text.rfind(" ", start, end)
                if space != -1:
                    end = space

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]


def format_for_training(
    query: str,
    response: str,
    instruction_prefix: str = "### Instruction:",
    response_prefix: str = "### Response:",
) -> str:
    """Format a Q&A pair as an instruction-tuning training sample.

    Args:
        query: The user query / instruction.
        response: The model response.
        instruction_prefix: Prefix label for the instruction block.
        response_prefix: Prefix label for the response block.

    Returns:
        A formatted string suitable for instruction fine-tuning.
    """
    return f"{instruction_prefix}\n{query.strip()}\n\n{response_prefix}\n{response.strip()}"


def detect_language(text: str) -> str:
    """Heuristically detect whether text is Chinese or English.

    Counts CJK characters versus ASCII letters and returns the majority.

    Args:
        text: Input text snippet (at least ~50 characters recommended).

    Returns:
        ``"zh"`` for Chinese, ``"en"`` for English, or ``"unknown"``.
    """
    if not text:
        return "unknown"

    cjk_count = sum(
        1
        for ch in text
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3400" <= ch <= "\u4dbf"
        or "\u20000" <= ch <= "\u2a6df"
    )
    ascii_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    if cjk_count == 0 and ascii_count == 0:
        return "unknown"
    if cjk_count > ascii_count:
        return "zh"
    return "en"
