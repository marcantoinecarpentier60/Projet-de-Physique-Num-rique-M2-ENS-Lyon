"""Word-boundary helpers, multi-script punctuation stripping, and repetition masks."""

from __future__ import annotations

import unicodedata
from typing import Sequence, Set

import numpy as np

NO_SPACE_LANGS = {"zh", "ja"}
FINAL_PUNCT_CHARS = set(".!?。！？．…।॥؟؛")
TRAILING_WRAP_CHARS = set(" \t\n\r\"'”’»」』)]")


def strip_final_punct(text: str) -> str:
    """Remove sentence-final punctuation across Latin, CJK, Devanagari, and Arabic scripts.

    Also strips closing quotes/brackets immediately preceded by final punctuation
    (including French thin non-breaking spaces before closing guillemets).
    """
    s = text.strip()
    while s:
        trimmed = s.rstrip(" \t\n\r\u00a0\u202f")
        if not trimmed:
            return ""
        if trimmed[-1] in FINAL_PUNCT_CHARS:
            s = trimmed[:-1]
        elif trimmed[-1] in TRAILING_WRAP_CHARS:
            inner = trimmed[:-1].rstrip(" \t\n\r\u00a0\u202f")
            if inner and inner[-1] in FINAL_PUNCT_CHARS:
                s = inner[:-1]
            else:
                return trimmed
        else:
            return trimmed
    return s


def is_punct_char(ch: str) -> bool:
    """Return True if ``ch`` belongs to any Unicode punctuation category (P*)."""
    return bool(ch) and unicodedata.category(ch).startswith("P")


def split_into_word_spans(
    token_ids: Sequence[int],
    bow_ids: Set[int],
) -> list[tuple[int, int]]:
    """Group subword indices into word spans ``(start, end)`` using bow-marking tokens.

    Parameters
    ----------
    token_ids : Sequence[int]
        Encoded token IDs of a target sentence.
    bow_ids : set[int]
        Set of vocabulary IDs that mark the beginning of a word (e.g. ``▁`` or ``Ġ``).

    Returns
    -------
    list[tuple[int, int]]
        Half-open index intervals ``[start, end)`` corresponding to individual words.
    """
    if not token_ids:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(token_ids)):
        if token_ids[idx] in bow_ids:
            spans.append((start, idx))
            start = idx
    spans.append((start, len(token_ids)))
    return spans


def repeated_ngram_char_mask(target: str, context: str, n: int = 4) -> np.ndarray:
    """Build a boolean character mask flagging characters in n-grams copied from context.

    Use ``n=4`` for space-delimited scripts (Wilcox et al., 2025) and ``n=2``
    for logographic scripts without spaces (``zh``, ``ja``).
    """
    mask = np.zeros(len(target), dtype=bool)
    if n <= 0 or len(target) < n or not context:
        return mask
    for pos in range(len(target) - n + 1):
        sub = target[pos : pos + n]
        if not sub.isspace() and sub in context:
            mask[pos : pos + n] = True
    return mask


def get_spm_boundary_sets(
    sp,
    special_ids: tuple[int, ...] = (0, 1, 2, 3),
    eos_id: int = 3,
) -> tuple[set[int], list[int], list[int]]:
    """Extract Pimentel & Meister boundary sets from a SentencePiece processor.

    Returns
    -------
    bow_only : set[int]
        Tokens marking word starts (``▁`` or space/tab/newline bytes).
    b_bow : list[int]
        ``V_bow U {EOS}`` used for internal word boundaries and full sentences.
    b_eow : list[int]
        ``V_bow U {EOS} U V_punct`` used at the right edge when final punctuation
        was stripped from the target sentence.
    """
    bow_only: set[int] = set()
    punct_nobow: set[int] = set()
    for idx in range(sp.get_piece_size()):
        piece = sp.id_to_piece(idx)
        if piece.startswith("\u2581") or piece in ("<0x20>", "<0x0A>", "<0x09>"):
            bow_only.add(idx)
        elif piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
            bval = int(piece[3:5], 16)
            if 33 <= bval <= 126 and is_punct_char(chr(bval)):
                punct_nobow.add(idx)
        elif idx not in special_ids and piece and is_punct_char(piece[0]):
            punct_nobow.add(idx)

    b_bow = sorted(bow_only | {eos_id})
    b_eow = sorted(bow_only | {eos_id} | punct_nobow)
    return bow_only, b_bow, b_eow
