"""Small, dependency-free companions to mini_bpe.py (Tier 6 concept files).

* wordpiece_encode  -- greedy longest-match-first (BERT-style, '##' continuations)
* unigram_segment   -- Viterbi best segmentation under a unigram LM
* split_digits      -- regex-free digit-by-digit pre-tokenizer
* fertility / tokens_per_char -- efficiency metrics
* normalize         -- Unicode normalization wrapper
* batch_pad         -- BOS/EOS/PAD batching with attention mask
"""
from __future__ import annotations

import math
import unicodedata

import numpy as np


def normalize(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text)


def wordpiece_encode(word: str, vocab: set[str], unk: str = "[UNK]") -> list[str]:
    """Greedy longest-match-first. Non-initial pieces carry the '##' prefix.
    If any position cannot be matched the *whole word* becomes [UNK] (BERT behaviour)."""
    pieces, start = [], 0
    while start < len(word):
        end, cur = len(word), None
        while start < end:
            sub = word[start:end] if start == 0 else "##" + word[start:end]
            if sub in vocab:
                cur = sub
                break
            end -= 1
        if cur is None:
            return [unk]
        pieces.append(cur)
        start = end
    return pieces


def unigram_segment(text: str, logp: dict[str, float]) -> list[str]:
    """Viterbi: the segmentation of `text` maximising sum of token log-probs.
    Returns [] if `text` cannot be segmented with the given vocabulary."""
    n = len(text)
    best = [-math.inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    maxlen = max(map(len, logp))
    for i in range(1, n + 1):
        for j in range(max(0, i - maxlen), i):
            piece = text[j:i]
            if piece in logp and best[j] + logp[piece] > best[i]:
                best[i], back[i] = best[j] + logp[piece], j
    if best[n] == -math.inf:
        return []
    out, i = [], n
    while i > 0:
        out.append(text[back[i]:i])
        i = back[i]
    return out[::-1]


def split_digits(text: str) -> list[str]:
    """Pre-tokenizer that isolates every digit; other runs stay together."""
    out, cur = [], ""
    for ch in text:
        if ch.isdigit():
            if cur:
                out.append(cur)
                cur = ""
            out.append(ch)
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def tokens_per_char(n_tokens: int, text: str) -> float:
    return n_tokens / max(1, len(text))


def fertility(n_tokens: int, text: str) -> float:
    """Tokens per whitespace-separated word."""
    return n_tokens / max(1, len(text.split()))


def batch_pad(seqs: list[list[int]], bos: int, eos: int, pad: int, side: str = "right"):
    """Add BOS/EOS, pad to a rectangle. Returns (ids, mask) as int arrays."""
    rows = [[bos] + s + [eos] for s in seqs]
    L = max(map(len, rows))
    ids = np.full((len(rows), L), pad, dtype=int)
    mask = np.zeros((len(rows), L), dtype=int)
    for r, row in enumerate(rows):
        sl = slice(0, len(row)) if side == "right" else slice(L - len(row), L)
        ids[r, sl], mask[r, sl] = row, 1
    return ids, mask
