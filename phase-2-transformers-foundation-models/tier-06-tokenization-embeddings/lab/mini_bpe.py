"""Miniature byte-level BPE tokenizer (Tier 6 build).

Pure Python, no dependencies. Works on UTF-8 bytes, so every string is
encodable (no UNK) -- the same idea GPT-2-style tokenizers use.

    tok = ByteBPE.train(corpus, num_merges=50)
    ids = tok.encode("low lower lowest")
    assert tok.decode(ids) == "low lower lowest"
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]  # ids 0..3, reserved


def pair_counts(seqs: list[list[int]], weights: list[int]) -> Counter:
    """Count adjacent pairs across sequences (weighted by word frequency)."""
    c: Counter = Counter()
    for seq, w in zip(seqs, weights):
        for a, b in zip(seq, seq[1:]):
            c[(a, b)] += w
    return c


def merge_pair(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` (left to right)."""
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


@dataclass
class ByteBPE:
    merges: list[tuple[int, int]] = field(default_factory=list)  # learned, in order
    offset: int = len(SPECIALS)  # byte b -> id b + offset

    # ---- vocabulary -------------------------------------------------------
    @property
    def vocab(self) -> dict[int, bytes]:
        v: dict[int, bytes] = {i + self.offset: bytes([i]) for i in range(256)}
        for k, (a, b) in enumerate(self.merges):
            v[self.offset + 256 + k] = v[a] + v[b]
        return v

    @property
    def vocab_size(self) -> int:
        return self.offset + 256 + len(self.merges)

    # ---- training ---------------------------------------------------------
    @classmethod
    def train(cls, text: str, num_merges: int) -> "ByteBPE":
        """Learn `num_merges` merges. Pre-tokenizes on whitespace so merges
        never cross word boundaries (a space stays attached to the next word)."""
        tok = cls()
        words = Counter(_pretokenize(text))
        seqs = [[b + tok.offset for b in w.encode("utf-8")] for w in words]
        weights = list(words.values())
        for k in range(num_merges):
            counts = pair_counts(seqs, weights)
            if not counts:
                break
            # ties broken deterministically: highest count, then smallest pair
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            if counts[best] < 2:  # nothing worth merging
                break
            new_id = tok.offset + 256 + len(tok.merges)
            tok.merges.append(best)
            seqs = [merge_pair(s, best, new_id) for s in seqs]
        return tok

    # ---- encode / decode --------------------------------------------------
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        for word in _pretokenize(text):
            seq = [b + self.offset for b in word.encode("utf-8")]
            # apply merges in the order they were learned (priority = rank)
            for k, pair in enumerate(self.merges):
                if len(seq) < 2:
                    break
                seq = merge_pair(seq, pair, self.offset + 256 + k)
            ids.extend(seq)
        if add_special:
            ids = [SPECIALS.index("<BOS>")] + ids + [SPECIALS.index("<EOS>")]
        return ids

    def decode(self, ids: list[int]) -> str:
        v = self.vocab
        data = b"".join(v[i] for i in ids if i >= self.offset)  # skip specials
        return data.decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        """Human-readable token strings (for visualisation)."""
        v = self.vocab
        return [v[i].decode("utf-8", errors="replace") for i in self.encode(text)]


def _pretokenize(text: str) -> list[str]:
    """Split into words, keeping the leading space on each word (GPT-2 style)."""
    out, cur = [], ""
    for ch in text:
        if ch.isspace() and cur and not cur[-1].isspace():
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def compression_ratio(tok: ByteBPE, text: str) -> float:
    """UTF-8 bytes per token (higher = better compression)."""
    return len(text.encode("utf-8")) / max(1, len(tok.encode(text)))


if __name__ == "__main__":
    corpus = "low lower lowest low low newer newest wider widest " * 20
    t = ByteBPE.train(corpus, 20)
    print("vocab size:", t.vocab_size)
    print(t.tokens("lowest newer"))
    print("bytes/token:", round(compression_ratio(t, corpus), 2))
