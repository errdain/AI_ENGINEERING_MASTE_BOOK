"""Tier 14 lab - toy CTC speech recogniser trained on synthetic 'phoneme' tones (numpy + torch, CPU).

Each symbol 'a'..'f' is a two-tone burst (a fake formant pair).  An utterance is a sequence of bursts
separated by quiet gaps.  A small Conv+BiGRU maps log-mel frames to per-frame label posteriors and is
trained with CTC loss (no frame-level labels).  Decoding: greedy, prefix beam search, forced alignment.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_core import log_mel, cmvn, tone, noise

SR = 8000
SYMBOLS = "abcdef"
BLANK = 0
VOCAB = ["<b>"] + list(SYMBOLS)           # index 0 = CTC blank
FREQS = {"a": (400, 1000), "b": (700, 1900), "c": (1000, 3000), "d": (1500, 2200), "e": (2300, 350), "f": (3200, 1500)}
HOP = 80                                  # 10 ms at 8 kHz
N_MELS = 24


def synth_utterance(text: str, rng: np.random.Generator, sr: int = SR, snr_noise: float = 0.05):
    """Returns (waveform, true_frame_labels[n_frames]) where label 0 = silence/gap."""
    parts, labels = [], []
    lead = int(rng.integers(2, 5))
    parts.append(np.zeros(lead * HOP)); labels += [0] * lead
    for ch in text:
        nfr = int(rng.integers(7, 13)); dur = nfr * HOP / sr
        f1, f2 = FREQS[ch]; jit = rng.uniform(0.97, 1.03)
        seg = 0.6 * tone(f1 * jit, dur, sr) + 0.4 * tone(f2 * jit, dur, sr)
        seg *= np.hanning(len(seg)) ** 0.3                       # soft edges
        parts.append(seg); labels += [VOCAB.index(ch)] * nfr
        gap = int(rng.integers(2, 4)); parts.append(np.zeros(gap * HOP)); labels += [0] * gap
    x = np.concatenate(parts) + noise(len(np.concatenate(parts)) / sr, sr, snr_noise, seed=int(rng.integers(1 << 30)))[: len(np.concatenate(parts))]
    return x.astype(np.float32), np.array(labels)


def make_dataset(n: int, seed: int = 0, min_len: int = 3, max_len: int = 6):
    rng = np.random.default_rng(seed); data = []
    for _ in range(n):
        text = "".join(rng.choice(list(SYMBOLS), int(rng.integers(min_len, max_len + 1))))
        x, lab = synth_utterance(text, rng)
        feats = cmvn(log_mel(x, SR, n_mels=N_MELS, win=256, hop=HOP)).astype(np.float32)
        data.append((feats, text, x, lab))
    return data


class FramewiseCTC(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden: int = 48, vocab: int = len(VOCAB), seed: int = 0):
        super().__init__(); torch.manual_seed(seed)
        self.conv = nn.Conv1d(n_mels, hidden, 5, padding=2)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, vocab)

    def forward(self, x):                      # x [B, T, n_mels] -> log-probs [B, T, V]
        h = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        h, _ = self.rnn(h)
        return F.log_softmax(self.out(h), dim=-1)


def collate(batch):
    T = max(len(b[0]) for b in batch)
    X = torch.zeros(len(batch), T, N_MELS); lens = torch.tensor([len(b[0]) for b in batch])
    for i, b in enumerate(batch): X[i, :len(b[0])] = torch.from_numpy(b[0])
    tgt = [torch.tensor([VOCAB.index(c) for c in b[1]]) for b in batch]
    return X, lens, torch.cat(tgt), torch.tensor([len(t) for t in tgt])


def train(model: FramewiseCTC, data, epochs: int = 12, lr: float = 5e-3, bs: int = 16, seed: int = 0) -> list[float]:
    torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed); losses = []
    for _ in range(epochs):
        order = rng.permutation(len(data)); tot = 0.0
        for i in range(0, len(data), bs):
            X, il, tg, tl = collate([data[j] for j in order[i:i + bs]])
            lp = model(X).transpose(0, 1)                       # CTC wants [T, B, V]
            loss = F.ctc_loss(lp, tg, il, tl, blank=BLANK, zero_infinity=True)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(tg)
        losses.append(tot / len(data))
    return losses

# ---------------------------------------------------------------- CTC decoding (14.10) -------------

def ctc_collapse(path) -> str:
    """The CTC many-to-one map B: merge repeated labels, then drop blanks."""
    out, prev = [], None
    for p in path:
        if p != prev and p != BLANK: out.append(VOCAB[p])
        prev = p
    return "".join(out)


def greedy_decode(logp: np.ndarray) -> str:
    return ctc_collapse(logp.argmax(-1))


def _logsumexp(a, b):
    m = max(a, b)
    return m if m == -np.inf else m + np.log(np.exp(a - m) + np.exp(b - m))


def prefix_beam_search(logp: np.ndarray, beam: int = 8) -> str:
    """CTC prefix beam search (no language model). Keeps, per prefix, log-prob of ending in blank / non-blank."""
    T, V = logp.shape
    beams = {"": (0.0, -np.inf)}                                  # prefix -> (p_blank, p_nonblank) in log
    for t in range(T):
        nxt: dict[str, list[float]] = {}
        def add(pref, pb, pnb):
            cur = nxt.setdefault(pref, [-np.inf, -np.inf])
            cur[0] = _logsumexp(cur[0], pb); cur[1] = _logsumexp(cur[1], pnb)
        for pref, (pb, pnb) in beams.items():
            ptot = _logsumexp(pb, pnb)
            add(pref, ptot + logp[t, BLANK], -np.inf)             # emit blank
            for v in range(1, V):
                p = logp[t, v]; ch = VOCAB[v]
                if pref and pref[-1] == ch:
                    add(pref, -np.inf, pnb + p)                   # repeat without blank: same prefix
                    add(pref + ch, -np.inf, pb + p)               # repeat after blank: new char
                else:
                    add(pref + ch, -np.inf, ptot + p)
        beams = {k: tuple(v) for k, v in sorted(nxt.items(), key=lambda kv: -_logsumexp(*kv[1]))[:beam]}
    return max(beams.items(), key=lambda kv: _logsumexp(*kv[1]))[0]


def forced_align(logp: np.ndarray, text: str) -> list[tuple[str, int, int]]:
    """Viterbi forced alignment of a known transcript to frames (14.15).
    Returns [(char, start_frame, end_frame_exclusive)] using the CTC topology (blank between labels)."""
    ids = [VOCAB.index(c) for c in text]
    ext = [BLANK]
    for i in ids: ext += [i, BLANK]
    S, T = len(ext), len(logp)
    dp = np.full((T, S), -np.inf); bp = np.zeros((T, S), dtype=int)
    dp[0, 0] = logp[0, ext[0]]
    if S > 1: dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(dp[t - 1, s], s)]
            if s >= 1: cands.append((dp[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]: cands.append((dp[t - 1, s - 2], s - 2))
            best, arg = max(cands); dp[t, s] = best + logp[t, ext[s]]; bp[t, s] = arg
    s = S - 1 if S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s; s = bp[t, s]
    spans: dict[int, list[int]] = {}
    for t, s in enumerate(path):
        if ext[s] != BLANK: spans.setdefault(s, []).append(t)
    return [(VOCAB[ext[s]], f[0], f[-1] + 1) for s, f in sorted(spans.items())]


def transcribe(model: FramewiseCTC, feats: np.ndarray, decoder: str = "greedy") -> str:
    with torch.no_grad():
        lp = model(torch.from_numpy(feats)[None])[0].numpy()
    return greedy_decode(lp) if decoder == "greedy" else prefix_beam_search(lp)


def streaming_greedy(model: FramewiseCTC, feats: np.ndarray, chunk: int = 20, context: int = 10) -> tuple[str, list[int]]:
    """Chunked pseudo-streaming: run the (bi-directional) model on each chunk plus `context` look-back frames and
    commit only the chunk's own frames.  Returns (text, number-of-frames-seen-when-each-char-was-emitted)."""
    committed: list[int] = []
    for start in range(0, len(feats), chunk):
        lo = max(0, start - context); hi = min(len(feats), start + chunk)
        with torch.no_grad():
            lp = model(torch.from_numpy(feats[lo:hi])[None])[0].numpy()
        committed += list(lp.argmax(-1)[start - lo:])
    return ctc_collapse(committed), committed
