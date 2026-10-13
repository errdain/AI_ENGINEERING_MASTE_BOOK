"""Tier 5 library: sequence models from scratch (NumPy cells, checked against torch.nn) and small PyTorch models for the comparison build.

NumPy:   rnn_cell / lstm_cell / gru_cell, an unrolled RNN with BPTT, one-hot, softmax.
PyTorch: CharModel (RNN / LSTM / GRU language model), MemoryTask (copy-after-delay), Seq2Seq (encoder-decoder, optional Bahdanau / Luong attention).
Everything is CPU, seeded and small, so the Tier 5 comparisons run in seconds.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


# ----------------------------------------------------------------------------------------------- NumPy cells (weights follow torch's layout)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)


def one_hot(idx, n):
    out = np.zeros((len(idx), n)); out[np.arange(len(idx)), idx] = 1.0; return out


def rnn_cell(x, h, Wih, Whh, bih, bhh):
    return np.tanh(x @ Wih.T + bih + h @ Whh.T + bhh)


def lstm_cell(x, h, c, Wih, Whh, bih, bhh):
    """torch gate order: input, forget, cell-candidate, output."""
    z = x @ Wih.T + bih + h @ Whh.T + bhh
    i, f, g, o = np.split(z, 4, axis=-1)
    i, f, o, g = sigmoid(i), sigmoid(f), sigmoid(o), np.tanh(g)
    c_new = f * c + i * g
    return o * np.tanh(c_new), c_new, (i, f, g, o)


def gru_cell(x, h, Wih, Whh, bih, bhh):
    """torch gate order: reset, update, new."""
    gi, gh = x @ Wih.T + bih, h @ Whh.T + bhh
    ir, iz, in_ = np.split(gi, 3, axis=-1); hr, hz, hn = np.split(gh, 3, axis=-1)
    r, z = sigmoid(ir + hr), sigmoid(iz + hz)
    n = np.tanh(in_ + r * hn)
    return (1 - z) * n + z * h, (r, z, n)


class NumpyRNN:
    """Vanilla RNN language model on one sequence with manual BPTT (used to check gradients and show the vanishing signal)."""

    def __init__(self, vocab, hidden, seed=0, scale=None):
        rng = np.random.default_rng(seed); s = scale if scale is not None else 1 / np.sqrt(hidden)
        self.V, self.H = vocab, hidden
        self.Wxh = rng.normal(0, s, (hidden, vocab)); self.Whh = rng.normal(0, s, (hidden, hidden)); self.bh = np.zeros(hidden)
        self.Why = rng.normal(0, s, (vocab, hidden)); self.by = np.zeros(vocab)

    def forward(self, xs, ys):
        hs = {-1: np.zeros(self.H)}; probs = {}; loss = 0.0
        for t, (x, y) in enumerate(zip(xs, ys)):
            hs[t] = np.tanh(self.Wxh[:, x] + self.Whh @ hs[t - 1] + self.bh)
            probs[t] = softmax(self.Why @ hs[t] + self.by); loss -= np.log(probs[t][y])
        return loss, hs, probs

    def backward(self, xs, ys, hs, probs):
        g = {k: np.zeros_like(v) for k, v in dict(Wxh=self.Wxh, Whh=self.Whh, bh=self.bh, Why=self.Why, by=self.by).items()}
        dh_next = np.zeros(self.H); dh_norms = []
        for t in reversed(range(len(xs))):
            dy = probs[t].copy(); dy[ys[t]] -= 1
            g["Why"] += np.outer(dy, hs[t]); g["by"] += dy
            dh = self.Why.T @ dy + dh_next
            dz = (1 - hs[t] ** 2) * dh
            g["bh"] += dz; g["Wxh"][:, xs[t]] += dz; g["Whh"] += np.outer(dz, hs[t - 1])
            dh_next = self.Whh.T @ dz; dh_norms.append(np.linalg.norm(dh_next))
        return g, dh_norms[::-1]


    def signal_from_last_step(self, xs, ys):
        """Norm of d(loss at the LAST step)/d h_t for every earlier step t: how much learning signal travels back through time."""
        _, hs, probs = self.forward(xs, ys)
        dy = probs[len(xs) - 1].copy(); dy[ys[-1]] -= 1
        dh = self.Why.T @ dy; norms = [np.linalg.norm(dh)]
        for t in reversed(range(1, len(xs))):
            dh = self.Whh.T @ ((1 - hs[t] ** 2) * dh); norms.append(np.linalg.norm(dh))
        return norms[::-1]                                                   # index 0 = the first time step


# ----------------------------------------------------------------------------------------------- PyTorch models
class CharModel(nn.Module):
    def __init__(self, vocab, hidden=64, kind="lstm", embed=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind](embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state


def memory_batch(batch, delay, vocab=8, generator=None):
    """Copy-after-delay: [symbol, delay random symbols, QUERY] -> predict the first symbol. QUERY token id = vocab."""
    first = torch.randint(0, vocab, (batch,), generator=generator)
    noise = torch.randint(0, vocab, (batch, delay), generator=generator)
    q = torch.full((batch, 1), vocab)
    return torch.cat([first[:, None], noise, q], dim=1), first


class MemoryModel(nn.Module):
    def __init__(self, kind, vocab=8, hidden=32):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, 16)
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind](16, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)
        if kind == "lstm":                                                  # common trick: start with the forget gate open
            for name, b in self.rnn.named_parameters():
                if name.startswith("bias"):
                    b.data[hidden:2 * hidden] = 1.0

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h[:, -1])


def fit_memory(kind, delay, steps=600, seed=0, lr=0.01, clip=1.0, batch=64):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed + 1)
    model = MemoryModel(kind); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        x, y = memory_batch(batch, delay, generator=g)
        opt.zero_grad(); nn.functional.cross_entropy(model(x), y).backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
    return model


def train_memory(kind, delay, steps=600, seed=0, lr=0.01, clip=1.0, batch=64):
    model = fit_memory(kind, delay, steps, seed, lr, clip, batch)
    x, y = memory_batch(1000, delay, generator=torch.Generator().manual_seed(999))
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


# ---- sequence-to-sequence: reverse a digit string
PAD, BOS, EOS = 10, 11, 12            # digits 0-9 are tokens 0-9
V = 13


def reverse_batch(batch, length, generator=None):
    digits = torch.randint(0, 10, (batch, length), generator=generator)
    src = digits
    tgt_in = torch.cat([torch.full((batch, 1), BOS), digits.flip(1)], dim=1)
    tgt_out = torch.cat([digits.flip(1), torch.full((batch, 1), EOS)], dim=1)
    return src, tgt_in, tgt_out


class Seq2Seq(nn.Module):
    """GRU encoder-decoder. attention: None | 'bahdanau' (additive) | 'luong' (dot)."""

    def __init__(self, hidden=64, attention=None):
        super().__init__()
        self.attention, self.H = attention, hidden
        self.emb = nn.Embedding(V, 16)
        self.enc = nn.GRU(16, hidden, batch_first=True)
        self.dec = nn.GRU(16 + (hidden if attention == "bahdanau" else 0), hidden, batch_first=True)
        self.out = nn.Linear(hidden * (2 if attention else 1), V)
        if attention == "bahdanau":
            self.Wq, self.Wk, self.v = nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, 1, bias=False)

    def encode(self, src):
        return self.enc(self.emb(src))                                     # (outputs, final hidden)

    def step(self, tok, h, enc_out):
        """One decoder step. tok: (B,), h: (1,B,H). Returns logits, new h, attention weights (or None)."""
        e = self.emb(tok)[:, None]                                         # (B,1,E)
        w = ctx = None
        if self.attention == "bahdanau":                                   # score uses the PREVIOUS decoder state
            score = self.v(torch.tanh(self.Wq(h[0])[:, None] + self.Wk(enc_out))).squeeze(-1)
            w = torch.softmax(score, dim=1); ctx = (w[:, :, None] * enc_out).sum(1)
            o, h = self.dec(torch.cat([e, ctx[:, None]], dim=-1), h)
            return self.out(torch.cat([o[:, 0], ctx], dim=-1)), h, w
        o, h = self.dec(e, h)
        if self.attention == "luong":                                      # score uses the CURRENT decoder state
            score = torch.bmm(enc_out, o.transpose(1, 2)).squeeze(-1)
            w = torch.softmax(score, dim=1); ctx = (w[:, :, None] * enc_out).sum(1)
            return self.out(torch.cat([o[:, 0], ctx], dim=-1)), h, w
        return self.out(o[:, 0]), h, None

    def forward(self, src, tgt_in, teacher_forcing=1.0):
        enc_out, h = self.encode(src); logits = []; tok = tgt_in[:, 0]
        for t in range(tgt_in.size(1)):
            lg, h, _ = self.step(tok, h, enc_out); logits.append(lg)
            use_truth = (torch.rand(()) < teacher_forcing).item() and t + 1 < tgt_in.size(1)
            tok = tgt_in[:, t + 1] if use_truth else lg.argmax(-1)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy(self, src, steps):
        enc_out, h = self.encode(src); tok = torch.full((src.size(0),), BOS); out, atts = [], []
        for _ in range(steps):
            lg, h, w = self.step(tok, h, enc_out); tok = lg.argmax(-1); out.append(tok); atts.append(w)
        return torch.stack(out, 1), (torch.stack(atts, 1) if atts[0] is not None else None)


def train_seq2seq(attention, length, steps=1500, seed=0, hidden=64, teacher_forcing=1.0, lr=0.005, batch=64):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed + 1)
    model = Seq2Seq(hidden, attention); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        src, ti, to = reverse_batch(batch, length, g)
        opt.zero_grad(); nn.functional.cross_entropy(model(src, ti, teacher_forcing).reshape(-1, V), to.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def seq2seq_accuracy(model, length, n=500, seed=999):
    src, _, to = reverse_batch(n, length, torch.Generator().manual_seed(seed))
    pred, _ = model.greedy(src, length + 1)
    return (pred == to).all(dim=1).float().mean().item()


# ----------------------------------------------------------------------------------------------- word2vec (NumPy, negative sampling)
def skipgram(sentences, dim=8, window=2, epochs=60, negatives=5, lr=None, seed=0, cbow=False):
    """Tiny word2vec. Returns (vocab list, input embeddings). cbow=True predicts the centre word from the averaged context."""
    lr = lr if lr is not None else (0.5 if cbow else 0.05)         # CBOW averages the context, so its per-word step is smaller: use a larger rate
    rng = np.random.default_rng(seed)
    vocab = sorted({w for s in sentences for w in s}); idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    W_in, W_out = rng.normal(0, 0.1, (n, dim)), np.zeros((n, dim))
    pairs = []
    for s in sentences:
        ids = [idx[w] for w in s]
        for i, c in enumerate(ids):
            ctx = [ids[j] for j in range(max(0, i - window), min(len(ids), i + window + 1)) if j != i]
            if cbow:
                if ctx: pairs.append((ctx, c))
            else:
                pairs += [([c], o) for o in ctx]
    for _ in range(epochs):
        for k in rng.permutation(len(pairs)):
            ctx, target = pairs[k]
            h = W_in[ctx].mean(0)
            for label, j in [(1.0, target)] + [(0.0, int(x)) for x in rng.integers(0, n, negatives) if x != target]:
                g = (sigmoid(h @ W_out[j]) - label) * lr
                dh = g * W_out[j]; W_out[j] -= g * h
                W_in[ctx] -= dh / len(ctx)
    return vocab, W_in
