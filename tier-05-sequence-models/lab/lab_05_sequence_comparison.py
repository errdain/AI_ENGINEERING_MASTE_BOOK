"""Tier 5 lab: char RNN -> LSTM -> attention Seq2Seq, compared on tasks that expose what each model can and cannot do.

    cd tier-05-sequence-models && PYTHONPATH=code python lab/lab_05_sequence_comparison.py
Each ex_* returns a dict; seeds are fixed. Numbers in the concept pages come from the same library (code/seqmodels.py).
"""
import numpy as np
import torch
from torch import nn

import seqmodels as S

def make_corpus(n_sentences=600, seed=0):
    """Random sentences from a tiny grammar, so there is real uncertainty to model (not a memorizable loop)."""
    rng = np.random.default_rng(seed); A = ["cat", "dog", "bird", "fox"]; V = ["saw", "chased", "liked"]; M = ["", " near the mat", " on the log"]
    return " ".join(f"the {A[rng.integers(4)]} {V[rng.integers(3)]} the {A[rng.integers(4)]}{M[rng.integers(3)]}." for _ in range(n_sentences))


def ex_5_10_char_language_models(steps=300, hidden=48, seed=0):
    """Next-character prediction with a vanilla RNN, an LSTM and a GRU (same size, same data, same steps)."""
    text = make_corpus(); chars = sorted(set(text)); idx = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([idx[c] for c in text]); T = 40; split = int(0.8 * len(data)); train, test = data[:split], data[split:]
    out = {}
    for kind in ("rnn", "lstm", "gru"):
        torch.manual_seed(seed); m = S.CharModel(len(chars), hidden, kind); opt = torch.optim.Adam(m.parameters(), lr=0.01)
        g = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            starts = torch.randint(0, len(train) - T - 1, (32,), generator=g)
            x = torch.stack([train[s:s + T] for s in starts]); y = torch.stack([train[s + 1:s + T + 1] for s in starts])
            opt.zero_grad(); nn.functional.cross_entropy(m(x)[0].reshape(-1, len(chars)), y.reshape(-1)).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        starts = torch.arange(0, len(test) - T - 1, 7)
        x = torch.stack([test[s:s + T] for s in starts]); y = torch.stack([test[s + 1:s + T + 1] for s in starts])
        with torch.no_grad():
            out[kind] = nn.functional.cross_entropy(m(x)[0].reshape(-1, len(chars)), y.reshape(-1)).item()
    counts = np.ones((len(chars), len(chars)))                                  # bigram baseline (previous character only)
    for a, b in zip(train[:-1].tolist(), train[1:].tolist()):
        counts[a, b] += 1
    P = counts / counts.sum(1, keepdims=True)
    out["bigram_baseline"] = float(-np.mean([np.log(P[a, b]) for a, b in zip(test[:-1].tolist(), test[1:].tolist())]))
    out["uniform_guess"] = float(np.log(len(chars)))
    return out


def ex_5_13_gradient_signal(length=40):
    """Norm of d(last-step loss)/d h_t, relative to the last step, for different recurrent-weight scales."""
    xs = list(np.random.default_rng(0).integers(0, 4, length)); ys = xs[1:] + [0]
    res = {}
    for scale in (0.2, 0.5, 0.8, 1.2):
        n = S.NumpyRNN(4, 16, 0, scale=scale).signal_from_last_step(xs, ys)
        res[scale] = [n[i] / n[-1] for i in range(length)]
    return res


def ex_5_14_memory_vs_delay(delays=(5, 10, 20, 30, 40), steps=1500):
    return {kind: {d: S.train_memory(kind, d, steps=steps) for d in delays} for kind in ("rnn", "lstm", "gru")}


def ex_5_21_seq2seq_vs_length(lengths=(4, 8, 16), steps=1200):
    res = {}
    for att in (None, "bahdanau", "luong"):
        res[att or "none"] = {L: S.seq2seq_accuracy(S.train_seq2seq(att, L, steps=steps), L) for L in lengths}
    return res


def ex_5_22_teacher_forcing(length=8, steps=1200):
    """Exact-match accuracy of a plain Seq2Seq trained with different teacher-forcing probabilities, decoding freely."""
    return {tf: S.seq2seq_accuracy(S.train_seq2seq(None, length, steps=steps, teacher_forcing=tf), length) for tf in (1.0, 0.5, 0.0)}


def ex_5_23_exposure_bias(length=12, steps=1200):
    """A model trained with full teacher forcing: per-token accuracy when fed the TRUE previous tokens vs when fed its OWN."""
    m = S.train_seq2seq(None, length, steps=steps, teacher_forcing=1.0)
    src, ti, to = S.reverse_batch(500, length, torch.Generator().manual_seed(999))
    with torch.no_grad():
        forced = (m(src, ti, teacher_forcing=1.0).argmax(-1) == to).float().mean().item()
        free = (m(src, ti, teacher_forcing=0.0).argmax(-1) == to).float().mean().item()
    return {"per_token_accuracy_with_true_history": forced, "per_token_accuracy_with_own_history": free}


def ex_5_26_sequential_depth(lengths=(10, 100, 1000)):
    """Operations that cannot run in parallel: an RNN needs T steps in a row; self-attention needs O(1) layers (but T^2 comparisons)."""
    return {T: {"rnn_sequential_steps": T, "attention_sequential_steps": 1, "attention_pairwise_scores": T * T, "path_between_far_tokens_rnn": T - 1, "path_between_far_tokens_attention": 1} for T in lengths}


if __name__ == "__main__":
    for fn in (ex_5_10_char_language_models, ex_5_13_gradient_signal, ex_5_14_memory_vs_delay, ex_5_21_seq2seq_vs_length, ex_5_22_teacher_forcing, ex_5_23_exposure_bias, ex_5_26_sequential_depth):
        print(fn.__name__, fn())
