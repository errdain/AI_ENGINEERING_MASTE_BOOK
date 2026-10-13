"""Synthetic stand-ins so Tier 15 needs no downloads/API keys: controllable classifier, bigram LM, biased judge."""
from __future__ import annotations
import math
import numpy as np
from metrics import softmax


def make_classifier_logits(n=2000, k=3, signal=2.0, sharpen=1.0, seed=0):
    """Return (logits, y). `signal` controls accuracy; `sharpen`>1 makes the model overconfident
    (logits multiplied), <1 underconfident. Accuracy is unchanged by sharpen (argmax is invariant)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += signal
    return logits * sharpen, y


class BigramLM:
    """Add-alpha smoothed bigram LM over whitespace tokens: the smallest honest 'language model' for perplexity."""
    def __init__(self, text: str, alpha: float = 0.1):
        toks = ["<s>"] + text.lower().split()
        self.vocab = sorted(set(toks) | {"<unk>"})
        self.alpha = alpha
        self.c = {}
        self.ctx = {}
        for a, b in zip(toks, toks[1:]):
            self.c[(a, b)] = self.c.get((a, b), 0) + 1
            self.ctx[a] = self.ctx.get(a, 0) + 1

    def prob(self, a, b):
        a = a if a in self.vocab else "<unk>"
        b = b if b in self.vocab else "<unk>"
        return (self.c.get((a, b), 0) + self.alpha) / (self.ctx.get(a, 0) + self.alpha * len(self.vocab))

    def token_nlls(self, text: str):
        toks = ["<s>"] + text.lower().split()
        return [-math.log(self.prob(a, b)) for a, b in zip(toks, toks[1:])]


class MockJudge:
    """Pairwise LLM-judge with injectable biases. P(pick A) = sigmoid(k*(qA-qB) + position_bias + verbosity_bias*(lenA-lenB)/100).
    position_bias>0 favours the first slot; verbosity_bias>0 favours the longer answer."""
    def __init__(self, position_bias=0.0, verbosity_bias=0.0, sharpness=4.0, seed=0):
        self.pb, self.vb, self.k = position_bias, verbosity_bias, sharpness
        self.rng = np.random.default_rng(seed)

    def p_first(self, qa, qb, la=100, lb=100):
        z = self.k * (qa - qb) + self.pb + self.vb * (la - lb) / 100
        return 1 / (1 + math.exp(-z))

    def prefers_first(self, qa, qb, la=100, lb=100):
        return bool(self.rng.random() < self.p_first(qa, qb, la, lb))


def swap_consistency(judge: MockJudge, qa, qb, la, lb, n=400):
    """Run each pair in both orders; return (consistency, win-rate of the first-listed slot)."""
    cons = first = 0
    for _ in range(n):
        a_first = judge.prefers_first(qa, qb, la, lb)     # A in slot 1
        b_first = judge.prefers_first(qb, qa, lb, la)     # B in slot 1
        a_wins_1, a_wins_2 = a_first, not b_first
        cons += a_wins_1 == a_wins_2
        first += a_first + b_first
    return cons / n, first / (2 * n)


def noisy_accuracy(base_acc: float, noise: float, rng_seed=0, n=4000):
    """Toy robustness model: each input is solved with prob base_acc*(1-noise)+ (1-...)*chance (chance=0.25)."""
    rng = np.random.default_rng(rng_seed)
    p = base_acc * (1 - noise) + 0.25 * noise
    return float((rng.random(n) < p).mean())
