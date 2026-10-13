"""Tier 15 metric library (numpy only). Every formula is small enough to hand-check; see test_tier15.py."""
from __future__ import annotations
import math
import numpy as np


# ---------- classification metrics (15.9, 15.10) ----------
def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def confusion_counts(y_true, y_pred, positive=1) -> dict:
    """Binary confusion counts for the `positive` label."""
    t = np.asarray(y_true) == positive
    p = np.asarray(y_pred) == positive
    return {"tp": int((t & p).sum()), "fp": int((~t & p).sum()),
            "fn": int((t & ~p).sum()), "tn": int((~t & ~p).sum())}


def prf1(y_true, y_pred, positive=1) -> dict:
    """precision = tp/(tp+fp), recall = tp/(tp+fn), F1 = harmonic mean. 0.0 when a denominator is 0."""
    c = confusion_counts(y_true, y_pred, positive)
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, **c}


def pr_curve(y_true, scores):
    """Precision/recall at every distinct threshold (predict positive if score >= t), thresholds descending."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    ts = np.unique(scores)[::-1]
    out = []
    for t in ts:
        m = prf1(y_true, (scores >= t).astype(int))
        out.append((float(t), m["precision"], m["recall"]))
    return out


# ---------- language-model metrics (15.5) ----------
def perplexity(token_nlls) -> float:
    """PPL = exp(mean negative log-likelihood in nats per token)."""
    return float(math.exp(np.mean(token_nlls)))


def perplexity_from_probs(probs) -> float:
    """PPL from the probabilities the model gave to the true tokens."""
    return perplexity(-np.log(np.asarray(probs, dtype=float)))


# ---------- calibration (15.11) ----------
def softmax(logits, T: float = 1.0):
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def reliability_bins(conf, correct, n_bins: int = 10):
    """Equal-width bins on [0,1]. Returns list of (lo, hi, count, mean_conf, accuracy)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.minimum(np.digitize(conf, edges[1:-1], right=True), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append((float(edges[b]), float(edges[b + 1]), n,
                     float(conf[m].mean()) if n else 0.0, float(correct[m].mean()) if n else 0.0))
    return rows


def ece(conf, correct, n_bins: int = 10) -> float:
    """Expected Calibration Error = sum_b (n_b/N) * |acc_b - conf_b|."""
    rows = reliability_bins(conf, correct, n_bins)
    N = sum(r[2] for r in rows)
    return float(sum(r[2] / N * abs(r[4] - r[3]) for r in rows if r[2]))


def brier(probs, y_onehot) -> float:
    probs, y = np.asarray(probs, float), np.asarray(y_onehot, float)
    return float(((probs - y) ** 2).sum(axis=1).mean())


def nll(logits, y, T: float = 1.0) -> float:
    p = softmax(logits, T)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def fit_temperature(logits, y, grid=None) -> float:
    """Temperature scaling: pick T minimising validation NLL (grid search keeps it dependency-free)."""
    grid = np.linspace(0.25, 4.0, 76) if grid is None else grid
    return float(min(grid, key=lambda T: nll(logits, y, T)))


# ---------- uncertainty on a benchmark score (15.6, 15.33) ----------
def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for stat(values), resampling examples with replacement."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    s = np.array([stat(v[i]) for i in idx])
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Closed-form CI for a proportion (accuracy)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired_bootstrap(scores_a, scores_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap on per-example scores of two models on the SAME examples.
    Returns mean diff (A-B), 95% CI and P(A>B) estimated over resamples."""
    d = np.asarray(scores_a, float) - np.asarray(scores_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(m, .025)),
            "ci_high": float(np.quantile(m, .975)), "p_a_better": float((m > 0).mean())}


# ---------- human agreement (15.27) ----------
def cohen_kappa(r1, r2) -> float:
    """kappa = (po - pe)/(1 - pe); po observed agreement, pe agreement expected by chance."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    labels = np.unique(np.concatenate([r1, r2]))
    po = (r1 == r2).mean()
    pe = sum((r1 == l).mean() * (r2 == l).mean() for l in labels)
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


# ---------- fairness (15.18, 15.19) ----------
def group_rates(y_true, y_pred, group) -> dict:
    """Per group: selection rate P(pred=1), TPR, FPR."""
    y_true, y_pred, group = map(np.asarray, (y_true, y_pred, group))
    out = {}
    for g in np.unique(group):
        m = group == g
        t, p = y_true[m], y_pred[m]
        out[g.item() if hasattr(g, "item") else g] = {
            "n": int(m.sum()), "selection": float(p.mean()),
            "tpr": float(p[t == 1].mean()) if (t == 1).any() else float("nan"),
            "fpr": float(p[t == 0].mean()) if (t == 0).any() else float("nan")}
    return out


def demographic_parity_diff(y_pred, group) -> float:
    r = [v["selection"] for v in group_rates(np.zeros(len(y_pred)), y_pred, group).values()]
    return float(max(r) - min(r))


def equalized_odds_diff(y_true, y_pred, group) -> float:
    """max over (TPR gap, FPR gap)."""
    g = group_rates(y_true, y_pred, group).values()
    tp = [v["tpr"] for v in g if not math.isnan(v["tpr"])]
    fp = [v["fpr"] for v in g if not math.isnan(v["fpr"])]
    return float(max(max(tp) - min(tp), max(fp) - min(fp)))


# ---------- contamination (15.21) ----------
def ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(test_text: str, train_texts, n: int = 4) -> float:
    """Fraction of the test item's n-grams that occur anywhere in the training corpus."""
    t = ngrams(test_text.lower().split(), n)
    if not t:
        return 0.0
    tr = set()
    for s in train_texts:
        tr |= ngrams(s.lower().split(), n)
    return len(t & tr) / len(t)
