"""Tier 2 reference implementations: a tiny ML training / evaluation laboratory in NumPy.

Everything here is written for clarity, not speed. scikit-learn is used ONLY in the tests, as an
independent reference to check these implementations against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np

# ---------------------------------------------------------------------------------------------
# Data generators (2.7-2.10)
# ---------------------------------------------------------------------------------------------


def make_blobs_binary(n: int = 400, sep: float = 2.0, pos_frac: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian clouds in 2-D. y=1 is the 'positive' class with fraction pos_frac."""
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * pos_frac))
    X = np.vstack([rng.standard_normal((n - n_pos, 2)), rng.standard_normal((n_pos, 2)) + [sep, sep]])
    y = np.r_[np.zeros(n - n_pos), np.ones(n_pos)].astype(int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_linear(n: int = 200, w: Sequence[float] = (2.0, -1.0), b: float = 0.5, noise: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(w)))
    y = X @ np.asarray(w) + b + noise * rng.standard_normal(n)
    return X, y


def make_sine(n: int = 30, noise: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    return x, np.sin(np.pi * x) + noise * rng.standard_normal(n)


# ---------------------------------------------------------------------------------------------
# Splitting (2.11-2.13, 2.44)
# ---------------------------------------------------------------------------------------------


def train_val_test_indices(n: int, fractions=(0.6, 0.2, 0.2), seed: int = 0, stratify: np.ndarray | None = None):
    """Disjoint index arrays; with `stratify` each class keeps (about) its global proportion."""
    if not np.isclose(sum(fractions), 1.0) or min(fractions) <= 0:
        raise ValueError("fractions must be positive and sum to 1")
    rng = np.random.default_rng(seed)
    groups = [np.arange(n)] if stratify is None else [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    parts = [[], [], []]
    for g in groups:
        g = rng.permutation(g)
        a = int(round(fractions[0] * len(g))); b = a + int(round(fractions[1] * len(g)))
        parts[0].append(g[:a]); parts[1].append(g[a:b]); parts[2].append(g[b:])
    return tuple(rng.permutation(np.concatenate(p)) for p in parts)


def kfold_indices(n: int, k: int = 5, seed: int = 0, stratify: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """List of (train_idx, val_idx). Every sample is in exactly one validation fold."""
    if not 2 <= k <= n:
        raise ValueError("need 2 <= k <= n")
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    groups = [np.arange(n)] if stratify is None else [np.flatnonzero(stratify == c) for c in np.unique(stratify)]
    offset = 0
    for g in groups:
        g = rng.permutation(g)
        for j, i in enumerate(g):
            folds[(j + offset) % k].append(int(i))
        offset += len(g)
    out = []
    for f in range(k):
        val = np.array(sorted(folds[f]))
        out.append((np.setdiff1d(np.arange(n), val), val))
    return out


# ---------------------------------------------------------------------------------------------
# Preprocessing (2.47-2.49)
# ---------------------------------------------------------------------------------------------


class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_, self.scale_ = X.mean(0), X.std(0)
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def fit(self, X):
        X = np.asarray(X, float)
        self.min_, rng = X.min(0), X.max(0) - X.min(0)
        self.range_ = np.where(rng < 1e-12, 1.0, rng)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.min_) / self.range_


def impute_median(X_train: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Replace NaN by the TRAINING median of each column."""
    med = np.nanmedian(X_train, axis=0)
    out = np.array(X, float, copy=True)
    r, c = np.where(np.isnan(out))
    out[r, c] = med[c]
    return out


def one_hot(codes: np.ndarray, n_categories: int) -> np.ndarray:
    codes = np.asarray(codes, int)
    if codes.min() < 0 or codes.max() >= n_categories:
        raise ValueError("category code out of range (unseen category?)")
    return np.eye(n_categories)[codes]


def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """[1, x, x^2, ..., x^degree] for a 1-D input."""
    return np.vander(np.asarray(x, float), degree + 1, increasing=True)


# ---------------------------------------------------------------------------------------------
# Losses / objectives (2.17, 2.18)
# ---------------------------------------------------------------------------------------------


def mse(y, yhat): return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))
def mae(y, yhat): return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def huber(y, yhat, delta: float = 1.0) -> float:
    r = np.abs(np.asarray(y) - np.asarray(yhat))
    return float(np.mean(np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))))


def sigmoid(z):
    z = np.asarray(z, float)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def bce_from_logits(y, z) -> float:
    """Mean binary cross-entropy, stable: softplus(z) - y*z."""
    y, z = np.asarray(y, float), np.asarray(z, float)
    return float(np.mean(np.logaddexp(0, z) - y * z))


# ---------------------------------------------------------------------------------------------
# Models: baselines, linear / logistic regression, k-means (2.14-2.16, 2.46)
# ---------------------------------------------------------------------------------------------


class MajorityBaseline:
    def fit(self, X, y):
        vals, counts = np.unique(y, return_counts=True)
        self.label_ = vals[np.argmax(counts)]
        self.prior_ = dict(zip(vals.tolist(), (counts / counts.sum()).tolist()))
        return self

    def predict(self, X): return np.full(len(X), self.label_)


class MeanBaseline:
    def fit(self, X, y): self.mean_ = float(np.mean(y)); return self
    def predict(self, X): return np.full(len(X), self.mean_)


class LinearRegression:
    """Least squares (optionally ridge) with an unpenalized intercept; solved by lstsq / normal equations."""

    def __init__(self, ridge: float = 0.0): self.ridge = ridge

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        A = np.hstack([np.ones((len(X), 1)), X])
        if self.ridge == 0:
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            P = self.ridge * np.eye(A.shape[1]); P[0, 0] = 0.0
            beta = np.linalg.solve(A.T @ A + P, A.T @ y)
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        return self

    def predict(self, X): return np.asarray(X, float) @ self.coef_ + self.intercept_


class LogisticRegression:
    """Binary logistic regression by full-batch gradient descent on the (weighted, L2-penalized) NLL."""

    def __init__(self, lr: float = 0.5, n_iter: int = 2000, l2: float = 0.0, class_weight: dict | None = None):
        self.lr, self.n_iter, self.l2, self.class_weight = lr, n_iter, l2, class_weight

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        sw = np.ones(len(y)) if self.class_weight is None else np.where(y == 1, self.class_weight.get(1, 1.0), self.class_weight.get(0, 1.0))
        sw = sw / sw.mean()
        w, b = np.zeros(X.shape[1]), 0.0
        self.losses_: list[float] = []
        for _ in range(self.n_iter):
            z = X @ w + b
            r = (sigmoid(z) - y) * sw
            w -= self.lr * (X.T @ r / len(y) + self.l2 * w)
            b -= self.lr * r.mean()
            self.losses_.append(float(np.mean(sw * (np.logaddexp(0, z) - y * z))))
        self.coef_, self.intercept_ = w, float(b)
        return self

    def decision_function(self, X): return np.asarray(X, float) @ self.coef_ + self.intercept_
    def predict_proba(self, X): return sigmoid(self.decision_function(X))
    def predict(self, X, threshold: float = 0.5): return (self.predict_proba(X) >= threshold).astype(int)


class KMeans:
    def __init__(self, k: int, n_init: int = 5, max_iter: int = 100, seed: int = 0):
        self.k, self.n_init, self.max_iter, self.seed = k, n_init, max_iter, seed

    def _run(self, X, rng):
        C = X[rng.choice(len(X), self.k, replace=False)].copy()
        for _ in range(self.max_iter):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else X[rng.integers(len(X))] for j in range(self.k)])
            if np.allclose(newC, C): break
            C = newC
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        return C, d.argmin(1), float(d.min(1).sum())

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        best = min((self._run(X, rng) for _ in range(self.n_init)), key=lambda t: t[2])
        self.centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        return (((np.asarray(X, float)[:, None, :] - self.centers_[None]) ** 2).sum(-1)).argmin(1)


# ---------------------------------------------------------------------------------------------
# Training loops: epochs / batches / iterations (2.19-2.25)
# ---------------------------------------------------------------------------------------------


def iterations_per_epoch(n_samples: int, batch_size: int, drop_last: bool = False) -> int:
    if batch_size <= 0 or n_samples <= 0:
        raise ValueError("n_samples and batch_size must be positive")
    return n_samples // batch_size if drop_last else -(-n_samples // batch_size)


def minibatches(n: int, batch_size: int, rng: np.random.Generator | None = None, drop_last: bool = False) -> Iterator[np.ndarray]:
    idx = np.arange(n) if rng is None else rng.permutation(n)
    for s in range(0, n, batch_size):
        b = idx[s:s + batch_size]
        if drop_last and len(b) < batch_size:
            return
        yield b


def sgd_linear(X, y, lr: float = 0.05, batch_size: int | None = None, epochs: int = 50, seed: int = 0, shuffle: bool = True):
    """(Mini-batch) SGD on mean-squared error. batch_size=None -> full batch GD, 1 -> pure SGD.

    Returns (w, b, history) where history has per-epoch loss and the running iteration count.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y); bs = n if batch_size is None else batch_size
    rng = np.random.default_rng(seed)
    w, b, it = np.zeros(X.shape[1]), 0.0, 0
    hist = {"epoch_loss": [], "iterations": []}
    for _ in range(epochs):
        for idx in minibatches(n, bs, rng if shuffle else None):
            r = X[idx] @ w + b - y[idx]
            w -= lr * 2 * X[idx].T @ r / len(idx)
            b -= lr * 2 * r.mean()
            it += 1
            if not np.isfinite(w).all():
                hist["epoch_loss"].append(float("inf")); hist["iterations"].append(it)
                return w, b, hist
        hist["epoch_loss"].append(mse(y, X @ w + b)); hist["iterations"].append(it)
    return w, b, hist


# ---------------------------------------------------------------------------------------------
# Metrics (2.37-2.43)
# ---------------------------------------------------------------------------------------------


def confusion_matrix_binary(y, yhat) -> np.ndarray:
    """[[TN, FP], [FN, TP]]"""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    return np.array([[np.sum((y == 0) & (yhat == 0)), np.sum((y == 0) & (yhat == 1))],
                     [np.sum((y == 1) & (yhat == 0)), np.sum((y == 1) & (yhat == 1))]])


def _safe_div(a, b, zero=0.0): return zero if b == 0 else a / b


def accuracy(y, yhat): return float(np.mean(np.asarray(y) == np.asarray(yhat)))


def precision(y, yhat, zero_division: float = 0.0):
    (tn, fp), (fn, tp) = confusion_matrix_binary(y, yhat); return _safe_div(tp, tp + fp, zero_division)


def recall(y, yhat, zero_division: float = 0.0):
    (tn, fp), (fn, tp) = confusion_matrix_binary(y, yhat); return _safe_div(tp, tp + fn, zero_division)


def fbeta(y, yhat, beta: float = 1.0):
    p, r = precision(y, yhat), recall(y, yhat)
    return _safe_div((1 + beta**2) * p * r, beta**2 * p + r)


def f1(y, yhat): return fbeta(y, yhat, 1.0)


def roc_curve(y, score):
    """Return (fpr, tpr, thresholds) sweeping the threshold from high to low."""
    y, score = np.asarray(y, int), np.asarray(score, float)
    order = np.argsort(-score, kind="mergesort")
    y, s = y[order], score[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of each distinct score
    tps, fps = np.cumsum(y)[distinct], (1 + distinct) - np.cumsum(y)[distinct]
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        raise ValueError("ROC needs both classes present")
    return np.r_[0, fps / N], np.r_[0, tps / P], np.r_[np.inf, s[distinct]]


def auc_trapezoid(x, y): return float(np.sum(np.diff(x) * (np.asarray(y)[1:] + np.asarray(y)[:-1]) / 2))


def roc_auc(y, score) -> float:
    fpr, tpr, _ = roc_curve(y, score); return auc_trapezoid(fpr, tpr)


def roc_auc_rank(y, score) -> float:
    """Mann-Whitney form: P(score_pos > score_neg) + 0.5 P(tie), via average ranks."""
    y, s = np.asarray(y, int), np.asarray(score, float)
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks over ties
        m = s == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    P, N = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def reliability_bins(y, p, n_bins: int = 10):
    """Per-bin (count, mean predicted prob, observed positive rate) over equal-width probability bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return [(int((b == i).sum()), float(p[b == i].mean()) if (b == i).any() else np.nan, float(y[b == i].mean()) if (b == i).any() else np.nan)
            for i in range(n_bins)]


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    bins, n = reliability_bins(y, p, n_bins), len(y)
    return float(sum(c / n * abs(conf - acc) for c, conf, acc in bins if c > 0))


def brier_score(y, p): return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


# ---------------------------------------------------------------------------------------------
# Imbalance (2.45, 2.50)
# ---------------------------------------------------------------------------------------------


def balanced_class_weights(y) -> dict:
    vals, counts = np.unique(y, return_counts=True)
    return {int(v): float(len(y) / (len(vals) * c)) for v, c in zip(vals, counts)}


def random_oversample(X, y, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals, counts = np.unique(y, return_counts=True)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == v), counts.max(), replace=True) for v in vals])
    return X[idx], y[idx]


def best_f1_threshold(y, score):
    ts = np.unique(score)
    f = [f1(y, (score >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(f))]), float(max(f))


# ---------------------------------------------------------------------------------------------
# Distribution shift (2.34)
# ---------------------------------------------------------------------------------------------


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with bin edges from the *expected* (training) quantiles."""
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.maximum(e, eps), np.maximum(a, eps)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b); allv = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(a, allv, side="right") / len(a) - np.searchsorted(b, allv, side="right") / len(b))))


# ---------------------------------------------------------------------------------------------
# Hyperparameter search (2.35, 2.36)
# ---------------------------------------------------------------------------------------------


def grid_search(objective: Callable[[dict], float], grid: dict[str, Sequence]) -> tuple[dict, float, list]:
    import itertools
    keys = list(grid)
    hist = [({k: v for k, v in zip(keys, combo)}, objective(dict(zip(keys, combo)))) for combo in itertools.product(*grid.values())]
    best = min(hist, key=lambda t: t[1])
    return best[0], best[1], hist


def random_search(objective: Callable[[dict], float], space: dict[str, Callable[[np.random.Generator], float]], n_trials: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    hist = []
    for _ in range(n_trials):
        params = {k: s(rng) for k, s in space.items()}
        hist.append((params, objective(params)))
    best = min(hist, key=lambda t: t[1])
    return best[0], best[1], hist


# ---------------------------------------------------------------------------------------------
# Bias-variance (2.26-2.31)
# ---------------------------------------------------------------------------------------------


def bias_variance_poly(degree: int, f_true: Callable[[np.ndarray], np.ndarray], n_train: int = 30, noise: float = 0.2,
                       n_datasets: int = 300, ridge: float = 0.0, seed: int = 0, fixed_design: bool = True) -> dict:
    """Monte-Carlo bias^2 / variance of a polynomial regressor on a fixed test grid.

    fixed_design=True keeps the training inputs on a regular grid so variance comes only from label noise
    (the textbook setting). With random inputs, extra variance appears from the sampling of x itself.
    """
    rng = np.random.default_rng(seed)
    xt = np.linspace(-1, 1, 101)
    preds = np.empty((n_datasets, len(xt)))
    for i in range(n_datasets):
        x = np.linspace(-1, 1, n_train) if fixed_design else rng.uniform(-1, 1, n_train)
        y = f_true(x) + noise * rng.standard_normal(n_train)
        m = LinearRegression(ridge).fit(poly_features(x, degree)[:, 1:], y)
        preds[i] = m.predict(poly_features(xt, degree)[:, 1:])
    mean_pred = preds.mean(0)
    bias2 = float(np.mean((mean_pred - f_true(xt)) ** 2)); var = float(np.mean(preds.var(0)))
    return {"bias2": bias2, "variance": var, "noise": noise**2, "expected_test_mse": bias2 + var + noise**2}


@dataclass
class FitResult:
    train_mse: float
    val_mse: float


def fit_poly_train_val(degree: int, x_tr, y_tr, x_va, y_va, ridge: float = 0.0) -> FitResult:
    m = LinearRegression(ridge).fit(poly_features(x_tr, degree)[:, 1:], y_tr)
    return FitResult(mse(y_tr, m.predict(poly_features(x_tr, degree)[:, 1:])), mse(y_va, m.predict(poly_features(x_va, degree)[:, 1:])))


# ---------------------------------------------------------------------------------------------
# Additional metrics: multiclass confusion matrix / averaging, average precision, time-series CV (2.40-2.44)
# ---------------------------------------------------------------------------------------------


def confusion_matrix_multiclass(y, yhat, k: int | None = None) -> np.ndarray:
    """cm[i, j] = number of samples with true class i predicted as class j."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    k = int(max(y.max(), yhat.max()) + 1) if k is None else k
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, yhat), 1)
    return cm


def precision_recall_f1_per_class(cm: np.ndarray):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return p, r, f


def f1_macro(cm: np.ndarray) -> float: return float(precision_recall_f1_per_class(cm)[2].mean())


def f1_micro(cm: np.ndarray) -> float:
    tp = np.diag(cm).sum(); return float(tp / cm.sum())  # for single-label multiclass micro-F1 == accuracy


def average_precision(y, score) -> float:
    """Area under the precision-recall curve as sum over thresholds of (R_n - R_{n-1}) * P_n (scikit-learn's definition)."""
    y, s = np.asarray(y, int), np.asarray(score, float)
    order = np.argsort(-s, kind="mergesort"); y, s = y[order], s[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[distinct]; fp = (1 + distinct) - tp
    prec, rec = tp / (tp + fp), tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def time_series_splits(n: int, n_splits: int = 4, min_train: int | None = None):
    """Expanding-window splits: always train on the past, validate on the next block."""
    min_train = min_train or n // (n_splits + 1)
    block = (n - min_train) // n_splits
    return [(np.arange(0, min_train + i * block), np.arange(min_train + i * block, min_train + (i + 1) * block)) for i in range(n_splits)]
