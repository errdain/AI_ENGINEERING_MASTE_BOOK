"""Tier 1 (1.32-1.47): covariance, Bayes, MLE, entropy, cross-entropy, KL divergence,
numerical stability (log-sum-exp), floating point, convexity and optimization landscapes.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# ---- 1.32 Covariance / correlation --------------------------------------------


def covariance_matrix(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    """X: (n_samples, n_features) -> (n_features, n_features)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] - ddof <= 0:
        raise ValueError("need a 2-D array with more samples than ddof")
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - ddof)


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    C = covariance_matrix(X)
    s = np.sqrt(np.diag(C))
    if (s == 0).any():
        raise ValueError("zero-variance feature: correlation undefined")
    return C / np.outer(s, s)


# ---- 1.33 / 1.34 Conditional probability and Bayes -------------------------------


def bayes_posterior(prior: float, p_pos_given_h: float, p_pos_given_not_h: float) -> float:
    """P(H | positive) from prior P(H), sensitivity P(+|H) and false-positive rate P(+|not H)."""
    for p in (prior, p_pos_given_h, p_pos_given_not_h):
        if not 0.0 <= p <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
    evidence = p_pos_given_h * prior + p_pos_given_not_h * (1 - prior)  # total probability
    if evidence == 0:
        raise ValueError("evidence has probability 0")
    return p_pos_given_h * prior / evidence


def posterior_categorical(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    post = np.asarray(prior, float) * np.asarray(likelihood, float)
    z = post.sum()
    if z == 0:
        raise ValueError("evidence is zero")
    return post / z


# ---- 1.35 Maximum likelihood ----------------------------------------------------


def bernoulli_nll(p: float, x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, float)
    p = float(np.clip(p, eps, 1 - eps))
    return float(-np.sum(x * np.log(p) + (1 - x) * np.log(1 - p)))


def mle_bernoulli(x: np.ndarray) -> float:
    return float(np.mean(x))


def mle_gaussian(x: np.ndarray) -> tuple[float, float]:
    """MLE mean and std (note: std uses 1/n, i.e. ddof=0, so it is biased)."""
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std())


# ---- 1.36 Entropy / 1.37 Cross-entropy / 1.38 KL --------------------------------


def _check_dist(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    if (p < 0).any() or not np.isclose(p.sum(), 1.0):
        raise ValueError("not a probability distribution")
    return p


def entropy(p: np.ndarray, base: float = 2.0) -> float:
    p = _check_dist(p)
    nz = p[p > 0]  # convention: 0 * log 0 = 0
    return float(-np.sum(nz * np.log(nz)) / np.log(base))


def cross_entropy(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    p, q = _check_dist(p), _check_dist(q)
    if ((q == 0) & (p > 0)).any():
        return float("inf")  # model says impossible, but it happened
    m = p > 0
    return float(-np.sum(p[m] * np.log(q[m])) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """KL(p || q) = cross_entropy(p, q) - entropy(p). Not symmetric."""
    ce = cross_entropy(p, q, base)
    return float("inf") if np.isinf(ce) else ce - entropy(p, base)


# ---- 1.39-1.44 Stability: log-sum-exp, softmax, cross-entropy from logits ------------


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # all -inf row -> log(0) = -inf, no NaN
    with np.errstate(divide="ignore"):  # log(0) = -inf is the correct answer for all -inf input
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, float)
    return z - logsumexp(z, axis=axis)[..., None] if axis == -1 else z - np.expand_dims(logsumexp(z, axis=axis), axis)


def naive_softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z)
    return e / e.sum()


def cross_entropy_from_logits(logits: np.ndarray, target: int) -> float:
    """-log softmax(logits)[target], computed stably."""
    return float(-log_softmax(np.asarray(logits, float))[target])


# ---- 1.45-1.47 Convexity, landscapes, minima ---------------------------------------


def midpoint_convex_violations(f: Callable[[float], float], lo: float, hi: float, n: int = 400, seed: int = 0) -> int:
    """Count random pairs where f((a+b)/2) > (f(a)+f(b))/2 + tol: any violation proves non-convexity."""
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(lo, hi, n), rng.uniform(lo, hi, n)
    viol = 0
    for x, y in zip(a, b):
        if f((x + y) / 2) > (f(x) + f(y)) / 2 + 1e-9:
            viol += 1
    return viol


def descend_1d(grad: Callable[[float], float], x0: float, lr: float = 0.01, steps: int = 5000) -> float:
    x = x0
    for _ in range(steps):
        x -= lr * grad(x)
    return x


def double_well(x: float) -> float:
    """Non-convex: two minima (one global, one local)."""
    return x**4 - 3 * x**2 + x


def double_well_grad(x: float) -> float:
    return 4 * x**3 - 6 * x + 1


def saddle(v: np.ndarray) -> float:
    """f(x, y) = x^2 - y^2: gradient is zero at the origin, but it is a saddle, not a minimum."""
    return float(v[0] ** 2 - v[1] ** 2)


if __name__ == "__main__":
    print("entropy fair coin", entropy([0.5, 0.5]))
    print("KL", kl_divergence([0.5, 0.5], [0.9, 0.1]), kl_divergence([0.9, 0.1], [0.5, 0.5]))
    print("bayes", bayes_posterior(0.01, 0.99, 0.05))
    z = np.array([1000.0, 1001.0, 1002.0])
    print("naive", naive_softmax(z), "stable", np.exp(log_softmax(z)))
    print("minima", descend_1d(double_well_grad, -2.0), descend_1d(double_well_grad, 2.0))
