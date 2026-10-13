"""Tier 1 (1.20-1.31): functions, derivatives, gradients, Jacobians, chain rule,
probability basics, distributions, expectation, variance, standard deviation.

Numerical derivatives exist to *check* analytic ones (gradient checking), not to train models.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Fn = Callable[[np.ndarray], np.ndarray | float]

# ---- 1.21 Derivatives ---------------------------------------------------------


def derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Central difference: error O(h^2); forward difference would be O(h)."""
    return (f(x + h) - f(x - h)) / (2 * h)


# ---- 1.22/1.23 Partial derivatives and gradients --------------------------------


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e.flat[i] = h
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def gradient_descent(grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, lr: float, steps: int) -> np.ndarray:
    x = np.asarray(x0, dtype=np.float64).copy()
    for _ in range(steps):
        x = x - lr * grad(x)
    return x


# ---- 1.24 Jacobians -------------------------------------------------------------


def numerical_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """J[i, j] = d f_i / d x_j, shape (m, n) for f: R^n -> R^m."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(f(x)).size
    J = np.zeros((m, x.size))
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        J[:, j] = (np.asarray(f(x + e)) - np.asarray(f(x - e))) / (2 * h)
    return J


# ---- 1.25 Chain rule: a 2-layer net with manual backprop ------------------------


def tiny_net_forward_backward(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, y: float):
    """L = (W2 . tanh(W1 x) - y)^2.  Returns loss, dL/dW1, dL/dW2 via the chain rule."""
    z1 = W1 @ x                 # (h,)
    a1 = np.tanh(z1)            # (h,)
    yhat = float(W2 @ a1)       # scalar
    loss = (yhat - y) ** 2
    d_yhat = 2 * (yhat - y)                 # dL/dyhat
    dW2 = d_yhat * a1                       # dL/dW2 = dL/dyhat * dyhat/dW2
    da1 = d_yhat * W2                       # dL/da1
    dz1 = da1 * (1 - a1 ** 2)               # tanh'(z) = 1 - tanh^2
    dW1 = np.outer(dz1, x)                  # dL/dW1
    return loss, dW1, dW2


# ---- 1.26-1.28 Probability, random variables, distributions ---------------------


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=-1, keepdims=True))  # shift for stability (see 1.39/1.43)
    return e / e.sum(axis=-1, keepdims=True)


def normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def bernoulli_pmf(k: int, p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if k not in (0, 1):
        return 0.0
    return p if k == 1 else 1.0 - p


# ---- 1.29-1.31 Expectation, variance, std --------------------------------------


def expectation(values: np.ndarray, probs: np.ndarray) -> float:
    values, probs = np.asarray(values, float), np.asarray(probs, float)
    if not np.isclose(probs.sum(), 1.0) or (probs < 0).any():
        raise ValueError("probs must be non-negative and sum to 1")
    return float(np.sum(values * probs))


def variance(values: np.ndarray, probs: np.ndarray) -> float:
    mu = expectation(values, probs)
    return expectation((np.asarray(values, float) - mu) ** 2, probs)  # E[(X-mu)^2]


def sample_variance(x: np.ndarray, ddof: int = 1) -> float:
    x = np.asarray(x, float)
    if x.size - ddof <= 0:
        raise ValueError("not enough samples for this ddof")
    return float(np.sum((x - x.mean()) ** 2) / (x.size - ddof))


class Welford:
    """Streaming mean/variance in one pass, numerically stable."""

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def var(self) -> float:  # sample variance
        return self.m2 / (self.n - 1) if self.n > 1 else float("nan")

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))


if __name__ == "__main__":
    print("d/dx x^3 at 2:", derivative(lambda x: x**3, 2.0))
    print("grad:", numerical_gradient(lambda v: v[0] ** 2 + 3 * v[0] * v[1], np.array([1.0, 2.0])))
    w = Welford()
    for v in [2, 4, 4, 4, 5, 5, 7, 9]:
        w.update(v)
    print("welford", w.mean, w.var, w.std)
