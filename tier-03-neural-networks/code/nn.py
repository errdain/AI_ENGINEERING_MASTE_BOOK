"""Tier 3 reference implementation: a small neural-network library in NumPy (manual forward and backward).

Conventions (chosen to match PyTorch so every function can be checked against it):
  * inputs are batch-first: X has shape (B, n_in)
  * Dense stores W with shape (n_out, n_in) and computes  y = X @ W.T + b
  * losses are means over the batch
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import erf

# ---------------------------------------------------------------------------------------------
# Activations and their derivatives with respect to the PRE-activation input (3.9-3.15)
# ---------------------------------------------------------------------------------------------


def sigmoid(x):
    x = np.asarray(x, float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def d_sigmoid(x): s = sigmoid(x); return s * (1 - s)
def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x) ** 2
def relu(x): return np.maximum(0.0, x)
def d_relu(x): return (np.asarray(x) > 0).astype(float)
def leaky_relu(x, a: float = 0.01): x = np.asarray(x, float); return np.where(x > 0, x, a * x)
def d_leaky_relu(x, a: float = 0.01): return np.where(np.asarray(x) > 0, 1.0, a)
def gelu(x): x = np.asarray(x, float); return 0.5 * x * (1 + erf(x / np.sqrt(2)))
def d_gelu(x):
    x = np.asarray(x, float)
    return 0.5 * (1 + erf(x / np.sqrt(2))) + x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
def silu(x): x = np.asarray(x, float); return x * sigmoid(x)
def d_silu(x): s = sigmoid(x); return s * (1 + np.asarray(x, float) * (1 - s))
def identity(x): return np.asarray(x, float)
def d_identity(x): return np.ones_like(np.asarray(x, float))


ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {
    "sigmoid": (sigmoid, d_sigmoid), "tanh": (tanh, d_tanh), "relu": (relu, d_relu), "leaky_relu": (leaky_relu, d_leaky_relu),
    "gelu": (gelu, d_gelu), "silu": (silu, d_silu), "identity": (identity, d_identity),
}


def softmax(z, axis: int = -1, temperature: float = 1.0):
    z = np.asarray(z, float) / temperature
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z, axis: int = -1):
    z = np.asarray(z, float); m = z.max(axis=axis, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=axis, keepdims=True))


# ---------------------------------------------------------------------------------------------
# Perceptron (3.2)
# ---------------------------------------------------------------------------------------------


class Perceptron:
    """Rosenblatt perceptron: prediction = step(w.x + b); update on mistakes only."""

    def __init__(self, lr: float = 1.0, max_epochs: int = 100): self.lr, self.max_epochs = lr, max_epochs

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.w_, self.b_, self.mistakes_per_epoch_ = np.zeros(X.shape[1]), 0.0, []
        for _ in range(self.max_epochs):
            mistakes = 0
            for xi, yi in zip(X, y):
                err = yi - self.predict(xi[None])[0]
                if err != 0:
                    self.w_ += self.lr * err * xi; self.b_ += self.lr * err; mistakes += 1
            self.mistakes_per_epoch_.append(mistakes)
            if mistakes == 0: self.converged_ = True; return self
        self.converged_ = False
        return self

    def predict(self, X): return (np.asarray(X, float) @ self.w_ + self.b_ > 0).astype(int)


# ---------------------------------------------------------------------------------------------
# Initialization (3.22-3.24) - defined here so layers can use it
# ---------------------------------------------------------------------------------------------


def init_weights(n_out: int, n_in: int, scheme: str = "he_normal", rng: np.random.Generator | None = None, gain: float = 1.0) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    if scheme == "zeros": return np.zeros((n_out, n_in))
    if scheme == "normal_0.01": return 0.01 * rng.standard_normal((n_out, n_in))
    if scheme == "normal_1": return rng.standard_normal((n_out, n_in))
    if scheme == "xavier_uniform":
        a = gain * np.sqrt(6.0 / (n_in + n_out)); return rng.uniform(-a, a, (n_out, n_in))
    if scheme == "xavier_normal": return gain * np.sqrt(2.0 / (n_in + n_out)) * rng.standard_normal((n_out, n_in))
    if scheme == "he_normal": return gain * np.sqrt(2.0 / n_in) * rng.standard_normal((n_out, n_in))
    raise ValueError(f"unknown init scheme {scheme}")


# ---------------------------------------------------------------------------------------------
# Layers (3.16)
# ---------------------------------------------------------------------------------------------


class Dense:
    def __init__(self, n_in: int, n_out: int, init: str = "he_normal", rng: np.random.Generator | None = None, bias: bool = True):
        self.W = init_weights(n_out, n_in, init, rng); self.b = np.zeros(n_out) if bias else None
        self.use_bias = bias; self.dW = np.zeros_like(self.W); self.db = None if self.b is None else np.zeros_like(self.b)

    def forward(self, X):
        self.X = np.asarray(X, float)
        return self.X @ self.W.T + (self.b if self.use_bias else 0.0)

    def backward(self, dY):
        self.dW = dY.T @ self.X
        if self.use_bias: self.db = dY.sum(0)
        return dY @ self.W

    def params(self): return [("W", self.W, "dW")] + ([("b", self.b, "db")] if self.use_bias else [])
    def n_params(self): return self.W.size + (self.b.size if self.use_bias else 0)


class Activation:
    def __init__(self, name: str): self.name = name; self.f, self.df = ACTIVATIONS[name]
    def forward(self, Z): self.Z = np.asarray(Z, float); return self.f(self.Z)
    def backward(self, dA): return dA * self.df(self.Z)
    def params(self): return []
    def n_params(self): return 0


# ---------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------


def softmax_cross_entropy(logits, y):
    """Mean cross-entropy from logits and integer labels; returns (loss, dlogits)."""
    B = len(y); lp = log_softmax(logits); loss = -lp[np.arange(B), y].mean()
    d = np.exp(lp); d[np.arange(B), y] -= 1
    return float(loss), d / B


def mse_loss(pred, target):
    diff = pred - target; return float((diff**2).mean()), 2 * diff / diff.size


def sigmoid_bce(logits, y):
    z = np.asarray(logits, float).ravel(); y = np.asarray(y, float).ravel()
    loss = np.mean(np.logaddexp(0, z) - y * z); d = (sigmoid(z) - y) / len(y)
    return float(loss), d.reshape(np.shape(logits))


# ---------------------------------------------------------------------------------------------
# MLP (3.17) with manual backpropagation (3.20)
# ---------------------------------------------------------------------------------------------


class MLP:
    def __init__(self, sizes: list[int], activation: str = "relu", init: str = "he_normal", seed: int = 0, bias: bool = True):
        rng = np.random.default_rng(seed); self.sizes, self.layers = list(sizes), []
        for i in range(len(sizes) - 1):
            self.layers.append(Dense(sizes[i], sizes[i + 1], init, rng, bias))
            if i < len(sizes) - 2: self.layers.append(Activation(activation))

    def forward(self, X):
        for l in self.layers: X = l.forward(X)
        return X

    def backward(self, dout):
        for l in reversed(self.layers): dout = l.backward(dout)
        return dout

    def parameters(self): return [(l, name, arr, g) for l in self.layers for name, arr, g in l.params()]
    def n_params(self): return sum(l.n_params() for l in self.layers)

    def step(self, lr: float, weight_decay: float = 0.0):
        for l, name, arr, g in self.parameters():
            arr -= lr * (getattr(l, g) + (weight_decay * arr if name == "W" else 0.0))

    def fit(self, X, y, loss: str = "softmax_ce", lr: float = 0.1, epochs: int = 200, batch_size: int | None = None, seed: int = 0, weight_decay: float = 0.0):
        fn = {"softmax_ce": softmax_cross_entropy, "mse": mse_loss, "bce": sigmoid_bce}[loss]
        rng = np.random.default_rng(seed); n = len(X); bs = batch_size or n; self.history_ = []
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, bs):
                b = idx[s:s + bs]; out = self.forward(X[b]); l, d = fn(out, y[b]); self.backward(d); self.step(lr, weight_decay)
            self.history_.append(fn(self.forward(X), y)[0])
        return self

    def predict_classes(self, X):
        out = self.forward(X); return out.argmax(1) if out.shape[1] > 1 else (out.ravel() > 0).astype(int)


def numerical_grad(f: Callable[[], float], arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of f() with respect to the entries of `arr` (modified in place and restored)."""
    g = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):
        old = arr[idx]; arr[idx] = old + eps; hi = f(); arr[idx] = old - eps; lo = f(); arr[idx] = old; g[idx] = (hi - lo) / (2 * eps)
    return g
