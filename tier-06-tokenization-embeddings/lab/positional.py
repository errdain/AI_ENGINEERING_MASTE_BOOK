"""Positional-information building blocks in NumPy (Tier 6.20-6.25)."""
from __future__ import annotations

import numpy as np


def sinusoidal(n_pos: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Original-Transformer sinusoidal table, shape (n_pos, d_model). d_model even."""
    assert d_model % 2 == 0
    pos = np.arange(n_pos)[:, None]
    freq = base ** (-np.arange(0, d_model, 2) / d_model)  # (d/2,)
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    return pe


def rope_angles(n_pos: int, d_head: int, base: float = 10000.0) -> np.ndarray:
    """Rotation angle theta_{m,i} = m * base^(-2i/d); shape (n_pos, d_head/2)."""
    freq = base ** (-np.arange(0, d_head, 2) / d_head)
    return np.arange(n_pos)[:, None] * freq


def apply_rope(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate consecutive pairs of x (n_pos, d_head) by position-dependent angles."""
    n, d = x.shape
    ang = rope_angles(n, d, base)
    cos, sin = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
    out[:, 1::2] = x[:, 0::2] * sin + x[:, 1::2] * cos
    return out


def alibi_slopes(n_heads: int) -> np.ndarray:
    """Geometric slopes 2^(-8/n * (h+1)) (n_heads power of two)."""
    start = 2 ** (-8 / n_heads)
    return start ** np.arange(1, n_heads + 1)


def alibi_bias(n_pos: int, slope: float) -> np.ndarray:
    """Causal ALiBi bias: -slope * (i - j) for j <= i, -inf above the diagonal."""
    i, j = np.indices((n_pos, n_pos))
    bias = -slope * (i - j).astype(float)
    bias[j > i] = -np.inf
    return bias


def embedding_lookup(table: np.ndarray, ids: list[int]) -> np.ndarray:
    """Embedding = row lookup = one-hot @ table."""
    return table[np.asarray(ids)]


def padding_mask(lengths: list[int]) -> np.ndarray:
    """(batch, max_len) 1 = real token, 0 = PAD."""
    m = max(lengths)
    return (np.arange(m)[None, :] < np.asarray(lengths)[:, None]).astype(int)
