"""Attention from matrix operations, NumPy only (Tier 7 build, step 1-2).

Shapes: B batch, H heads, T queries, S keys, D head dim, E model dim.
"""
from __future__ import annotations
import numpy as np

NEG = -1e30  # large negative instead of -inf: avoids NaN on fully masked rows


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exp)."""
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(t: int, s: int | None = None) -> np.ndarray:
    """Boolean (t, s): True = may attend. Query i sees keys j <= i (+ offset if s > t)."""
    s = t if s is None else s
    return np.tril(np.ones((t, s), dtype=bool), k=s - t)


def padding_mask(lengths, max_len: int | None = None) -> np.ndarray:
    """Boolean (B, max_len): True = real token, False = PAD."""
    max_len = max_len or max(lengths)
    return np.arange(max_len)[None, :] < np.asarray(lengths)[:, None]


def sliding_window_mask(t: int, window: int, sinks: int = 0) -> np.ndarray:
    """Causal + only the last `window` keys (incl. self); optionally keep first `sinks` keys."""
    i = np.arange(t)[:, None]; j = np.arange(t)[None, :]
    m = (j <= i) & (j > i - window)
    if sinks:
        m |= (j < sinks) & (j <= i)
    return m


def scaled_dot_product_attention(q, k, v, mask=None, scale: float | None = None):
    """softmax(Q K^T / sqrt(D) + mask) V.   q:(...,T,D) k:(...,S,D) v:(...,S,Dv).
    mask: boolean broadcastable to (...,T,S), True = keep.  Returns (out, weights)."""
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d) if scale is None else scale
    scores = (q @ np.swapaxes(k, -1, -2)) * scale            # (...,T,S)
    if mask is not None:
        scores = np.where(mask, scores, NEG)
    w = softmax(scores, axis=-1)
    return w @ v, w


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """(B,T,E) -> (B,H,T,E/H)."""
    b, t, e = x.shape
    assert e % n_heads == 0, "model dim must be divisible by n_heads"
    return x.reshape(b, t, n_heads, e // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """(B,H,T,D) -> (B,T,H*D)."""
    b, h, t, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * d)


class MultiHeadAttention:
    """MHA / GQA / MQA in NumPy.  n_kv_heads == n_heads -> MHA; 1 -> MQA; else GQA."""

    def __init__(self, d_model, n_heads, n_kv_heads=None, seed=0):
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h, self.hk, self.d = n_heads, n_kv_heads, d_model // n_heads
        r = np.random.default_rng(seed)
        s = 1 / np.sqrt(d_model)
        self.Wq = r.normal(0, s, (d_model, n_heads * self.d))
        self.Wk = r.normal(0, s, (d_model, n_kv_heads * self.d))
        self.Wv = r.normal(0, s, (d_model, n_kv_heads * self.d))
        self.Wo = r.normal(0, s, (n_heads * self.d, d_model))

    def __call__(self, x, context=None, mask=None):
        """x:(B,T,E) queries; context:(B,S,E) for cross-attention (default = x, self-attention).
        mask: bool broadcastable to (B,H,T,S)."""
        context = x if context is None else context
        q = split_heads(x @ self.Wq, self.h)                  # (B,H,T,D)
        k = split_heads(context @ self.Wk, self.hk)           # (B,Hk,S,D)
        v = split_heads(context @ self.Wv, self.hk)
        rep = self.h // self.hk
        k = np.repeat(k, rep, axis=1)                         # share each KV head across `rep` query heads
        v = np.repeat(v, rep, axis=1)
        out, self.last_weights = scaled_dot_product_attention(q, k, v, mask)
        return merge_heads(out) @ self.Wo


def kv_cache_bytes(layers, n_kv_heads, head_dim, seq_len, batch=1, bytes_per=2):
    """K and V cache size: 2 * L * Hk * D * S * B * bytes."""
    return 2 * layers * n_kv_heads * head_dim * seq_len * batch * bytes_per


def score_matrix_bytes(seq_len, n_heads, batch=1, bytes_per=2):
    """Materialised attention-score matrix per layer: B*H*S*S*bytes (what FlashAttention avoids storing)."""
    return batch * n_heads * seq_len * seq_len * bytes_per


if __name__ == "__main__":
    x = np.random.default_rng(1).normal(size=(2, 5, 16))
    mha = MultiHeadAttention(16, 4)
    print(mha(x, mask=causal_mask(5)).shape, mha.last_weights[0, 0].round(2))
