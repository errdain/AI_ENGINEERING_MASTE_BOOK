"""Tier 1 (1.1-1.10) reference implementations: manual Python vs NumPy.

Each function is minimal and typed. Manual versions exist to show the mechanics;
NumPy versions are what you use in practice.
"""
from __future__ import annotations

import numpy as np

# ---- 1.8 Vector addition / 1.9 Scalar multiplication -----------------------


def vec_add_manual(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return [x + y for x, y in zip(a, b)]


def vec_scale_manual(c: float, a: list[float]) -> list[float]:
    return [c * x for x in a]


# ---- 1.10 Dot product -------------------------------------------------------


def dot_manual(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


# ---- 1.11-style helper used to illustrate 1.3 / 1.5 -------------------------


def matmul_manual(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """(m,n) @ (n,p) -> (m,p). Raises on inner-dimension mismatch."""
    n = len(A[0])
    if n != len(B):
        raise ValueError(f"inner dims differ: {n} vs {len(B)}")
    p = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(len(A))]


# ---- 1.7 Broadcasting -------------------------------------------------------


def broadcast_shape(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the NumPy broadcast result shape by the right-aligned rule."""
    out: list[int] = []
    for d1, d2 in zip(reversed(s1), reversed(s2)):
        if d1 == d2 or d2 == 1:
            out.append(d1)
        elif d1 == 1:
            out.append(d2)
        else:
            raise ValueError(f"cannot broadcast {s1} with {s2}")
    longer = s1 if len(s1) > len(s2) else s2
    out.extend(reversed(longer[: abs(len(s1) - len(s2))]))
    return tuple(reversed(out))


# ---- 1.4-1.6 Tensors, shapes, axes -----------------------------------------


def describe(x: np.ndarray) -> dict[str, object]:
    return {"ndim": x.ndim, "shape": x.shape, "size": x.size, "dtype": str(x.dtype)}


if __name__ == "__main__":
    a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    print("add   ", a + b, vec_add_manual(a.tolist(), b.tolist()))
    print("scale ", 2 * a, vec_scale_manual(2, a.tolist()))
    print("dot   ", a @ b, dot_manual(a.tolist(), b.tolist()))
    batch = np.zeros((32, 128, 768))  # (batch, seq_len, d_model)
    print("tensor", describe(batch))
    print("bcast ", broadcast_shape((32, 1, 768), (128, 1)))
