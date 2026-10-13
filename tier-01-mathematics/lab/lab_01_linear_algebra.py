"""Tier 1 Lab, part 1: solutions to the hands-on exercises of concepts 1.1-1.19.

Each `ex_1_N()` returns a dict of results so tests (and readers) can inspect them.
Run `python lab_01_linear_algebra.py` to print every result.
"""
from __future__ import annotations

import time

import numpy as np

from linalg_basics import matmul_manual
from linalg_core import cosine_matrix, low_rank, norm_p


def ex_1_1() -> dict:
    """0-D vs (1,) arrays: shapes and what `a + b` broadcasts to."""
    a, b = np.array(2.0), np.array([3.0])
    return {"a_shape": a.shape, "b_shape": b.shape, "sum_shape": (a + b).shape}


def ex_1_2(n: int = 1_000_000, d: int = 768) -> dict:
    """Memory of n vectors of dim d in float32/float16 (formula checked against nbytes)."""
    f32 = np.zeros((1000, d), np.float32).nbytes / 1000 * n  # per-row bytes scaled
    f16 = np.zeros((1000, d), np.float16).nbytes / 1000 * n
    assert f32 == n * d * 4 and f16 == n * d * 2
    return {"float32_GB": f32 / 1e9, "float16_GB": f16 / 1e9}


def ex_1_3() -> dict:
    W, x = np.ones((3, 4)), np.ones(4)
    return {"expected": (3,), "actual": (W @ x).shape}


def ex_1_4() -> dict:
    x32 = np.zeros((8, 1024, 4096), np.float32)
    x16 = x32.astype(np.float16)
    return {"float32_MiB": x32.nbytes / 2**20, "float16_MiB": x16.nbytes / 2**20}


def ex_1_5(seed: int = 0) -> dict:
    """A single attention head with a shape assertion on every line."""
    rng = np.random.default_rng(seed)
    B, T, D, d = 2, 5, 16, 8
    x = rng.standard_normal((B, T, D)); assert x.shape == (B, T, D)
    Wq, Wk, Wv = (rng.standard_normal((D, d)) for _ in range(3))
    q, k, v = x @ Wq, x @ Wk, x @ Wv; assert q.shape == k.shape == v.shape == (B, T, d)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d); assert scores.shape == (B, T, T)
    w = np.exp(scores - scores.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    out = w @ v; assert out.shape == (B, T, d)
    return {"out_shape": out.shape, "rows_sum_to_one": bool(np.allclose(w.sum(-1), 1))}


def ex_1_6(seed: int = 0) -> dict:
    """Softmax rows sum to 1 along the chosen axis, not along the other one."""
    s = np.random.default_rng(seed).standard_normal((4, 6))
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    return {"axis_last_ok": bool(np.allclose(p.sum(axis=-1), 1)), "axis_first_ok": bool(np.allclose(p.sum(axis=0), 1))}


def ex_1_7(B: int = 8, seed: int = 0) -> dict:
    """The silent (B,1) vs (B,) broadcasting bug, and the shape-assert guard."""
    rng = np.random.default_rng(seed)
    pred, target = rng.standard_normal((B, 1)), rng.standard_normal(B)
    buggy_shape = (pred - target).shape  # (B, B) !
    buggy_mse = float(((pred - target) ** 2).mean())

    def mse(p: np.ndarray, t: np.ndarray) -> float:
        if p.shape != t.shape:
            raise ValueError(f"shape mismatch {p.shape} vs {t.shape}")
        return float(((p - t) ** 2).mean())

    try:
        mse(pred, target)
        guarded = False
    except ValueError:
        guarded = True
    correct = mse(pred[:, 0], target)
    return {"buggy_shape": buggy_shape, "buggy_mse": buggy_mse, "correct_mse": correct, "guard_raises": guarded}


def ex_1_8(seed: int = 0) -> dict:
    r = np.random.default_rng(seed)
    a, b, c = (r.standard_normal(100) for _ in range(3))
    return {"commutative": bool(np.allclose(a + b, b + a)), "associative": bool(np.allclose((a + b) + c, a + (b + c)))}


def ex_1_9() -> dict:
    """Softmax of fixed logits at three temperatures."""
    z = np.array([2.0, 1.0, 0.1])
    out = {}
    for T in (0.25, 1.0, 4.0):
        e = np.exp((z - z.max()) / T)
        out[T] = e / e.sum()
    return out


def ex_1_10(seed: int = 0) -> dict:
    """Var(q.k) ~ d for N(0,1) entries; dividing by sqrt(d) gives ~1."""
    rng = np.random.default_rng(seed)
    res = {}
    for d in (16, 256, 4096):
        q, k = rng.standard_normal((20000, d)), rng.standard_normal((20000, d))
        s = (q * k).sum(1)
        res[d] = (float(s.var()), float((s / np.sqrt(d)).var()))
    return res


def _timed(f) -> float:
    t0 = time.perf_counter(); f(); return time.perf_counter() - t0


def ex_1_11(sizes=(32, 64, 96), seed: int = 0) -> dict:
    """Manual triple loop vs `@`: correctness and speed ratio."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        A, B = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        best = lambda f: min(_timed(f) for _ in range(3))  # best-of-3 reduces timing noise
        t_manual = best(lambda: matmul_manual(A.tolist(), B.tolist()))
        t_numpy = best(lambda: A @ B)
        out[n] = {"match": bool(np.allclose(matmul_manual(A.tolist(), B.tolist()), A @ B)), "slowdown": t_manual / max(t_numpy, 1e-9)}
    return out


def ex_1_12() -> dict:
    A = np.arange(12.0).reshape(3, 4)
    return {"A_contig": A.flags["C_CONTIGUOUS"], "AT_contig": A.T.flags["C_CONTIGUOUS"],
            "ascontig_contig": np.ascontiguousarray(A.T).flags["C_CONTIGUOUS"], "shares_memory": bool(np.shares_memory(A, A.T))}


def ex_1_13(seed: int = 0) -> dict:
    """Rank-deficient X^T X (duplicated feature) fixed by + lambda I."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((50, 1))
    X = np.hstack([x, x, rng.standard_normal((50, 1))])  # column 2 duplicates column 1
    G = X.T @ X
    rank_before = np.linalg.matrix_rank(G)
    cond_before = np.linalg.cond(G)
    Gr = G + 1e-3 * np.eye(3)
    return {"rank_before": int(rank_before), "cond_before": float(cond_before), "cond_after": float(np.linalg.cond(Gr)),
            "full_rank_after": bool(np.linalg.matrix_rank(Gr) == 3)}


def ex_1_14(seed: int = 0) -> dict:
    """Solution sensitivity: ill-conditioned system amplifies a tiny perturbation of b."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    V, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    s = np.logspace(0, -10, 6)  # condition number 1e10
    A = U @ np.diag(s) @ V.T
    b = rng.standard_normal(6)
    x = np.linalg.solve(A, b)
    db = 1e-8 * U[:, -1]  # worst case: perturb b along the weakest singular direction
    x2 = np.linalg.solve(A, b + db)
    return {"cond": float(np.linalg.cond(A)), "abs_change_b": float(np.linalg.norm(db)),
            "abs_change_x": float(np.linalg.norm(x2 - x)),
            "amplification": float(np.linalg.norm(x2 - x) / np.linalg.norm(db))}


def global_norm_clip(grads: list[np.ndarray], c: float) -> list[np.ndarray]:
    total = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))
    scale = min(1.0, c / (total + 1e-12))
    return [g * scale for g in grads]


def ex_1_15(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    big = [rng.standard_normal((3, 3)) * 10, rng.standard_normal(5) * 10]
    small = [np.array([0.1, 0.2]), np.array([0.05])]
    gn = lambda gs: float(np.sqrt(sum((g ** 2).sum() for g in gs)))
    clipped = global_norm_clip(big, 5.0)
    unchanged = global_norm_clip(small, 5.0)
    return {"before": gn(big), "after": gn(clipped), "small_unchanged": all(np.array_equal(a, b) for a, b in zip(small, unchanged))}


def ex_1_16(seed: int = 0) -> dict:
    """Distance concentration: (max-min)/min shrinks as d grows."""
    rng = np.random.default_rng(seed)
    out = {}
    for d in (2, 20, 500):
        X, q = rng.standard_normal((1000, d)), rng.standard_normal(d)
        dist = np.linalg.norm(X - q, axis=1)
        out[d] = float((dist.max() - dist.min()) / dist.min())
    return out


def ex_1_17(seed: int = 0) -> dict:
    """Top-5 neighbours by cosine, dot and L2, with and without normalization."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((100, 16)) * rng.uniform(0.2, 5.0, (100, 1))  # varying norms
    q = rng.standard_normal(16)
    top = lambda s, largest=True: tuple(int(i) for i in np.argsort(-s if largest else s)[:5])
    cos = cosine_matrix(q[None], X)[0]
    dot = X @ q
    l2 = np.linalg.norm(X - q, axis=1)
    Xn, qn = X / np.linalg.norm(X, axis=1, keepdims=True), q / np.linalg.norm(q)
    return {"cos": top(cos), "dot": top(dot), "l2": top(l2, False),
            "unit_dot": top(Xn @ qn), "unit_l2": top(np.linalg.norm(Xn - qn, axis=1), False)}


def ex_1_18(seed: int = 0) -> dict:
    """Covariance eigenvectors of correlated 2-D data = principal axes."""
    rng = np.random.default_rng(seed)
    angle = np.deg2rad(30)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X = (rng.standard_normal((5000, 2)) * [3.0, 0.5]) @ R.T
    w, Q = np.linalg.eigh(np.cov(X, rowvar=False))
    major = Q[:, -1]
    return {"major_axis_angle_deg": float(np.rad2deg(np.arctan2(major[1], major[0])) % 180),  # an axis has no sign
            "explained_variance_ratio": (w / w.sum())[::-1].tolist()}


def ex_1_19(seed: int = 0) -> dict:
    """Low-rank + noise matrix: smallest k retaining 95% energy, and reconstruction errors."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((60, 5)) @ rng.standard_normal((5, 40)) + 0.05 * rng.standard_normal((60, 40))
    s = np.linalg.svd(M, compute_uv=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    k95 = int(np.searchsorted(energy, 0.95) + 1)
    errs = {k: float(np.linalg.norm(M - low_rank(M, k))) for k in (1, 5, 20)}
    return {"k95": k95, "errors": errs}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
