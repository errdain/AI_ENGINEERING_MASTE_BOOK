"""Tier 1 (1.11-1.19): matmul, transpose, identity, inverse, norms, distances,
cosine similarity, eigen-decomposition (power iteration), SVD low-rank.

Manual versions show the mechanics; NumPy is the reference they are tested against.
"""
from __future__ import annotations

import numpy as np

Matrix = list[list[float]]

# ---- 1.12 Transpose / 1.13 Identity ------------------------------------------


def transpose_manual(A: Matrix) -> Matrix:
    return [list(col) for col in zip(*A)]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ---- 1.14 Matrix inverse (Gauss-Jordan with partial pivoting) -----------------


def inverse_manual(A: Matrix, tol: float = 1e-12) -> Matrix:
    n = len(A)
    if any(len(r) != n for r in A):
        raise ValueError("matrix must be square")
    M = [list(map(float, A[i])) + identity(n)[i] for i in range(n)]  # [A | I]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < tol:
            raise np.linalg.LinAlgError("matrix is singular (or numerically singular)")
        M[col], M[pivot] = M[pivot], M[col]
        p = M[col][col]
        M[col] = [v / p for v in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [row[n:] for row in M]


# ---- 1.15 Norms / 1.16 Distances ----------------------------------------------


def norm_p(x: np.ndarray, p: float) -> float:
    if p == np.inf:
        return float(np.max(np.abs(x)))
    if p < 1:
        raise ValueError("p must be >= 1 for a true norm")
    return float(np.sum(np.abs(x) ** p) ** (1.0 / p))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return norm_p(a - b, 2)


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return norm_p(a - b, 1)


def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
    return norm_p(a - b, np.inf)


# ---- 1.17 Cosine similarity ---------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))  # clip guards float rounding


def cosine_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """All-pairs cosine similarity: (n,d),(m,d) -> (n,m). One matmul after normalizing."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Xn @ Yn.T


# ---- 1.18 Eigenvalues / eigenvectors ------------------------------------------


def power_iteration(A: np.ndarray, iters: int = 1000, tol: float = 1e-12, seed: int = 0):
    """Dominant eigenpair of a square matrix (assumes a unique largest |eigenvalue|)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0, v
        w /= nw
        new_lam = float(w @ A @ w)  # Rayleigh quotient
        if abs(new_lam - lam) < tol:
            v, lam = w, new_lam
            break
        v, lam = w, new_lam
    return lam, v


# ---- 1.19 SVD ----------------------------------------------------------------


def low_rank(A: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation (Eckart-Young) from the truncated SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k]


if __name__ == "__main__":
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    print("inverse  ", np.round(inverse_manual(A.tolist()), 4).tolist(), np.linalg.inv(A).round(4).tolist())
    print("norms    ", norm_p(np.array([3.0, -4.0]), 1), norm_p(np.array([3.0, -4.0]), 2), norm_p(np.array([3.0, -4.0]), np.inf))
    print("cosine   ", cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 1.0])))
    print("eig      ", power_iteration(A)[0], np.linalg.eigvals(A))
    M = np.random.default_rng(0).standard_normal((50, 30))
    print("rank-5 err", np.linalg.norm(M - low_rank(M, 5)))
