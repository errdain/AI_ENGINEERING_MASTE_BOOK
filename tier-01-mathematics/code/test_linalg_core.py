import numpy as np
import pytest

from linalg_basics import matmul_manual
from linalg_core import (
    chebyshev,
    cosine_matrix,
    cosine_similarity,
    euclidean,
    identity,
    inverse_manual,
    low_rank,
    manhattan,
    norm_p,
    power_iteration,
    transpose_manual,
)

RNG = np.random.default_rng(42)


# 1.11 matmul
def test_matmul_not_commutative_but_associative():
    A, B, C = (RNG.standard_normal((3, 3)) for _ in range(3))
    assert not np.allclose(A @ B, B @ A)
    np.testing.assert_allclose((A @ B) @ C, A @ (B @ C), atol=1e-10)


def test_matmul_manual_rectangular():
    A, B = RNG.standard_normal((4, 5)), RNG.standard_normal((5, 2))
    np.testing.assert_allclose(matmul_manual(A.tolist(), B.tolist()), A @ B)


def test_matmul_flops_formula():
    m, n, p = 4, 5, 2
    assert 2 * m * n * p == 80  # multiply + add per output term


# 1.12 transpose
def test_transpose_properties():
    A, B = RNG.standard_normal((2, 3)), RNG.standard_normal((3, 4))
    np.testing.assert_allclose(transpose_manual(A.tolist()), A.T)
    np.testing.assert_allclose((A @ B).T, B.T @ A.T)  # order reverses
    np.testing.assert_allclose(A.T.T, A)


# 1.13 identity
def test_identity_neutral():
    A = RNG.standard_normal((3, 3))
    I = np.array(identity(3))
    np.testing.assert_allclose(A @ I, A)
    np.testing.assert_allclose(I @ A, A)


# 1.14 inverse
def test_inverse_matches_numpy_and_roundtrip():
    A = RNG.standard_normal((5, 5)) + 5 * np.eye(5)
    Ai = np.array(inverse_manual(A.tolist()))
    np.testing.assert_allclose(Ai, np.linalg.inv(A), atol=1e-9)
    np.testing.assert_allclose(A @ Ai, np.eye(5), atol=1e-9)


def test_inverse_singular_raises():
    with pytest.raises(np.linalg.LinAlgError):
        inverse_manual([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(ValueError):
        inverse_manual([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_solve_preferred_over_inverse_is_accurate():
    A = RNG.standard_normal((6, 6)) + 6 * np.eye(6)
    b = RNG.standard_normal(6)
    np.testing.assert_allclose(A @ np.linalg.solve(A, b), b, atol=1e-10)


# 1.15 norms
def test_norms_known_values():
    x = np.array([3.0, -4.0])
    assert norm_p(x, 1) == 7 and norm_p(x, 2) == 5 and norm_p(x, np.inf) == 4
    for p in (1, 2, np.inf):
        assert norm_p(x, p) == pytest.approx(np.linalg.norm(x, p))
    with pytest.raises(ValueError):
        norm_p(x, 0.5)


def test_norm_axioms():
    a, b = RNG.standard_normal(10), RNG.standard_normal(10)
    assert norm_p(a + b, 2) <= norm_p(a, 2) + norm_p(b, 2) + 1e-12  # triangle
    assert norm_p(-3 * a, 2) == pytest.approx(3 * norm_p(a, 2))  # homogeneity


# 1.16 distances
def test_distances():
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert euclidean(a, b) == 5 and manhattan(a, b) == 7 and chebyshev(a, b) == 4


def test_euclidean_on_unit_vectors_relates_to_cosine():
    a, b = RNG.standard_normal(8), RNG.standard_normal(8)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    assert euclidean(a, b) ** 2 == pytest.approx(2 - 2 * cosine_similarity(a, b))


# 1.17 cosine
def test_cosine_basic_and_scale_invariance():
    a, b = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(np.sqrt(2) / 2)
    assert cosine_similarity(a, 100 * b) == pytest.approx(cosine_similarity(a, b))
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_zero_vector_raises():
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(3), np.ones(3))


def test_cosine_matrix_matches_pairwise():
    X, Y = RNG.standard_normal((4, 6)), RNG.standard_normal((3, 6))
    S = cosine_matrix(X, Y)
    assert S.shape == (4, 3)
    assert S[2, 1] == pytest.approx(cosine_similarity(X[2], Y[1]))


# 1.18 eigen
def test_power_iteration_matches_numpy():
    A = np.array([[4.0, 1.0], [2.0, 3.0]])
    lam, v = power_iteration(A)
    assert lam == pytest.approx(max(np.linalg.eigvals(A).real), rel=1e-9)
    np.testing.assert_allclose(A @ v, lam * v, atol=1e-8)


def test_eigh_symmetric_reconstruction():
    S = RNG.standard_normal((4, 4))
    S = S + S.T
    w, Q = np.linalg.eigh(S)
    np.testing.assert_allclose(Q @ np.diag(w) @ Q.T, S, atol=1e-10)
    np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-10)  # orthonormal


# 1.19 SVD
def test_svd_reconstruction_and_eckart_young():
    M = RNG.standard_normal((20, 12))
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    np.testing.assert_allclose((U * s) @ Vt, M, atol=1e-10)
    k = 4
    err = np.linalg.norm(M - low_rank(M, k))  # Frobenius
    assert err == pytest.approx(np.sqrt(np.sum(s[k:] ** 2)))
    assert np.linalg.matrix_rank(low_rank(M, k)) == k


def test_low_rank_param_savings():
    m, n, k = 4096, 4096, 16
    assert k * (m + n) < m * n // 100  # >100x fewer parameters
