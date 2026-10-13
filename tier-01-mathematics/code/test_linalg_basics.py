import numpy as np
import pytest

from linalg_basics import (
    broadcast_shape,
    dot_manual,
    matmul_manual,
    vec_add_manual,
    vec_scale_manual,
)


def test_add_and_scale_match_numpy():
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    assert vec_add_manual(a, b) == (np.array(a) + np.array(b)).tolist()
    assert vec_scale_manual(-2, a) == (-2 * np.array(a)).tolist()


def test_add_length_mismatch_raises():
    with pytest.raises(ValueError):
        vec_add_manual([1.0], [1.0, 2.0])


def test_dot_matches_numpy_and_orthogonality():
    assert dot_manual([1, 2, 3], [4, 5, 6]) == 32
    assert dot_manual([1, 0], [0, 1]) == 0  # orthogonal
    with pytest.raises(ValueError):
        dot_manual([1], [1, 2])


def test_matmul_matches_numpy_and_shape_error():
    A = np.arange(6.0).reshape(2, 3)
    B = np.arange(12.0).reshape(3, 4)
    np.testing.assert_allclose(matmul_manual(A.tolist(), B.tolist()), A @ B)
    with pytest.raises(ValueError):
        matmul_manual(B.tolist(), A.tolist())  # (3,4)@(2,3)


@pytest.mark.parametrize(
    "s1,s2",
    [((3,), (4, 3)), ((4, 1), (1, 5)), ((32, 1, 768), (128, 1)), ((), (2, 2)), ((5, 1, 3), (4, 1))],
)
def test_broadcast_shape_matches_numpy(s1, s2):
    expected = np.broadcast_shapes(s1, s2)
    assert broadcast_shape(s1, s2) == expected


def test_broadcast_incompatible():
    with pytest.raises(ValueError):
        broadcast_shape((3,), (4,))
    with pytest.raises(ValueError):
        np.broadcast_shapes((3,), (4,))


def test_batch_matmul_shapes():
    x = np.zeros((32, 128, 768))
    w = np.zeros((768, 64))
    assert (x @ w).shape == (32, 128, 64)
