import numpy as np
import pytest

from calculus_prob import (
    Welford,
    bernoulli_pmf,
    derivative,
    expectation,
    gradient_descent,
    normal_pdf,
    numerical_gradient,
    numerical_jacobian,
    sample_variance,
    softmax,
    tiny_net_forward_backward,
    variance,
)

RNG = np.random.default_rng(7)


# 1.21 derivatives
def test_derivative_known_functions():
    assert derivative(lambda x: x**3, 2.0) == pytest.approx(12.0, rel=1e-6)
    assert derivative(np.sin, 0.0) == pytest.approx(1.0, rel=1e-6)
    assert derivative(np.exp, 1.0) == pytest.approx(np.e, rel=1e-6)


def test_derivative_h_too_small_is_noisy():
    good = abs(derivative(np.exp, 1.0, h=1e-5) - np.e)
    bad = abs(derivative(np.exp, 1.0, h=1e-13) - np.e)
    assert bad > good  # float cancellation: smaller h is not always better


def test_relu_kink_has_no_derivative_at_zero():
    relu = lambda x: max(x, 0.0)
    assert derivative(relu, 1.0) == pytest.approx(1.0)
    assert derivative(relu, -1.0) == pytest.approx(0.0)
    assert derivative(relu, 0.0, h=1e-3) == pytest.approx(0.5)  # symmetric average, not a true derivative


# 1.22/1.23 partials and gradients
def test_gradient_matches_analytic():
    f = lambda v: v[0] ** 2 + 3 * v[0] * v[1]
    np.testing.assert_allclose(numerical_gradient(f, np.array([1.0, 2.0])), [2 * 1 + 3 * 2, 3 * 1], atol=1e-6)


def test_gradient_points_uphill_and_descent_converges():
    f = lambda v: float((v ** 2).sum())
    x = np.array([3.0, -4.0])
    g = numerical_gradient(f, x)
    assert f(x + 1e-3 * g) > f(x)  # ascent direction
    out = gradient_descent(lambda v: 2 * v, x, lr=0.1, steps=200)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_gradient_descent_diverges_with_large_lr():
    out = gradient_descent(lambda v: 2 * v, np.array([1.0]), lr=1.5, steps=20)
    assert abs(out[0]) > 1e3  # |1 - 2*lr| = 2 > 1 => blows up


# 1.24 Jacobian
def test_jacobian_linear_map_equals_matrix():
    A = RNG.standard_normal((3, 4))
    J = numerical_jacobian(lambda x: A @ x, RNG.standard_normal(4))
    assert J.shape == (3, 4)
    np.testing.assert_allclose(J, A, atol=1e-6)


def test_jacobian_of_elementwise_is_diagonal():
    x = RNG.standard_normal(5)
    J = numerical_jacobian(np.tanh, x)
    np.testing.assert_allclose(J, np.diag(1 - np.tanh(x) ** 2), atol=1e-6)


def test_softmax_jacobian_matches_formula():
    z = RNG.standard_normal(4)
    p = softmax(z)
    expected = np.diag(p) - np.outer(p, p)
    np.testing.assert_allclose(numerical_jacobian(softmax, z), expected, atol=1e-6)
    np.testing.assert_allclose(expected.sum(axis=0), 0.0, atol=1e-12)  # probabilities sum to 1


# 1.25 chain rule via manual backprop = gradient check
def test_backprop_matches_numerical_gradient():
    x, y = RNG.standard_normal(3), 0.7
    W1, W2 = RNG.standard_normal((4, 3)), RNG.standard_normal(4)
    _, dW1, dW2 = tiny_net_forward_backward(x, W1, W2, y)

    f1 = lambda w: tiny_net_forward_backward(x, w.reshape(4, 3), W2, y)[0]
    f2 = lambda w: tiny_net_forward_backward(x, W1, w, y)[0]
    np.testing.assert_allclose(dW1.ravel(), numerical_gradient(f1, W1.ravel()), atol=1e-6)
    np.testing.assert_allclose(dW2, numerical_gradient(f2, W2), atol=1e-6)


# 1.26-1.28 probability and distributions
def test_softmax_is_distribution_and_shift_invariant():
    z = RNG.standard_normal(6)
    p = softmax(z)
    assert p.sum() == pytest.approx(1.0) and (p > 0).all()
    np.testing.assert_allclose(p, softmax(z + 1000.0))  # no overflow thanks to the shift
    assert np.isfinite(softmax(np.array([1000.0, 0.0]))).all()


def test_normal_pdf_integrates_to_one_and_peak():
    xs = np.linspace(-8, 8, 20001)
    area = np.trapezoid(normal_pdf(xs), xs)
    assert area == pytest.approx(1.0, abs=1e-6)
    assert normal_pdf(np.array(0.0)) == pytest.approx(1 / np.sqrt(2 * np.pi))
    assert normal_pdf(np.array(0.0), sigma=0.1) > 1.0  # a density can exceed 1
    with pytest.raises(ValueError):
        normal_pdf(0.0, sigma=0.0)


def test_bernoulli_pmf():
    assert bernoulli_pmf(1, 0.3) == 0.3 and bernoulli_pmf(0, 0.3) == pytest.approx(0.7)
    assert bernoulli_pmf(2, 0.3) == 0.0
    with pytest.raises(ValueError):
        bernoulli_pmf(1, 1.2)


# 1.29-1.31 expectation, variance, std
def test_fair_die_expectation_variance():
    v, p = np.arange(1, 7), np.full(6, 1 / 6)
    assert expectation(v, p) == pytest.approx(3.5)
    assert variance(v, p) == pytest.approx(35 / 12)


def test_expectation_rejects_bad_probs():
    with pytest.raises(ValueError):
        expectation([1, 2], [0.7, 0.7])


def test_linearity_and_variance_scaling():
    x = RNG.standard_normal(100_000)
    a, b = 3.0, 5.0
    assert (a * x + b).mean() == pytest.approx(a * x.mean() + b)
    assert (a * x + b).var() == pytest.approx(a**2 * x.var())  # shift does not matter


def test_sample_variance_ddof_matches_numpy():
    x = np.array([2.0, 4, 4, 4, 5, 5, 7, 9])
    assert sample_variance(x, ddof=0) == pytest.approx(np.var(x))  # = 4.0
    assert sample_variance(x, ddof=1) == pytest.approx(np.var(x, ddof=1))
    with pytest.raises(ValueError):
        sample_variance(np.array([1.0]))


def test_welford_matches_numpy_and_is_stable_with_large_offset():
    x = RNG.standard_normal(10_000) + 1e9
    w = Welford()
    for v in x:
        w.update(float(v))
    assert w.var == pytest.approx(np.var(x, ddof=1), rel=1e-6)
    assert w.std == pytest.approx(np.sqrt(w.var))
    naive = np.mean(x**2) - np.mean(x) ** 2  # catastrophic cancellation
    assert abs(naive - 1.0) > abs(w.var - 1.0)


# 1.20 functions: stacked linear layers collapse; nonlinearity prevents it
def test_linear_layers_collapse_without_nonlinearity():
    x = RNG.standard_normal(5)
    W1, W2 = RNG.standard_normal((4, 5)), RNG.standard_normal((3, 4))
    np.testing.assert_allclose(W2 @ (W1 @ x), (W2 @ W1) @ x)  # one matrix does the same job
    relu = lambda v: np.maximum(v, 0)
    two_layer = W2 @ relu(W1 @ x)
    assert not np.allclose(two_layer, (W2 @ W1) @ x)


# 1.27 random variables: law of large numbers
def test_law_of_large_numbers_die():
    rolls = np.random.default_rng(0).integers(1, 7, size=200_000)
    assert rolls.mean() == pytest.approx(3.5, abs=0.02)
    same = np.random.default_rng(0).integers(1, 7, size=200_000)
    np.testing.assert_array_equal(rolls, same)  # seeded => reproducible


# 1.31 standard deviation: z-scores and the 68-95-99.7 rule
def test_zscore_and_empirical_rule():
    x = np.random.default_rng(1).normal(10, 3, size=200_000)
    z = (x - x.mean()) / x.std()
    assert z.mean() == pytest.approx(0.0, abs=1e-9) and z.std() == pytest.approx(1.0)
    assert (np.abs(z) < 1).mean() == pytest.approx(0.6827, abs=0.01)
    assert (np.abs(z) < 2).mean() == pytest.approx(0.9545, abs=0.01)


def test_standard_error_shrinks_with_sqrt_n():
    rng = np.random.default_rng(2)
    means_100 = rng.normal(0, 1, size=(5000, 100)).mean(axis=1)
    means_400 = rng.normal(0, 1, size=(5000, 400)).mean(axis=1)
    assert means_100.std() == pytest.approx(1 / np.sqrt(100), rel=0.05)
    assert means_100.std() / means_400.std() == pytest.approx(2.0, rel=0.1)
