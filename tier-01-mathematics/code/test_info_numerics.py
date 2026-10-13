import warnings

import numpy as np
import pytest

from info_numerics import (
    bayes_posterior,
    bernoulli_nll,
    correlation_matrix,
    covariance_matrix,
    cross_entropy,
    cross_entropy_from_logits,
    descend_1d,
    double_well,
    double_well_grad,
    entropy,
    kl_divergence,
    log_softmax,
    logsumexp,
    midpoint_convex_violations,
    mle_bernoulli,
    mle_gaussian,
    naive_softmax,
    posterior_categorical,
    saddle,
)

RNG = np.random.default_rng(11)


# ---- 1.32 covariance
def test_covariance_matches_numpy_and_symmetry():
    X = RNG.standard_normal((200, 4))
    C = covariance_matrix(X)
    np.testing.assert_allclose(C, np.cov(X, rowvar=False))
    np.testing.assert_allclose(C, C.T)
    assert (np.linalg.eigvalsh(C) >= -1e-12).all()  # positive semi-definite


def test_correlation_range_scale_invariance_and_zero_var():
    x = RNG.standard_normal(500)
    X = np.column_stack([x, 2 * x + 0.1 * RNG.standard_normal(500), RNG.standard_normal(500)])
    R = correlation_matrix(X)
    assert np.all(np.abs(R) <= 1 + 1e-12) and np.allclose(np.diag(R), 1)
    assert R[0, 1] > 0.95 and abs(R[0, 2]) < 0.2
    X2 = X.copy(); X2[:, 0] *= 1000
    np.testing.assert_allclose(correlation_matrix(X2), R, atol=1e-10)
    with pytest.raises(ValueError):
        correlation_matrix(np.column_stack([np.ones(10), RNG.standard_normal(10)]))


def test_uncorrelated_is_not_independent():
    x = RNG.standard_normal(200_000)
    y = x**2  # fully determined by x
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.01


# ---- 1.33/1.34 conditional probability and Bayes
def test_bayes_base_rate_neglect_example():
    post = bayes_posterior(0.01, 0.99, 0.05)  # rare disease, good test
    assert post == pytest.approx(0.0099 / (0.0099 + 0.0495))  # = 1/6
    assert post == pytest.approx(1 / 6)
    assert bayes_posterior(0.5, 0.99, 0.05) > 0.9  # base rate matters


def test_bayes_validation_and_categorical_posterior():
    with pytest.raises(ValueError):
        bayes_posterior(1.2, 0.9, 0.1)
    with pytest.raises(ValueError):
        bayes_posterior(0.0, 0.9, 0.0)
    post = posterior_categorical([0.5, 0.3, 0.2], [0.1, 0.4, 0.5])
    assert post.sum() == pytest.approx(1) and post.argmax() == 1


def test_conditional_probability_by_simulation():
    rolls = np.random.default_rng(3).integers(1, 7, size=(300_000, 2))
    s = rolls.sum(axis=1)
    p = ((s == 7) & (rolls[:, 0] == 3)).mean() / (rolls[:, 0] == 3).mean()
    assert p == pytest.approx(1 / 6, abs=0.01)  # P(sum=7 | first=3)


# ---- 1.35 MLE
def test_mle_bernoulli_minimizes_nll():
    x = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1.0])
    p_hat = mle_bernoulli(x)
    grid = np.linspace(0.01, 0.99, 99)
    best = grid[np.argmin([bernoulli_nll(p, x) for p in grid])]
    assert p_hat == pytest.approx(0.7) and best == pytest.approx(p_hat, abs=0.01)


def test_mle_gaussian_std_is_biased_low():
    rng = np.random.default_rng(5)
    stds = [mle_gaussian(rng.normal(0, 1, 5))[1] ** 2 for _ in range(20_000)]
    assert np.mean(stds) == pytest.approx(0.8, abs=0.03)  # E = (n-1)/n * sigma^2 = 0.8


# ---- 1.36-1.38 entropy / cross-entropy / KL
def test_entropy_known_values():
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([1.0, 0.0]) == 0.0
    assert entropy(np.full(8, 1 / 8)) == pytest.approx(3.0)
    assert entropy(np.full(8, 1 / 8), base=np.e) == pytest.approx(np.log(8))
    with pytest.raises(ValueError):
        entropy([0.6, 0.6])


def test_uniform_has_max_entropy():
    for _ in range(20):
        p = RNG.dirichlet(np.ones(6))
        assert entropy(p) <= np.log2(6) + 1e-12


def test_cross_entropy_decomposition_and_gibbs():
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.5, 0.3, 0.2])
    assert cross_entropy(p, q) == pytest.approx(entropy(p) + kl_divergence(p, q))
    assert kl_divergence(p, q) >= 0 and kl_divergence(p, p) == pytest.approx(0)
    assert cross_entropy(p, q) >= entropy(p)


def test_kl_is_asymmetric_and_infinite_when_q_zero():
    p, q = [0.5, 0.5], [0.9, 0.1]
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf")
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(1.0)  # p=0 terms are skipped


def test_onehot_cross_entropy_equals_neg_log_prob():
    logits = np.array([2.0, 0.5, -1.0])
    p = np.exp(log_softmax(logits))
    assert cross_entropy_from_logits(logits, 0) == pytest.approx(-np.log(p[0]))
    onehot = np.array([1.0, 0, 0])
    assert cross_entropy(onehot, p, base=np.e) == pytest.approx(-np.log(p[0]))


# ---- 1.39-1.44 numerical stability
def test_naive_softmax_overflows_stable_does_not():
    z = np.array([1000.0, 1001.0, 1002.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(naive_softmax(z)).any()
    stable = np.exp(log_softmax(z))
    assert np.isfinite(stable).all() and stable.sum() == pytest.approx(1)
    np.testing.assert_allclose(stable, np.exp(log_softmax(z - 1000)))  # shift invariance


def test_logsumexp_matches_naive_when_safe_and_handles_extremes():
    x = RNG.standard_normal(10)
    assert logsumexp(x) == pytest.approx(np.log(np.sum(np.exp(x))))
    assert logsumexp(np.array([1000.0, 1000.0])) == pytest.approx(1000 + np.log(2))
    assert logsumexp(np.array([-1000.0, -1000.0])) == pytest.approx(-1000 + np.log(2))
    assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf


def test_log_space_avoids_underflow_of_probability_products():
    p = np.full(400, 1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.prod(p) == 0.0  # underflow: 1e-1200 is not representable
    assert np.sum(np.log(p)) == pytest.approx(400 * np.log(1e-3))


def test_cross_entropy_on_extreme_logits_is_finite():
    assert np.isfinite(cross_entropy_from_logits(np.array([-1000.0, 1000.0]), 0))
    assert cross_entropy_from_logits(np.array([-1000.0, 1000.0]), 0) == pytest.approx(2000.0)


def test_log_and_exp_identities():
    a, b = 3.7, 0.25
    assert np.log(a * b) == pytest.approx(np.log(a) + np.log(b))
    assert np.exp(np.log(a)) == pytest.approx(a)
    assert np.log1p(1e-20) == pytest.approx(1e-20) and np.log(1 + 1e-20) == 0.0  # log1p is exact for tiny x
    assert np.expm1(1e-20) == pytest.approx(1e-20) and np.exp(1e-20) - 1 == 0.0


# ---- 1.40 floating point
def test_floating_point_facts():
    assert 0.1 + 0.2 != 0.3 and abs(0.1 + 0.2 - 0.3) < 1e-15
    assert np.finfo(np.float64).eps == pytest.approx(2.220446049250313e-16)
    assert np.finfo(np.float32).eps == pytest.approx(1.1920929e-07)
    assert np.finfo(np.float16).max == 65504
    assert np.float32(16777216) + np.float32(1) == np.float32(16777216)  # 2^24: +1 is absorbed
    assert np.float64(1e16) + 1 == 1e16
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isinf(np.float16(70000.0))
        assert np.float16(1e-8) == 0  # underflow to zero


def test_sum_order_and_error_accumulation():
    x = np.float32(1e8)
    small = np.full(1000, 1.0, dtype=np.float32)
    naive = x
    for v in small:
        naive = np.float32(naive + v)  # each +1 is absorbed (1e8 has spacing 8 in float32)
    assert naive == np.float32(1e8)
    assert np.sum(small, dtype=np.float64) + 1e8 == pytest.approx(1e8 + 1000)


# ---- 1.45-1.47 convexity and landscapes
def test_convex_functions_have_no_midpoint_violations():
    assert midpoint_convex_violations(lambda x: x**2, -5, 5) == 0
    assert midpoint_convex_violations(abs, -5, 5) == 0
    assert midpoint_convex_violations(lambda x: np.exp(x), -3, 3) == 0


def test_nonconvex_function_has_violation():
    assert midpoint_convex_violations(double_well, -2.5, 2.5) > 0
    assert midpoint_convex_violations(np.sin, 0, 2 * np.pi) > 0


def test_gradient_descent_on_nonconvex_depends_on_start():
    left = descend_1d(double_well_grad, -2.0)
    right = descend_1d(double_well_grad, 2.0)
    assert left == pytest.approx(-1.3, abs=0.01) and right == pytest.approx(1.13, abs=0.01)
    assert double_well(left) < double_well(right)  # left = global min, right = only a local min
    assert abs(double_well_grad(left)) < 1e-6 and abs(double_well_grad(right)) < 1e-6  # both look "converged"


def test_saddle_gradient_zero_but_not_minimum():
    g = lambda v: np.array([2 * v[0], -2 * v[1]])
    assert np.allclose(g(np.zeros(2)), 0)
    assert saddle(np.array([0.0, 0.1])) < saddle(np.zeros(2)) < saddle(np.array([0.1, 0.0]))
    H = np.diag([2.0, -2.0])  # Hessian: mixed-sign eigenvalues => saddle
    w = np.linalg.eigvalsh(H)
    assert w.min() < 0 < w.max()


def test_gradient_descent_escapes_saddle_with_tiny_perturbation():
    v = np.array([0.0, 1e-8])  # off-axis start: y grows, so GD leaves the saddle
    for _ in range(200):  # y grows by 1.2x per step: 1e-8 * 1.2^200 ~ 7
        v = v - 0.1 * np.array([2 * v[0], -2 * v[1]])
    assert abs(v[1]) > 1.0
