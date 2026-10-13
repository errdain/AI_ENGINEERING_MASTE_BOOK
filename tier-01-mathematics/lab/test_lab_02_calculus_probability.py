import numpy as np
import pytest

import lab_02_calculus_probability as L


def test_activation_ranges_and_flat_regions():
    r = L.ex_1_20()
    assert 0 < r["sigmoid"]["min"] < r["sigmoid"]["max"] < 1
    assert r["tanh"]["min"] > -1 and r["relu"]["min"] == 0
    assert r["tanh"]["flat_fraction"] > r["sigmoid"]["flat_fraction"] > 0
    assert r["relu"]["flat_fraction"] == pytest.approx(0.5, abs=0.01)


def test_derivative_step_size_sweet_spot():
    r = L.ex_1_21()
    assert r["best_h_central"] == pytest.approx(1e-5) and r["best_h_forward"] == pytest.approx(1e-8)
    assert r["best_err_central"] < r["best_err_forward"]


def test_partials_and_gd_stability_threshold():
    r = L.ex_1_22(); np.testing.assert_allclose(r["analytic"], r["numerical"], atol=1e-6)
    g = L.ex_1_23()
    assert g[0.09] < g[0.05] < g[0.01] and g[0.11] > 1e10  # eta < 0.1 converges, 0.11 diverges


def test_jacobian_and_backprop_with_bias():
    r = L.ex_1_24(); np.testing.assert_allclose(r["by_hand"], r["numerical"], atol=1e-6)
    b = L.ex_1_25()
    assert b["tanh"] < 1e-6 and b["relu"] < 1e-6


def test_probability_labs():
    r = L.ex_1_26(); assert r["estimate"] == pytest.approx(r["exact"], abs=0.005)
    s = L.ex_1_27(); assert s["diff_within_noise"]
    d = L.ex_1_28()
    assert d["normal"]["mean"] == pytest.approx(d["lognormal"]["mean"], rel=0.05)
    assert d["lognormal"]["p50"] < 0.7 * d["normal"]["p50"] and d["lognormal"]["p99"] > d["normal"]["p99"]


def test_robust_means_and_bessel():
    m = L.ex_1_29()
    assert m["mean"] > 1000 and m["median"] == pytest.approx(100, abs=1) and m["trimmed_mean"] == pytest.approx(100, abs=1)
    v = L.ex_1_30()
    assert v["mean_var_ddof0"] == pytest.approx(0.8, abs=0.03) and v["mean_var_ddof1"] == pytest.approx(1.0, abs=0.03)


def test_standard_scaler_no_leakage_and_zero_variance():
    r = L.ex_1_31()
    assert np.allclose(r["train_mean"], 0, atol=1e-9) and r["train_std"][0] == pytest.approx(1.0)
    assert r["test_mean_col0"] > 0.3 and r["const_col_finite"]  # test shift is NOT erased (fit on train only)
