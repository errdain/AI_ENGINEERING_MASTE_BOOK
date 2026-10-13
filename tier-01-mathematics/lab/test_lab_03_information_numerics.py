import numpy as np
import pytest

import lab_03_information_numerics as L


def test_covariance_bayes_chain_rule():
    r = L.ex_1_32(); assert r["major_axis_deg"] == pytest.approx(45, abs=2) and r["explained_ratio"] == pytest.approx(7 / 8, abs=0.02)
    c = L.ex_1_33(); assert c["prob"] == pytest.approx(c["direct"]) and c["log_prob"] == pytest.approx(np.log(0.42))
    b = L.ex_1_34()
    assert b["posterior_at_1pct"] == pytest.approx(1 / 6)
    assert b["prevalence_for_50pct"] == pytest.approx(b["analytic_prevalence"], abs=0.003)


def test_mle_entropy_cross_entropy_kl():
    assert L.ex_1_35()["closed_form"] == pytest.approx(L.ex_1_35()["grid"], abs=0.002)
    h = L.ex_1_36(); ts = sorted(h); assert all(h[a] < h[b] for a, b in zip(ts, ts[1:]))  # entropy rises with T
    assert L.ex_1_37()["max_abs_diff"] < 1e-7
    k = L.ex_1_38()
    assert abs(k["forward_fit"][0]) < 0.6 and k["forward_fit"][1] > 2.5  # mass-covering: wide, centred
    assert abs(abs(k["reverse_fit"][0]) - 3) < 0.6 and k["reverse_fit"][1] < 1.6  # mode-seeking: narrow, on one mode


def test_numerics_labs():
    v = L.ex_1_39()
    assert abs(v["two_pass"] - v["reference"]) < 1e-6 and abs(v["welford"] - v["reference"]) < 1e-6
    assert abs(v["naive"] - v["reference"]) > 10  # naive formula is wildly wrong at a 1e9 offset
    f = L.ex_1_40(); assert f["pairwise_f32_err"] < f["naive_f32_err"] / 5
    p = L.ex_1_41(); assert p["direct_product"] == 0.0 and np.isfinite(p["log_prob"])
    for name, r in L.ex_1_42().items():
        assert r["largest_finite_arg"] == pytest.approx(r["ln_max"], abs=0.01), name
    o = L.ex_1_43(); assert o["online"] == pytest.approx(o["batch"])
    g = L.ex_1_44(); assert g["stopped_at"] == 3 and "nan" in g["message"]


def test_online_logsumexp_edge_cases():
    o = L.OnlineLogSumExp(); assert o.value == float("-inf")
    o.update(np.array([-np.inf, -np.inf])); assert o.value == float("-inf")
    o.update(np.array([1000.0, 1000.0])); assert o.value == pytest.approx(1000 + np.log(2))


def test_convexity_and_minima_labs():
    c = L.ex_1_45()
    assert c["logistic_spread"] < 1e-6  # convex: same optimum from every init
    assert c["mlp_spread"] > 0.01 and c["mlp_mean"] < c["logistic_mean"]  # non-convex: init-dependent but far better fit
    m = L.ex_1_46()
    assert m["gd_unstable_lr_0.11"] is None and m["momentum_lr_0.05_beta_0.8"] < m["gd_best_lr_0.099"]
    b = L.ex_1_47(); assert b["start_rule_holds"] and b["to_global_min"] > 0 and b["to_local_min"] > 0
    assert b["basin_boundary"] == pytest.approx(0.17, abs=0.005)
