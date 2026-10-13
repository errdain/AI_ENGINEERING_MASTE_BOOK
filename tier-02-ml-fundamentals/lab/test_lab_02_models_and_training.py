import numpy as np
import pytest

import lab_02_models_and_training as L


def test_classification_multiclass_and_cost_sensitive_threshold():
    r = L.ex_2_14()
    assert r["multiclass_acc"] > 0.95 > r["chance"] + 0.6 and r["probs_sum_to_1"]
    assert r["best_cost_threshold"] < 0.5 and r["cost_at_best"] < r["cost_at_0.5"]  # misses cost more => lower threshold


def test_regression_recovers_coefficients_and_beats_mean_baseline():
    r = L.ex_2_15()
    assert r["coef"] == pytest.approx([2.0, -1.0], abs=0.1) and r["intercept"] == pytest.approx(0.5, abs=0.1)
    assert r["test_mse"] < 0.05 * r["baseline_mse"] and r["r2"] > 0.95 and abs(r["residual_mean"]) < 1e-9


def test_clustering_scaling_and_initialization():
    r = L.ex_2_16()
    s = r["silhouette_by_k"]; assert s[2] > s[3] > s[4]
    assert r["agree_clean"] > 0.99 and r["agree_unscaled_huge_units"] < 0.9 and r["agree_after_scaling"] > 0.95
    i = r["init_sensitivity_5_blobs"]
    assert i["n_init_1"][1] > 2 * i["n_init_1"][0]  # a single random init can be much worse
    assert i["n_init_10"][1] == pytest.approx(i["n_init_10"][0], rel=1e-6)  # best-of-10 is stable here


def test_silhouette_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    X, _ = L.make_blobs_multi(200, 3, seed=2); lab = L.M.KMeans(3, seed=0).fit(X).labels_
    assert L.silhouette(X, lab) == pytest.approx(sk.silhouette_score(X, lab))


def test_loss_choice_outliers_and_saturation():
    r = L.ex_2_17(); f = r["slope_and_intercept_by_loss(true=(3, 1))"]
    assert abs(f["mse"][1] - 1) > 1.5 and abs(f["mae"][1] - 1) < 0.1 and abs(f["huber"][1] - 1) < 0.2  # robust losses ignore the 5 gross outliers
    assert abs(r["bce_grad_wrt_logit"]) > 100 * abs(r["mse_on_sigmoid_grad_wrt_logit"])  # MSE-on-sigmoid saturates when confidently wrong


def test_objective_is_loss_plus_penalty_and_ridge_shrinks():
    r = L.ex_2_18()["ridge"]
    assert r[0.0]["coef_norm"] > r[1.0]["coef_norm"] > r[10.0]["coef_norm"] and r[10.0]["data_loss"] > r[0.0]["data_loss"]
    assert r[10.0]["objective"] == pytest.approx(r[10.0]["data_loss"] * 40 + r[10.0]["penalty"])
    assert L.ex_2_18()["f1_metric_at_0.5"] < 0.8  # the minimized objective (log loss) is not the metric we report


def test_gradient_descent_reaches_closed_form_solution_monotonically():
    r = L.ex_2_19()
    assert r["gd_w"] == pytest.approx(r["closed_form_w"], abs=2e-3) and r["gd_b"] == pytest.approx(r["closed_form_b"], abs=2e-3)
    assert r["monotone_decrease"] and r["steps_to_within_5pct_of_optimum"] < 100


def test_minibatch_gradient_unbiased_variance_falls_with_batch_and_decay_beats_constant():
    r = L.ex_2_20_21()
    assert all(v < 1e-9 for v in r["unbiased_check"].values())
    v = r["variance_by_batch"]; assert v[1] > v[8] > v[64] and v[1] / v[64] == pytest.approx(64, rel=0.5)  # ~ 1/B
    d = r["distance_to_optimum_after_6000_sgd_steps"]; assert d["decay"] < d["const_0.05"]


def test_epoch_batch_iteration_accounting():
    r = L.ex_2_22_24()
    assert r["iterations_per_epoch"] == 11 and r["iterations_per_epoch_drop_last"] == 10 and r["total_iterations_after_3_epochs"] == 33
    assert r["batch_sizes_in_one_epoch"][-1] == 50 and sum(r["batch_sizes_in_one_epoch"]) == 1050
    assert r["pure_sgd_iterations_per_epoch"] == 1050


def test_learning_rate_stability_limit_from_curvature():
    r = L.ex_2_25()
    assert r["loss_at_0.9_limit"] < 0.02 and r["loss_at_1.1_limit"] > 1e6  # threshold is 2 / lambda_max
    s = r["sweep"]; assert s[1e-4] > s[1e-3] > s[1e-2] > s[1e-1] - 1e-9
    assert r["stability_limit_with_unscaled_feature"] < r["stability_limit"] / 100  # unscaled features shrink the usable LR
