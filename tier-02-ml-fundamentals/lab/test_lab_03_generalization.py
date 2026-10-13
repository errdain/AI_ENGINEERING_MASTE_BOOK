import numpy as np
import pytest

import lab_03_generalization as L


def test_underfit_good_fit_overfit_by_degree():
    r = L.ex_2_26_27(); t = r["train_val_mse_by_degree"]
    assert t[0][1] > 5 * t[3][1] and t[1][1] > 3 * t[3][1]             # underfit: high error on both sets
    assert t[15][0] < t[3][0] and t[15][1] > 100 * t[3][1]             # overfit: train falls, validation explodes
    assert r["best_degree_by_val"] in (3, 5, 9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_early_stopping_when_params_exceed_samples(seed):
    r = L.ex_2_27_early_stopping(seed=seed)
    assert r["best_epoch"] < 300 and r["val_at_end"] > r["val_at_best"] * 1.02 and r["train_at_end"] < 1e-6 < r["train_at_best"]


def test_bias_systematic_error_and_sampling_bias():
    r = L.ex_2_28()
    assert r["degree1_bias2"] > 1000 * r["degree5_bias2"]
    assert r["mse_outside_training_range"] > 1000 * r["mse_inside_training_range"]


def test_variance_grows_with_flexibility_and_shrinks_with_data():
    r = L.ex_2_29(); d = r["variance_by_degree"]; n = r["variance_by_n_train(degree3)"]
    assert d[1] < d[3] < d[9] and n[15] > n[30] > n[120] > n[480]
    assert n[30] / n[480] == pytest.approx(480 / 30, rel=0.5)           # roughly 1/n


def test_bias_variance_decomposition_and_optimum():
    r = L.ex_2_30()
    assert r["decomposition_error"] < 1e-12 and r["best_degree"] in (3, 4, 5, 6)
    rows = r["rows"]; assert rows[0]["bias2"] > rows[3]["bias2"] and rows[10]["variance"] > rows[3]["variance"]
    assert rows[1]["bias2"] == pytest.approx(rows[2]["bias2"], abs=1e-3)  # sin is odd: even-degree terms add no bias
    ridge = r["degree10_test_mse_by_ridge"]; assert min(ridge.values()) < ridge[0.0]


def test_generalization_gap_shrinks_with_data_and_memorization_of_random_labels():
    r = L.ex_2_31(); g = r["gap_by_n"]
    assert g[20]["gap"] > g[100]["gap"] > g[1000]["gap"] and g[1000]["gap"] < 0.01
    assert r["random_labels_train_acc_1nn"] == 1.0 and abs(r["random_labels_test_acc_1nn"] - 0.5) < 0.05


def test_regularization_ridge_lasso():
    r = L.ex_2_32(); v = r["val_mse_by_ridge_degree15"]
    assert v[0.0] > 100 * v[r["best_lambda"]] and v[10.0] > v[r["best_lambda"]]  # U-shape in lambda
    assert r["lasso_support"] == [0, 1] and r["ridge_nonzero"] == 10 and r["ridge_max_abs_noise_coef"] < 0.5


def test_lasso_matches_sklearn():
    sk = pytest.importorskip("sklearn.linear_model")
    X = np.random.default_rng(0).standard_normal((100, 6)); y = X @ [3, -2, 0, 0, 1, 0] + 0.5 * np.random.default_rng(1).standard_normal(100)
    w, b = L.lasso_ista(X, y, 0.2, steps=20000); ref = sk.Lasso(alpha=0.2, tol=1e-12, max_iter=100000).fit(X, y)
    np.testing.assert_allclose(w, ref.coef_, atol=1e-4); assert b == pytest.approx(ref.intercept_, abs=1e-4)


def test_leakage_demos():
    r = L.ex_2_33()
    assert r["selection_leak_cv_acc"] > 0.8 and r["selection_inside_fold_cv_acc"] < 0.6        # pure noise, yet 90% when selection leaks
    assert r["target_leak_test_acc"] > 0.95 and abs(r["no_leak_test_acc"] - 0.5) < 0.1
    assert r["forward_split_mse"] > 5 * r["random_split_mse"]


def test_distribution_shift_detection_and_effects():
    r = L.ex_2_34()
    assert r["covariate_shift_shifted_mean_2_mse"] > 10 * r["covariate_shift_same_mse"]
    assert r["psi_same"] < 0.05 and r["psi_shifted"] > 0.25 and r["ks_shifted"] > 10 * r["ks_same"]
    assert abs(r["domain_auc_same"] - 0.5) < 0.05 and r["domain_auc_shifted"] > 0.8
    assert r["precision_at_prevalence_0.05"] < r["precision_at_prevalence_0.5"] - 0.3 and r["recall_at_prevalence_0.05"] == pytest.approx(r["recall_at_prevalence_0.5"], abs=0.05)


def test_random_search_beats_grid_when_one_parameter_matters_and_selection_optimism():
    r = L.ex_2_35_36()
    assert r["mean_best_random"] < r["mean_best_grid"] and r["random_wins_fraction"] > 0.6
    assert r["best_cv_score_over_300_configs_on_noise"] > 0.58 and abs(r["that_config_on_fresh_data"] - 0.5) < 0.03
