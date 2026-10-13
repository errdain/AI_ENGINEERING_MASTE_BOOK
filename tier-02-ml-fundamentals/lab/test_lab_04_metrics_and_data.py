import numpy as np
import pytest

import lab_04_metrics_and_data as L


def test_accuracy_paradox():
    r = L.ex_2_37()
    assert r["accuracy_always_negative"] > 0.98 and r["recall_always_negative"] == 0 and r["balanced_accuracy_always_negative"] == 0.5
    assert r["balanced_accuracy_model"] > 0.7


def test_precision_recall_tradeoff_and_prevalence_dependence():
    r = L.ex_2_38_39(); s = r["(precision, recall) by threshold"]
    ts = sorted(s); assert all(s[a][0] <= s[b][0] and s[a][1] >= s[b][1] for a, b in zip(ts, ts[1:]))   # precision up, recall down as threshold rises
    ppv = r["ppv_by_prevalence"]; assert ppv[0.5] > 0.9 and ppv[0.01] < 0.2 and ppv[0.001] < 0.02


def test_f1_properties_and_averaging():
    r = L.ex_2_40(); h, a = r["harmonic_vs_arithmetic_mean"]
    assert h == pytest.approx(0.18) and a == pytest.approx(0.5)
    f2, f1 = r["f2_vs_f1_for_recall_heavy"]; assert f2 > f1
    assert r["imbalanced_multiclass_macro_f1"] < r["imbalanced_multiclass_micro_f1_equals_accuracy"]


def test_roc_auc_hides_what_pr_auc_shows_under_imbalance():
    r = L.ex_2_41()
    assert max(v["roc_auc"] for v in r.values()) - min(v["roc_auc"] for v in r.values()) < 0.01
    assert r[0.5]["average_precision"] > r[0.05]["average_precision"] > r[0.005]["average_precision"] + 0.3
    assert r[0.005]["recall_at_0.5"] < 0.1


def test_confusion_matrix_and_cost():
    r = L.ex_2_42(); cm = np.array(r["cm"]); assert cm.sum() == 3000 and r["expected_cost_per_sample"] == pytest.approx((cm[0, 1] + 10 * cm[1, 0]) / 3000)
    assert np.array(r["multiclass_cm"]).sum() == 300


def test_calibration_fix_changes_probabilities_not_ranking():
    r = L.ex_2_43()
    assert r["raw"]["ece"] > 2 * r["temperature_scaled"]["ece"] and r["raw"]["ece"] > 2 * r["platt_scaled"]["ece"]
    assert r["temperature_scaled"]["auc"] == pytest.approx(r["raw"]["auc"], abs=1e-9) and r["platt_scaled"]["auc"] == pytest.approx(r["raw"]["auc"], abs=1e-3)
    assert r["temperature"] > 1                                                  # raw model was overconfident
    first, last = r["reliability_raw"][0], r["reliability_raw"][-1]; assert first[1] < first[2] and last[1] > last[2]   # under-predicts low, over-predicts high


def test_cross_validation_reduces_estimate_variance_and_respects_time():
    r = L.ex_2_44()
    assert r["cv5_30_repeats"][1] < r["holdout_30_splits"][1] / 3
    assert abs(r["cv5_30_repeats"][0] - r["true_accuracy_on_big_set"]) < 0.06
    assert r["time_series_random_kfold_mse"] < r["time_series_forward_mse"]       # random folds are optimistic for time-ordered data
    assert r["folds_stratified_positive_counts"] == [10] * 5


def test_class_imbalance_methods_move_operating_point_not_ranking():
    r = L.ex_2_45()
    assert r["class_weights"]["recall"] > r["plain"]["recall"] + 0.2 and r["class_weights"]["precision"] < r["plain"]["precision"] - 0.3
    assert r["class_weights"]["pr_auc"] == pytest.approx(r["plain"]["pr_auc"], abs=0.02) and r["oversample"]["pr_auc"] == pytest.approx(r["plain"]["pr_auc"], abs=0.02)
    assert r["plain"]["accuracy"] > r["class_weights"]["accuracy"]                  # accuracy rewards ignoring the minority
    assert r["plain+tuned_threshold(train)"]["recall"] > r["plain"]["recall"]


def test_baselines_can_be_hard_to_beat():
    r = L.ex_2_46(); f = r["forecast_mse"]
    assert f["naive_yesterday"] < f["mean"] and f["naive_yesterday"] < f["linear_trend"]
    assert r["classification_acc"]["logistic"] > r["classification_acc"]["majority"] + 0.05


def test_scaling_effects():
    r = L.ex_2_47(); s = r["spread_of_normal_points_after_scaling_with_one_outlier"]
    assert s["robust"] > 10 * s["standard"] and s["standard"] > 10 * s["minmax"]
    e = r["epochs_to_loss_below_0.02"]; assert e["raw_scales_1_and_100"] is None and e["standardized"] < 50
    assert r["ols_prediction_invariant"] and abs(r["skew_before_after_log"][0]) > 3 and abs(r["skew_before_after_log"][1]) < 0.3


def test_robust_scaler_matches_sklearn():
    pre = pytest.importorskip("sklearn.preprocessing"); X = np.random.default_rng(0).lognormal(0, 1, (200, 3))
    np.testing.assert_allclose(L.RobustScaler().fit(X).transform(X), pre.RobustScaler().fit_transform(X))


def test_feature_engineering_cyclic_and_target_encoding_leak():
    r = L.ex_2_48()
    assert r["hour_distance_23_to_0_raw_vs_cyclic"][1] < 0.3 < 1 and r["hour_distance_12_to_0_raw_vs_cyclic"][1] == pytest.approx(2.0)
    tr_acc, te_acc = r["target_encoding_train_vs_test_acc_naive"]; assert tr_acc > te_acc + 0.15 and abs(te_acc - 0.5) < 0.08
    assert r["target_encoding_naive_corr_with_label"] > 10 * abs(r["target_encoding_oof_corr_with_label"])


def test_preprocessing_pipeline_and_informative_missingness():
    r = L.ex_2_49()
    assert r["pipeline_equals_manual_train_only_fit"] and r["acc_with_missing_indicator"] > r["acc_median_impute_only"] + 0.03
    assert r["acc_with_gross_outliers_clipped"] > r["acc_with_gross_outliers_unclipped"] + 0.02


def test_dataset_prevalence_shift_and_prior_correction():
    r = L.ex_2_50()
    assert r["mean_predicted_prob_raw"] > 5 * r["production_prevalence"] and r["mean_predicted_prob_corrected"] == pytest.approx(r["production_prevalence"], abs=0.005)
    assert r["ece_corrected"] < 0.1 * r["ece_raw"] and r["auc_corrected"] == pytest.approx(r["auc_raw"], abs=1e-12)
    assert r["precision_at_0.5_corrected"] > r["precision_at_0.5_raw"] + 0.4


def test_metric_choice_depends_on_decision_context():
    r = L.ex_2_51(); m = r["metrics"]
    assert m["plain"]["roc_auc"] == pytest.approx(m["weighted"]["roc_auc"], abs=0.002) and m["plain"]["pr_auc"] == pytest.approx(m["weighted"]["pr_auc"], abs=0.005)
    assert m["plain"]["f1@0.5"] > m["weighted"]["f1@0.5"] and m["plain"]["log_loss"] < m["weighted"]["log_loss"]
    c = r["expected_cost_at_fixed_threshold_0.5"]
    assert c["fp_costly(1:5)"]["plain"] < c["fp_costly(1:5)"]["weighted"] and c["fn_costly(20:1)"]["weighted"] < c["fn_costly(20:1)"]["plain"]   # winner flips with the cost ratio
