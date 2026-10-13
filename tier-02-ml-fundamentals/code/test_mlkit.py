import numpy as np
import pytest

import mlkit as M

sk = pytest.importorskip("sklearn")
from sklearn import linear_model, metrics as skm


# ---------- data and splits (2.7-2.13) ----------
def test_blobs_shapes_labels_and_prevalence():
    X, y = M.make_blobs_binary(400, pos_frac=0.1, seed=1)
    assert X.shape == (400, 2) and set(y) == {0, 1} and y.sum() == 40


def test_split_is_disjoint_complete_and_reproducible():
    tr, va, te = M.train_val_test_indices(1000, seed=3)
    assert len(tr) == 600 and len(va) == 200 and len(te) == 200
    allidx = np.concatenate([tr, va, te])
    assert len(set(allidx)) == 1000 and set(allidx) == set(range(1000))
    tr2, _, _ = M.train_val_test_indices(1000, seed=3)
    np.testing.assert_array_equal(tr, tr2)
    with pytest.raises(ValueError):
        M.train_val_test_indices(10, fractions=(0.5, 0.5, 0.5))


def test_stratified_split_keeps_prevalence():
    y = np.r_[np.ones(50), np.zeros(950)].astype(int)
    for part in M.train_val_test_indices(1000, seed=0, stratify=y):
        assert y[part].mean() == pytest.approx(0.05, abs=0.01)


def test_kfold_partitions_and_stratification():
    y = np.r_[np.ones(20), np.zeros(80)].astype(int)
    folds = M.kfold_indices(100, 5, seed=0, stratify=y)
    vals = np.concatenate([v for _, v in folds])
    assert sorted(vals) == list(range(100))  # each sample validated exactly once
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0 and y[va].sum() == 4
    with pytest.raises(ValueError):
        M.kfold_indices(5, 1)


# ---------- preprocessing (2.47-2.49) ----------
def test_scalers_fit_on_train_only_and_constant_column():
    tr = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    sc = M.StandardScaler().fit(tr)
    z = sc.transform(tr)
    assert np.allclose(z[:, 0].mean(), 0) and np.isfinite(z).all()
    mm = M.MinMaxScaler().fit(tr).transform(np.array([[7.0, 5.0]]))
    assert mm[0, 0] == pytest.approx(1.5)  # test value outside train range is NOT clipped


def test_impute_uses_training_median_and_one_hot_rejects_unseen():
    tr = np.array([[1.0], [2.0], [100.0], [np.nan]])
    out = M.impute_median(tr, np.array([[np.nan], [4.0]]))
    assert out[0, 0] == 2.0 and out[1, 0] == 4.0
    assert M.one_hot(np.array([0, 2]), 3).tolist() == [[1, 0, 0], [0, 0, 1]]
    with pytest.raises(ValueError):
        M.one_hot(np.array([3]), 3)


# ---------- losses (2.17/2.18) ----------
def test_losses_known_values_and_outlier_sensitivity():
    y, p = np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 10.0])
    assert M.mse(y, p) == 25.0 and M.mae(y, p) == 2.5
    assert M.huber(y, p, 1.0) == pytest.approx((9.5) / 4)
    assert M.huber(y, p, 1.0) < M.mae(y, p) < M.mse(y, p)


def test_bce_from_logits_stable_and_matches_sklearn():
    z, y = np.array([-1000.0, 1000.0, 0.3]), np.array([1, 0, 1])
    assert np.isfinite(M.bce_from_logits(y, z))
    z2 = np.array([-2.0, 0.5, 3.0]); y2 = np.array([0, 1, 1])
    assert M.bce_from_logits(y2, z2) == pytest.approx(skm.log_loss(y2, 1 / (1 + np.exp(-z2))))
    assert np.isfinite(M.sigmoid(np.array([-1000.0, 1000.0]))).all()


# ---------- models (2.14-2.16, 2.46) ----------
def test_linear_regression_matches_sklearn_and_ridge():
    X, y = M.make_linear(300, noise=0.5, seed=2)
    ours = M.LinearRegression().fit(X, y)
    ref = linear_model.LinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-8); assert ours.intercept_ == pytest.approx(ref.intercept_)
    r_o = M.LinearRegression(ridge=5.0).fit(X, y); r_s = linear_model.Ridge(alpha=5.0).fit(X, y)
    np.testing.assert_allclose(r_o.coef_, r_s.coef_, atol=1e-8); assert r_o.intercept_ == pytest.approx(r_s.intercept_)
    assert np.linalg.norm(r_o.coef_) < np.linalg.norm(ours.coef_)  # ridge shrinks


def test_logistic_regression_matches_sklearn_unpenalized():
    X, y = M.make_blobs_binary(400, sep=1.5, seed=4)
    ours = M.LogisticRegression(lr=0.5, n_iter=20000).fit(X, y)
    ref = linear_model.LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_[0], atol=2e-2)
    assert (ours.predict(X) == ref.predict(X)).mean() > 0.99
    assert ours.losses_[-1] < ours.losses_[0]  # training loss goes down


def test_kmeans_recovers_blobs_and_inertia_decreases_with_k():
    X, y = M.make_blobs_binary(300, sep=8.0, seed=1)
    km = M.KMeans(2, seed=0).fit(X)
    agree = max((km.labels_ == y).mean(), (km.labels_ != y).mean())
    assert agree > 0.99
    assert M.KMeans(4, seed=0).fit(X).inertia_ < km.inertia_
    assert (km.predict(X) == km.labels_).all()


def test_baselines():
    X, y = M.make_blobs_binary(200, pos_frac=0.1, seed=0)
    b = M.MajorityBaseline().fit(X, y)
    assert b.label_ == 0 and M.accuracy(y, b.predict(X)) == pytest.approx(0.9)
    assert M.recall(y, b.predict(X)) == 0.0  # 90% accurate, finds nothing
    assert M.MeanBaseline().fit(X, y.astype(float)).predict(X)[0] == pytest.approx(y.mean())


# ---------- training loops (2.19-2.25) ----------
@pytest.mark.parametrize("n,bs,drop,expected", [(1000, 100, False, 10), (1050, 100, False, 11), (1050, 100, True, 10), (10, 100, False, 1)])
def test_iterations_per_epoch(n, bs, drop, expected):
    assert M.iterations_per_epoch(n, bs, drop) == expected
    assert len(list(M.minibatches(n, bs, np.random.default_rng(0), drop))) == expected


def test_minibatches_cover_every_sample_once_per_epoch():
    got = np.concatenate(list(M.minibatches(103, 10, np.random.default_rng(1))))
    assert sorted(got) == list(range(103))


def test_sgd_variants_converge_and_count_iterations():
    X, y = M.make_linear(256, noise=0.05, seed=5)
    for bs in (None, 32, 1):
        w, b, h = M.sgd_linear(X, y, lr=0.02 if bs == 1 else 0.1, batch_size=bs, epochs=60, seed=0)
        assert h["epoch_loss"][-1] < 0.01
        per_epoch = 1 if bs is None else M.iterations_per_epoch(256, bs)
        assert h["iterations"][-1] == 60 * per_epoch
    assert w == pytest.approx([2.0, -1.0], abs=0.1) and b == pytest.approx(0.5, abs=0.1)


def test_learning_rate_too_large_diverges_and_too_small_is_slow():
    X, y = M.make_linear(200, noise=0.1, seed=6)
    _, _, big = M.sgd_linear(X, y, lr=5.0, epochs=30)
    _, _, tiny = M.sgd_linear(X, y, lr=1e-4, epochs=30)
    _, _, ok = M.sgd_linear(X, y, lr=0.1, epochs=30)
    assert not np.isfinite(big["epoch_loss"][-1]) or big["epoch_loss"][-1] > 1e3
    assert tiny["epoch_loss"][-1] > 10 * ok["epoch_loss"][-1]


def test_minibatch_noise_larger_than_full_batch():
    X, y = M.make_linear(256, noise=0.5, seed=7)
    _, _, full = M.sgd_linear(X, y, lr=0.05, batch_size=None, epochs=80)
    _, _, single = M.sgd_linear(X, y, lr=0.05, batch_size=1, epochs=80)
    assert np.std(np.diff(single["epoch_loss"][-20:])) > np.std(np.diff(full["epoch_loss"][-20:]))


# ---------- metrics (2.37-2.43) ----------
def test_confusion_and_scores_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500); yhat = np.where(rng.random(500) < 0.8, y, 1 - y)
    cm = M.confusion_matrix_binary(y, yhat)
    np.testing.assert_array_equal(cm, skm.confusion_matrix(y, yhat))
    assert M.accuracy(y, yhat) == pytest.approx(skm.accuracy_score(y, yhat))
    assert M.precision(y, yhat) == pytest.approx(skm.precision_score(y, yhat))
    assert M.recall(y, yhat) == pytest.approx(skm.recall_score(y, yhat))
    assert M.f1(y, yhat) == pytest.approx(skm.f1_score(y, yhat))
    assert M.fbeta(y, yhat, 2) == pytest.approx(skm.fbeta_score(y, yhat, beta=2))


def test_metrics_edge_cases():
    y = np.array([0, 0, 1, 1])
    assert M.precision(y, np.zeros(4, int)) == 0.0 and M.precision(y, np.zeros(4, int), zero_division=1.0) == 1.0
    assert M.f1(y, np.zeros(4, int)) == 0.0
    assert M.f1(y, y) == 1.0


def test_roc_auc_matches_sklearn_ties_and_rank_form():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400); s = np.round(rng.random(400) * 0.5 + 0.4 * y, 1)  # many ties
    assert M.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert M.roc_auc_rank(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    f, t, _ = M.roc_curve(y, s); f2, t2, _ = skm.roc_curve(y, s, drop_intermediate=False)
    np.testing.assert_allclose(np.sort(f), np.sort(f2)) if len(f) == len(f2) else None
    assert M.roc_auc(y, np.zeros(400)) == pytest.approx(0.5)  # constant score = chance
    assert M.roc_auc(y, -s) == pytest.approx(1 - M.roc_auc(y, s))
    with pytest.raises(ValueError):
        M.roc_auc(np.ones(5, int), np.arange(5.0))


def test_auc_is_rank_based_so_invariant_to_monotone_transform():
    rng = np.random.default_rng(2); y = rng.integers(0, 2, 300); s = rng.random(300) + 0.5 * y
    assert M.roc_auc(y, s) == pytest.approx(M.roc_auc(y, np.exp(5 * s)))


def test_calibration_metrics_and_overconfidence():
    rng = np.random.default_rng(3)
    p_true = rng.random(20000); y = (rng.random(20000) < p_true).astype(int)
    assert M.expected_calibration_error(y, p_true) < 0.02  # calibrated
    logit = np.log(p_true / (1 - p_true)); over = 1 / (1 + np.exp(-3 * logit))  # strictly monotone sharpening
    assert M.expected_calibration_error(y, over) > 0.1  # pushed toward 0/1: overconfident
    assert M.brier_score(y, p_true) == pytest.approx(skm.brier_score_loss(y, p_true))
    assert M.roc_auc(y, over) == pytest.approx(M.roc_auc(y, p_true), abs=1e-9)  # ranking unchanged, calibration ruined


# ---------- imbalance (2.45/2.50) ----------
def test_accuracy_paradox_and_class_weights():
    X, y = M.make_blobs_binary(2000, sep=2.5, pos_frac=0.02, seed=8)
    tr, _, te = M.train_val_test_indices(2000, (0.7, 0.1, 0.2), seed=0, stratify=y)
    Xs = M.StandardScaler().fit(X[tr])
    plain = M.LogisticRegression(n_iter=3000).fit(Xs.transform(X[tr]), y[tr])
    wtd = M.LogisticRegression(n_iter=3000, class_weight=M.balanced_class_weights(y[tr])).fit(Xs.transform(X[tr]), y[tr])
    maj = M.accuracy(y[te], np.zeros(len(te), int))
    assert maj > 0.97
    assert M.recall(y[te], wtd.predict(Xs.transform(X[te]))) > M.recall(y[te], plain.predict(Xs.transform(X[te])))
    assert M.balanced_class_weights(np.r_[np.zeros(90), np.ones(10)]) == {0: pytest.approx(100 / 180), 1: pytest.approx(5.0)}


def test_oversample_balances_and_threshold_tuning_improves_f1():
    X, y = M.make_blobs_binary(1000, sep=2.0, pos_frac=0.05, seed=9)
    Xo, yo = M.random_oversample(X, y)
    assert yo.mean() == pytest.approx(0.5) and len(yo) == 2 * (y == 0).sum()
    m = M.LogisticRegression(n_iter=3000).fit(X, y); s = m.predict_proba(X)
    t, f = M.best_f1_threshold(y, s)
    assert f >= M.f1(y, (s >= 0.5).astype(int)) and t < 0.5


# ---------- shift (2.34) ----------
def test_psi_and_ks_detect_shift():
    rng = np.random.default_rng(0)
    a, same, shifted = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000), rng.normal(0.8, 1.3, 5000)
    assert M.psi(a, same) < 0.05 and M.psi(a, shifted) > 0.25
    from scipy.stats import ks_2samp
    assert M.ks_statistic(a, shifted) == pytest.approx(ks_2samp(a, shifted).statistic)
    assert M.ks_statistic(a, a) == 0.0


# ---------- hyperparameters (2.35/2.36) ----------
def test_grid_and_random_search_find_good_region():
    obj = lambda p: (np.log10(p["lr"]) + 2) ** 2 + (p["l2"] - 0.1) ** 2
    bg, sg, hist = M.grid_search(obj, {"lr": [1e-4, 1e-3, 1e-2, 1e-1], "l2": [0.0, 0.1, 1.0]})
    assert bg == {"lr": 1e-2, "l2": 0.1} and sg == 0 and len(hist) == 12
    br, sr, h2 = M.random_search(obj, {"lr": lambda r: 10 ** r.uniform(-5, 0), "l2": lambda r: r.uniform(0, 1)}, 200, seed=0)
    assert sr < 0.05 and len(h2) == 200
    assert M.random_search(obj, {"lr": lambda r: 10 ** r.uniform(-5, 0), "l2": lambda r: r.uniform(0, 1)}, 200, seed=0)[1] == sr  # reproducible


# ---------- bias / variance / overfitting (2.26-2.32) ----------
def test_under_and_overfitting_curves():
    f = lambda x: np.sin(np.pi * x)
    x, y = M.make_sine(30, 0.2, seed=1); xv, yv = M.make_sine(300, 0.2, seed=2)
    res = {d: M.fit_poly_train_val(d, x, y, xv, yv) for d in (1, 3, 15)}
    assert res[1].train_mse > res[3].train_mse  # underfit: high error everywhere
    assert res[15].train_mse < res[3].train_mse  # train error keeps falling...
    assert res[15].val_mse > res[3].val_mse  # ...but validation error is worse: overfitting
    r_ridge = M.fit_poly_train_val(15, x, y, xv, yv, ridge=1e-3)
    assert r_ridge.val_mse < res[15].val_mse  # regularization helps


def test_bias_variance_tradeoff_with_degree_and_ridge():
    f = lambda x: np.sin(np.pi * x)
    lo, mid, hi = (M.bias_variance_poly(d, f, n_datasets=300, seed=0) for d in (1, 3, 12))
    assert lo["bias2"] > mid["bias2"] > 0 and hi["variance"] > mid["variance"] > lo["variance"]
    assert lo["variance"] == pytest.approx(0.04 * 2 / 30, rel=0.25) and hi["variance"] == pytest.approx(0.04 * 13 / 30, rel=0.25)  # sigma^2 * p / n
    assert mid["expected_test_mse"] < lo["expected_test_mse"] and mid["expected_test_mse"] < hi["expected_test_mse"]
    hi_r = M.bias_variance_poly(12, f, n_datasets=300, ridge=1e-2, seed=0)
    assert hi_r["variance"] < hi["variance"]


# ---------- data leakage (2.33) ----------
def test_scaling_before_split_changes_statistics_and_feature_leak_inflates_score():
    X, y = M.make_blobs_binary(400, sep=0.0, seed=10)  # labels carry NO signal
    leak = np.column_stack([X, y + 0.05 * np.random.default_rng(0).standard_normal(400)])  # feature derived from the label
    tr, _, te = M.train_val_test_indices(400, (0.7, 0.1, 0.2), seed=0)
    clean = M.LogisticRegression(n_iter=2000).fit(X[tr], y[tr])
    leaky = M.LogisticRegression(n_iter=2000).fit(leak[tr], y[tr])
    assert M.accuracy(y[te], clean.predict(X[te])) < 0.65  # honest: no signal, near chance
    assert M.accuracy(y[te], leaky.predict(leak[te])) > 0.95  # leaked target: "great" score that fails in production
    all_scaled = (X - X.mean(0)) / X.std(0)
    assert not np.allclose(all_scaled[tr].mean(0), 0, atol=1e-12)  # train stats were polluted by test rows


# ---------- additional metrics (2.40-2.44) ----------
def test_multiclass_confusion_and_f1_averages_match_sklearn():
    rng = np.random.default_rng(0); y = rng.integers(0, 4, 400); yhat = np.where(rng.random(400) < 0.7, y, rng.integers(0, 4, 400))
    cm = M.confusion_matrix_multiclass(y, yhat)
    np.testing.assert_array_equal(cm, skm.confusion_matrix(y, yhat))
    assert M.f1_macro(cm) == pytest.approx(skm.f1_score(y, yhat, average="macro"))
    assert M.f1_micro(cm) == pytest.approx(skm.f1_score(y, yhat, average="micro")) == pytest.approx(M.accuracy(y, yhat))
    p, r, f = M.precision_recall_f1_per_class(cm)
    np.testing.assert_allclose(p, skm.precision_score(y, yhat, average=None)); np.testing.assert_allclose(r, skm.recall_score(y, yhat, average=None))


def test_average_precision_matches_sklearn_and_is_prevalence_sensitive():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int); s = rng.normal(0, 1, 2000) + 2.0 * y
    assert M.average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))
    tied = np.round(s)  # ties
    assert M.average_precision(y, tied) == pytest.approx(skm.average_precision_score(y, tied))
    assert M.average_precision(y, rng.random(2000)) == pytest.approx(y.mean(), abs=0.03)  # random scores: AP ~ prevalence, AUC ~ 0.5


def test_time_series_splits_never_train_on_the_future():
    for tr, va in M.time_series_splits(100, 4):
        assert tr.max() < va.min() and len(va) > 0
