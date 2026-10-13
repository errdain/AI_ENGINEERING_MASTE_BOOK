"""Tier 2 Lab, part 4: concepts 2.37-2.51 (metrics, calibration, cross-validation, imbalance, baselines, scaling, features, preprocessing, metric selection)."""
from __future__ import annotations

import numpy as np

import mlkit as M


def _scores(seed=0, n=6000, pos_frac=0.01, sep=2.5, n_iter=1500):
    X, y = M.make_blobs_binary(n, sep=sep, pos_frac=pos_frac, seed=seed)
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    m = M.LogisticRegression(n_iter=n_iter).fit(X[tr], y[tr])
    return y[te], m.predict_proba(X[te]), m


# ---- 2.37 accuracy -------------------------------------------------------------------------------
def ex_2_37(seed: int = 0) -> dict:
    yte, p, _ = _scores(seed, pos_frac=0.01)
    pred = (p >= 0.5).astype(int); maj = np.zeros_like(yte)
    bal = lambda y, yh: 0.5 * (M.recall(y, yh) + (1 - M.confusion_matrix_binary(y, yh)[0, 1] / max((y == 0).sum(), 1)))
    return {"prevalence": float(yte.mean()), "accuracy_model": M.accuracy(yte, pred), "accuracy_always_negative": M.accuracy(yte, maj),
            "recall_always_negative": M.recall(yte, maj), "balanced_accuracy_model": float(bal(yte, pred)), "balanced_accuracy_always_negative": float(bal(yte, maj))}


# ---- 2.38 / 2.39 precision and recall ------------------------------------------------------------
def ex_2_38_39(seed: int = 0) -> dict:
    yte, p, _ = _scores(seed, pos_frac=0.1, sep=2.0)
    sweep = {t: (round(M.precision(yte, (p >= t).astype(int)), 3), round(M.recall(yte, (p >= t).astype(int)), 3)) for t in (0.1, 0.3, 0.5, 0.7, 0.9)}
    # precision from TPR, FPR and prevalence: PPV = TPR*pi / (TPR*pi + FPR*(1-pi))
    tpr, fpr = 0.9, 0.05
    ppv = lambda pi: tpr * pi / (tpr * pi + fpr * (1 - pi))
    return {"(precision, recall) by threshold": sweep, "ppv_by_prevalence": {pi: ppv(pi) for pi in (0.5, 0.1, 0.01, 0.001)}}


# ---- 2.40 F1 ---------------------------------------------------------------------------------------
def ex_2_40(seed: int = 0) -> dict:
    p, r = 0.9, 0.1
    rng = np.random.default_rng(seed); y = rng.integers(0, 4, 600); yhat = np.where(rng.random(600) < 0.7, y, rng.integers(0, 4, 600))
    y2 = np.where(rng.random(600) < 0.8, 0, rng.integers(1, 4, 600)); yh2 = np.where(rng.random(600) < 0.85, y2, 0)  # imbalanced multiclass
    cm = M.confusion_matrix_multiclass(y2, yh2, 4)
    return {"harmonic_vs_arithmetic_mean": (2 * p * r / (p + r), (p + r) / 2), "f2_vs_f1_for_recall_heavy": (M.fbeta(np.array([1, 1, 1, 1, 0]), np.array([1, 1, 1, 1, 1]), 2), M.f1(np.array([1, 1, 1, 1, 0]), np.array([1, 1, 1, 1, 1]))),
            "imbalanced_multiclass_macro_f1": M.f1_macro(cm), "imbalanced_multiclass_micro_f1_equals_accuracy": M.f1_micro(cm), "per_class_recall": M.precision_recall_f1_per_class(cm)[1].round(2).tolist()}


# ---- 2.41 ROC / AUC ----------------------------------------------------------------------------------
def ex_2_41(seed: int = 0) -> dict:
    out = {}
    for pf in (0.5, 0.05, 0.005):
        yte, p, _ = _scores(seed, n=40000, pos_frac=pf, sep=2.0)
        out[pf] = {"roc_auc": M.roc_auc(yte, p), "average_precision": M.average_precision(yte, p), "precision_at_0.5": M.precision(yte, (p >= 0.5).astype(int)), "recall_at_0.5": M.recall(yte, (p >= 0.5).astype(int))}
    return out


# ---- 2.42 confusion matrix ---------------------------------------------------------------------------
def ex_2_42(seed: int = 0) -> dict:
    yte, p, _ = _scores(seed, pos_frac=0.2, sep=2.0)
    cm = M.confusion_matrix_binary(yte, (p >= 0.5).astype(int)); cost = np.array([[0, 1], [10, 0]])  # a miss costs 10, a false alarm costs 1
    k, y3 = 3, np.random.default_rng(seed).integers(0, 3, 300)
    yh3 = np.where(np.random.default_rng(seed + 1).random(300) < 0.75, y3, (y3 + 1) % 3)
    return {"cm": cm.tolist(), "expected_cost_per_sample": float((cm * cost).sum() / cm.sum()), "multiclass_cm": M.confusion_matrix_multiclass(y3, yh3).tolist()}


# ---- 2.43 calibration -------------------------------------------------------------------------------------
def fit_temperature(logits, y) -> float:
    grid = np.exp(np.linspace(-2, 3, 400)); return float(grid[np.argmin([M.bce_from_logits(y, logits / T) for T in grid])])


def fit_platt(logits, y):
    m = M.LogisticRegression(lr=0.5, n_iter=3000).fit(np.asarray(logits)[:, None], y); return lambda z: m.predict_proba(np.asarray(z)[:, None])


def ex_2_43(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); n, d = 300, 40                  # many features + few samples -> overconfident
    w = np.zeros(d); w[:3] = 1.0
    gen = lambda m: (lambda X: (X, (rng.random(m) < M.sigmoid(X @ w)).astype(int)))(rng.standard_normal((m, d)))
    Xtr, ytr = gen(n); Xva, yva = gen(2000); Xte, yte = gen(5000)
    m = M.LogisticRegression(lr=0.3, n_iter=3000).fit(Xtr, ytr)
    zva, zte = m.decision_function(Xva), m.decision_function(Xte)
    T = fit_temperature(zva, yva); platt = fit_platt(zva, yva)
    res = lambda p: {"ece": M.expected_calibration_error(yte, p), "brier": M.brier_score(yte, p), "auc": M.roc_auc(yte, p)}
    return {"raw": res(M.sigmoid(zte)), "temperature_scaled": res(M.sigmoid(zte / T)), "platt_scaled": res(platt(zte)), "temperature": T,
            "reliability_raw": [(c, round(a, 2), round(b, 2)) for c, a, b in M.reliability_bins(yte, M.sigmoid(zte), 5) if c > 0]}


# ---- 2.44 cross-validation ----------------------------------------------------------------------------------
def _cv_score(X, y, k, seed, stratify=None):
    s = []
    for tr, va in M.kfold_indices(len(y), k, seed=seed, stratify=stratify):
        sc = M.StandardScaler().fit(X[tr]); m = M.LogisticRegression(n_iter=500).fit(sc.transform(X[tr]), y[tr])
        s.append(M.accuracy(y[va], m.predict(sc.transform(X[va]))))
    return float(np.mean(s)), float(np.std(s))


def ex_2_44(seed: int = 0) -> dict:
    X, y = M.make_blobs_binary(2500, sep=1.5, seed=seed); Xs, ys = X[:100], y[:100]; Xbig, ybig = X[100:], y[100:]
    full = M.LogisticRegression(n_iter=500).fit(M.StandardScaler().fit_transform(Xs), ys); sc = M.StandardScaler().fit(Xs)
    truth = M.accuracy(ybig, full.predict(sc.transform(Xbig)))
    holdout = []
    for s in range(30):
        perm = np.random.default_rng(s).permutation(100); tr, va = perm[:70], perm[70:]
        sc2 = M.StandardScaler().fit(Xs[tr]); holdout.append(M.accuracy(ys[va], M.LogisticRegression(n_iter=500).fit(sc2.transform(Xs[tr]), ys[tr]).predict(sc2.transform(Xs[va]))))
    cv5 = [_cv_score(Xs, ys, 5, s)[0] for s in range(30)]
    n, rng = 500, np.random.default_rng(seed); t = np.arange(n); yts = np.sin(t / 30) + 0.3 * rng.standard_normal(n); feat = t[:, None] / n
    forward, rand = [], []
    for tr, va in M.time_series_splits(n, 4):
        nn = lambda a, b: yts[a][np.abs(feat[b] - feat[a].T).argmin(1)]
        forward.append(M.mse(yts[va], nn(tr, va)))
    for tr, va in M.kfold_indices(n, 4, seed=0):
        rand.append(M.mse(yts[va], yts[tr][np.abs(feat[va] - feat[tr].T).argmin(1)]))
    return {"true_accuracy_on_big_set": truth, "holdout_30_splits": (float(np.mean(holdout)), float(np.std(holdout))), "cv5_30_repeats": (float(np.mean(cv5)), float(np.std(cv5))),
            "time_series_forward_mse": float(np.mean(forward)), "time_series_random_kfold_mse": float(np.mean(rand)), "folds_stratified_positive_counts": [int(y[v].sum()) for _, v in M.kfold_indices(100, 5, stratify=y[:100])]}


# ---- 2.45 class imbalance ------------------------------------------------------------------------------------
def ex_2_45(seed: int = 0) -> dict:
    X, y = M.make_blobs_binary(20000, sep=2.2, pos_frac=0.02, seed=seed); tr, te = np.arange(10000), np.arange(10000, 20000)
    sc = M.StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[te]); yt, ye = y[tr], y[te]
    res = {}
    def report(name, m, thr=0.5):
        pr = m.predict_proba(B); yh = (pr >= thr).astype(int)
        res[name] = {"precision": M.precision(ye, yh), "recall": M.recall(ye, yh), "f1": M.f1(ye, yh), "pr_auc": M.average_precision(ye, pr), "accuracy": M.accuracy(ye, yh)}
    base = M.LogisticRegression(n_iter=1500).fit(A, yt); report("plain", base)
    report("class_weights", M.LogisticRegression(n_iter=1500, class_weight=M.balanced_class_weights(yt)).fit(A, yt))
    Ao, yo = M.random_oversample(A, yt); report("oversample", M.LogisticRegression(n_iter=1500).fit(Ao, yo))
    t, _ = M.best_f1_threshold(yt, base.predict_proba(A)); report("plain+tuned_threshold(train)", base, t)
    return res


# ---- 2.46 baselines ----------------------------------------------------------------------------------------------
def ex_2_46(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); n = 600; level = np.cumsum(rng.standard_normal(n) * 0.1) + 10
    series = level + 0.3 * rng.standard_normal(n)
    X, y = np.arange(1, n)[:, None] / n, series[1:]; tr, te = np.arange(400), np.arange(400, n - 1)
    mean_pred = np.full(len(te), y[tr].mean()); naive = series[te]  # naive: tomorrow = today
    lin = M.LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    Xc, yc = M.make_blobs_binary(1000, sep=1.0, pos_frac=0.3, seed=seed)
    maj = M.accuracy(yc[500:], M.MajorityBaseline().fit(Xc[:500], yc[:500]).predict(Xc[500:]))
    lr_ = M.accuracy(yc[500:], M.LogisticRegression(n_iter=1500).fit(Xc[:500], yc[:500]).predict(Xc[500:]))
    return {"forecast_mse": {"mean": M.mse(y[te], mean_pred), "naive_yesterday": M.mse(y[te], naive), "linear_trend": M.mse(y[te], lin)},
            "classification_acc": {"majority": maj, "logistic": lr_}}


# ---- 2.47 feature scaling -------------------------------------------------------------------------------------------
class RobustScaler:
    def fit(self, X):
        X = np.asarray(X, float); self.med_ = np.median(X, 0); q = np.percentile(X, [25, 75], axis=0); self.iqr_ = np.where((q[1] - q[0]) < 1e-12, 1.0, q[1] - q[0]); return self
    def transform(self, X): return (np.asarray(X, float) - self.med_) / self.iqr_


def ex_2_47(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); x = rng.normal(50, 10, 1000); x_out = np.r_[x, [10000.0]]
    std, mm, rob = M.StandardScaler().fit(x_out[:, None]), M.MinMaxScaler().fit(x_out[:, None]), RobustScaler().fit(x_out[:, None])
    spread = lambda sc: float(np.std(sc.transform(x[:, None])))
    X, y = M.make_linear(300, noise=0.1, seed=seed); Xb = X * np.array([1.0, 100.0])
    steps = {}
    for name, A, lr in (("raw_scales_1_and_100", Xb, 0.00009), ("standardized", M.StandardScaler().fit_transform(Xb), 0.1)):
        _, _, h = M.sgd_linear(A, y, lr=lr, epochs=400); steps[name] = next((i + 1 for i, l in enumerate(h["epoch_loss"]) if l < 0.02), None)
    ols_raw, ols_scaled = M.LinearRegression().fit(Xb, y), M.LinearRegression().fit(M.StandardScaler().fit_transform(Xb), y)
    skew = rng.lognormal(0, 1.2, 5000)
    return {"spread_of_normal_points_after_scaling_with_one_outlier": {"standard": spread(std), "minmax": spread(mm), "robust": spread(rob)}, "epochs_to_loss_below_0.02": steps,
            "ols_prediction_invariant": bool(np.allclose(ols_raw.predict(Xb), ols_scaled.predict(M.StandardScaler().fit(Xb).transform(Xb)))),
            "skew_before_after_log": (float(((skew - skew.mean()) ** 3).mean() / skew.std() ** 3), float(((np.log(skew) - np.log(skew).mean()) ** 3).mean() / np.log(skew).std() ** 3))}


# ---- 2.48 feature engineering ------------------------------------------------------------------------------------------
def out_of_fold_target_encoding(cat, y, k=5, seed=0):
    enc = np.zeros(len(y)); prior = y.mean()
    for tr, va in M.kfold_indices(len(y), k, seed=seed):
        means = {c: y[tr][cat[tr] == c].mean() for c in np.unique(cat[tr])}
        enc[va] = [means.get(c, prior) for c in cat[va]]
    return enc


def ex_2_48(seed: int = 0) -> dict:
    hours = np.arange(24); ang = 2 * np.pi * hours / 24; feat = np.c_[np.sin(ang), np.cos(ang)]
    d = lambda a, b: float(np.linalg.norm(feat[a] - feat[b]))
    rng = np.random.default_rng(seed); n, ncat = 1000, 300; cat = rng.integers(0, ncat, n); y = rng.integers(0, 2, n)  # category carries NO signal
    prior = lambda tr: y[tr].mean()
    naive = np.array([y[cat == c].mean() for c in cat])                                   # uses each row's own label
    oof = out_of_fold_target_encoding(cat, y)
    tr, te = np.arange(700), np.arange(700, 1000)
    def acc(enc_all): return M.accuracy(y[te], (M.LogisticRegression(n_iter=800).fit(enc_all[tr, None], y[tr]).predict(enc_all[te, None])).astype(int))
    naive_tr = np.array([y[tr][cat[tr] == c].mean() if (cat[tr] == c).any() else y[tr].mean() for c in cat])  # encoding computed from train labels, applied to all rows
    return {"hour_distance_23_to_0_raw_vs_cyclic": (23.0, d(23, 0)), "hour_distance_12_to_0_raw_vs_cyclic": (12.0, d(12, 0)),
            "target_encoding_train_vs_test_acc_naive": (M.accuracy(y[tr], (naive_tr[tr] > y[tr].mean()).astype(int)), M.accuracy(y[te], (naive_tr[te] > y[tr].mean()).astype(int))),
            "target_encoding_oof_corr_with_label": float(np.corrcoef(oof, y)[0, 1]), "target_encoding_naive_corr_with_label": float(np.corrcoef(naive, y)[0, 1])}


# ---- 2.49 preprocessing ---------------------------------------------------------------------------------------------------
class Pipeline:
    """Fit every step on training data only; apply the same fitted steps to anything else."""
    def __init__(self, steps): self.steps = steps
    def fit(self, X, y):
        for s in self.steps[:-1]: X = s.fit(X).transform(X)
        self.steps[-1].fit(X, y); return self
    def predict(self, X):
        for s in self.steps[:-1]: X = s.transform(X)
        return self.steps[-1].predict(X)


class MedianImputer:
    def fit(self, X): self.med_ = np.nanmedian(X, 0); return self
    def transform(self, X): X = np.array(X, float); r, c = np.where(np.isnan(X)); X[r, c] = self.med_[c]; return X


class Clip:
    def __init__(self, lo=1, hi=99): self.lo, self.hi = lo, hi
    def fit(self, X): self.a_, self.b_ = np.nanpercentile(X, [self.lo, self.hi], axis=0); return self
    def transform(self, X): return np.clip(X, self.a_, self.b_)


def ex_2_49(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); n = 4000
    X, y = M.make_blobs_binary(n, sep=1.5, seed=seed); X = X.copy()
    miss = (rng.random(n) < np.where(y == 1, 0.5, 0.05)); X[miss, 0] = np.nan             # missingness depends on the label (informative)
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    plain = Pipeline([MedianImputer(), M.StandardScaler(), M.LogisticRegression(n_iter=800)]).fit(X[tr], y[tr])
    ind = lambda A: np.c_[A, np.isnan(A[:, 0]).astype(float)]
    withind = Pipeline([MedianImputer(), M.StandardScaler(), M.LogisticRegression(n_iter=800)]).fit(np.c_[X[tr], np.isnan(X[tr, 0])], y[tr])
    # pipeline fitted on train == manual fit on train; leaking statistics from test would differ
    imp = MedianImputer().fit(X[tr]); sc = M.StandardScaler().fit(imp.transform(X[tr])); manual = M.LogisticRegression(n_iter=800).fit(sc.transform(imp.transform(X[tr])), y[tr])
    same = bool(np.array_equal(manual.predict(sc.transform(imp.transform(X[te]))), plain.predict(X[te])))
    outlier = X.copy(); outlier[:20, 1] = 1e4
    clipped = Pipeline([MedianImputer(), Clip(5, 95), M.StandardScaler(), M.LogisticRegression(n_iter=800)]).fit(outlier[tr], y[tr])
    unclipped = Pipeline([MedianImputer(), M.StandardScaler(), M.LogisticRegression(n_iter=800)]).fit(outlier[tr], y[tr])
    return {"acc_median_impute_only": M.accuracy(y[te], plain.predict(X[te])), "acc_with_missing_indicator": M.accuracy(y[te], withind.predict(np.c_[X[te], np.isnan(X[te, 0])])),
            "pipeline_equals_manual_train_only_fit": same, "acc_with_gross_outliers_unclipped": M.accuracy(y[te], unclipped.predict(outlier[te])), "acc_with_gross_outliers_clipped": M.accuracy(y[te], clipped.predict(outlier[te]))}


# ---- 2.50 dataset imbalance (prevalence of the dataset vs the world) ------------------------------------------------------
def ex_2_50(seed: int = 0) -> dict:
    X, y = M.make_blobs_binary(60000, sep=2.0, pos_frac=0.02, seed=seed); tr, te = np.arange(30000), np.arange(30000, 60000)
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]; keep = np.r_[pos, np.random.default_rng(seed).choice(neg, len(pos), replace=False)]   # 50/50 training set
    m = M.LogisticRegression(n_iter=2000).fit(X[keep], y[keep]); z = m.decision_function(X[te])
    pi_train, pi_prod = 0.5, y[tr].mean()
    z_corr = z + np.log(pi_prod / (1 - pi_prod)) - np.log(pi_train / (1 - pi_train))      # prior (base-rate) correction of the logit
    ece = lambda zz: M.expected_calibration_error(y[te], M.sigmoid(zz))
    return {"train_prevalence": 0.5, "production_prevalence": float(pi_prod), "mean_predicted_prob_raw": float(M.sigmoid(z).mean()), "mean_predicted_prob_corrected": float(M.sigmoid(z_corr).mean()),
            "ece_raw": ece(z), "ece_corrected": ece(z_corr), "auc_raw": M.roc_auc(y[te], z), "auc_corrected": M.roc_auc(y[te], z_corr),
            "precision_at_0.5_raw": M.precision(y[te], (z >= 0).astype(int)), "precision_at_0.5_corrected": M.precision(y[te], (z_corr >= 0).astype(int))}


# ---- 2.51 metric selection ------------------------------------------------------------------------------------------------------
def ex_2_51(seed: int = 0) -> dict:
    X, y = M.make_blobs_binary(30000, sep=1.8, pos_frac=0.05, seed=seed); tr, te = np.arange(15000), np.arange(15000, 30000)
    sc = M.StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[te])
    models = {"plain": M.LogisticRegression(n_iter=1500).fit(A, y[tr]), "weighted": M.LogisticRegression(n_iter=1500, class_weight=M.balanced_class_weights(y[tr])).fit(A, y[tr])}
    metrics = {k: {"roc_auc": M.roc_auc(y[te], m.predict_proba(B)), "pr_auc": M.average_precision(y[te], m.predict_proba(B)), "f1@0.5": M.f1(y[te], m.predict(B)),
                   "log_loss": M.bce_from_logits(y[te], m.decision_function(B)), "brier": M.brier_score(y[te], m.predict_proba(B))} for k, m in models.items()}
    def best_cost(m, c_fn, c_fp):
        p = m.predict_proba(B); ts = np.linspace(0.02, 0.98, 49)
        return min((c_fn * np.sum((p < t) & (y[te] == 1)) + c_fp * np.sum((p >= t) & (y[te] == 0))) / len(te) for t in ts)
    scen = {"fp_costly(1:5)": (1, 5), "equal(1:1)": (1, 1), "fn_costly(20:1)": (20, 1)}
    costs = {name: {k: best_cost(m, fn, fp) for k, m in models.items()} for name, (fn, fp) in scen.items()}
    at_half = {name: {k: float((fn * np.sum((m.predict(B) == 0) & (y[te] == 1)) + fp * np.sum((m.predict(B) == 1) & (y[te] == 0))) / len(te)) for k, m in models.items()} for name, (fn, fp) in scen.items()}
    return {"metrics": metrics, "min_expected_cost_by_scenario": costs, "expected_cost_at_fixed_threshold_0.5": at_half}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
