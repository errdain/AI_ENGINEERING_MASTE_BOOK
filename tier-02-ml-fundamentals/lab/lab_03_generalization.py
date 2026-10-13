"""Tier 2 Lab, part 3: concepts 2.26-2.36 (under/overfitting, bias/variance, generalization, regularization, leakage, shift, tuning)."""
from __future__ import annotations

import numpy as np

import mlkit as M

F_TRUE = lambda x: np.sin(np.pi * x)


# ---- 2.26 / 2.27 under- and overfitting ---------------------------------------------------------
def ex_2_26_27(seed: int = 1, degrees=(0, 1, 3, 5, 9, 15)) -> dict:
    x, y = M.make_sine(30, 0.2, seed=seed); xv, yv = M.make_sine(500, 0.2, seed=seed + 100)
    table = {d: (lambda r: (r.train_mse, r.val_mse))(M.fit_poly_train_val(d, x, y, xv, yv)) for d in degrees}
    best = min(table, key=lambda d: table[d][1])
    return {"train_val_mse_by_degree": {d: (round(a, 4), round(c, 4)) for d, (a, c) in table.items()}, "best_degree_by_val": best}


def ex_2_27_early_stopping(seed: int = 0, n: int = 30, p: int = 60, epochs: int = 3000) -> dict:
    """More parameters than samples + label noise: gradient descent from w=0 first learns signal, then fits noise."""
    rng = np.random.default_rng(seed); w_true = np.zeros(p); w_true[:5] = 1.0
    gen = lambda m: (lambda X: (X, X @ w_true + 1.0 * rng.standard_normal(m)))(rng.standard_normal((m, p)))
    X, y = gen(n); Xv, yv = gen(2000); w = np.zeros(p); tr, va = [], []
    for _ in range(epochs):
        w -= 0.01 * 2 * X.T @ (X @ w - y) / n
        tr.append(M.mse(y, X @ w)); va.append(M.mse(yv, Xv @ w))
    return {"best_epoch": int(np.argmin(va)) + 1, "val_at_best": float(min(va)), "val_at_end": float(va[-1]), "train_at_best": tr[int(np.argmin(va))], "train_at_end": float(tr[-1])}


# ---- 2.28 bias ----------------------------------------------------------------------------------
def ex_2_28(seed: int = 0) -> dict:
    lo = M.bias_variance_poly(1, F_TRUE, seed=seed); mid = M.bias_variance_poly(5, F_TRUE, seed=seed)
    # sampling bias: train only on x < 0, evaluate everywhere
    rng = np.random.default_rng(seed); x = rng.uniform(-1, 0, 60); y = F_TRUE(x) + 0.1 * rng.standard_normal(60)
    xt = np.linspace(-1, 1, 400)
    m = M.LinearRegression().fit(M.poly_features(x, 5)[:, 1:], y); p = m.predict(M.poly_features(xt, 5)[:, 1:])
    return {"degree1_bias2": lo["bias2"], "degree5_bias2": mid["bias2"],
            "mse_inside_training_range": M.mse(F_TRUE(xt[xt < 0]), p[xt < 0]), "mse_outside_training_range": M.mse(F_TRUE(xt[xt > 0]), p[xt > 0])}


# ---- 2.29 variance ------------------------------------------------------------------------------
def ex_2_29(seed: int = 0) -> dict:
    by_degree = {d: M.bias_variance_poly(d, F_TRUE, n_train=30, seed=seed)["variance"] for d in (1, 3, 9)}
    by_n = {n: M.bias_variance_poly(3, F_TRUE, n_train=n, seed=seed)["variance"] for n in (15, 30, 120, 480)}
    return {"variance_by_degree": by_degree, "variance_by_n_train(degree3)": by_n}


# ---- 2.30 bias-variance trade-off ---------------------------------------------------------------
def ex_2_30(seed: int = 0) -> dict:
    rows = {d: M.bias_variance_poly(d, F_TRUE, n_datasets=300, seed=seed) for d in range(0, 11)}
    best = min(rows, key=lambda d: rows[d]["expected_test_mse"])
    ridge = {lam: M.bias_variance_poly(10, F_TRUE, n_datasets=300, ridge=lam, seed=seed)["expected_test_mse"] for lam in (0.0, 1e-4, 1e-2, 1.0)}
    return {"rows": {d: {k: round(v, 4) for k, v in r.items()} for d, r in rows.items()}, "best_degree": best, "decomposition_error": max(abs(r["bias2"] + r["variance"] + r["noise"] - r["expected_test_mse"]) for r in rows.values()),
            "degree10_test_mse_by_ridge": ridge}


# ---- 2.31 generalization ------------------------------------------------------------------------
def ex_2_31(seed: int = 0) -> dict:
    gap = {}
    for n in (20, 100, 1000):
        accs = []
        for s in range(10):
            X, y = M.make_blobs_binary(n + 2000, sep=1.0, seed=seed + s)
            m = M.LogisticRegression(n_iter=800).fit(X[:n], y[:n])
            accs.append((M.accuracy(y[:n], m.predict(X[:n])), M.accuracy(y[n:], m.predict(X[n:]))))
        a = np.mean(accs, 0); gap[n] = {"train": float(a[0]), "test": float(a[1]), "gap": float(a[0] - a[1])}
    rng = np.random.default_rng(seed)
    Xr, yr = rng.standard_normal((200, 20)), rng.integers(0, 2, 200); Xt, yt = rng.standard_normal((2000, 20)), rng.integers(0, 2, 2000)
    nn = lambda A, ya, B: ya[((B[:, None] - A[None]) ** 2).sum(-1).argmin(1)]
    return {"gap_by_n": gap, "random_labels_train_acc_1nn": M.accuracy(yr, nn(Xr, yr, Xr)), "random_labels_test_acc_1nn": M.accuracy(yt, nn(Xr, yr, Xt))}


# ---- 2.32 regularization ------------------------------------------------------------------------
def lasso_ista(X, y, alpha: float, steps: int = 5000):
    """Minimize (1/2n)||y - b - Xw||^2 + alpha*||w||_1 by proximal gradient (ISTA)."""
    X, y = np.asarray(X, float), np.asarray(y, float); n = len(y)
    xm, ym = X.mean(0), y.mean(); Xc, yc = X - xm, y - ym
    L = np.linalg.eigvalsh(Xc.T @ Xc / n).max(); w = np.zeros(X.shape[1])
    for _ in range(steps):
        z = w - (Xc.T @ (Xc @ w - yc) / n) / L
        w = np.sign(z) * np.maximum(np.abs(z) - alpha / L, 0)
    return w, float(ym - xm @ w)


def ex_2_32(seed: int = 1) -> dict:
    x, y = M.make_sine(30, 0.2, seed=seed); xv, yv = M.make_sine(500, 0.2, seed=seed + 100)
    lams = (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
    val = {lam: M.fit_poly_train_val(15, x, y, xv, yv, ridge=lam).val_mse for lam in lams}
    X, yy = M.make_linear(100, w=(3, -2, 0, 0, 0, 0, 0, 0, 0, 0), noise=1.0, seed=seed)
    w_l, _ = lasso_ista(X, yy, 0.3); w_r = M.LinearRegression(ridge=20.0).fit(X, yy).coef_
    return {"val_mse_by_ridge_degree15": val, "best_lambda": min(val, key=val.get), "lasso_nonzero": int(np.sum(np.abs(w_l) > 1e-8)),
            "lasso_support": np.flatnonzero(np.abs(w_l) > 1e-8).tolist(), "ridge_nonzero": int(np.sum(np.abs(w_r) > 1e-8)), "ridge_max_abs_noise_coef": float(np.abs(w_r[2:]).max())}


# ---- 2.33 data leakage --------------------------------------------------------------------------
def _nearest_centroid_acc(Xtr, ytr, Xte, yte):
    c0, c1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    pred = (((Xte - c1) ** 2).sum(1) < ((Xte - c0) ** 2).sum(1)).astype(int)
    return M.accuracy(yte, pred)


def select_top_k(X, y, k):
    corr = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    return np.argsort(-corr)[:k]


def ex_2_33(seed: int = 0) -> dict:
    # (a) feature selection on ALL data before CV, with pure-noise features and labels
    rng = np.random.default_rng(seed); n, p = 60, 1000
    X, y = rng.standard_normal((n, p)), rng.integers(0, 2, n)
    leaky_idx = select_top_k(X, y, 20)
    leaky, honest = [], []
    for tr, va in M.kfold_indices(n, 5, seed=0, stratify=y):
        leaky.append(_nearest_centroid_acc(X[tr][:, leaky_idx], y[tr], X[va][:, leaky_idx], y[va]))
        idx = select_top_k(X[tr], y[tr], 20)
        honest.append(_nearest_centroid_acc(X[tr][:, idx], y[tr], X[va][:, idx], y[va]))
    # (b) target leakage: a feature recorded after the outcome
    Xb, yb = M.make_blobs_binary(600, sep=0.0, seed=seed)
    leak = np.c_[Xb, yb + 0.05 * rng.standard_normal(600)]
    m = M.LogisticRegression(n_iter=2000).fit(leak[:400], yb[:400]); m2 = M.LogisticRegression(n_iter=2000).fit(Xb[:400], yb[:400])
    # (c) temporal leakage: 1-NN on the time index interpolates neighbours under a random split, but must extrapolate in a forward split
    t = np.arange(400.0)[:, None]; series = np.cumsum(rng.standard_normal(400))
    nn = lambda tr_i, te_i: series[tr_i][np.abs(t[te_i] - t[tr_i].T).argmin(1)]
    rnd = rng.permutation(400); tr_r, te_r = rnd[:300], rnd[300:]; tr_f, te_f = np.arange(300), np.arange(300, 400)
    return {"selection_leak_cv_acc": float(np.mean(leaky)), "selection_inside_fold_cv_acc": float(np.mean(honest)),
            "target_leak_test_acc": M.accuracy(yb[400:], m.predict(leak[400:])), "no_leak_test_acc": M.accuracy(yb[400:], m2.predict(Xb[400:])),
            "random_split_mse": M.mse(series[te_r], nn(tr_r, te_r)), "forward_split_mse": M.mse(series[te_f], nn(tr_f, te_f))}


# ---- 2.34 distribution shift --------------------------------------------------------------------
def ex_2_34(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    f = lambda x: x ** 2  # true relation is nonlinear
    xtr = rng.normal(0, 1, 2000); ytr = f(xtr) + 0.1 * rng.standard_normal(2000)
    m = M.LinearRegression().fit(xtr[:, None], ytr)
    out = {"in_dist_mse": M.mse(f(xtr), m.predict(xtr[:, None]))}
    for name, mu in (("same", 0.0), ("shifted_mean_2", 2.0)):
        xs = rng.normal(mu, 1, 2000); out[f"covariate_shift_{name}_mse"] = M.mse(f(xs) + 0.1 * rng.standard_normal(2000), m.predict(xs[:, None]))
    xs = rng.normal(2.0, 1, 2000)
    out["psi_same"], out["psi_shifted"] = M.psi(xtr, rng.normal(0, 1, 2000)), M.psi(xtr, xs)
    out["ks_same"], out["ks_shifted"] = M.ks_statistic(xtr, rng.normal(0, 1, 2000)), M.ks_statistic(xtr, xs)
    out["domain_auc_same"] = domain_classifier_auc(xtr, rng.normal(0, 1, 2000)); out["domain_auc_shifted"] = domain_classifier_auc(xtr, xs)
    # label (prior) shift: precision collapses when the positive rate drops, with a fixed classifier
    Xa, ya = M.make_blobs_binary(4000, sep=2.5, pos_frac=0.5, seed=seed + 1); clf = M.LogisticRegression(n_iter=1500).fit(Xa, ya)
    for pf in (0.5, 0.05):
        Xb, yb = M.make_blobs_binary(8000, sep=2.5, pos_frac=pf, seed=seed + 2); pr = clf.predict(Xb)
        out[f"precision_at_prevalence_{pf}"] = M.precision(yb, pr); out[f"recall_at_prevalence_{pf}"] = M.recall(yb, pr)
    return out


def domain_classifier_auc(A: np.ndarray, B: np.ndarray, seed: int = 0) -> float:
    """AUC of a classifier that tries to tell sample A (training) from sample B (production): ~0.5 means no detectable covariate shift."""
    Z = np.r_[A, B][:, None]; d = np.r_[np.zeros(len(A)), np.ones(len(B))].astype(int)
    idx = np.random.default_rng(seed).permutation(len(d)); Z, d = Z[idx], d[idx]; h = len(d) // 2
    return M.roc_auc(d[h:], M.LogisticRegression(n_iter=800).fit(Z[:h], d[:h]).decision_function(Z[h:]))


# ---- 2.35 / 2.36 hyperparameters and tuning ------------------------------------------------------
def ex_2_35_36(n_seeds: int = 300) -> dict:
    """Objective depends strongly on lr (log-scale) and weakly on l2. Compare 3x3 grid vs 9 random trials, averaged over objective noise."""
    best_grid, best_rand = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(s); lr_opt = rng.uniform(-4, -1); noise = lambda: 0.02 * rng.standard_normal()
        obj = lambda lr, l2: (np.log10(lr) - lr_opt) ** 2 + 0.05 * (l2 - 0.3) ** 2 + noise()
        grid = [(10 ** a, b) for a in (-4, -2.5, -1) for b in (0.0, 0.5, 1.0)]
        rand = [(10 ** rng.uniform(-4, -1), rng.uniform(0, 1)) for _ in range(9)]
        best_grid.append(min(obj(*p) for p in grid)); best_rand.append(min(obj(*p) for p in rand))
    # selection optimism: choose the best of 30 random configs by CV on pure noise, then evaluate honestly on fresh data
    rng = np.random.default_rng(0); X, y = rng.standard_normal((120, 5)), rng.integers(0, 2, 120)
    folds = M.kfold_indices(120, 5, seed=0, stratify=y)
    def cv(w): return float(np.mean([M.accuracy(y[v], ((X[v] @ w) > 0).astype(int)) for _, v in folds]))
    ws = [rng.standard_normal(5) for _ in range(300)]; scores = [cv(w) for w in ws]; wbest = ws[int(np.argmax(scores))]
    Xf, yf = rng.standard_normal((20000, 5)), rng.integers(0, 2, 20000)
    return {"mean_best_grid": float(np.mean(best_grid)), "mean_best_random": float(np.mean(best_rand)), "random_wins_fraction": float(np.mean(np.array(best_rand) < np.array(best_grid))),
            "best_cv_score_over_300_configs_on_noise": float(max(scores)), "that_config_on_fresh_data": M.accuracy(yf, ((Xf @ wbest) > 0).astype(int))}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
