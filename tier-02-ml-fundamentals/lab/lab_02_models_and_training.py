"""Tier 2 Lab, part 2: concepts 2.14-2.25 (task types, losses/objectives, gradient descent family, training vocabulary, learning rate)."""
from __future__ import annotations

import numpy as np

import mlkit as M


# ---- helpers -------------------------------------------------------------------------------------
def make_blobs_multi(n: int = 600, k: int = 3, radius: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(k) / k
    centers = radius * np.c_[np.cos(ang), np.sin(ang)]
    y = rng.integers(0, k, n)
    return centers[y] + rng.standard_normal((n, 2)), y


def softmax_rows(Z):
    e = np.exp(Z - Z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def fit_softmax_regression(X, y, k, lr: float = 0.5, steps: int = 1500):
    W, b = np.zeros((X.shape[1], k)), np.zeros(k); Y = np.eye(k)[y]
    for _ in range(steps):
        P = softmax_rows(X @ W + b); G = (P - Y) / len(y)          # gradient wrt logits = softmax - onehot (Tier 1.37)
        W -= lr * X.T @ G; b -= lr * G.sum(0)
    return lambda Z: softmax_rows(Z @ W + b)


def silhouette(X, labels) -> float:
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); ks = np.unique(labels); s = []
    for i in range(len(X)):
        same = (labels == labels[i]); same[i] = False
        if not same.any(): s.append(0.0); continue
        a = D[i, same].mean(); b = min(D[i, labels == c].mean() for c in ks if c != labels[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def fit_linear_gd(X, y, loss: str = "mse", lr: float = 0.05, steps: int = 4000, delta: float = 1.0):
    """Linear regression by gradient descent with a selectable loss (gradient = dL/dprediction)."""
    w, b = np.zeros(X.shape[1]), 0.0
    for t in range(steps):
        r = X @ w + b - y
        g = {"mse": 2 * r, "mae": np.sign(r), "huber": np.clip(r, -delta, delta)}[loss]
        step = lr / np.sqrt(1 + t / 100) if loss == "mae" else lr  # subgradient needs a decaying step
        w -= step * X.T @ g / len(y); b -= step * g.mean()
    return w, b


# ---- 2.14 classification -------------------------------------------------------------------------
def ex_2_14(seed: int = 0) -> dict:
    X, y = make_blobs_multi(900, 3, seed=seed)
    tr, te = np.arange(600), np.arange(600, 900)
    P = fit_softmax_regression(X[tr], y[tr], 3)(X[te])
    Xb, yb = M.make_blobs_binary(600, sep=2.0, seed=seed)
    m = M.LogisticRegression(n_iter=3000).fit(Xb[:400], yb[:400]); p = m.predict_proba(Xb[400:])
    cost = lambda t: 5 * np.sum((p < t) & (yb[400:] == 1)) + 1 * np.sum((p >= t) & (yb[400:] == 0))  # a miss costs 5x a false alarm
    ts = np.linspace(0.05, 0.95, 19)
    return {"multiclass_acc": M.accuracy(y[te], P.argmax(1)), "probs_sum_to_1": bool(np.allclose(P.sum(1), 1)), "chance": 1 / 3,
            "best_cost_threshold": float(ts[np.argmin([cost(t) for t in ts])]), "cost_at_0.5": float(cost(0.5)), "cost_at_best": float(min(cost(t) for t in ts))}


# ---- 2.15 regression -----------------------------------------------------------------------------
def ex_2_15(seed: int = 0) -> dict:
    X, y = M.make_linear(300, noise=0.5, seed=seed)
    tr, te = np.arange(200), np.arange(200, 300)
    m = M.LinearRegression().fit(X[tr], y[tr]); pred = m.predict(X[te])
    sst = np.sum((y[te] - y[te].mean()) ** 2)
    res = {"coef": m.coef_.round(2).tolist(), "intercept": round(m.intercept_, 2), "test_mse": M.mse(y[te], pred),
           "baseline_mse": M.mse(y[te], M.MeanBaseline().fit(X[tr], y[tr]).predict(X[te])), "r2": float(1 - np.sum((y[te] - pred) ** 2) / sst)}
    res["residual_mean"] = float(np.mean(y[tr] - m.predict(X[tr])))
    return res


# ---- 2.16 clustering -----------------------------------------------------------------------------
def ex_2_16(seed: int = 0) -> dict:
    X, y = M.make_blobs_binary(300, sep=5.0, seed=seed)
    km = M.KMeans(2, seed=seed).fit(X)
    Xu = X.copy(); Xu[:, 1] = Xu[:, 1] * 1000 + np.random.default_rng(seed).standard_normal(300) * 3000  # a feature in huge units
    kmu = M.KMeans(2, seed=seed).fit(Xu)
    Xs = M.StandardScaler().fit_transform(Xu); kms = M.KMeans(2, seed=seed).fit(Xs)
    agree = lambda l: float(max((l == y).mean(), (l != y).mean()))
    return {"silhouette_by_k": {k: silhouette(X, M.KMeans(k, seed=seed).fit(X).labels_) for k in (2, 3, 4)}, "agree_clean": agree(km.labels_),
            "agree_unscaled_huge_units": agree(kmu.labels_), "agree_after_scaling": agree(kms.labels_),
            "init_sensitivity_5_blobs": ex_2_16_init()}


def ex_2_16_init() -> dict:
    """Random initialization can land in a poor local optimum: compare 20 single-init runs with 20 best-of-10 runs."""
    X, _ = make_blobs_multi(500, 5, radius=7.0, seed=1)
    one = [M.KMeans(5, n_init=1, seed=s).fit(X).inertia_ for s in range(20)]
    ten = [M.KMeans(5, n_init=10, seed=s).fit(X).inertia_ for s in range(20)]
    return {"n_init_1": (min(one), max(one)), "n_init_10": (min(ten), max(ten))}


# ---- 2.17 loss functions -------------------------------------------------------------------------
def ex_2_17(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((100, 1)); y = 3 * X[:, 0] + 1 + 0.2 * rng.standard_normal(100); y[:5] += 40  # 5 gross outliers
    fits = {l: fit_linear_gd(X, y, l, lr=0.05 if l != "mae" else 0.3, steps=3000) for l in ("mse", "mae", "huber")}
    slopes = {l: (round(float(w[0]), 3), round(float(b), 3)) for l, (w, b) in fits.items()}
    z, yt = -6.0, 1.0                                       # confident and wrong
    s = 1 / (1 + np.exp(-z))
    return {"slope_and_intercept_by_loss(true=(3, 1))": slopes, "bce_grad_wrt_logit": float(s - yt), "mse_on_sigmoid_grad_wrt_logit": float(2 * (s - yt) * s * (1 - s))}


# ---- 2.18 objective functions --------------------------------------------------------------------
def ex_2_18(seed: int = 0) -> dict:
    X, y = M.make_linear(40, w=(2.0, -1.0, 0.0, 0.0, 0.0, 0.0), noise=1.0, seed=seed)
    out = {}
    for lam in (0.0, 1.0, 10.0):
        m = M.LinearRegression(ridge=lam).fit(X, y)
        data = M.mse(y, m.predict(X)); pen = lam * float(np.sum(m.coef_ ** 2))
        out[lam] = {"data_loss": data, "penalty": pen, "objective": data * len(y) + pen, "coef_norm": float(np.linalg.norm(m.coef_))}
    Xb, yb = M.make_blobs_binary(600, sep=1.2, pos_frac=0.2, seed=seed)  # objective (log loss) vs metric (F1 at 0.5)
    m = M.LogisticRegression(n_iter=3000).fit(Xb, yb)
    return {"ridge": out, "log_loss_objective": float(M.bce_from_logits(yb, m.decision_function(Xb))), "f1_metric_at_0.5": M.f1(yb, m.predict(Xb))}


# ---- 2.19 gradient descent -----------------------------------------------------------------------
def ex_2_19(seed: int = 0) -> dict:
    X, y = M.make_linear(300, noise=0.3, seed=seed)
    w, b, h = M.sgd_linear(X, y, lr=0.1, batch_size=None, epochs=300)
    ref = M.LinearRegression().fit(X, y)
    first = next(i for i, l in enumerate(h["epoch_loss"]) if l < 1.05 * M.mse(y, ref.predict(X)))
    return {"gd_w": w.round(4).tolist(), "closed_form_w": ref.coef_.round(4).tolist(), "gd_b": round(b, 4), "closed_form_b": round(ref.intercept_, 4),
            "steps_to_within_5pct_of_optimum": first + 1, "monotone_decrease": bool(np.all(np.diff(h["epoch_loss"]) <= 1e-12))}


# ---- 2.20 SGD / 2.21 mini-batch ------------------------------------------------------------------
def grad_mse(X, y, w, b, idx):
    r = X[idx] @ w + b - y[idx]; return np.r_[2 * X[idx].T @ r / len(idx), 2 * r.mean()]


def ex_2_20_21(seed: int = 0) -> dict:
    X, y = M.make_linear(512, noise=1.0, seed=seed)
    w, b = np.array([0.5, 0.5]), 0.0
    full = grad_mse(X, y, w, b, np.arange(512))
    rng = np.random.default_rng(seed)
    out = {"unbiased_check": {}, "variance_by_batch": {}}
    perm = rng.permutation(512)
    for B in (1, 8, 64):
        g = np.array([grad_mse(X, y, w, b, perm[s:s + B]) for s in range(0, 512, B)])
        out["unbiased_check"][B] = float(np.abs(g.mean(0) - full).max())  # exact for equal-size batches that partition the data
        gs = np.array([grad_mse(X, y, w, b, rng.choice(512, B, replace=False)) for _ in range(3000)])
        out["variance_by_batch"][B] = float(gs[:, 0].var())
    floor = {}
    for name, lr_fn in (("const_0.05", lambda t: 0.05), ("decay", lambda t: 0.05 / (1 + t / 200))):
        r = np.random.default_rng(seed); ww, bb = np.zeros(2), 0.0
        for t in range(6000):
            g = grad_mse(X, y, ww, bb, r.integers(0, 512, 1)); ww -= lr_fn(t) * g[:2]; bb -= lr_fn(t) * g[2]
        floor[name] = float(np.linalg.norm(ww - M.LinearRegression().fit(X, y).coef_))
    out["distance_to_optimum_after_6000_sgd_steps"] = floor
    return out


# ---- 2.22-2.24 epoch / batch / iteration ----------------------------------------------------------
def ex_2_22_24() -> dict:
    n, bs, ep = 1050, 100, 3
    X, y = M.make_linear(n, noise=0.1, seed=0)
    _, _, h = M.sgd_linear(X, y, lr=0.05, batch_size=bs, epochs=ep)
    sizes = [len(b) for b in M.minibatches(n, bs, np.random.default_rng(0))]
    return {"iterations_per_epoch": M.iterations_per_epoch(n, bs), "iterations_per_epoch_drop_last": M.iterations_per_epoch(n, bs, True),
            "total_iterations_after_3_epochs": h["iterations"][-1], "batch_sizes_in_one_epoch": sizes, "full_batch_gd_iterations_per_epoch": 1,
            "pure_sgd_iterations_per_epoch": M.iterations_per_epoch(n, 1)}


# ---- 2.25 learning rate ---------------------------------------------------------------------------
def ex_2_25(seed: int = 0) -> dict:
    X, y = M.make_linear(200, noise=0.1, seed=seed)
    A = np.hstack([X, np.ones((200, 1))]); lam_max = float(np.linalg.eigvalsh(2 * A.T @ A / 200).max())
    limit = 2 / lam_max
    final = lambda lr, ep=200: M.sgd_linear(X, y, lr=lr, epochs=ep)[2]["epoch_loss"][-1]
    sweep = {lr: final(lr) for lr in (1e-4, 1e-3, 1e-2, 1e-1)}
    Xbad = X * np.array([1.0, 30.0])  # one feature on a 30x larger scale
    Ab = np.hstack([Xbad, np.ones((200, 1))]); limit_bad = 2 / float(np.linalg.eigvalsh(2 * Ab.T @ Ab / 200).max())
    return {"lambda_max": lam_max, "stability_limit": limit, "loss_at_0.9_limit": final(0.9 * limit), "loss_at_1.1_limit": final(1.1 * limit),
            "sweep": sweep, "stability_limit_with_unscaled_feature": limit_bad}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
