"""Tier 2 Lab, part 1: concepts 2.1-2.13 (paradigms, features/labels/samples, dataset construction, splits)."""
from __future__ import annotations

import numpy as np

import mlkit as M


# ---- 2.1 AI vs ML vs DL ------------------------------------------------------------------------
def tiny_mlp_fit(X, y, hidden: int = 8, lr: float = 0.5, steps: int = 4000, seed: int = 0):
    """One-hidden-layer tanh network for binary classification (manual backprop, see Tier 1.25)."""
    r = np.random.default_rng(seed)
    W1, b1, w2, b2 = r.standard_normal((X.shape[1], hidden)) * 0.8, np.zeros(hidden), r.standard_normal(hidden) * 0.8, 0.0
    for _ in range(steps):
        h = np.tanh(X @ W1 + b1); p = M.sigmoid(h @ w2 + b2); d = (p - y) / len(y)
        dh = np.outer(d, w2) * (1 - h ** 2)
        W1 -= lr * X.T @ dh; b1 -= lr * dh.sum(0); w2 -= lr * h.T @ d; b2 -= lr * d.sum()
    return lambda Z: (M.sigmoid(np.tanh(Z @ W1 + b1) @ w2 + b2) >= 0.5).astype(int)


def ex_2_1(seed: int = 0) -> dict:
    """Same XOR-like task: hand-written rule (AI, no learning), linear ML, ML + engineered feature, small neural net (DL)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((600, 2)); y = (X[:, 0] * X[:, 1] > 0).astype(int)
    tr, te = np.arange(400), np.arange(400, 600)
    rule = lambda Z: (Z[:, 0] > 0).astype(int)  # a plausible but wrong hand-written rule
    lin = M.LogisticRegression(n_iter=3000).fit(X[tr], y[tr])
    feats = lambda Z: np.column_stack([Z, Z[:, 0] * Z[:, 1]])
    eng = M.LogisticRegression(n_iter=3000).fit(feats(X[tr]), y[tr])
    mlp = tiny_mlp_fit(X[tr], y[tr], seed=seed)
    return {"rule_based": M.accuracy(y[te], rule(X[te])), "linear_ml": M.accuracy(y[te], lin.predict(X[te])),
            "ml_with_engineered_feature": M.accuracy(y[te], eng.predict(feats(X[te]))), "neural_net": M.accuracy(y[te], mlp(X[te]))}


# ---- 2.2 supervised ----------------------------------------------------------------------------
def ex_2_2(seed: int = 0) -> dict:
    X, y = M.make_blobs_binary(600, sep=2.0, seed=seed)
    tr, va, te = M.train_val_test_indices(600, seed=seed, stratify=y)
    m = M.LogisticRegression(n_iter=3000).fit(X[tr], y[tr])
    return {"train_acc": M.accuracy(y[tr], m.predict(X[tr])), "test_acc": M.accuracy(y[te], m.predict(X[te])),
            "baseline_acc": M.accuracy(y[te], M.MajorityBaseline().fit(X[tr], y[tr]).predict(X[te]))}


# ---- 2.3 unsupervised --------------------------------------------------------------------------
def ex_2_3(seed: int = 0) -> dict:
    X, y = M.make_blobs_binary(400, sep=6.0, seed=seed)
    inertia = {k: M.KMeans(k, seed=seed).fit(X).inertia_ for k in (1, 2, 3, 4, 6)}
    km = M.KMeans(2, seed=seed).fit(X)
    return {"inertia_by_k": inertia, "agreement_with_hidden_labels": float(max((km.labels_ == y).mean(), (km.labels_ != y).mean()))}


# ---- 2.4 semi-supervised: self-training --------------------------------------------------------
def ex_2_4(seed: int = 0, n_labeled: int = 10, threshold: float = 0.95) -> dict:
    X, y = M.make_blobs_binary(1200, sep=3.0, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(1200); te, pool = idx[:400], idx[400:]
    lab = np.r_[pool[y[pool] == 0][:n_labeled // 2], pool[y[pool] == 1][:n_labeled // 2]]
    unl = np.setdiff1d(pool, lab)
    base = M.LogisticRegression(n_iter=3000).fit(X[lab], y[lab])
    Xc, yc = X[lab], y[lab]
    model = base
    for _ in range(5):
        p = model.predict_proba(X[unl]); conf = np.maximum(p, 1 - p) >= threshold
        Xc = np.vstack([X[lab], X[unl][conf]]); yc = np.r_[y[lab], (p[conf] >= 0.5).astype(int)]
        model = M.LogisticRegression(n_iter=3000).fit(Xc, yc)
    return {"labeled_only_acc": M.accuracy(y[te], base.predict(X[te])), "self_training_acc": M.accuracy(y[te], model.predict(X[te])),
            "pseudo_labels_used": int(len(yc) - len(lab)), "pseudo_label_accuracy": float(np.mean(yc[len(lab):] == y[unl][conf])) if conf.any() else None}


# ---- 2.5 self-supervised: next-character bigram model --------------------------------------------
def ex_2_5(text: str | None = None) -> dict:
    text = text or ("the cat sat on the mat. the dog sat on the log. the cat saw the dog. " * 20)
    chars = sorted(set(text)); idx = {c: i for i, c in enumerate(chars)}
    ids = np.array([idx[c] for c in text])
    inputs, targets = ids[:-1], ids[1:]  # labels come from the data itself: the next character
    counts = np.ones((len(chars), len(chars))) * 0.1  # smoothing
    np.add.at(counts, (inputs, targets), 1)
    P = counts / counts.sum(1, keepdims=True)
    nll = -np.mean(np.log(P[inputs, targets]))
    return {"n_training_pairs": len(inputs), "n_chars": len(chars), "uniform_nll": float(np.log(len(chars))), "bigram_nll": float(nll),
            "perplexity": float(np.exp(nll)), "most_likely_after_t": chars[int(P[idx['t']].argmax())]}


# ---- 2.6 reinforcement learning overview: epsilon-greedy bandit ---------------------------------
def ex_2_6(seed: int = 0, steps: int = 3000) -> dict:
    means = np.array([0.2, 0.5, 0.8])
    out = {}
    for eps in (0.0, 0.1):
        rng = np.random.default_rng(seed)
        Q, N, total = np.zeros(3), np.zeros(3), 0.0
        for _ in range(steps):
            a = int(rng.integers(3)) if rng.random() < eps else int(Q.argmax())
            r = float(rng.random() < means[a]); N[a] += 1; Q[a] += (r - Q[a]) / N[a]; total += r
        out[eps] = {"avg_reward": total / steps, "best_arm_share": float(N[2] / steps)}
    return out


# ---- 2.7-2.9 features, labels, samples ---------------------------------------------------------
def ex_2_7_to_2_9(seeds=range(10), n_train: int = 100) -> dict:
    """Averages over seeds: junk features, label noise and sample size each change held-out accuracy."""
    res = {"clean": [], "junk60": [], "noise40": [], **{f"n={n}": [] for n in (5, 20, 100, 400)}}
    for seed in seeds:
        X, y = M.make_blobs_binary(1000, sep=2.0, seed=seed)
        rng = np.random.default_rng(seed)
        tr, te = np.arange(n_train), np.arange(500, 1000)
        fit_acc = lambda Xa, ya, idx: M.accuracy(y[te], M.LogisticRegression(n_iter=2000).fit(Xa[idx], ya[idx]).predict(Xa[te]))
        res["clean"].append(fit_acc(X, y, tr))
        res["junk60"].append(fit_acc(np.column_stack([X, rng.standard_normal((1000, 60))]), y, tr))
        noisy = y.copy(); flip = rng.random(1000) < 0.4; noisy = np.where(flip, 1 - y, y)  # 40% of TRAINING labels wrong
        res["noise40"].append(fit_acc(X, noisy, tr))
        for n in (5, 20, 100, 400):
            res[f"n={n}"].append(fit_acc(X, y, np.arange(n)))
    X, y = M.make_blobs_binary(1000, sep=2.0, seed=0)
    return {"X_shape": X.shape, "y_shape": y.shape, "first_sample": (X[0].round(2).tolist(), int(y[0])), **{k: float(np.mean(v)) for k, v in res.items()}}


# ---- 2.10 dataset construction: duplicates, groups, leakage across splits -----------------------
def ex_2_10(seed: int = 0) -> dict:
    """Users have several rows each. Random row split puts the same user in train and test (optimistic); a group split does not."""
    rng = np.random.default_rng(seed)
    n_users, rows = 60, 10
    user_effect = rng.standard_normal((n_users, 5)) * 2.0               # user-specific fingerprint
    users = np.repeat(np.arange(n_users), rows)
    X = user_effect[users] + 0.3 * rng.standard_normal((n_users * rows, 5))
    y_user = rng.integers(0, 2, n_users)                                # label is a property of the USER, unrelated to X's meaning
    y = y_user[users]
    def knn1(Xtr, ytr, Xte):
        d = ((Xte[:, None] - Xtr[None]) ** 2).sum(-1); return ytr[d.argmin(1)]
    perm = rng.permutation(len(y)); row_tr, row_te = perm[:400], perm[400:]
    g_tr_users = rng.permutation(n_users)[:40]; g_tr = np.flatnonzero(np.isin(users, g_tr_users)); g_te = np.flatnonzero(~np.isin(users, g_tr_users))
    return {"random_row_split_acc": M.accuracy(y[row_te], knn1(X[row_tr], y[row_tr], X[row_te])),
            "group_split_acc": M.accuracy(y[g_te], knn1(X[g_tr], y[g_tr], X[g_te])),
            "shared_users_in_row_split": int(len(np.intersect1d(users[row_tr], users[row_te]))), "shared_users_in_group_split": int(len(np.intersect1d(users[g_tr], users[g_te])))}


# ---- 2.11-2.13 train / validation / test roles --------------------------------------------------
def ex_2_11_to_2_13(seed: int = 0, n_models: int = 200) -> dict:
    """Pick the best of many pure-noise models on the test set: the winning test score is optimistic. A fresh set tells the truth."""
    rng = np.random.default_rng(seed)
    y_test, y_fresh = rng.integers(0, 2, 100), rng.integers(0, 2, 5000)
    scores = [M.accuracy(y_test, rng.integers(0, 2, 100)) for _ in range(n_models)]  # each "model" guesses
    best = int(np.argmax(scores))
    fresh = M.accuracy(y_fresh, np.random.default_rng(seed + 1000 + best).integers(0, 2, 5000))
    return {"best_test_acc_after_peeking": float(max(scores)), "mean_test_acc": float(np.mean(scores)), "fresh_data_acc_of_winner": fresh}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("ex_"):
            print(name, fn())
