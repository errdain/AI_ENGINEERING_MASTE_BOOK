"""Tier 2 figures: AI/ML/DL (2.1), learning curve (2.9), split leakage (2.10), test-set peeking (2.13), decision boundary and threshold (2.14), losses (2.17)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from _common import OUT, fs
import mlkit as M
from lab_01_paradigms_and_data import ex_2_1, ex_2_10

# ---- 2.1 AI ⊃ ML ⊃ DL, with the XOR ladder measured in the lab
r = ex_2_1(); fig = plt.figure(figsize=(12.5, 5.6)); ax = fig.add_axes([0.0, 0.0, 0.5, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
for (x, y, w, h, c, name, sub) in ((0.3, 0.6, 9.4, 8.6, fs.PURPLE, "Artificial Intelligence", "any system that acts intelligently: rules, search, planning, learning"), (1.3, 1.1, 7.4, 5.6, fs.BLUE, "Machine Learning", "behaviour learned from data by fitting parameters"), (2.3, 1.6, 5.4, 2.7, fs.TEAL, "Deep Learning", "many-layer neural networks that learn their own features")):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.35", fc=c + "22", ec=c, lw=2.6)); ax.text(x + w / 2, y + h - 0.55, name, ha="center", va="center", color=c, fontsize=14, fontweight="bold"); ax.text(x + w / 2, y + h - 1.1, sub, ha="center", va="center", color=fs.MUTED, fontsize=8.8)
ax.text(5, 7.55, "hand-written rules · expert systems · A* search", ha="center", color=fs.FG, fontsize=10); ax.text(5, 5.15, "linear / logistic regression · k-means · trees", ha="center", color=fs.FG, fontsize=10); ax.text(5, 2.45, "CNNs · Transformers · LLMs", ha="center", color=fs.FG, fontsize=10)
ax2 = fig.add_axes([0.58, 0.16, 0.39, 0.66]); names = ["hand-written\nrule (AI)", "linear\nlearner (ML)", "ML + engineered\nfeature x₁·x₂", "small neural\nnet (DL)"]; vals = [r["rule_based"], r["linear_ml"], r["ml_with_engineered_feature"], r["neural_net"]]; cols = [fs.PURPLE, fs.BLUE, fs.BLUE, fs.TEAL]
ax2.bar(range(4), vals, color=cols, width=0.62); ax2.axhline(0.5, color=fs.MUTED, ls=(0, (4, 3))); ax2.text(3.45, 0.51, "chance", color=fs.MUTED, ha="right", fontsize=9)
for i, v in enumerate(vals): ax2.text(i, v + 0.02, f"{v:.1%}", ha="center", color=fs.FG, fontsize=10.5, fontweight="bold")
ax2.set_xticks(range(4)); ax2.set_xticklabels(names, fontsize=8.8); ax2.set_ylim(0, 1.1); ax2.set_ylabel("test accuracy"); ax2.grid(True, axis="y", alpha=0.3); ax2.set_title("Same XOR-like task (measured in the lab)", fontsize=11.5)
fig.suptitle("AI ⊃ ML ⊃ DL: who supplies the knowledge — a human, the weights, or the learned features", fontsize=13.5, fontweight="bold", y=0.99); fig.savefig(OUT / "fig_2_01_ai_ml_dl.svg"); plt.close(fig)

# ---- 2.9 learning curve and what hurts it (average over 10 seeds)
ns = [5, 10, 20, 50, 100, 200, 400, 800]; curves = {"clean features, clean labels": [], "+ 60 junk features": [], "40% of training labels flipped": []}
for n in ns:
    a = [[], [], []]
    for seed in range(10):
        X, y = M.make_blobs_binary(1600, sep=2.0, seed=seed); rng = np.random.default_rng(seed); te = np.arange(800, 1600); tr = np.arange(n)
        fit = lambda Xa, ya: M.accuracy(y[te], M.LogisticRegression(n_iter=1500).fit(Xa[tr], ya[tr]).predict(Xa[te]))
        a[0].append(fit(X, y)); a[1].append(fit(np.column_stack([X, rng.standard_normal((1600, 60))]), y)); a[2].append(fit(X, np.where(rng.random(1600) < 0.4, 1 - y, y)))
    for k, key in enumerate(curves): curves[key].append(np.mean(a[k]))
fig, ax = plt.subplots(figsize=(10, 5.4))
for (k, v), c in zip(curves.items(), (fs.TEAL, fs.ORANGE, fs.RED)): ax.semilogx(ns, v, "-o", color=c, lw=2.6, ms=6, label=k)
ax.axhline(0.5, color=fs.MUTED, ls=(0, (4, 3))); ax.set_xlabel("number of training samples n"); ax.set_ylabel("held-out accuracy (mean of 10 seeds)"); ax.set_ylim(0.4, 1.0); ax.grid(True, which="both", alpha=0.3); ax.legend(loc="lower right", fontsize=10)
ax.set_title("Learning curve: more data helps with diminishing returns; junk features and noisy labels slow learning", fontsize=12, fontweight="bold"); fig.savefig(OUT / "fig_2_09_learning_curve.svg"); plt.close(fig)

# ---- 2.10 random row split vs group split
r10 = ex_2_10(); rng = np.random.default_rng(2); users, rows = 12, 6; uid = np.repeat(np.arange(users), rows); colors = [fs.PALETTE[u % 7] for u in uid]
perm = rng.permutation(users * rows); row_train = set(perm[:48]); g_train_users = set(rng.permutation(users)[:8]); grp_train = {i for i in range(users * rows) if uid[i] in g_train_users}
fig, axs = plt.subplots(2, 1, figsize=(12, 6.6))
for ax, tr, title, acc, shared in ((axs[0], row_train, "random ROW split", r10["random_row_split_acc"], r10["shared_users_in_row_split"]), (axs[1], grp_train, "GROUP split (all rows of a user stay together)", r10["group_split_acc"], r10["shared_users_in_group_split"])):
    ax.set_xlim(-0.2, users * rows + 0.2); ax.set_ylim(-0.2, 2.4); ax.axis("off")
    for i in range(users * rows):
        for lab, y0, sel in (("train", 1.3, i in tr), ("test", 0.0, i not in tr)):
            if sel: ax.add_patch(Rectangle((i, y0), 0.85, 0.9, fc=colors[i], ec="none"))
    ax.text(-0.1, 2.25, "train", color=fs.FG, fontsize=11); ax.text(-0.1, 1.15 - 0.0, "test", color=fs.FG, fontsize=11, va="top")
    ax.set_title(f"{title}:  1-NN accuracy {acc:.1%};  users appearing in both train and test: {shared}", fontsize=12, color=fs.FG, loc="left", pad=2)
fig.suptitle("Same data, two splits (each colour = one user, each bar = one row)\nRandom rows leak the user's fingerprint; labels are random per user, so the true accuracy is chance", fontsize=11.5, y=1.0)
fig.savefig(OUT / "fig_2_10_split_leakage.svg"); plt.close(fig)

# ---- 2.13 test-set peeking: best of N guessing models
rng = np.random.default_rng(0); N_models, trials = 200, 3000; best = []; fresh = []
for _ in range(trials):
    y = rng.integers(0, 2, 100); accs = (rng.integers(0, 2, (N_models, 100)) == y).mean(1); best.append(accs.max()); fresh.append((rng.integers(0, 2, 5000) == rng.integers(0, 2, 5000)).mean())
fig, ax = plt.subplots(figsize=(10, 5.2)); bins = np.linspace(0.42, 0.72, 62); ax.hist(best, bins, color=fs.RED, alpha=0.75, label=f"test accuracy of the model picked as best of {N_models}\n(mean {np.mean(best):.1%})"); ax.hist(fresh, bins, color=fs.TEAL, alpha=0.75, label=f"the same kind of model on 5,000 fresh items\n(mean {np.mean(fresh):.1%})")
ax.axvline(0.5, color=fs.FG, ls=(0, (4, 3))); ax.text(0.505, ax.get_ylim()[1] * 0.92, "chance = 50%", color=fs.FG, fontsize=10); ax.set_xlabel("accuracy"); ax.set_ylabel("number of repeated experiments"); ax.legend(loc="upper right", fontsize=9.5)
ax.set_title("Peeking at the test set: choosing the best of 200 coin-flip models looks like skill on 100 items", fontsize=12, fontweight="bold"); fig.savefig(OUT / "fig_2_13_test_peeking.svg"); plt.close(fig)

# ---- 2.14 logistic regression: probability map and decision thresholds
X, y = M.make_blobs_binary(400, sep=2.0, pos_frac=0.3, seed=1); sc = M.StandardScaler().fit(X); m = M.LogisticRegression(n_iter=3000).fit(sc.transform(X), y)
g = np.linspace(-3.2, 5.2, 300); G1, G2 = np.meshgrid(g, np.linspace(-3.2, 5.2, 300)); P = m.predict_proba(sc.transform(np.c_[G1.ravel(), G2.ravel()])).reshape(G1.shape)
fig, ax = plt.subplots(figsize=(8.8, 6.4)); cf = ax.contourf(G1, G2, P, levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.75); ax.contour(G1, G2, P, levels=[0.1], colors=[fs.YELLOW], linewidths=2.6, linestyles="--"); ax.contour(G1, G2, P, levels=[0.5], colors=["white"], linewidths=2.6)
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, color=fs.BLUE, edgecolor=fs.BG, linewidth=0.4); ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, color=fs.RED, edgecolor=fs.BG, linewidth=0.4); fig.colorbar(cf, label="predicted probability of class 1")
ax.plot([], [], color="white", lw=2.6, label="threshold 0.5 (default)"); ax.plot([], [], color=fs.YELLOW, lw=2.6, ls="--", label="threshold 0.1 (misses cost 5× false alarms)"); ax.legend(loc="lower right", fontsize=9.5); ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
ax.set_title("Classification: probabilities out,\nthreshold chosen from costs", fontsize=12, fontweight="bold"); fig.savefig(OUT / "fig_2_14_decision_boundary.png", dpi=150); plt.close(fig)

# ---- 2.17 loss functions
e = np.linspace(-4, 4, 800); fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(e, e**2, color=fs.RED, lw=2.8, label="MSE  e²"); axs[0].plot(e, np.abs(e), color=fs.TEAL, lw=2.8, label="MAE  |e|"); axs[0].plot(e, np.where(np.abs(e) <= 1, 0.5 * e**2, np.abs(e) - 0.5), color=fs.YELLOW, lw=2.8, label="Huber (δ = 1)"); axs[0].set_ylim(0, 9); axs[0].set_xlabel("prediction error  e = ŷ − y"); axs[0].set_ylabel("loss"); axs[0].legend(); axs[0].grid(True, alpha=0.3)
axs[0].set_title("Regression losses: MSE punishes large errors (outliers) most", fontsize=11.5)
z = np.linspace(-8, 4, 500); s = 1 / (1 + np.exp(-z)); axs[1].plot(z, np.abs(s - 1), color=fs.BLUE, lw=2.8, label="BCE: |∂L/∂logit|  =  1 − σ(z)"); axs[1].plot(z, np.abs(2 * (s - 1) * s * (1 - s)), color=fs.ORANGE, lw=2.8, label="MSE on sigmoid:  2(1−σ)·σ(1−σ)")
axs[1].set_xlabel("logit z   (true label = 1; left = confidently WRONG)"); axs[1].set_ylabel("gradient size"); axs[1].legend(loc="upper right"); axs[1].grid(True, alpha=0.3); axs[1].annotate("at z = −6: BCE gradient 1.00,\nMSE gradient 0.005 (≈200× weaker)", xy=(-6, 0.005), xytext=(-7.6, 0.55), color=fs.ORANGE, fontsize=10, arrowprops=dict(arrowstyle="->", color=fs.ORANGE))
axs[1].set_title("Classification: MSE barely corrects a confident mistake", fontsize=11.5); fig.suptitle("Loss functions: what the optimizer is told to care about", fontsize=13.5, fontweight="bold", y=1.02); fig.savefig(OUT / "fig_2_17_loss_functions.svg", bbox_inches="tight"); plt.close(fig)
print("t2 basics ok", {k: round(v[-1], 3) for k, v in curves.items()})
