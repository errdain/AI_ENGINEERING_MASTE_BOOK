"""Tier 2 metric figures: precision/recall vs threshold (2.38/2.39), ROC and PR under imbalance (2.41), confusion matrix (2.42), calibration (2.43), cross-validation schemes (2.44), imbalance methods (2.45), cost vs threshold (2.51)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from _common import OUT, fs
import mlkit as M
import lab_04_metrics_and_data as L4

# ---- 2.38/2.39 precision and recall as the threshold moves (10% positives)
yte, p, _ = L4._scores(0, pos_frac=0.1, sep=2.0); ts = np.linspace(0.02, 0.98, 97); P = np.array([M.precision(yte, (p >= t).astype(int), zero_division=1.0) for t in ts]); R = np.array([M.recall(yte, (p >= t).astype(int)) for t in ts]); F1 = np.array([M.f1(yte, (p >= t).astype(int)) for t in ts])
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5))
axs[0].plot(ts, P, color=fs.BLUE, lw=3, label="precision"); axs[0].plot(ts, R, color=fs.ORANGE, lw=3, label="recall"); axs[0].plot(ts, F1, color=fs.YELLOW, lw=2, ls="--", label="F1"); b = int(np.argmax(F1)); axs[0].scatter([ts[b]], [F1[b]], color=fs.YELLOW, s=90, zorder=5, edgecolor="white")
axs[0].set_xlabel("decision threshold"); axs[0].set_ylabel("score"); axs[0].legend(loc="lower center", fontsize=10); axs[0].grid(True, alpha=0.3); axs[0].set_title(f"Raising the threshold: precision ↑, recall ↓  (best F1 = {F1[b]:.2f} at {ts[b]:.2f})", fontsize=11)
ppv = lambda pi, tpr=0.9, fpr=0.05: tpr * pi / (tpr * pi + fpr * (1 - pi)); pis = np.logspace(-3, np.log10(0.5), 100); axs[1].semilogx(pis, ppv(pis), color=fs.BLUE, lw=3); axs[1].set_xlabel("prevalence of positives in the population"); axs[1].set_ylabel("precision of the same classifier"); axs[1].grid(True, which="both", alpha=0.3)
for pi in (0.5, 0.1, 0.01, 0.001): axs[1].scatter([pi], [ppv(pi)], color=fs.YELLOW, s=70, zorder=5); axs[1].text(pi * 1.15, ppv(pi) + 0.04, f"{pi:g}: {ppv(pi):.2f}", color=fs.YELLOW, fontsize=10)
axs[1].set_title("Precision depends on prevalence (TPR 0.90, FPR 0.05 held fixed)", fontsize=11); fig.suptitle("Precision and recall", fontsize=13.5, fontweight="bold", y=1.02); fig.savefig(OUT / "fig_2_38_precision_recall.svg", bbox_inches="tight"); plt.close(fig)

# ---- 2.41 ROC vs PR under imbalance
fig, axs = plt.subplots(1, 2, figsize=(13, 5.4)); cols = {0.5: fs.TEAL, 0.05: fs.ORANGE, 0.005: fs.RED}
for pf, c in cols.items():
    y_, s_, _ = L4._scores(0, n=40000, pos_frac=pf, sep=2.0); fpr, tpr, _ = M.roc_curve(y_, s_); axs[0].plot(fpr, tpr, color=c, lw=2.6, label=f"{pf:.1%} positives:  AUC {M.roc_auc(y_, s_):.3f}")
    o = np.argsort(-s_); tp = np.cumsum(y_[o]); prec = tp / np.arange(1, len(y_) + 1); rec = tp / y_.sum(); axs[1].plot(rec, prec, color=c, lw=2.6, label=f"{pf:.1%} positives:  AP {M.average_precision(y_, s_):.3f}"); axs[1].axhline(pf, color=c, lw=1, ls=(0, (3, 3)), alpha=0.7)
axs[0].plot([0, 1], [0, 1], color=fs.MUTED, ls=(0, (4, 3))); axs[0].set_xlabel("false-positive rate"); axs[0].set_ylabel("true-positive rate (recall)"); axs[0].legend(loc="lower right"); axs[0].grid(True, alpha=0.3); axs[0].set_title("ROC curve: barely changes with prevalence", fontsize=11.5)
axs[1].set_xlabel("recall"); axs[1].set_ylabel("precision"); axs[1].legend(loc="upper right", fontsize=9.5); axs[1].grid(True, alpha=0.3); axs[1].set_ylim(0, 1.03); axs[1].set_title("Precision–recall curve: collapses for rare positives\n(dashed = precision of a random guess)", fontsize=11.5)
fig.suptitle("Same classifier quality, three prevalences: ROC looks fine, precision–recall tells the truth", fontsize=13.5, fontweight="bold", y=1.03); fig.savefig(OUT / "fig_2_41_roc_vs_pr.svg", bbox_inches="tight"); plt.close(fig)

# ---- 2.42 confusion matrix (3 classes, counts and row-normalized recall)
rng = np.random.default_rng(1); y3 = rng.integers(0, 3, 600); yh = np.where(rng.random(600) < 0.78, y3, (y3 + rng.integers(1, 3, 600)) % 3); cm = M.confusion_matrix_multiclass(y3, yh, 3); names = ["billing", "refund", "cancel"]; pr, rc, f1 = M.precision_recall_f1_per_class(cm)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, mat, title, fmt in ((axs[0], cm, "counts  cm[i, j] = true class i predicted as j", "{:.0f}"), (axs[1], cm / cm.sum(1, keepdims=True), "each row ÷ row total = per-class RECALL on the diagonal", "{:.0%}")):
    im = ax.imshow(mat, cmap="viridis"); ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(names); ax.set_yticklabels(names); ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.set_title(title, fontsize=10.5)
    for i in range(3):
        for j in range(3): ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", color="white" if mat[i, j] < mat.max() * 0.6 else fs.BG, fontsize=15, fontweight="bold")
fig.suptitle(f"Confusion matrix (3-class intent classifier): recall {', '.join(f'{n} {r:.0%}' for n, r in zip(names, rc))};  precision {', '.join(f'{n} {q:.0%}' for n, q in zip(names, pr))}", fontsize=11.5, fontweight="bold", y=1.03)
fig.savefig(OUT / "fig_2_42_confusion_matrix.svg", bbox_inches="tight"); plt.close(fig)

# ---- 2.43 reliability diagram: raw vs temperature scaled (same model as lab ex_2_43)
rng = np.random.default_rng(0); n, d = 300, 40; w = np.zeros(d); w[:3] = 1.0; gen = lambda m: (lambda X: (X, (rng.random(m) < M.sigmoid(X @ w)).astype(int)))(rng.standard_normal((m, d)))
Xtr, ytr = gen(n); Xva, yva = gen(2000); Xte, yte2 = gen(5000); mm = M.LogisticRegression(lr=0.3, n_iter=3000).fit(Xtr, ytr); zva, zte = mm.decision_function(Xva), mm.decision_function(Xte); T = L4.fit_temperature(zva, yva)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4))
for ax, pr_, name, c in ((axs[0], M.sigmoid(zte), "raw model", fs.RED), (axs[1], M.sigmoid(zte / T), f"after temperature scaling (T = {T:.2f})", fs.TEAL)):
    bins = M.reliability_bins(yte2, pr_, 10); xs_ = [b[1] for b in bins if b[0] > 0]; ys_ = [b[2] for b in bins if b[0] > 0]; cnt = [b[0] for b in bins if b[0] > 0]
    ax.plot([0, 1], [0, 1], color=fs.MUTED, ls=(0, (4, 3)), label="perfect calibration"); ax.plot(xs_, ys_, "-o", color=c, lw=2.8, ms=8, label=name); ax.fill_between(xs_, xs_, ys_, color=c, alpha=0.2)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction of positives"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=10); ax.set_title(f"{name}\nECE = {M.expected_calibration_error(yte2, pr_):.3f}   Brier = {M.brier_score(yte2, pr_):.3f}   AUC = {M.roc_auc(yte2, pr_):.3f}", fontsize=10.5)
fig.suptitle("Calibration: when the model says 90%, is it right 90% of the time? (overconfidence is fixed without changing the ranking)", fontsize=12, fontweight="bold", y=1.03); fig.savefig(OUT / "fig_2_43_calibration.svg", bbox_inches="tight"); plt.close(fig)

# ---- 2.44 cross-validation schemes (drawn from the actual index sets)
fig, axs = plt.subplots(3, 1, figsize=(12, 7.6)); n = 40; ybal = np.r_[np.ones(8), np.zeros(32)].astype(int)
def strip(ax, tr, va, y0, label=None, cols=(fs.BLUE, fs.ORANGE)):
    for i in range(n):
        ax.add_patch(Rectangle((i, y0), 0.9, 0.8, fc=(cols[1] if i in va else cols[0]) if (i in va or i in tr) else fs.GRID, ec="none"))
    if label: ax.text(-0.6, y0 + 0.4, label, ha="right", va="center", color=fs.FG, fontsize=10)
for ax in axs: ax.set_xlim(-6, n + 0.5); ax.set_ylim(-0.3, 5.7); ax.axis("off")
for f, (tr, va) in enumerate(M.kfold_indices(n, 5, seed=0)): strip(axs[0], set(tr), set(va), 4.6 - f * 1.0, f"fold {f + 1}")
axs[0].set_title("k-fold (k = 5): every sample is validated exactly once", fontsize=11.5, loc="left"); axs[0].add_patch(Rectangle((30, 5.45), 0.9, 0.5, fc=fs.BLUE, ec="none", clip_on=False)); axs[0].text(31.2, 5.7, "train", color=fs.FG, va="center", fontsize=10); axs[0].add_patch(Rectangle((35, 5.45), 0.9, 0.5, fc=fs.ORANGE, ec="none", clip_on=False)); axs[0].text(36.2, 5.7, "validation", color=fs.FG, va="center", fontsize=10)
for f, (tr, va) in enumerate(M.kfold_indices(n, 5, seed=0, stratify=ybal)): strip(axs[1], set(tr), set(va), 4.6 - f * 1.0, f"fold {f + 1}", cols=(fs.BLUE, fs.ORANGE)); [axs[1].add_patch(Rectangle((i, 4.6 - f * 1.0 + 0.8), 0.9, 0.12, fc=fs.RED, ec="none")) for i in np.flatnonzero(ybal == 1) if i in set(va)]
axs[1].set_title("stratified k-fold: every validation fold keeps the class ratio (red marks = rare positives: exactly 1–2 per fold)", fontsize=11.5, loc="left")
for f, (tr, va) in enumerate(M.time_series_splits(n, 4)): strip(axs[2], set(tr), set(va), 4.3 - f * 1.1, f"split {f + 1}")
axs[2].text(n / 2, 0.35, "time →", color=fs.MUTED, ha="center"); axs[2].set_title("time-series split (expanding window): always train on the past, validate on the future", fontsize=11.5, loc="left")
fig.suptitle("Cross-validation schemes", fontsize=14, fontweight="bold", y=0.995); fig.savefig(OUT / "fig_2_44_cross_validation.svg", bbox_inches="tight"); plt.close(fig)

# ---- 2.45 imbalance methods move the operating point, not the ranking
r45 = L4.ex_2_45(); names = ["plain", "class_weights", "oversample", "plain+tuned_threshold(train)"]; labs = ["plain\n(threshold 0.5)", "class weights", "oversampling", "plain +\ntuned threshold"]; metrics = ["precision", "recall", "f1", "pr_auc"]; cols = [fs.BLUE, fs.ORANGE, fs.YELLOW, fs.TEAL]
fig, ax = plt.subplots(figsize=(12, 5.6)); wbar = 0.2
for k, (m_, c) in enumerate(zip(metrics, cols)): vals = [r45[nm][m_] for nm in names]; ax.bar(np.arange(4) + (k - 1.5) * wbar, vals, wbar * 0.92, color=c, label={"pr_auc": "PR-AUC (ranking quality)", "f1": "F1"}.get(m_, m_)); [ax.text(i + (k - 1.5) * wbar, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=fs.FG) for i, v in enumerate(vals)]
ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylim(0, 1.08); ax.grid(True, axis="y", alpha=0.3); ax.legend(ncol=4, loc="upper center", fontsize=10); ax.set_ylabel("score (2% positives)")
ax.set_title("Class imbalance: weights and resampling trade precision for recall; PR-AUC (the ranking) does not move", fontsize=12, fontweight="bold"); fig.savefig(OUT / "fig_2_45_imbalance_methods.svg"); plt.close(fig)

# ---- 2.51 expected cost vs threshold for three cost ratios
X, y = M.make_blobs_binary(30000, sep=1.8, pos_frac=0.05, seed=0); tr, te = np.arange(15000), np.arange(15000, 30000); sc = M.StandardScaler().fit(X[tr]); mod = M.LogisticRegression(n_iter=1500).fit(sc.transform(X[tr]), y[tr]); pp = mod.predict_proba(sc.transform(X[te])); ys = y[te]; ts = np.linspace(0.01, 0.99, 98)
cost = lambda fn, fp: np.array([(fn * np.sum((pp < t) & (ys == 1)) + fp * np.sum((pp >= t) & (ys == 0))) / len(ys) for t in ts])
fig, ax = plt.subplots(figsize=(10.5, 5.6))
for (fn, fp, c, name) in ((1, 5, fs.TEAL, "false alarm costs 5× a miss"), (1, 1, fs.BLUE, "equal costs"), (20, 1, fs.RED, "a miss costs 20× a false alarm")):
    cu = cost(fn, fp) / (fn if fn > fp else fp); b = int(np.argmin(cu)); ax.plot(ts, cu, color=c, lw=2.8, label=name); ax.scatter([ts[b]], [cu[b]], s=110, color=c, edgecolor="white", zorder=5); ax.text(ts[b] + 0.02, cu[b] + 0.012, f"best threshold {ts[b]:.2f}", color=c, fontsize=10)
ax.axvline(0.5, color=fs.MUTED, ls=(0, (4, 3))); ax.text(0.505, 0.2, "default threshold 0.5", color=fs.MUTED, fontsize=10); ax.set_xlabel("decision threshold"); ax.set_ylabel("expected cost per case (scaled by the larger unit cost)"); ax.grid(True, alpha=0.3); ax.legend(loc="upper center", fontsize=10)
ax.set_title("Metric selection starts from costs: the best threshold moves with the cost ratio (5% positives)", fontsize=11.5, fontweight="bold"); fig.savefig(OUT / "fig_2_51_cost_vs_threshold.svg"); plt.close(fig)
print("metrics ok")
