"""Tier 2 figures: under/overfitting (2.27), bias-variance (2.30), regularization paths (2.32), distribution shift (2.34)."""
import numpy as np, matplotlib.pyplot as plt
from _common import OUT, fs
import mlkit as M
from lab_03_generalization import lasso_ista, domain_classifier_auc

F = lambda x: np.sin(np.pi * x)
# ---- 2.27 three fits to the same 30 points + train/validation error curve
x, y = M.make_sine(30, 0.2, seed=1); xv, yv = M.make_sine(500, 0.2, seed=101); xs = np.linspace(-1, 1, 400); degs = list(range(0, 16))
tab = {d: M.fit_poly_train_val(d, x, y, xv, yv) for d in degs}
fig = plt.figure(figsize=(16.5, 5.4)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.28)
for k, (d, name, c) in enumerate(((1, "underfit (degree 1)", fs.BLUE), (3, "good fit (degree 3)", fs.TEAL), (15, "overfit (degree 15)", fs.RED))):
    ax = fig.add_subplot(gs[0, k]); m = M.LinearRegression().fit(M.poly_features(x, d)[:, 1:], y); ax.scatter(x, y, s=24, color=fs.FG, zorder=4); ax.plot(xs, F(xs), color=fs.MUTED, lw=1.6, ls=(0, (4, 3)), label="true curve"); ax.plot(xs, np.clip(m.predict(M.poly_features(xs, d)[:, 1:]), -2.5, 2.5), color=c, lw=2.8, label="fit")
    ax.set_ylim(-2.2, 2.2); ax.set_title(f"{name}\ntrain {tab[d].train_mse:.3f}  ·  validation {tab[d].val_mse:.3g}", fontsize=10.5, color=c); ax.grid(True, alpha=0.3)
    if k == 0: ax.legend(fontsize=8.5, loc="lower right")
ax = fig.add_subplot(gs[0, 3]); ax.semilogy(degs, [tab[d].train_mse for d in degs], "-o", color=fs.BLUE, lw=2.4, ms=5, label="training error"); ax.semilogy(degs, [tab[d].val_mse for d in degs], "-o", color=fs.RED, lw=2.4, ms=5, label="validation error")
ax.axvline(3, color=fs.TEAL, ls=(0, (4, 3))); ax.text(3.2, 3e-2, "sweet spot", color=fs.TEAL, fontsize=10); ax.set_xlabel("model complexity (polynomial degree)"); ax.set_ylabel("mean squared error (log scale)"); ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3); ax.set_title("training error keeps falling,\nvalidation error turns up", fontsize=10.5)
fig.suptitle("Underfitting vs overfitting on the same 30 noisy points", fontsize=14, fontweight="bold", y=1.03); fig.savefig(OUT / "fig_2_27_under_over_fitting.svg", bbox_inches="tight"); plt.close(fig)

# ---- 2.30 bias^2 + variance + noise vs complexity (Monte-Carlo, fixed design)
rows = {d: M.bias_variance_poly(d, F, n_datasets=300, seed=0) for d in range(0, 11)}; ds = list(rows)
b2, var, noise, tot = (np.array([rows[d][k] for d in ds]) for k in ("bias2", "variance", "noise", "expected_test_mse"))
fig, ax = plt.subplots(figsize=(10, 5.6)); ax.stackplot(ds, b2, var, noise, colors=[fs.RED + "bb", fs.ORANGE + "bb", fs.MUTED + "88"], labels=["bias²  (systematic error)", "variance  (sensitivity to the training sample)", "irreducible noise σ² = 0.04"])
ax.plot(ds, tot, color=fs.FG, lw=2.6, marker="o", ms=5, label="expected test MSE (sum)"); best = int(np.argmin(tot)); ax.scatter([ds[best]], [tot[best]], s=160, color=fs.YELLOW, edgecolor="white", zorder=6); ax.annotate(f"minimum at degree {ds[best]}\nMSE {tot[best]:.4f}", xy=(ds[best], tot[best]), xytext=(ds[best] + 1.2, 0.22), color=fs.YELLOW, arrowprops=dict(arrowstyle="->", color=fs.YELLOW), fontsize=10.5)
ax.set_xlabel("model complexity (polynomial degree)"); ax.set_ylabel("expected squared error"); ax.set_ylim(0, 0.6); ax.legend(loc="upper right", fontsize=9.5); ax.grid(True, alpha=0.3)
ax.set_title("Bias–variance trade-off (noisy sine, 300 simulated datasets)", fontsize=12.5, fontweight="bold"); fig.savefig(OUT / "fig_2_30_bias_variance.svg"); plt.close(fig)

# ---- 2.32 regularization paths: ridge shrinks, lasso zeroes
Xr, yr = M.make_linear(100, w=(3, -2, 1.2, 0, 0, 0, 0, 0), noise=1.0, seed=1); lams = np.logspace(-3, 3, 40)
ridge = np.array([M.LinearRegression(ridge=l * 100).fit(Xr, yr).coef_ for l in lams]); alphas = np.logspace(-3, 0.6, 40); lasso = np.array([lasso_ista(Xr, yr, a, steps=3000)[0] for a in alphas])
fig, axs = plt.subplots(1, 2, figsize=(13, 5)); cols = [fs.PALETTE[i % 7] if i < 3 else fs.MUTED for i in range(8)]
for ax, path, xs_, name, xl in ((axs[0], ridge, lams * 100, "ridge (L2): all coefficients shrink smoothly, none reach exactly 0", "penalty strength λ"), (axs[1], lasso, alphas, "lasso (L1): irrelevant coefficients hit exactly 0 first", "penalty strength α")):
    for j in range(8): ax.semilogx(xs_, path[:, j], color=cols[j], lw=2.6 if j < 3 else 1.6, label=(f"true weight {[3, -2, 1.2][j]}" if j < 3 else ("5 noise features (true weight 0)" if j == 3 else None)))
    ax.axhline(0, color=fs.GRID, lw=1); ax.set_xlabel(xl); ax.set_ylabel("fitted coefficient"); ax.grid(True, alpha=0.3); ax.set_title(name, fontsize=11)
axs[0].legend(loc="lower left", fontsize=9); fig.suptitle("Regularization paths: the coefficients as the penalty grows", fontsize=13.5, fontweight="bold", y=1.02); fig.savefig(OUT / "fig_2_32_regularization_paths.svg", bbox_inches="tight"); plt.close(fig)

# ---- 2.34 distribution shift: densities, detectors and the model's error
rng = np.random.default_rng(0); tr = rng.normal(0, 1, 4000); prod = rng.normal(2.0, 1, 4000); xx = np.linspace(-4, 6, 400); pdf = lambda m_: np.exp(-0.5 * (xx - m_) ** 2) / np.sqrt(2 * np.pi)
psi_same, psi_shift = M.psi(tr, rng.normal(0, 1, 4000)), M.psi(tr, prod); ks_shift = M.ks_statistic(tr, prod); auc_shift = domain_classifier_auc(tr, prod)
fm = M.LinearRegression().fit(tr[:, None], tr**2 + 0.1 * rng.standard_normal(4000)); f_true = lambda t: t**2
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5))
axs[0].fill_between(xx, pdf(0), color=fs.BLUE, alpha=0.45, label="training data  x ~ N(0, 1)"); axs[0].fill_between(xx, pdf(2.0), color=fs.ORANGE, alpha=0.45, label="production data  x ~ N(2, 1)"); axs[0].legend(fontsize=9.5); axs[0].set_xlabel("feature x"); axs[0].set_ylabel("density"); axs[0].grid(True, alpha=0.3)
axs[0].set_title(f"The input distribution moved: PSI = {psi_shift:.2f} (vs {psi_same:.3f} for same-distribution),\nKS = {ks_shift:.2f}, domain-classifier AUC = {auc_shift:.2f}", fontsize=10.5)
xg = np.linspace(-3, 5, 300); axs[1].plot(xg, f_true(xg), color=fs.TEAL, lw=2.8, label="true relation  y = x²"); axs[1].plot(xg, fm.predict(xg[:, None]), color=fs.RED, lw=2.8, label="linear model fitted on training data"); axs[1].axvspan(-2.5, 2.5, color=fs.BLUE, alpha=0.12); axs[1].text(-2.4, 12, "training range", color=fs.BLUE, fontsize=10)
axs[1].axvspan(1, 3.5, color=fs.ORANGE, alpha=0.12, ymin=0, ymax=0.55); axs[1].text(1.05, 3.0, "production\nrange", color=fs.ORANGE, fontsize=10); axs[1].legend(loc="upper center", fontsize=9.5); axs[1].set_xlabel("feature x"); axs[1].set_ylabel("y"); axs[1].grid(True, alpha=0.3)
e_in, e_out = M.mse(f_true(tr), fm.predict(tr[:, None])), M.mse(f_true(prod), fm.predict(prod[:, None])); axs[1].set_title(f"…and the model is wrong where the data now lives:\nMSE {e_in:.1f} on training data → {e_out:.1f} in production", fontsize=10.5)
fig.suptitle("Covariate shift: detect it from inputs alone, before labels arrive", fontsize=13.5, fontweight="bold", y=1.03); fig.savefig(OUT / "fig_2_34_distribution_shift.svg", bbox_inches="tight"); plt.close(fig)
print("generalization ok")
