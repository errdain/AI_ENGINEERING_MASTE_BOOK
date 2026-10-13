"""Tier 2 training-dynamics figures: k-means animation (2.16), SGD paths (2.20), learning-rate animation (2.25), feature-scaling contours (2.47)."""
import numpy as np, matplotlib.pyplot as plt
from _common import OUT, fs
import mlkit as M
from lab_02_models_and_training import make_blobs_multi

# ---- 2.16 k-means, one assignment/update step at a time (same algorithm as mlkit.KMeans, recorded)
X, _ = make_blobs_multi(240, 3, radius=3.2, seed=4); X = X + np.random.default_rng(1).normal(0, 0.15, X.shape); rng = np.random.default_rng(7); C = X[rng.choice(len(X), 3, replace=False)].copy()
hist = [("start: pick 3 random points as centres", None, C.copy())]
for it in range(1, 7):
    lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1); hist.append((f"iteration {it}a: assign every point to its nearest centre", lab, C.copy()))
    newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(3)]); hist.append((f"iteration {it}b: move each centre to the mean of its points", lab, newC.copy()))
    if np.allclose(newC, C): break
    C = newC
inertia = lambda lab, C: float(((X - C[lab]) ** 2).sum()); frames = []
for name, lab, C in hist:
    for rep in range(3):
        fig, ax = plt.subplots(figsize=(7.6, 6.2)); ax.set_aspect("equal"); ax.set_xlim(-5.4, 5.4); ax.set_ylim(-5, 5); ax.grid(True, alpha=0.25)
        ax.scatter(X[:, 0], X[:, 1], s=22, c=[fs.MUTED if lab is None else fs.PALETTE[l] for l in (lab if lab is not None else np.zeros(len(X), int))], alpha=0.85, edgecolor=fs.BG, linewidth=0.3)
        for j in range(3): ax.scatter(*C[j], marker="*", s=420, color=fs.PALETTE[j], edgecolor="white", linewidth=1.6, zorder=6)
        ax.set_title("k-means (k = 3)" + (f"   inertia = {inertia(lab, C):.0f}" if lab is not None else ""), fontsize=13, fontweight="bold"); fig.text(0.5, 0.02, name, ha="center", color=fs.YELLOW, fontsize=11)
        frames.append(fs.fig_to_image(fig)); plt.close(fig)
ref = M.KMeans(3, n_init=10, seed=0).fit(X); print("kmeans final", inertia(lab, C), "library best", ref.inertia_, "frames", len(frames))
print("kmeans bytes", fs.save_gif(frames, OUT / "fig_2_16_kmeans.gif", fps=4, colors=48, hold_last=6))

# ---- 2.25 learning rate on a narrow valley f = x^2 + 10 y^2 (stability limit 2/20 = 0.1): three runs side by side (animation)
f = lambda p: p[0] ** 2 + 10 * p[1] ** 2; grad = lambda p: np.array([2 * p[0], 20 * p[1]]); runs = []
for lr, name, c in ((0.01, "η = 0.01  too small: crawls", fs.BLUE), (0.09, "η = 0.09  near the limit: zig-zags", fs.TEAL), (0.11, "η = 0.11  above 2/λmax = 0.1: diverges", fs.RED)):
    p = np.array([-1.8, 1.0]); path = [p.copy()]
    for _ in range(40): p = p - lr * grad(p); path.append(p.copy())
    runs.append((name, c, np.array(path)))
gx, gy = np.meshgrid(np.linspace(-2.2, 2.2, 200), np.linspace(-1.3, 1.3, 200)); Z = gx ** 2 + 10 * gy ** 2; frames = []
for k in list(range(0, 41, 2)) + [40] * 6:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
    for ax, (name, c, P) in zip(axs, runs):
        ax.contour(gx, gy, Z, levels=np.geomspace(0.05, 40, 14), colors=fs.GRID, linewidths=1); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
        Q = np.clip(P[:k + 1], -2.2, 2.2); ax.plot(Q[:, 0], Q[:, 1], "-o", color=c, lw=2.2, ms=3.5); ax.scatter(*np.clip(P[k], -2.2, 2.2), s=110, color=c, edgecolor="white", zorder=6); ax.scatter([0], [0], marker="*", s=240, color=fs.YELLOW, zorder=5)
        fv = f(P[k]); ax.set_title(name, fontsize=11.5, color=c); ax.text(0, -1.22, f"step {min(k, 40)}   loss = {fv:.3g}", ha="center", color=fs.FG, fontsize=10)
    fig.suptitle("Learning rate: three gradient-descent runs on f(x, y) = x² + 10y²   (★ = minimum)", fontsize=13.5, fontweight="bold", y=0.97); fig.subplots_adjust(top=0.82, bottom=0.08); frames.append(fs.fig_to_image(fig)); plt.close(fig)
print("lr bytes", fs.save_gif(frames, OUT / "fig_2_25_learning_rate.gif", fps=6, colors=32, hold_last=5))

# ---- 2.20 full-batch GD vs mini-batch SGD vs pure SGD on a 2-weight regression loss
rng = np.random.default_rng(3); Xd = rng.standard_normal((256, 2)) * np.array([1.0, 0.6]); w_true = np.array([2.0, -1.0]); y = Xd @ w_true + 0.8 * rng.standard_normal(256)
mse = lambda w: float(np.mean((Xd @ w - y) ** 2)); g1, g2 = np.meshgrid(np.linspace(-1, 3.5, 150), np.linspace(-3, 1.5, 150)); L = np.array([[mse(np.array([a, b])) for a in g1[0]] for b in g2[:, 0]]); wopt = np.linalg.lstsq(Xd, y, rcond=None)[0]
def run(bs, lr, steps, seed=0):
    r = np.random.default_rng(seed); w = np.array([-0.5, 1.0]); path = [w.copy()]
    for _ in range(steps):
        idx = r.choice(256, bs, replace=False) if bs < 256 else np.arange(256); w = w - lr * 2 * Xd[idx].T @ (Xd[idx] @ w - y[idx]) / len(idx); path.append(w.copy())
    return np.array(path)
fig, ax = plt.subplots(figsize=(9.6, 6.4)); ax.contour(g1, g2, L, levels=np.geomspace(L.min() * 1.02, L.max(), 18), colors=fs.GRID, linewidths=1.1)
for bs, lr, st, c, name in ((256, 0.1, 60, fs.TEAL, "full batch: smooth, exact gradient"), (16, 0.1, 150, fs.ORANGE, "mini-batch of 16: a little noisy"), (1, 0.04, 400, fs.RED, "pure SGD (1 sample): very noisy")):
    P = run(bs, lr, st); ax.plot(P[:, 0], P[:, 1], color=c, lw=1.8 if bs > 1 else 1.1, alpha=0.95 if bs > 1 else 0.8, label=name); ax.scatter(*P[-1], color=c, s=60, edgecolor="white", zorder=6)
ax.scatter(*wopt, marker="*", s=300, color=fs.YELLOW, zorder=7, label="least-squares optimum"); ax.scatter([-0.5], [1.0], s=80, color=fs.FG, zorder=7); ax.text(-0.45, 1.08, "start", color=fs.FG)
ax.legend(loc="lower right", fontsize=9.5); ax.set_xlabel("weight w₁"); ax.set_ylabel("weight w₂"); ax.set_title("Gradient noise: smaller batches take cheaper but noisier steps and hover around the optimum", fontsize=11.5, fontweight="bold")
fig.savefig(OUT / "fig_2_20_sgd_paths.png", dpi=150); plt.close(fig)

# ---- 2.47 feature scaling reshapes the loss surface (condition number) — same linear regression, raw vs standardized features
rng = np.random.default_rng(5); Z1 = rng.standard_normal((200, 2)); yv = Z1 @ np.array([1.0, 1.0]) + 0.2 * rng.standard_normal(200); raw = Z1 * np.array([1.0, 12.0]); std = M.StandardScaler().fit_transform(raw)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4))
for ax, Xf, name in ((axs[0], raw, "raw features (scales 1 and 12)"), (axs[1], std, "standardized features")):
    H = 2 * Xf.T @ Xf / len(Xf); ev = np.linalg.eigvalsh(H); lr = 0.9 * 2 / ev.max(); w_opt = np.linalg.lstsq(Xf, yv - yv.mean(), rcond=None)[0]; span = np.array([0.8, 0.8 / 12 if Xf is raw else 0.8]) * 1.1
    a, b = np.meshgrid(np.linspace(w_opt[0] - span[0] * 2, w_opt[0] + span[0] * 2, 200), np.linspace(w_opt[1] - span[1] * 2, w_opt[1] + span[1] * 2, 200)); W = np.stack([a, b], -1); Lz = np.mean(((W @ Xf.T) - (yv - yv.mean())) ** 2, axis=-1)
    ax.contour(a, b, Lz, levels=np.geomspace(Lz.min() + 1e-3, Lz.max(), 14), colors=fs.GRID, linewidths=1.1); w = w_opt + np.array([span[0] * 1.5, -span[1] * 1.7]); path = [w.copy()]
    for _ in range(25): w = w - lr * 2 * Xf.T @ (Xf @ w - (yv - yv.mean())) / len(Xf); path.append(w.copy())
    L0, L25 = float(np.mean((Xf @ path[0] - (yv - yv.mean())) ** 2)), float(np.mean((Xf @ path[-1] - (yv - yv.mean())) ** 2))
    P = np.array(path); ax.plot(P[:, 0], P[:, 1], "-o", color=fs.YELLOW, lw=2, ms=4); ax.scatter(*w_opt, marker="*", s=240, color=fs.TEAL, zorder=6); ax.set_title(f"{name}\ncondition number κ = {ev.max() / ev.min():.0f}, largest stable η = {2 / ev.max():.3f}", fontsize=11); ax.text(0.5, 0.04, f"after 25 steps (η = 0.9 × limit) the loss fell by {1 - L25 / L0:.1%}", transform=ax.transAxes, ha="center", color=fs.YELLOW, fontsize=10.5)
    ax.set_xlabel("w₁   (axes not drawn to scale)"); ax.set_ylabel("w₂")
fig.suptitle("Why scale features: a long narrow valley (left) forces tiny steps and zig-zags; a round bowl (right) lets gradient descent go straight", fontsize=12, fontweight="bold", y=1.03)
fig.savefig(OUT / "fig_2_47_scaling_conditioning.svg", bbox_inches="tight"); plt.close(fig)
print("training ok")
