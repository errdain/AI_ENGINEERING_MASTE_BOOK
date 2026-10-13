"""Tier 1: eigenvectors animation (1.18), SVD steps and low-rank images (1.19), derivative-as-limit animation (1.21), gradient on a 3-D surface (1.23)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from _common import OUT, fs, arrow, clean2d
from linalg_core import low_rank, power_iteration
from calculus_prob import derivative, numerical_gradient

# ---- 1.18 eigenvectors: the directions a matrix only stretches (animation)
A = np.array([[2.0, 1.0], [1.0, 2.0]]); w, V = np.linalg.eigh(A); lam, v1 = power_iteration(A); assert abs(lam - w.max()) < 1e-6              # dominant eigenpair agrees with power iteration
th = np.linspace(0, 2 * np.pi, 200); circle = np.vstack([np.cos(th), np.sin(th)]); gl = [np.vstack([np.linspace(-1.5, 1.5, 40), np.full(40, c)]) for c in np.linspace(-1.5, 1.5, 7)] + [np.vstack([np.full(40, c), np.linspace(-1.5, 1.5, 40)]) for c in np.linspace(-1.5, 1.5, 7)]
g = np.array([0.9, -0.2]); frames = []; NF = 36
for k in range(NF + 12):
    t = fs.ease(min(1, k / (NF * 0.8))); M = np.eye(2) + t * (A - np.eye(2)); fig, ax = plt.subplots(figsize=(8, 6.2)); clean2d(ax, ((-4.6, 4.6), (-3.8, 3.8)), grid=False)
    for L in gl: P = M @ L; ax.plot(P[0], P[1], color=fs.GRID, lw=1.1)
    C = M @ circle; ax.fill(C[0], C[1], color=fs.BLUE, alpha=0.18); ax.plot(C[0], C[1], color=fs.BLUE, lw=2.4)
    for i, c in ((1, fs.YELLOW), (0, fs.TEAL)):
        ev = V[:, i]; ax.plot([-4 * ev[0], 4 * ev[0]], [-4 * ev[1], 4 * ev[1]], color=c, lw=1, ls=(0, (4, 4)), alpha=0.7); arrow(ax, (0, 0), M @ ev, c, lw=3.4, label=(f"eigenvector, λ = {w[i]:.0f}" if k > NF * 0.8 else None), label_off=(0.1, 0.15))
    arrow(ax, (0, 0), g, fs.PINK, lw=2, alpha=0.5); arrow(ax, (0, 0), M @ g, fs.PINK, lw=3.2, label="a generic vector\nchanges direction" if k > NF * 0.8 else None, label_off=(0.15, -0.9))
    fig.text(0.5, 0.945, "Eigenvectors: directions the matrix only stretches", ha="center", color=fs.FG, fontsize=15, fontweight="bold"); fig.text(0.5, 0.895, f"A = [[2, 1], [1, 2]]   →   A v = λ v   with λ = {w[1]:.0f} (along 45°) and λ = {w[0]:.0f} (along 135°)", ha="center", color=fs.MUTED, fontsize=10.5)
    fig.text(0.5, 0.05, "the unit circle becomes an ellipse whose axes are the eigenvectors;\nthe pink vector is not an eigenvector, so it is rotated as well as stretched", ha="center", color=fs.MUTED, fontsize=9.5)
    frames.append(fs.fig_to_image(fig)); plt.close(fig)
print("eigen bytes", fs.save_gif(frames, OUT / "fig_1_18_eigenvectors.gif", fps=12, colors=48, hold_last=12))

# ---- 1.19 SVD as rotate -> stretch -> rotate, and low-rank approximation
M0 = np.array([[2.2, 0.9], [-0.4, 1.3]]); U, s, Vt = np.linalg.svd(M0); assert np.allclose(U @ np.diag(s) @ Vt, M0)
stages = [("input: unit circle\nwith the right singular\nvectors v₁, v₂", np.eye(2)), ("① rotate by Vᵀ\n(v₁, v₂ → coordinate axes)", Vt), ("② stretch by Σ\n(σ₁ = %.2f, σ₂ = %.2f)" % tuple(s), np.diag(s) @ Vt), ("③ rotate by U\n= A x", M0)]
fig, axs = plt.subplots(1, 4, figsize=(16, 4.6)); circle = np.vstack([np.cos(th), np.sin(th)])
for ax, (name, T) in zip(axs, stages):
    clean2d(ax, ((-3, 3), (-3, 3))); C = T @ circle; ax.fill(C[0], C[1], color=fs.BLUE, alpha=0.2); ax.plot(C[0], C[1], color=fs.BLUE, lw=2.4)
    arrow(ax, (0, 0), T @ Vt[0], fs.YELLOW, lw=3); arrow(ax, (0, 0), T @ Vt[1], fs.TEAL, lw=3); ax.set_title(name, fontsize=10.5)
fig.suptitle("Singular value decomposition  A = U Σ Vᵀ : any matrix is a rotation, a stretch along the axes, and another rotation", fontsize=13, fontweight="bold", y=1.06); fig.savefig(OUT / "fig_1_19_svd_steps.svg", bbox_inches="tight"); plt.close(fig)
yy, xx = np.mgrid[0:64, 0:64]; img = (np.sin(xx / 6.0) + 0.6 * np.cos(yy / 9.0) + ((xx - 32) ** 2 + (yy - 28) ** 2 < 150) * 1.4 + 0.5 * np.sin((xx + yy) / 12.0)); img = img + 0.08 * np.random.default_rng(0).standard_normal(img.shape)
sv = np.linalg.svd(img, compute_uv=False); en = np.cumsum(sv ** 2) / np.sum(sv ** 2); ks = (1, 3, 6, 15)
fig = plt.figure(figsize=(14, 6.4)); gs = fig.add_gridspec(2, 5, height_ratios=[3.2, 2.3], hspace=0.55); ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img, cmap="magma"); ax0.set_title("original 64×64\n4,096 numbers", fontsize=10); ax0.axis("off")
for j, k in enumerate(ks): a = fig.add_subplot(gs[0, j + 1]); a.imshow(low_rank(img, k), cmap="magma"); err = np.linalg.norm(img - low_rank(img, k)) / np.linalg.norm(img); a.set_title(f"rank {k}: {k * 128} numbers ({k * 128 / 4096:.0%})\nerror {err:.0%}", fontsize=10); a.axis("off")
axb = fig.add_subplot(gs[1, :3]); axb.semilogy(np.arange(1, 65), sv, "-o", color=fs.BLUE, lw=2.2, ms=4); axb.set_xlabel("index i"); axb.set_ylabel("singular value σᵢ"); axb.grid(True, alpha=0.3); axb.set_title("singular values: a few large ones carry the structure", fontsize=10.5)
axc = fig.add_subplot(gs[1, 3:]); axc.plot(np.arange(1, 65), en, color=fs.YELLOW, lw=2.6); axc.axhline(0.95, color=fs.MUTED, ls=(0, (4, 3))); k95 = int(np.searchsorted(en, 0.95) + 1); axc.scatter([k95], [en[k95 - 1]], color=fs.RED, zorder=5); axc.text(k95 + 2, 0.55, f"{k95} of 64 singular values\ncapture 95% of the energy", color=fs.RED, fontsize=10); axc.set_xlabel("rank k"); axc.set_ylabel("cumulative energy Σσᵢ² / Σσ²"); axc.grid(True, alpha=0.3); axc.set_title("how many components are enough?", fontsize=10.5)
fig.suptitle("Low-rank approximation with the SVD (best rank-k approximation, Eckart–Young)", fontsize=13.5, fontweight="bold", y=0.99); fig.savefig(OUT / "fig_1_19_lowrank_image.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ---- 1.21 the derivative as the limit of secant slopes (animation)
f = lambda x: x ** 3; x0 = 0.7; true = 3 * x0 ** 2; assert abs(derivative(f, x0) - true) < 1e-6; xs = np.linspace(-0.2, 2.2, 300); frames = []; NF = 40
for k in range(NF + 10):
    h = 10 ** (0.0 - 3.0 * fs.ease(min(1, k / NF))); sl = (f(x0 + h) - f(x0)) / h; fig, ax = plt.subplots(figsize=(8.6, 5.6)); ax.plot(xs, f(xs), color=fs.BLUE, lw=3)
    L = np.array([-0.2, 2.2]); ax.plot(L, f(x0) + true * (L - x0), color=fs.MUTED, lw=1.4, ls=(0, (5, 4)), label=f"tangent (slope {true:.3f})"); ax.plot(L, f(x0) + sl * (L - x0), color=fs.YELLOW, lw=2.6, label=f"secant through x₀ and x₀+h")
    ax.scatter([x0, x0 + h], [f(x0), f(x0 + h)], color=[fs.TEAL, fs.YELLOW], s=[90, 70], zorder=6); ax.set_xlim(-0.2, 2.2); ax.set_ylim(-1.2, 7); ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=10)
    ax.set_title(f"f(x) = x³ at x₀ = 0.7:   h = {h:.3g}   →   slope of the secant = {sl:.4f}", fontsize=12.5, fontweight="bold"); fig.text(0.5, 0.01, "as h → 0 the secant turns into the tangent and its slope tends to f′(x₀) = 3x₀² = 1.47", ha="center", color=fs.MUTED, fontsize=10)
    frames.append(fs.fig_to_image(fig)); plt.close(fig)
print("tangent bytes", fs.save_gif(frames, OUT / "fig_1_21_derivative_limit.gif", fps=10, colors=48, hold_last=10))

# ---- 1.23 gradient on a 3-D surface: arrows point uphill, descent steps go the other way
fz = lambda v: float(v[0] ** 2 + 10 * v[1] ** 2); gx = np.linspace(-2.2, 2.2, 60); X, Y = np.meshgrid(gx, gx); Z = X ** 2 + 10 * Y ** 2; Zc = np.where(Z <= 30, Z, np.nan)
cmap = LinearSegmentedColormap.from_list("gr", [fs.BLUE, "#2b3a67", fs.ORANGE]); fig = plt.figure(figsize=(11, 7)); ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(fs.BG)
ax.plot_surface(X, Y, Zc, cmap=cmap, alpha=0.62, linewidth=0, rstride=1, cstride=1, vmin=0, vmax=30); ax.contour(X, Y, Zc, zdir="z", offset=-1, levels=10, colors=fs.MUTED, linewidths=0.8, alpha=0.8)
p = np.array([-2.0, 1.1]); path = [p.copy()]
for _ in range(25): p = p - 0.085 * numerical_gradient(fz, p); path.append(p.copy())
path = np.array(path); zz = path[:, 0] ** 2 + 10 * path[:, 1] ** 2; ax.plot(path[:, 0], path[:, 1], zz + 0.4, color=fs.YELLOW, lw=2.6, marker="o", ms=4, label="gradient descent steps  w ← w − η∇f", zorder=9); ax.plot(path[:, 0], path[:, 1], np.full(len(path), -1), color=fs.YELLOW, lw=1.5, alpha=0.8)
for q in ((-1.6, -0.9), (1.4, 0.7), (-0.4, 1.3), (1.9, -0.5)):
    q = np.array(q); gq = numerical_gradient(fz, q); n = gq / (np.linalg.norm(gq) + 1e-9); ax.quiver(q[0], q[1], fz(q) + 0.3, 0.55 * n[0], 0.55 * n[1], 0.0, color=fs.RED, linewidth=3, arrow_length_ratio=0.4)
ax.plot([], [], color=fs.RED, lw=3, label="gradient ∇f points UPHILL (steepest ascent)")
for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.set_pane_color((0.06, 0.07, 0.12, 1.0)); pane._axinfo["grid"]["color"] = fs.GRID
ax.set_zlim(-1, 30); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x, y) = x² + 10y²"); ax.view_init(elev=30, azim=-56); ax.legend(loc="upper left", fontsize=10)
ax.set_title("The gradient points uphill; minimizing means stepping the other way\n(narrow valley: steps zig-zag across it, as in the learning-rate discussion)", fontsize=12.5, fontweight="bold"); fig.savefig(OUT / "fig_1_23_gradient_surface.png", dpi=140); plt.close(fig)
print("linalg/calculus ok")
