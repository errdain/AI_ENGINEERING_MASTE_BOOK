"""Tier 1 geometry figures: vector addition/scaling (1.8), dot product (1.10), norm balls (1.15), distance concentration (1.16), cosine (1.17), covariance ellipse (1.32), convexity (1.45)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle
from _common import OUT, fs, arrow, clean2d
from linalg_core import cosine_similarity, norm_p
from info_numerics import covariance_matrix, midpoint_convex_violations

# ---- 1.8 vector addition (tip-to-tail) and scalar multiplication
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8))
u, v = np.array([3.0, 1.0]), np.array([1.0, 2.5]); s = u + v
clean2d(a, ((-0.5, 5.2), (-0.5, 4.2))); arrow(a, (0, 0), u, fs.BLUE, label="a = [3, 1]", label_off=(0.05, -0.35)); arrow(a, (0, 0), v, fs.TEAL, label="b = [1, 2.5]", label_off=(-0.9, 0.12))
arrow(a, u, s, fs.TEAL, ls=(0, (4, 3)), alpha=0.7); arrow(a, v, s, fs.BLUE, ls=(0, (4, 3)), alpha=0.7); arrow(a, (0, 0), s, fs.YELLOW, lw=3.2, label="a + b = [4, 3.5]", label_off=(0.08, 0.1))
a.set_title("Vector addition: place b at the tip of a", fontsize=12)
clean2d(b, ((-3.2, 4.2), (-2.2, 2.2)))
for k, c in ((2, fs.TEAL), (1, fs.BLUE), (0.5, fs.YELLOW), (-1, fs.RED)): arrow(b, (0, 0), k * np.array([1.5, 0.6]), c, lw=2.4, label=f"{k}·a", label_off=(0.05, 0.1 if k > 0 else -0.35))
b.set_title("Scalar multiplication: stretch, shrink or flip", fontsize=12); fig.suptitle("Adding vectors and scaling vectors", fontsize=14, fontweight="bold"); fig.savefig(OUT / "fig_1_08_vector_add_scale.svg"); plt.close(fig)

# ---- 1.10 dot product geometry: projection and the angle
a_v, b_v = np.array([3.0, 0.8]), np.array([1.5, 2.2]); dot = float(a_v @ b_v); cos = cosine_similarity(a_v, b_v); ang = np.degrees(np.arccos(cos)); proj = (dot / (a_v @ a_v)) * a_v
fig, ax = plt.subplots(figsize=(9, 5.4)); clean2d(ax, ((-0.4, 4.4), (-0.6, 3.2)))
arrow(ax, (0, 0), a_v, fs.BLUE, label="a", label_off=(0.05, -0.3)); arrow(ax, (0, 0), b_v, fs.TEAL, label="b", label_off=(-0.2, 0.1)); ax.plot([b_v[0], proj[0]], [b_v[1], proj[1]], color=fs.MUTED, ls=(0, (3, 3)), lw=1.6)
ax.plot([0, proj[0]], [0, proj[1]], color=fs.YELLOW, lw=6, alpha=0.8, solid_capstyle="round"); ax.text(proj[0] / 2 + 0.3, proj[1] / 2 - 0.7, "projection of b on a\n(length ‖b‖cosθ)", color=fs.YELLOW, fontsize=10, ha="center")
th = np.linspace(0, np.radians(np.degrees(np.arctan2(b_v[1], b_v[0])) - np.degrees(np.arctan2(a_v[1], a_v[0]))) + np.arctan2(a_v[1], a_v[0]), 40); ax.plot(0.8 * np.cos(th), 0.8 * np.sin(th), color=fs.PINK, lw=2); ax.text(0.9, 0.55, f"θ = {ang:.0f}°", color=fs.PINK, fontsize=11)
ax.set_title(f"Dot product  a·b = Σ aᵢbᵢ = ‖a‖‖b‖cosθ = {dot:.2f}   (cos θ = {cos:.2f})", fontsize=12.5, fontweight="bold"); fig.savefig(OUT / "fig_1_10_dot_product.svg"); plt.close(fig)

# ---- 1.15 norm unit balls
fig, axs = plt.subplots(1, 4, figsize=(13, 3.9)); th = np.linspace(0, 2 * np.pi, 400)
for ax, (p, name, c) in zip(axs, ((1, "L1  (diamond)", fs.TEAL), (2, "L2  (circle)", fs.BLUE), (np.inf, "L∞  (square)", fs.ORANGE), (0.5, "p = 0.5 (not a norm)", fs.RED))):
    pts = np.array([[np.cos(t), np.sin(t)] for t in th]); r = np.array([1 / norm_p(q, p) if p >= 1 else 1 / (np.abs(q[0]) ** 0.5 + np.abs(q[1]) ** 0.5) ** 2 for q in pts]); P = pts * r[:, None]
    ax.fill(P[:, 0], P[:, 1], color=c, alpha=0.25); ax.plot(P[:, 0], P[:, 1], color=c, lw=2.6); clean2d(ax, ((-1.35, 1.35), (-1.35, 1.35))); ax.set_title(name, fontsize=11.5); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
fig.suptitle("Unit balls {x : ‖x‖ ≤ 1}: the shape of 'distance 1' depends on the norm (p < 1 gives a non-convex shape, so it is not a norm)", fontsize=12.5, fontweight="bold", y=1.04); fig.savefig(OUT / "fig_1_15_norm_balls.svg", bbox_inches="tight"); plt.close(fig)

# ---- 1.16 distance concentration (relative contrast shrinks with dimension)
rng = np.random.default_rng(0); dims = [2, 5, 10, 20, 50, 100, 200, 500, 1000]; con = []
for d in dims:
    X, q = rng.standard_normal((1000, d)), rng.standard_normal(d); dist = np.linalg.norm(X - q, axis=1); con.append((dist.max() - dist.min()) / dist.min())
fig, ax = plt.subplots(figsize=(9, 5)); ax.semilogx(dims, con, "-o", color=fs.ORANGE, lw=2.6, ms=7); ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.35)
ax.set_xlabel("dimension d"); ax.set_ylabel("(max distance − min distance) / min distance"); ax.set_title("Distance concentration: in high dimension all points look equally far away", fontsize=12.5, fontweight="bold")
ax.annotate(f"d = 2: farthest point is ~{con[0]:.0f}× farther than the nearest", xy=(2, con[0]), xytext=(4, con[0] * 0.5), color=fs.FG, arrowprops=dict(arrowstyle="->", color=fs.MUTED), fontsize=10); ax.annotate(f"d = 500: only {con[7]:.2f}× contrast", xy=(500, con[7]), xytext=(25, 0.6), color=fs.FG, arrowprops=dict(arrowstyle="->", color=fs.MUTED), fontsize=10)
fig.savefig(OUT / "fig_1_16_distance_concentration.svg"); plt.close(fig)

# ---- 1.17 cosine similarity: direction, not length
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.9)); base = np.array([1.0, 0.6])
clean2d(axs[0], ((-0.3, 3.2), (-1.4, 2.2)))
for k, c in ((0.6, fs.BLUE), (1.4, fs.TEAL), (2.6, fs.YELLOW)): arrow(axs[0], (0, 0), k * base, c, label=f"{k}·a", label_off=(0.04, 0.08))
q = np.array([1.0, -0.9]); arrow(axs[0], (0, 0), q, fs.RED, label="c", label_off=(0.05, -0.3)); axs[0].set_title(f"Same direction, different lengths: cos = {cosine_similarity(base, 2.6 * base):.2f} for all of them\n(cosine to c = {cosine_similarity(base, q):.2f})", fontsize=11)
ang = np.linspace(0, 180, 361); axs[1].plot(ang, np.cos(np.radians(ang)), color=fs.BLUE, lw=3); axs[1].set_ylim(-1.2, 1.3); axs[1].set_xlabel("angle between vectors θ (degrees)"); axs[1].set_ylabel("cosine similarity = cos θ"); axs[1].grid(True, alpha=0.35)
for t, lab in ((0, "same direction  (1)"), (90, "unrelated / orthogonal  (0)"), (180, "opposite  (−1)")): axs[1].scatter([t], [np.cos(np.radians(t))], s=80, color=fs.YELLOW, zorder=5); axs[1].text(t + (4 if t < 150 else -78), np.cos(np.radians(t)) + (0.07 if t != 180 else 0.14), lab, color=fs.YELLOW, fontsize=10)
axs[1].set_title("Cosine similarity depends only on the angle", fontsize=11); fig.savefig(OUT / "fig_1_17_cosine.svg"); plt.close(fig)

# ---- 1.32 covariance: the data cloud, its covariance ellipse and principal axes
rng = np.random.default_rng(3); C0 = np.array([[3.0, 2.2], [2.2, 2.0]]); X = rng.multivariate_normal([0, 0], C0, 400); C = covariance_matrix(X); w, V = np.linalg.eigh(C); V[:, 1] *= np.sign(V[0, 1])
fig, ax = plt.subplots(figsize=(8.2, 6.2)); ax.scatter(X[:, 0], X[:, 1], s=14, color=fs.BLUE, alpha=0.55)
for k, a in ((1, 0.5), (2, 0.3)): ax.add_patch(Ellipse((0, 0), 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])), fc="none", ec=fs.YELLOW, lw=2, alpha=a + 0.4))
for i, (c, nm) in enumerate(((fs.RED, "minor axis"), (fs.TEAL, "major axis (1st principal component)"))): arrow(ax, (0, 0), 2 * np.sqrt(w[i]) * V[:, i], c, lw=3, label=f"{nm}\nvariance {w[i]:.2f}", label_off=(0.15, 0.25 if i else -0.9))
clean2d(ax, ((-6.5, 6.5), (-5, 5))); ax.set_title(f"Covariance matrix  C = [[{C[0,0]:.2f}, {C[0,1]:.2f}], [{C[1,0]:.2f}, {C[1,1]:.2f}]]   corr = {C[0,1]/np.sqrt(C[0,0]*C[1,1]):.2f}", fontsize=12, fontweight="bold")
ax.text(-6.3, -4.6, "ellipses: 1σ and 2σ contours; eigenvectors of C are the axes, eigenvalues the variances along them", color=fs.MUTED, fontsize=9); fig.savefig(OUT / "fig_1_32_covariance_ellipse.svg"); plt.close(fig)

# ---- 1.45 convexity: the chord test
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6)); x = np.linspace(-2.2, 2.2, 400)
for ax, f, name, col in ((axs[0], lambda t: t**2, "convex:  f(x) = x²", fs.BLUE), (axs[1], lambda t: t**4 - 3 * t**2 + t, "not convex:  f(x) = x⁴ − 3x² + x", fs.RED)):
    ax.plot(x, f(x), color=col, lw=3); a_, b_ = -1.6, 1.3 if col == fs.BLUE else 1.5
    xs = np.linspace(a_, b_, 2); ax.plot(xs, f(xs), color=fs.YELLOW, lw=2.2, ls=(0, (5, 3))); m = (a_ + b_) / 2; ax.scatter([a_, b_], f(xs), color=fs.YELLOW, s=60, zorder=5); ax.scatter([m], [f(m)], color=col, s=70, zorder=6); ax.scatter([m], [(f(a_) + f(b_)) / 2], color=fs.YELLOW, s=70, zorder=6, marker="D")
    ok = f(m) <= (f(a_) + f(b_)) / 2; ax.set_title(name + ("\nchord lies ABOVE the curve ✓" if ok else "\nchord dips BELOW the curve ✗ → not convex"), fontsize=11.5); ax.grid(True, alpha=0.35); ax.set_ylim(-4.5, 8)
v1, v2 = midpoint_convex_violations(lambda t: t**2, -5, 5), midpoint_convex_violations(lambda t: t**4 - 3 * t**2 + t, -2.5, 2.5)
fig.suptitle(f"Convexity test: random chord checks found {v1} violations for x² and {v2} for the double well (any violation proves non-convexity)", fontsize=11.5, y=1.02); fig.savefig(OUT / "fig_1_45_convexity.svg", bbox_inches="tight"); plt.close(fig)
print("geometry ok")
