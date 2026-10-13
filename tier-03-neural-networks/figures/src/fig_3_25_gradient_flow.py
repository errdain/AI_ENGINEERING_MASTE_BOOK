"""Figure 3.25-1 (SVG) and 3.25-2 (PNG, 3D): per-layer gradient norms at initialization, and the vanish/explode landscape over depth x weight gain."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from _common import OUT, figstyle as fs
from lab_02_training_internals import grad_norms_by_layer

# ---- 3.25-1: gradient norm of each layer's weights, first layer = 1 ... last layer = 21 (20 hidden blocks + output Dense)
cfgs = [("sigmoid, Xavier", dict(act="sigmoid", init="xavier_normal"), fs.RED), ("sigmoid + LayerNorm", dict(act="sigmoid", init="xavier_normal", norm="layer"), fs.ORANGE),
        ("sigmoid + residual connections", dict(act="sigmoid", init="xavier_normal", residual=True), fs.TEAL), ("tanh, Xavier", dict(act="tanh", init="xavier_normal"), fs.YELLOW),
        ("ReLU, He", dict(act="relu", init="he_normal"), fs.BLUE)]
fig, ax = plt.subplots(figsize=(11, 6.4)); ratios = {}
for name, kw, c in cfgs:
    g = np.array(grad_norms_by_layer(depth=20, width=64, seed=0, **kw)); ratios[name] = g[0] / g[-1]
    ax.semilogy(np.arange(1, len(g) + 1), g, "-o", color=c, lw=2.4, ms=4.5, label=f"{name}   (first/last = {g[0] / g[-1]:.1e})")
ax.set_xlabel("layer index   (1 = closest to the input, 21 = output layer)"); ax.set_ylabel("gradient norm of the layer's weights (log scale)")
ax.set_title("Vanishing gradients: how much learning signal reaches each layer (20 blocks, width 64, at initialization)", fontsize=13, fontweight="bold", pad=12)
ax.grid(True, which="both", alpha=0.35); ax.legend(loc="lower right", fontsize=10)
ax.annotate("sigmoid: the signal shrinks ≈4× at every layer\n(1e-13 after 20 layers)", xy=(5, 1e-10), xytext=(7.5, 1e-12), color=fs.RED, fontsize=11, arrowprops=dict(arrowstyle="->", color=fs.RED))
ax.set_xticks(range(1, 22, 2))
fig.savefig(OUT / "fig_3_25_gradient_norms.svg"); plt.close(fig)

# ---- 3.25-2: log10 global gradient norm of a ReLU network vs depth and weight gain
depths = np.arange(4, 33, 4); gains = np.round(np.arange(0.5, 2.01, 0.125), 3); Z = np.zeros((len(gains), len(depths)))
for i, g in enumerate(gains):
    for j, d in enumerate(depths): Z[i, j] = np.log10(np.sqrt(sum(v**2 for v in grad_norms_by_layer("relu", "he_normal", depth=int(d), width=64, gain=float(g), seed=0))))
D, G = np.meshgrid(depths, gains)
cmap = LinearSegmentedColormap.from_list("vx", [fs.BLUE, "#2b3a67", fs.BG if False else "#243056", fs.ORANGE, fs.RED])
fig = plt.figure(figsize=(11.5, 7.2)); ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(fs.BG)
surf = ax.plot_surface(D, G, Z, cmap=cmap, edgecolor=(1, 1, 1, 0.08), linewidth=0.3, alpha=0.95, rstride=1, cstride=1, antialiased=True)
ax.plot_surface(D, G, np.zeros_like(Z), color=fs.MUTED, alpha=0.12, linewidth=0)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.set_pane_color((0.06, 0.07, 0.12, 1.0)); pane._axinfo["grid"]["color"] = fs.GRID
ax.set_xlabel("depth (layers)", labelpad=10); ax.set_ylabel("weight gain (1 = He init)", labelpad=10); ax.set_zlabel("log₁₀ global gradient norm", labelpad=8)
ax.view_init(elev=24, azim=-128)
ax.set_title("Gradient scale vs depth and initial weight scale (ReLU network)", fontsize=14, fontweight="bold", pad=6)
ax.text(33.5, 0.4, Z[0, -1] + 0.8, "vanishes\n(gain < 1)", color=fs.BLUE, fontsize=11); ax.text(30, 1.95, Z[-1, -1] - 0.2, "explodes\n(gain > 1)", color=fs.RED, fontsize=11)
fig.colorbar(surf, shrink=0.55, pad=0.08, label="log₁₀ norm"); fig.text(0.5, 0.03, "grey plane = norm 1. Each point is a freshly initialized network: gradient norms change exponentially with depth when the weight scale is off.", ha="center", color=fs.MUTED, fontsize=9.5)
fig.savefig(OUT / "fig_3_25_gain_depth_surface.png", dpi=150); print("ratios", {k: f"{v:.1e}" for k, v in ratios.items()}, "grid", Z.min().round(1), Z.max().round(1))
