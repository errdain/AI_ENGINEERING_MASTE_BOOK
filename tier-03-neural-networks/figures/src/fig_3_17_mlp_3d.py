"""Figure 3.17-1: a trained MLP drawn in 3D. Edge colour = sign of the learned weight, opacity/width = |weight|."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from _common import OUT, figstyle as fs
import nn as N

sizes = [4, 6, 6, 3]
rng = np.random.default_rng(3)
centers = rng.standard_normal((3, 4)) * 2.0
y = rng.integers(0, 3, 300); X = centers[y] + rng.standard_normal((300, 4))
net = N.MLP(sizes, activation="tanh", init="xavier_normal", seed=1).fit(X, y, lr=0.2, epochs=300)
acc = float((net.predict_classes(X) == y).mean())
Ws = [l.W for l in net.layers if isinstance(l, N.Dense)]                     # each (n_out, n_in)

def grid(n, x):                                                                 # arrange n nodes on a small 2-D grid in the y-z plane
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols)); pts = []
    for i in range(n): pts.append((x, 1.7 * ((i % cols) - (cols - 1) / 2), -1.7 * ((i // cols) - (rows - 1) / 2)))
    return np.array(pts)

pos = [grid(n, 4.6 * k) for k, n in enumerate(sizes)]
fig = plt.figure(figsize=(12, 6.2)); ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(fs.BG)
wmax = max(np.abs(W).max() for W in Ws)
for k, W in enumerate(Ws):
    for j in range(W.shape[0]):
        for i in range(W.shape[1]):
            w = W[j, i]; a = pos[k][i]; b = pos[k + 1][j]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=fs.BLUE if w > 0 else fs.RED, alpha=0.12 + 0.75 * abs(w) / wmax, lw=0.4 + 2.0 * abs(w) / wmax)
names = ["Input layer\n(4 features)", "Hidden layer 1\n(6 tanh units)", "Hidden layer 2\n(6 tanh units)", "Output layer\n(3 logits)"]
cols = [fs.TEAL, fs.YELLOW, fs.YELLOW, fs.ORANGE]
for k, p in enumerate(pos):
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=210, color=cols[k], edgecolor=fs.BG, linewidth=1.2, depthshade=False, zorder=5)
    ax.text(p[:, 0].mean(), 0, 3.0, names[k], color=fs.FG, ha="center", fontsize=11)
ax.set_xlim(-0.8, 14.6); ax.set_ylim(-2.6, 2.6); ax.set_zlim(-2.6, 3.8); ax.view_init(elev=20, azim=-58)
ax.set_axis_off(); ax.set_box_aspect((15.4, 5.2, 6.4), zoom=1.12); ax.set_position([0.0, 0.02, 1.0, 0.9])
fig.text(0.5, 0.955, "A trained multi-layer perceptron (MLP) in 3D", ha="center", color=fs.FG, fontsize=15, fontweight="bold")
fig.text(0.5, 0.915, f"4 → 6 → 6 → 3, tanh hidden units, trained with backprop (training accuracy {acc:.0%}); {sum(W.size for W in Ws)} weights + {6+6+3} biases", ha="center", color=fs.MUTED, fontsize=10)
fig.text(0.06, 0.07, "● positive weight", color=fs.BLUE, fontsize=10); fig.text(0.21, 0.07, "● negative weight", color=fs.RED, fontsize=10)
fig.text(0.37, 0.07, "line opacity and thickness = |weight| (a stronger connection is brighter)", color=fs.MUTED, fontsize=9)
fig.savefig(OUT / "fig_3_17_mlp_3d.png", dpi=170); print("saved", acc)
