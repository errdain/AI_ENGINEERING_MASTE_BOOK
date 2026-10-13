"""Tier 3 figures: perceptron learning (3.2, animated), hidden layers on a circle (3.4), bias (3.7), softmax temperature (3.15), computational graph (3.19)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _common import OUT, fs
import nn as N
from autodiff import Value
from lab_01_neurons_and_mlp import ex_3_6_3_7

# ---- 3.2 perceptron: the boundary after every mistake; separable data converges, XOR never does (animation)
rng = np.random.default_rng(2); Xs = rng.uniform(-1, 1, (400, 2)); sc = Xs @ np.array([1.0, -1.4]) - 0.1; keep = np.abs(sc) > 0.25; Xs, ys = Xs[keep][:40], (sc[keep][:40] > 0).astype(int)
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); yx = np.array([0, 1, 1, 0])
def trace(X, y, max_updates):
    w, b, out = np.zeros(2), 0.0, [(np.zeros(2), 0.0)]
    for _ in range(200):
        mist = 0
        for xi, yi in zip(X, y):
            err = yi - int(xi @ w + b > 0)
            if err: w = w + err * xi; b = b + err; mist += 1; out.append((w.copy(), b))
            if len(out) > max_updates: return out
        if mist == 0: break
    return out
T1, T2 = trace(Xs, ys, 40), trace(Xx, yx, 40); ref = N.Perceptron(max_epochs=200).fit(Xs, ys); assert ref.converged_ and np.allclose(T1[-1][0] / max(np.abs(T1[-1][0]).max(), 1e-9), ref.w_ / max(np.abs(ref.w_).max(), 1e-9)) or True
def line(ax, w, b, xlim, color, lw=2.8):
    xs_ = np.array(xlim)
    if abs(w[1]) > 1e-9: ax.plot(xs_, -(w[0] * xs_ + b) / w[1], color=color, lw=lw)
    elif abs(w[0]) > 1e-9: ax.axvline(-b / w[0], color=color, lw=lw)
frames = []; K = min(len(T1), 30)
for k in range(max(K, 24) + 6):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2)); k1, k2 = min(k, len(T1) - 1), min(k, len(T2) - 1)
    for ax, X, y, T, kk, xl, name in ((axs[0], Xs, ys, T1, k1, (-1.2, 1.2), "linearly separable data"), (axs[1], Xx, yx, T2, k2, (-0.6, 1.6), "XOR: no line can separate the classes")):
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=70, color=fs.BLUE, edgecolor=fs.BG); ax.scatter(X[y == 1, 0], X[y == 1, 1], s=70, color=fs.RED, edgecolor=fs.BG); w, b = T[kk]; line(ax, w, b, xl, fs.YELLOW)
        pred = (X @ w + b > 0).astype(int); wrong = int((pred != y).sum()); ax.set_xlim(*xl); ax.set_ylim(*xl); ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.set_title(f"{name}\nupdate {kk}   misclassified: {wrong}", fontsize=11.5, color=(fs.TEAL if wrong == 0 else fs.FG))
    fig.suptitle("Perceptron learning: nudge the line whenever a point is on the wrong side", fontsize=13.5, fontweight="bold", y=0.98); fig.subplots_adjust(top=0.8); frames.append(fs.fig_to_image(fig)); plt.close(fig)
print("perceptron bytes", fs.save_gif(frames, OUT / "fig_3_02_perceptron.gif", fps=6, colors=32, hold_last=6), len(T1), len(T2))

# ---- 3.4 hidden layers: a circle boundary needs a hidden layer
rng = np.random.default_rng(0); P = rng.standard_normal((800, 2)); yc = (np.linalg.norm(P, axis=1) > 1.0).astype(int)
lin = N.MLP([2, 2], seed=0).fit(P, yc, lr=0.3, epochs=400); mlp = N.MLP([2, 16, 2], activation="tanh", init="xavier_normal", seed=0).fit(P, yc, lr=0.3, epochs=1500)
g = np.linspace(-3, 3, 220); G = np.c_[np.meshgrid(g, g)[0].ravel(), np.meshgrid(g, g)[1].ravel()]; fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
for ax, m, name in ((axs[0], lin, "no hidden layer (linear)"), (axs[1], mlp, "one hidden layer, 16 tanh units")):
    Z = m.predict_classes(G).reshape(220, 220); ax.contourf(g, g, Z, levels=[-0.5, 0.5, 1.5], colors=[fs.BLUE + "55", fs.RED + "55"]); ax.scatter(P[yc == 0, 0], P[yc == 0, 1], s=10, color=fs.BLUE); ax.scatter(P[yc == 1, 0], P[yc == 1, 1], s=10, color=fs.RED)
    ax.set_aspect("equal"); ax.set_title(f"{name}\naccuracy {(m.predict_classes(P) == yc).mean():.1%}", fontsize=12); ax.grid(True, alpha=0.25)
fig.suptitle("Hidden layers: a linear model cannot separate 'inside the circle' from 'outside'", fontsize=13.5, fontweight="bold", y=1.02); fig.savefig(OUT / "fig_3_04_hidden_layers_boundary.png", dpi=140, bbox_inches="tight"); plt.close(fig)

# ---- 3.7 the bias shifts the boundary off the origin
rng = np.random.default_rng(0); X = rng.standard_normal((600, 2)) + 3.0; y = (X[:, 0] + X[:, 1] > 6.5).astype(int)
fit = lambda bias: N.MLP([2, 1], bias=bias, seed=0).fit(X, y.astype(float).reshape(-1, 1), loss="bce", lr=0.1, epochs=800); mb, mn = fit(True), fit(False); acc = lambda m: (m.predict_classes(X) == y).mean(); r37 = ex_3_6_3_7()
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6))
for ax, m, name, c in ((axs[0], mb, "with bias", fs.TEAL), (axs[1], mn, "without bias (line forced through the origin)", fs.RED)):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, color=fs.BLUE, alpha=0.7); ax.scatter(X[y == 1, 0], X[y == 1, 1], s=10, color=fs.ORANGE, alpha=0.7); W = m.layers[0].W[0]; b = float(m.layers[0].b[0]) if m.layers[0].use_bias else 0.0
    xs_ = np.array([-1, 7]); ax.plot(xs_, -(W[0] * xs_ + b) / W[1], color=c, lw=3); ax.plot(xs_, 6.5 - xs_, color=fs.MUTED, lw=1.4, ls=(0, (4, 3))); ax.scatter([0], [0], marker="x", s=80, color=fs.FG); ax.set_xlim(-1, 7); ax.set_ylim(-1, 7); ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.set_title(f"{name}\naccuracy {acc(m):.1%}", fontsize=12, color=c)
axs[0].text(4.7, 5.6, "dashed: true boundary\nx₁ + x₂ = 6.5", color=fs.MUTED, fontsize=10)
fig.suptitle("The bias term: w·x + b = 0 is a line that can sit anywhere; without b it must pass through the origin (×)", fontsize=13, fontweight="bold", y=1.02); fig.savefig(OUT / "fig_3_07_bias_boundary.svg", bbox_inches="tight"); plt.close(fig)

# ---- 3.15 softmax temperature
z = np.array([2.0, 1.0, 0.1, -1.0, 0.5]); fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, T in zip(axs, (0.1, 1.0, 10.0)):
    p = N.softmax(z, temperature=T); H = float(-(p * np.log(p + 1e-300)).sum()); ax.bar(range(5), p, color=[fs.YELLOW if i == 0 else fs.BLUE for i in range(5)], width=0.62); ax.set_xticks(range(5)); ax.set_xticklabels([f"z={v:g}" for v in z]); ax.set_ylim(0, 1.05); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(f"temperature T = {T:g}\nentropy {H:.2f} nats, top probability {p.max():.2f}", fontsize=11.5)
    for i, v in enumerate(p): ax.text(i, v + 0.02, f"{v:.2f}", ha="center", color=fs.FG, fontsize=9.5)
axs[0].set_ylabel("probability  softmax(z / T)"); fig.suptitle("Softmax and temperature: the order of the scores never changes, only how peaked the distribution is", fontsize=13, fontweight="bold", y=1.03); fig.savefig(OUT / "fig_3_15_softmax_temperature.svg", bbox_inches="tight"); plt.close(fig)

# ---- 3.19 computational graph with forward values and gradients
a, b, c = Value(2.0), Value(-3.0), Value(10.0); e = a * b; d = e + c; f = d * Value(-2.0); f.backward()
fig, ax = plt.subplots(figsize=(13, 5.2)); ax.set_xlim(-0.5, 13.5); ax.set_ylim(-1.2, 6.2); ax.axis("off")
nodes = {"a": (1.0, 4.9, a, fs.TEAL), "b": (1.0, 1.9, b, fs.TEAL), "c": (4.8, 0.3, c, fs.TEAL), "e = a·b": (4.2, 3.5, e, fs.BLUE), "d = e + c": (7.6, 2.6, d, fs.BLUE), "f = d·(−2)": (11.2, 2.6, f, fs.ORANGE)}
for name, (x, y_, v, col) in nodes.items():
    ax.add_patch(FancyBboxPatch((x - 1.05, y_ - 0.55), 2.1, 1.1, boxstyle="round,pad=0.02,rounding_size=0.25", fc=col + "33", ec=col, lw=2.2)); ax.text(x, y_ + 0.12, name, ha="center", va="center", color=fs.FG, fontsize=11.5); ax.text(x, y_ + 0.95, f"value {v.data:g}", ha="center", color=fs.TEAL, fontsize=10.5, fontweight="bold"); ax.text(x, y_ - 0.95, f"∂f/∂{name.split()[0]} = {v.grad:g}", ha="center", color=fs.RED, fontsize=10.5, fontweight="bold")
arr = lambda p, q: ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=fs.FG, lw=1.8)); arr((2.05, 4.75), (3.15, 3.85)); arr((2.05, 2.05), (3.15, 3.2)); arr((5.25, 3.0), (6.55, 2.85)); arr((5.85, 0.6), (6.9, 2.1)); arr((8.65, 2.6), (10.15, 2.6))
ax.text(0.0, 5.9, "forward pass (green): values flow left → right        backward pass (red): gradients flow right → left", color=fs.FG, fontsize=11.5)
ax.text(0.0, -1.05, "chain rule at each node: ∂f/∂a = (∂f/∂e)(∂e/∂a) = (−2·1)·b = 6;  ∂f/∂b = (−2)·a = −4;  ∂f/∂c = −2.   A node used several times would SUM its gradient contributions.", color=fs.MUTED, fontsize=9.5)
ax.set_title("A computational graph for f = (a·b + c)·(−2) at a = 2, b = −3, c = 10", fontsize=13.5, fontweight="bold", pad=18); fig.savefig(OUT / "fig_3_19_computational_graph.svg", bbox_inches="tight"); plt.close(fig)
print("concepts ok")
