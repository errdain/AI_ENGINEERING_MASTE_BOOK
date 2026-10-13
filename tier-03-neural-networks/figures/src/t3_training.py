"""Tier 3 figures: initialization (3.22/3.24), exploding gradients and clipping (3.26), residual connections (3.27), weight decay (3.29)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from _common import OUT, fs
import nn as N
from lab_02_training_internals import activation_std_by_layer, train_deep, grad_norms_by_layer, ex_3_29

# ---- 3.22 / 3.24 activation scale through 30 layers for different initializations
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2)); L = np.arange(1, 31)
for ax, act, schemes in ((axs[0], "tanh", ("zeros", "normal_0.01", "normal_1", "xavier_normal", "he_normal")), (axs[1], "relu", ("normal_0.01", "xavier_normal", "he_normal"))):
    for sch, c in zip(schemes, (fs.MUTED, fs.RED, fs.ORANGE, fs.BLUE, fs.TEAL)):
        s = np.array(activation_std_by_layer(256, 30, act, sch)); ax.semilogy(L, np.maximum(s, 1e-30), "-o", color=c, lw=2.4, ms=3.5, label={"normal_0.01": "N(0, 0.01²)", "normal_1": "N(0, 1)", "xavier_normal": "Xavier", "he_normal": "He", "zeros": "all zeros"}[sch] + f"   (layer 30: {s[-1]:.1e})")
    ax.axhline(1, color=fs.GRID, lw=1); ax.set_xlabel("layer"); ax.set_ylabel("standard deviation of the activations (log scale)"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9, loc="lower left"); ax.set_ylim(1e-30 if act == "relu" else 1e-6, 3)
axs[0].set_title("tanh network, width 256: only a well-scaled init keeps the signal alive", fontsize=11); axs[1].set_title("ReLU network: Xavier shrinks ≈ √2 per layer, He keeps the scale", fontsize=11)
fig.suptitle("Initialization decides whether the signal survives 30 layers", fontsize=13.5, fontweight="bold", y=1.02); fig.savefig(OUT / "fig_3_22_initialization.svg", bbox_inches="tight"); plt.close(fig)

# ---- 3.26 gradient norm during training with and without clipping
stable, n0 = train_deep(1.0, None, steps=40); big, n1 = train_deep(1.3, None, steps=40); clip, n2 = train_deep(1.3, 1.0, steps=40)
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2))
axs[0].semilogy(np.arange(1, len(n0) + 1), n0, "-o", color=fs.TEAL, lw=2.4, ms=4, label="gain 1.0 (He scale)"); axs[0].semilogy(np.arange(1, len(n1) + 1), np.where(np.isfinite(n1), n1, np.nan), "-o", color=fs.RED, lw=2.4, ms=5, label="gain 1.3, no clipping"); axs[0].semilogy(np.arange(1, len(n2) + 1), n2, "-o", color=fs.ORANGE, lw=2.4, ms=4, label="gain 1.3, norm measured BEFORE clipping")
axs[0].axhline(1.0, color=fs.YELLOW, ls=(0, (4, 3))); axs[0].text(25, 1.4, "clip threshold = 1", color=fs.YELLOW, fontsize=10); axs[0].set_xlabel("training step"); axs[0].set_ylabel("global gradient norm"); axs[0].legend(fontsize=9.5, loc="upper right"); axs[0].grid(True, which="both", alpha=0.3); axs[0].set_title("without clipping the run blows up after 5 steps", fontsize=11)
axs[1].plot(np.arange(len(stable)), stable, color=fs.TEAL, lw=2.6, label="gain 1.0"); axs[1].plot(np.arange(len(clip)), clip, color=fs.ORANGE, lw=2.6, label="gain 1.3 with clipping at 1.0"); axs[1].plot(np.arange(len(big)), np.where(np.isfinite(big), big, np.nan), "-o", color=fs.RED, lw=2.6, label="gain 1.3, no clipping (loss → ∞)")
axs[1].set_xlabel("training step"); axs[1].set_ylabel("training loss"); axs[1].set_ylim(0, 6); axs[1].legend(fontsize=9.5); axs[1].grid(True, alpha=0.3); axs[1].set_title("clipping keeps training finite (it treats the symptom; fix the scale too)", fontsize=11)
fig.suptitle("Exploding gradients and global-norm clipping (12-layer ReLU net)", fontsize=13.5, fontweight="bold", y=1.02); fig.savefig(OUT / "fig_3_26_exploding_clipping.svg", bbox_inches="tight"); plt.close(fig)

# ---- 3.27 residual block diagram + plain vs residual 30-layer training
def train30(residual, steps=150, seed=0):
    net = N.build_deep_net(32, 30, "tanh", "xavier_normal", residual=residual, n_out=3, seed=seed)
    for blk in net.layers[:-1]: (blk.block if residual else blk).layers[0].W *= (0.5 if residual else 1.0)
    rng = np.random.default_rng(seed); X = rng.standard_normal((200, 32)); y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int); losses = []
    for _ in range(steps): out = net.forward(X); l, d = N.softmax_cross_entropy(out, y); net.backward(d); net.step(0.1); losses.append(l)
    return losses
lp, lr_ = train30(False), train30(True)
fig = plt.figure(figsize=(14, 5.6)); a0 = fig.add_axes([0.02, 0.08, 0.42, 0.78]); a0.set_xlim(0, 10); a0.set_ylim(0, 8); a0.axis("off")
def box(ax, x, y, w, h, text, c): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", fc=c + "33", ec=c, lw=2)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=fs.FG, fontsize=11)
box(a0, 0.4, 3.4, 2.6, 1.2, "plain block\nF(x)", fs.BLUE); a0.annotate("", xy=(0.4, 4.0), xytext=(-0.1, 4.0), arrowprops=dict(arrowstyle="-|>", color=fs.FG, lw=2)); a0.annotate("", xy=(3.9, 4.0), xytext=(3.0, 4.0), arrowprops=dict(arrowstyle="-|>", color=fs.FG, lw=2)); a0.text(2.0, 5.0, "y = F(x)\ngradient ×∂F/∂x", ha="center", color=fs.BLUE, fontsize=10)
box(a0, 5.9, 3.4, 2.2, 1.2, "F(x)", fs.TEAL); a0.add_patch(Circle((9.0, 4.0), 0.4, fc=fs.YELLOW + "33", ec=fs.YELLOW, lw=2)); a0.text(9.0, 4.0, "+", ha="center", va="center", color=fs.YELLOW, fontsize=16, fontweight="bold")
a0.annotate("", xy=(5.9, 4.0), xytext=(5.2, 4.0), arrowprops=dict(arrowstyle="-|>", color=fs.FG, lw=2)); a0.annotate("", xy=(8.6, 4.0), xytext=(8.1, 4.0), arrowprops=dict(arrowstyle="-|>", color=fs.FG, lw=2)); a0.annotate("", xy=(9.0, 3.6), xytext=(9.0, 2.2), arrowprops=dict(arrowstyle="-", color=fs.YELLOW, lw=2.6)); a0.plot([5.2, 5.2, 9.0], [4.0, 2.2, 2.2], color=fs.YELLOW, lw=2.6); a0.text(7.1, 1.6, "skip connection (identity)", ha="center", color=fs.YELLOW, fontsize=10.5)
a0.annotate("", xy=(9.9, 4.0), xytext=(9.4, 4.0), arrowprops=dict(arrowstyle="-|>", color=fs.FG, lw=2)); a0.text(7.5, 5.0, "y = x + F(x)\ngradient = dY + dF", ha="center", color=fs.TEAL, fontsize=10); a0.text(5.0, 7.4, "a residual block keeps a direct path for the signal and the gradient", ha="center", color=fs.FG, fontsize=11.5)
a1 = fig.add_axes([0.52, 0.12, 0.46, 0.74]); a1.plot(lp, color=fs.RED, lw=2.8, label="plain 30-layer tanh network"); a1.plot(lr_, color=fs.TEAL, lw=2.8, label="same network with residual connections"); a1.axhline(np.log(3), color=fs.MUTED, ls=(0, (4, 3))); a1.text(80, np.log(3) + 0.03, "chance level  ln 3 = 1.10", color=fs.MUTED, fontsize=10)
a1.set_ylim(-0.1, 3.5); a1.text(30, 3.1, f"early loss spike up to {max(lr_):.0f} (off scale), then it settles", color=fs.TEAL, fontsize=9.5); a1.set_xlabel("training step"); a1.set_ylabel("training loss"); a1.legend(fontsize=10); a1.grid(True, alpha=0.3); a1.set_title(f"after 150 steps: plain {lp[-1]:.2f} (not learning) vs residual {lr_[-1]:.4f}", fontsize=11)
fig.suptitle("Residual connections: y = x + F(x)", fontsize=14, fontweight="bold", y=0.98); fig.savefig(OUT / "fig_3_27_residual_connections.svg"); plt.close(fig)

# ---- 3.29 weight decay: shrinkage curve and effect on a small-data task
r29 = ex_3_29(); wds = list(r29["by_weight_decay"]); fig, axs = plt.subplots(1, 3, figsize=(15, 4.8)); t = np.arange(0, 301)
for lam, c in ((0.0, fs.MUTED), (0.01, fs.BLUE), (0.1, fs.ORANGE), (0.5, fs.RED)): axs[0].plot(t, (1 - 0.1 * lam) ** t, color=c, lw=2.6, label=f"λ = {lam:g}")
axs[0].set_xlabel("steps (gradient = 0)"); axs[0].set_ylabel("remaining weight  (1 − ηλ)ᵗ  (η = 0.1)"); axs[0].legend(); axs[0].grid(True, alpha=0.3); axs[0].set_title("pure decay: every step shrinks the weights", fontsize=11)
x_ = np.arange(len(wds)); axs[1].bar(x_, [r29["by_weight_decay"][w]["weight_norm"] for w in wds], color=fs.TEAL, width=0.6); axs[1].set_xticks(x_); axs[1].set_xticklabels([f"{w:g}" for w in wds]); axs[1].set_xlabel("weight decay λ"); axs[1].set_ylabel("weight norm after training"); axs[1].grid(True, axis="y", alpha=0.3); axs[1].set_title("trained weights get smaller", fontsize=11)
for i, w in enumerate(wds): axs[1].text(i, r29["by_weight_decay"][w]["weight_norm"] + 0.3, f"{r29['by_weight_decay'][w]['weight_norm']:.1f}", ha="center", color=fs.FG, fontsize=10)
axs[2].plot(x_, [r29["by_weight_decay"][w]["test_acc"] for w in wds], "-o", color=fs.YELLOW, lw=2.8, ms=8); axs[2].set_xticks(x_); axs[2].set_xticklabels([f"{w:g}" for w in wds]); axs[2].set_xlabel("weight decay λ"); axs[2].set_ylabel("test accuracy (mean of 6 seeds)"); axs[2].grid(True, alpha=0.3); axs[2].set_title("…and helps on this small-data task", fontsize=11)
fig.suptitle("Weight decay: w ← (1 − ηλ)w − η∇L", fontsize=13.5, fontweight="bold", y=1.03); fig.savefig(OUT / "fig_3_29_weight_decay.svg", bbox_inches="tight"); plt.close(fig)
print("training figs ok", lp[-1], lr_[-1])
