"""Tier 3 figures: dropout (3.28, animated), batch normalization (3.30), which axis each normalizer uses (3.31), LayerNorm vs RMSNorm (3.32)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from _common import OUT, fs
import nn as N
from lab_02_training_internals import ex_3_30, ex_3_32

# ---- 3.28 dropout masks over successive training steps (animation)
d = N.Dropout(0.5, seed=3); x = np.ones((1, 12)); frames = []; means = []
for step in range(16):
    y = d.forward(x)[0]; means.append(y.mean())
    for rep in range(3):
        fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(-0.5, 12.5); ax.set_ylim(-1.6, 3.2); ax.set_aspect("equal"); ax.axis("off")
        for j in range(12):
            kept = y[j] > 0; ax.add_patch(plt.Circle((j + 0.5, 1.5), 0.42, fc=(fs.TEAL if kept else "#2a2f4e"), ec=(fs.FG if kept else fs.GRID), lw=1.8, alpha=1.0 if kept else 0.6)); ax.text(j + 0.5, 1.5, f"{y[j]:.0f}" if kept else "✕", ha="center", va="center", color=fs.BG if kept else fs.MUTED, fontsize=12, fontweight="bold")
            ax.text(j + 0.5, 0.55, "1" , ha="center", color=fs.MUTED, fontsize=9)
        ax.text(-0.45, 2.45, "dropout p = 0.5 — a fresh random mask at every training step", color=fs.FG, fontsize=13, fontweight="bold"); ax.text(-0.45, 0.0, "activation before dropout (all 1)", color=fs.MUTED, fontsize=9.5)
        ax.text(-0.45, -0.7, f"step {step + 1}: {int((y > 0).sum())} of 12 units kept, survivors scaled by 1/(1−p) = 2  →  mean activation this step {y.mean():.2f}", color=fs.YELLOW, fontsize=10.5)
        ax.text(-0.45, -1.3, f"average over the steps so far: {np.mean(means):.2f}  (→ 1, the expectation is preserved)   |   at inference dropout is off", color=fs.MUTED, fontsize=9.5)
        frames.append(fs.fig_to_image(fig)); plt.close(fig)
print("dropout bytes", fs.save_gif(frames, OUT / "fig_3_28_dropout.gif", fps=6, colors=32, hold_last=6))

# ---- 3.30 batch norm: re-centering a shifted feature, and dependence on the other samples in the batch
r30 = ex_3_30(); rng = np.random.default_rng(0); X = rng.standard_normal((256, 8)) * 5 + 3; bn = N.BatchNorm1d(8); Y = bn.forward(X)
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.9)); bins = np.linspace(-12, 18, 70)
axs[0].hist(X[:, 0], bins, color=fs.ORANGE, alpha=0.8, label=f"before: mean {X[:, 0].mean():.1f}, std {X[:, 0].std():.1f}"); axs[0].hist(Y[:, 0], bins, color=fs.TEAL, alpha=0.8, label=f"after BatchNorm: mean {Y[:, 0].mean():.1f}, std {Y[:, 0].std():.1f}"); axs[0].legend(fontsize=9); axs[0].set_xlabel("value of one feature over the batch"); axs[0].set_ylabel("count"); axs[0].grid(True, alpha=0.3); axs[0].set_title("normalizes each feature over the batch", fontsize=11)
sp = r30["output_of_one_sample_std_across_random_batches_by_batch_size"]; bs = list(sp); axs[1].loglog(bs, [sp[b] for b in bs], "-o", color=fs.RED, lw=2.8, ms=8, label="measured"); axs[1].loglog(bs, [sp[2] * np.sqrt(2 / b) for b in bs], color=fs.MUTED, ls=(0, (4, 3)), label="∝ 1/√B"); axs[1].set_xlabel("batch size B"); axs[1].set_ylabel("spread of ONE sample's output across random batches"); axs[1].legend(); axs[1].grid(True, which="both", alpha=0.3); axs[1].set_title("…so a sample's output depends on its batch-mates", fontsize=11)
lrs = r30["final_loss_lr0.5"]; axs[2].bar(["no normalization", "with BatchNorm"], [1.0, lrs["batchnorm"]], color=[fs.RED, fs.TEAL], width=0.5); axs[2].set_yscale("log"); axs[2].text(0, 1.3, "diverges (loss = ∞)", ha="center", color=fs.RED, fontsize=11); axs[2].text(1, lrs["batchnorm"] * 1.5, f"final loss {lrs['batchnorm']:.4f}", ha="center", color=fs.TEAL, fontsize=11); axs[2].set_ylim(1e-3, 10); axs[2].set_ylabel("training loss after 60 steps (log)"); axs[2].grid(True, axis="y", alpha=0.3); axs[2].set_title("deep ReLU net at learning rate 0.5", fontsize=11)
fig.suptitle("Batch normalization: stable and fast to train, but each sample depends on its batch", fontsize=13.5, fontweight="bold", y=1.03); fig.savefig(OUT / "fig_3_30_batch_normalization.svg", bbox_inches="tight"); plt.close(fig)

# ---- 3.31 which numbers are normalized together (the same (B, D) array)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6)); B, D = 6, 8; rng = np.random.default_rng(1); M = rng.standard_normal((B, D))
for ax, kind in zip(axs, ("batch", "layer", "rms")):
    ax.set_xlim(-0.6, D + 0.2); ax.set_ylim(-1.6, B + 1.0); ax.axis("off")
    for i in range(B):
        for j in range(D):
            hl = (kind == "batch" and j == 2) or (kind in ("layer", "rms") and i == 2); ax.add_patch(Rectangle((j, B - 1 - i), 0.9, 0.9, fc=(fs.YELLOW if hl else "#2b3050"), ec="none", alpha=1.0 if hl else 0.9))
    ax.annotate("features (D) →", xy=(D / 2, B + 0.25), ha="center", color=fs.MUTED, fontsize=9.5); ax.text(-0.5, B / 2, "samples (B) ↓", rotation=90, va="center", color=fs.MUTED, fontsize=9.5)
    if kind == "batch": ax.set_title("BatchNorm: one feature, all samples\nmean/var over the BATCH axis", fontsize=11, color=fs.YELLOW); ax.text(D / 2, -0.9, "statistics depend on the other samples;\nrunning averages are used at inference", ha="center", color=fs.FG, fontsize=9.5)
    if kind == "layer": ax.set_title("LayerNorm: one sample, all features\nmean and var over the FEATURE axis", fontsize=11, color=fs.YELLOW); ax.text(D / 2, -0.9, "each sample is independent of the batch;\nsame behaviour in training and inference", ha="center", color=fs.FG, fontsize=9.5)
    if kind == "rms": ax.set_title("RMSNorm: one sample, all features\nroot-mean-square only (no mean subtraction)", fontsize=11, color=fs.YELLOW); ax.text(D / 2, -0.9, "cheaper than LayerNorm; scale-invariant\nbut not shift-invariant", ha="center", color=fs.FG, fontsize=9.5)
fig.suptitle("Which numbers are normalized together? (the same (B, D) activation array, yellow = one normalization group)", fontsize=13, fontweight="bold", y=1.04); fig.savefig(OUT / "fig_3_31_normalization_axes.svg", bbox_inches="tight"); plt.close(fig)

# ---- 3.32 LayerNorm vs RMSNorm: identical on centered inputs, different when the inputs have an offset
rng = np.random.default_rng(0); D = 1024; Xc = rng.standard_normal((200, D)) * 3; Xs = Xc + 5.0; ln, rn = N.LayerNorm(D), N.RMSNorm(D); r32 = ex_3_32()
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, Xin, name in ((axs[0], Xc, "inputs with mean ≈ 0"), (axs[1], Xs, "inputs shifted by +5")):
    a, b = ln.forward(Xin)[:5].ravel(), rn.forward(Xin)[:5].ravel(); ax.scatter(a, b, s=6, color=fs.BLUE, alpha=0.55); lim = [-4.5, 4.5]; ax.plot(lim, lim, color=fs.MUTED, ls=(0, (4, 3))); ax.set_xlim(lim); ax.set_ylim(-1.5 if Xin is Xs else -4.5, 4.5 if Xin is Xc else 3); ax.set_xlabel("LayerNorm output"); ax.set_ylabel("RMSNorm output"); ax.grid(True, alpha=0.3); ax.set_title(f"{name}\nmax |difference| = {np.abs(ln.forward(Xin) - rn.forward(Xin)).max():.2f}", fontsize=11)
fig.suptitle(f"LayerNorm vs RMSNorm: same when the mean is ≈ 0; RMSNorm keeps a common offset (shift error {r32['shift_invariance_rmsnorm(not invariant)']:.1f} vs {r32['shift_invariance_layernorm']:.0e} for LayerNorm)", fontsize=12, fontweight="bold", y=1.03)
fig.savefig(OUT / "fig_3_32_layernorm_vs_rmsnorm.svg", bbox_inches="tight"); plt.close(fig)
print("norm ok")
