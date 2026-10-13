"""Tier 4 figures (numpy only, so the numbers are computed here, not typed in):
4.8-1 optimizer paths on a ravine, 4.12-1 learning-rate schedules, 4.18-1 number-format ranges, 4.35-1 training/validation curves."""
import math
from _common import OUT, fs, plt, np

# ---------- 4.8-1: SGD vs momentum vs Adam on a narrow ravine f(x, y) = 0.5 * (x^2 + 25 y^2)
def grad(p):
    return np.array([p[0], 25.0 * p[1]])

def run(step, n=60):
    p, state, path = np.array([-4.0, 1.5]), {}, []
    for t in range(1, n + 1):
        path.append(p.copy()); p = p - step(grad(p), state, t)
    path.append(p.copy()); return np.array(path)

def sgd(g, s, t):
    return 0.03 * g

def momentum(g, s, t):
    s["v"] = 0.9 * s.get("v", 0) + g; return 0.02 * s["v"]

def adam(g, s, t, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g; s["v"] = b2 * s.get("v", 0) + (1 - b2) * g * g
    return 0.1 * (s["m"] / (1 - b1 ** t)) / (np.sqrt(s["v"] / (1 - b2 ** t)) + eps)

paths = {"SGD": (run(sgd), fs.RED), "SGD + momentum": (run(momentum), fs.YELLOW), "Adam": (run(adam), fs.TEAL)}
fig, ax = plt.subplots(figsize=(10.5, 5.6))
X, Y = np.meshgrid(np.linspace(-5, 1.5, 300), np.linspace(-2, 2, 300))
ax.contour(X, Y, 0.5 * (X ** 2 + 25 * Y ** 2), levels=[0.2, 0.8, 2, 4, 8, 14], colors=fs.GRID, linewidths=1)
for name, (pth, c) in paths.items():
    ax.plot(pth[:, 0], pth[:, 1], "-o", ms=3, lw=1.6, color=c, label=f"{name} (loss after 60 steps: {0.5 * (pth[-1, 0] ** 2 + 25 * pth[-1, 1] ** 2):.1e})")
ax.plot(0, 0, "*", color="white", ms=14, zorder=5); ax.text(0.1, 0.12, "minimum", color="white")
ax.set_xlim(-5, 1.5); ax.set_ylim(-2, 2); ax.set_xlabel("weight 1 (gentle direction)"); ax.set_ylabel("weight 2 (steep direction)")
ax.set_title("Three optimizers on a narrow valley: the steep walls make plain SGD zig-zag", fontsize=13, fontweight="bold"); ax.legend(loc="lower right", fontsize=9.5)
fig.savefig(OUT / "fig_4_08_optimizer_paths.svg"); plt.close(fig)
print({k: float(0.5 * (v[0][-1, 0] ** 2 + 25 * v[0][-1, 1] ** 2)) for k, v in paths.items()})

# ---------- 4.12-1: learning-rate schedules
T, warm, base = 1000, 100, 1.0
steps = np.arange(T + 1)
const = np.full(T + 1, base)
step_decay = base * 0.5 ** (steps // 250)
cos = 0.5 * base * (1 + np.cos(math.pi * steps / T))
wc = np.where(steps < warm, base * steps / warm, 0.5 * base * (1 + np.cos(math.pi * (steps - warm) / (T - warm))))
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for y, name, c in [(const, "constant", fs.MUTED), (step_decay, "step decay (halve every 250)", fs.ORANGE), (cos, "cosine decay", fs.BLUE), (wc, "warm-up (100 steps) + cosine", fs.TEAL)]:
    ax.plot(steps, y, lw=2.4, color=c, label=name)
ax.axvspan(0, warm, color=fs.YELLOW, alpha=0.1); ax.text(8, 0.05, "warm-up", color=fs.YELLOW)
ax.set_xlabel("training step"); ax.set_ylabel("learning rate (relative to the peak)"); ax.set_title("Learning-rate schedules", fontsize=13, fontweight="bold"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
fig.savefig(OUT / "fig_4_12_lr_schedules.svg"); plt.close(fig)

# ---------- 4.18-1: how far each number format reaches (log10 of the largest value and the smallest normal value)
fmt = [("FP32", 8, 23), ("FP16", 5, 10), ("BF16", 8, 7)]
def info(e, m):
    bias = 2 ** (e - 1) - 1
    return math.log10((2 - 2 ** -m) * 2.0 ** bias), math.log10(2.0 ** (1 - bias)), math.log10(2.0 ** -m)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
for i, (n, e, m) in enumerate(fmt):
    hi, lo, _ = info(e, m); c = [fs.BLUE, fs.ORANGE, fs.TEAL][i]
    ax.barh(i, hi - lo, left=lo, color=c, alpha=0.85, height=0.5)
    ax.text(hi + 0.8, i, f"{e} exponent bits, {m} fraction bits\nlargest ≈ 10^{hi:.1f}", va="center", color=fs.FG, fontsize=9.5)
ax.set_yticks(range(3)); ax.set_yticklabels([f[0] for f in fmt], fontsize=12); ax.invert_yaxis(); ax.set_xlim(-50, 75)
ax.set_xlabel("log10 of the value: the bar spans from the smallest normal number to the largest"); ax.set_title("Range of the number formats: BF16 keeps FP32's range, FP16 does not", fontsize=13, fontweight="bold"); ax.grid(axis="x", alpha=0.3)
fig.savefig(OUT / "fig_4_18_float_formats.svg"); plt.close(fig)

# ---------- 4.35-1: reading training and validation curves (underfit / good / overfit), synthetic but illustrative
rng = np.random.default_rng(0); e = np.arange(1, 61)
good_tr = 0.12 + 1.0 * np.exp(-e / 12) + 0.01 * rng.normal(size=60); good_va = 0.2 + 1.0 * np.exp(-e / 11) + 0.015 * rng.normal(size=60)
over_tr = 0.02 + 1.0 * np.exp(-e / 8) + 0.01 * rng.normal(size=60); over_va = 0.35 + 0.9 * np.exp(-e / 7) + 0.006 * np.maximum(e - 18, 0) + 0.015 * rng.normal(size=60)
under_tr = 0.9 + 0.3 * np.exp(-e / 15) + 0.01 * rng.normal(size=60); under_va = under_tr + 0.05 + 0.01 * rng.normal(size=60)
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
for ax, (tr, va), t in zip(axs, [(under_tr, under_va), (good_tr, good_va), (over_tr, over_va)], ["underfitting: both high", "healthy: both fall together", "overfitting: validation turns up"]):
    ax.plot(e, tr, color=fs.BLUE, lw=2.2, label="training loss"); ax.plot(e, va, color=fs.ORANGE, lw=2.2, label="validation loss"); ax.set_title(t, fontsize=11.5); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
axs[0].set_ylabel("loss"); axs[0].legend(fontsize=9)
fig.suptitle("Reading the curves (illustrative shapes)", fontsize=13, fontweight="bold"); fig.tight_layout()
fig.savefig(OUT / "fig_4_35_training_curves.svg"); plt.close(fig)
