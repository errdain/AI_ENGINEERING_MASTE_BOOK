"""Tier 1 probability / information / numerics figures: LLN (1.27), distributions (1.28), empirical rule (1.31), Bayes waffle (1.34), entropy (1.36), KL directions (1.38), float spacing (1.40)."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from _common import OUT, fs
from calculus_prob import normal_pdf, softmax
from info_numerics import entropy, kl_divergence, bayes_posterior

# ---- 1.27 law of large numbers
fig, ax = plt.subplots(figsize=(9.5, 5.2)); n = np.arange(1, 3001)
for k, c in enumerate(fs.PALETTE[:6]):
    r = np.random.default_rng(k).integers(1, 7, 3000); ax.plot(n, np.cumsum(r) / n, color=c, lw=1.6, alpha=0.9)
sd = np.sqrt(35 / 12); ax.fill_between(n, 3.5 - sd / np.sqrt(n), 3.5 + sd / np.sqrt(n), color=fs.YELLOW, alpha=0.14, label="±1 standard error  σ/√n"); ax.axhline(3.5, color=fs.YELLOW, lw=1.8, ls=(0, (5, 3)), label="true mean 3.5")
ax.set_xscale("log"); ax.set_ylim(1.5, 5.5); ax.set_xlabel("number of rolls n"); ax.set_ylabel("running average of a fair die"); ax.grid(True, which="both", alpha=0.3); ax.legend(loc="upper right")
ax.set_title("Law of large numbers: six independent experiments settle on the expectation", fontsize=12.5, fontweight="bold"); fig.savefig(OUT / "fig_1_27_law_of_large_numbers.svg"); plt.close(fig)

# ---- 1.28 distributions
fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
axs[0].bar([0, 1], [0.7, 0.3], color=[fs.MUTED, fs.BLUE], width=0.5); axs[0].set_xticks([0, 1]); axs[0].set_title("Bernoulli(p = 0.3)\nPMF: heights are probabilities", fontsize=10.5)
z = np.array([2.0, 1.0, 0.1, -1.0]); p = softmax(z); axs[1].bar(range(4), p, color=fs.TEAL, width=0.6); axs[1].set_xticks(range(4)); axs[1].set_title(f"Categorical = softmax of logits\nsums to {p.sum():.0f}", fontsize=10.5)
x = np.linspace(-4, 4, 800); axs[2].plot(x, normal_pdf(x), color=fs.BLUE, lw=2.6, label="σ = 1"); axs[2].plot(x, normal_pdf(x, 0, 0.1), color=fs.RED, lw=2.6, label=f"σ = 0.1 (peak {normal_pdf(np.array(0.0), 0, 0.1):.2f} > 1)")
axs[2].set_ylim(0, 4.3); axs[2].legend(fontsize=8.5, loc="upper right"); axs[2].set_title("Normal PDF: a density can exceed 1;\nonly AREAS are probabilities", fontsize=10.5)
from math import erf, sqrt
cdf = 0.5 * (1 + np.array([erf(t / sqrt(2)) for t in x])); axs[3].plot(x, cdf, color=fs.YELLOW, lw=2.6); axs[3].set_ylim(-0.02, 1.05); axs[3].set_title("Normal CDF  F(x) = P(X ≤ x)\nrises from 0 to 1", fontsize=10.5)
for a in axs: a.grid(True, alpha=0.3)
fig.suptitle("Probability distributions: PMF (discrete), PDF (continuous), CDF (cumulative)", fontsize=13, fontweight="bold", y=1.04); fig.savefig(OUT / "fig_1_28_distributions.svg", bbox_inches="tight"); plt.close(fig)

# ---- 1.31 empirical rule
rng = np.random.default_rng(1); sample = rng.normal(10, 3, 200_000); zz = (sample - sample.mean()) / sample.std(); frac = [(np.abs(zz) < k).mean() for k in (1, 2, 3)]
fig, ax = plt.subplots(figsize=(10, 5)); x = np.linspace(-4, 4, 800); y = normal_pdf(x); ax.plot(x, y, color=fs.FG, lw=2.2)
for k, c, f in zip((3, 2, 1), (fs.PURPLE, fs.TEAL, fs.BLUE), frac[::-1]): m = np.abs(x) <= k; ax.fill_between(x[m], y[m], color=c, alpha=0.55, label=f"within ±{k}σ: {f:.1%} of sampled points")
ax.set_xticks(range(-4, 5)); ax.set_xticklabels([f"{t}σ" if t else "mean" for t in range(-4, 5)]); ax.set_ylabel("density"); ax.legend(loc="upper right", fontsize=10); ax.grid(True, alpha=0.3)
ax.set_title("Standard deviation as a ruler: the 68–95–99.7 rule (percentages measured on 200,000 simulated values)", fontsize=12, fontweight="bold"); fig.savefig(OUT / "fig_1_31_empirical_rule.svg"); plt.close(fig)

# ---- 1.34 Bayes: 1,000 people (the disease-test example)
post = bayes_posterior(0.01, 0.99, 0.05); sick, tp, fp = 10, 10, 50                       # per 1,000 people (counts rounded): 10 sick, ~10 true positives, 50 false positives
cells = [("sick, test positive", tp, fs.RED), ("healthy, test positive", fp, fs.ORANGE), ("everyone else (test negative)", 1000 - tp - fp, "#2b3050")]
fig, ax = plt.subplots(figsize=(11, 6.6)); cols, rows = 50, 20; idx = 0; order = []
for name, k, c in cells: order += [c] * k
for i, c in enumerate(order):
    r, cc = divmod(i, cols); ax.add_patch(Rectangle((cc, rows - 1 - r), 0.86, 0.86, fc=c, ec="none"))
ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(f"Bayes in pictures: of 1,000 people, 60 test positive — only 10 are actually sick (posterior = {post:.3f} ≈ 1/6)", fontsize=12.5, fontweight="bold", pad=10)
for j, (name, k, c) in enumerate(cells): ax.add_patch(Rectangle((1 + j * 15.5, -2.6), 1, 1, fc=c, clip_on=False)); ax.text(2.4 + j * 15.5, -2.1, f"{name}: {k}", color=fs.FG, fontsize=10.5, va="center")
ax.text(0, -4.2, "prevalence 1%, test sensitivity 99%, false-positive rate 5%.  P(sick | positive) = 10 / (10 + 50): a rare condition makes most positives false.", color=fs.MUTED, fontsize=9.5)
fig.savefig(OUT / "fig_1_34_bayes_waffle.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 1.36 entropy
p = np.linspace(0.001, 0.999, 500); H = np.array([entropy([q, 1 - q]) for q in p]); fig, ax = plt.subplots(figsize=(9.5, 5)); ax.plot(p, H, color=fs.BLUE, lw=3)
for q, lab, dy in ((0.5, "fair coin: 1 bit", 0.08), (0.99, f"p = 0.99: {entropy([0.99, 0.01]):.3f} bits", 0.1), (0.9, f"p = 0.9: {entropy([0.9, 0.1]):.2f} bits", 0.08)): ax.scatter([q], [entropy([q, 1 - q])], color=fs.YELLOW, s=70, zorder=5); ax.text(q - (0.02 if q > 0.6 else 0), entropy([q, 1 - q]) + dy, lab, color=fs.YELLOW, fontsize=10, ha="right" if q > 0.6 else "center")
ax.set_xlabel("probability of heads p"); ax.set_ylabel("entropy H(p) in bits"); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.2)
ax.set_title("Entropy of a coin: most uncertain when fair, nearly zero when predictable", fontsize=12.5, fontweight="bold"); ax.text(0.02, 0.07, "for comparison: a uniform choice among 8 outcomes = 3 bits", color=fs.MUTED, fontsize=9.5); fig.savefig(OUT / "fig_1_36_entropy.svg"); plt.close(fig)

# ---- 1.38 forward vs reverse KL
x = np.linspace(-10, 10, 2001); pdf = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)); P = 0.5 * pdf(-3, 1) + 0.5 * pdf(3, 1); P /= P.sum(); best = {"f": (np.inf, None), "r": (np.inf, None)}
for m in np.linspace(-5, 5, 41):
    for s in np.linspace(0.5, 5, 19):
        Q = np.maximum(pdf(m, s) / pdf(m, s).sum(), 1e-300); fk = float(np.sum(P[P > 0] * np.log(P[P > 0] / Q[P > 0]))); rk = float(np.sum(Q * np.log(Q / np.maximum(P, 1e-300))))
        if fk < best["f"][0]: best["f"] = (fk, (m, s))
        if rk < best["r"][0]: best["r"] = (rk, (m, s))
fig, ax = plt.subplots(figsize=(10, 5.2)); ax.fill_between(x, P / (x[1] - x[0]), color=fs.MUTED, alpha=0.5, label="target p: two modes")
(mf, sf), (mr, sr) = best["f"][1], best["r"][1]; ax.plot(x, pdf(mf, sf), color=fs.TEAL, lw=3, label=f"forward KL(p‖q): q covers both modes (μ={mf:.1f}, σ={sf:.2f})"); ax.plot(x, pdf(mr, sr), color=fs.PINK, lw=3, label=f"reverse KL(q‖p): q locks onto one mode (μ={mr:.1f}, σ={sr:.2f})")
ax.legend(loc="upper center", fontsize=9.5); ax.set_ylim(0, 0.62); ax.grid(True, alpha=0.3); ax.set_title("KL divergence is asymmetric: which direction you minimize decides the fit", fontsize=12.5, fontweight="bold"); fig.savefig(OUT / "fig_1_38_kl_directions.svg"); plt.close(fig)

# ---- 1.40 floating point spacing (plotted on log10 axes computed by hand: matplotlib's log scale overflows for 1e-300..1e300)
fig, ax = plt.subplots(figsize=(10, 5.4)); mags = {"float16": (np.float16, np.logspace(-4, np.log10(60000), 400)), "float32": (np.float32, np.logspace(-10, 38, 400)), "float64": (np.float64, np.logspace(-10, 40, 400))}
for (name, (dt, xs)), c in zip(mags.items(), (fs.RED, fs.YELLOW, fs.BLUE)): ax.plot(np.log10(xs), np.log10(np.spacing(xs.astype(dt)).astype(float)), color=c, lw=2.6, label=f"{name}  (ε = {np.finfo(dt).eps:.2g})")
sp = float(np.spacing(np.float32(1e8))); ax.scatter([8], [np.log10(sp)], color=fs.YELLOW, s=90, zorder=5, edgecolor="white"); ax.annotate(f"float32 near 10⁸: neighbours are {sp:.0f} apart,\nso 1e8 + 1 == 1e8", xy=(8, np.log10(sp)), xytext=(17, -17), color=fs.YELLOW, fontsize=10, arrowprops=dict(arrowstyle="->", color=fs.YELLOW))
ax.axvline(np.log10(65504), color=fs.RED, lw=1.3, ls=(0, (4, 3))); ax.text(np.log10(65504) + 0.6, -9.5, "← float16 overflows above 65504", color=fs.RED, fontsize=9.5)
ax.set_xlabel("log₁₀ of the number"); ax.set_ylabel("log₁₀ of the gap to the next representable number"); ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=10)
ax.set_title("Floating point: the gap between neighbours grows in proportion to the number", fontsize=12.5, fontweight="bold"); fig.savefig(OUT / "fig_1_40_float_spacing.svg"); plt.close(fig)
print("prob ok", best["f"][1], best["r"][1])
