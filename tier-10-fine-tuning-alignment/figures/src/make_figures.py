"""Generate Tier 10 figures + GIFs.  Run:  python figures/src/make_figures.py   (~2-3 min CPU; trains the tiny lab models)
Figures 05-08 and GIF 10 use REAL numbers from lab/ (seeded). Others are analytic/illustrative as captioned."""
import copy
import pathlib
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lab"))
import torch
import run_lab as RL
from data import split_combos, sft_batch, base_batch
from tiny_lm import lm_loss, n_params
from lora import finetune_memory_gb, lora_param_count
from merge import lerp_state

OUT = HERE.parents[1] / "figures"
OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def save(fig, name, dpi=100):
    fig.savefig(OUT / name, dpi=dpi); plt.close(fig)


def box(ax, x, y, w, h, text, color, fs=10, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=color, ec="white", alpha=.95))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, weight="bold")


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color="#667", lw=1.6))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + .18, text, ha="center", fontsize=9, color="#445")


# ------------------------------------------------------------------ 01 pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    box(ax, 0.1, 2.6, 2.1, 1.3, "Base model\n(next-token\npredictor)", C[0])
    box(ax, 3.1, 2.6, 2.1, 1.3, "SFT model\n(follows\ninstructions)", C[1])
    box(ax, 6.2, 3.3, 2.4, 1.1, "Reward model\n(Bradley-Terry)", C[5])
    box(ax, 9.3, 3.3, 2.5, 1.1, "RLHF / PPO\n(policy vs reward\n+ KL to ref)", C[3])
    box(ax, 6.2, 1.1, 5.6, 1.1, "DPO  (no reward model, no sampling loop:\nclassification loss on chosen vs rejected)", C[2])
    box(ax, 3.1, 0.1, 2.1, .7, "Adapter (LoRA)", C[4], fs=9)
    arrow(ax, 2.2, 3.25, 3.1, 3.25, "SFT on\n(prompt, answer)")
    arrow(ax, 5.2, 3.5, 6.2, 3.85, "preference\npairs")
    arrow(ax, 8.6, 3.85, 9.3, 3.85)
    arrow(ax, 5.2, 2.9, 6.2, 1.8, "same pairs")
    arrow(ax, 4.15, .8, 4.15, 2.6, "can train any stage\ncheaply")
    ax.text(0.1, 1.4, "Data needed:\n 1. raw text (Tier 9)\n 2. 10^3-10^5 demonstrations\n 3. 10^4-10^6 preference pairs",
            fontsize=10, va="center", color="#445")
    ax.text(.1, 4.75, "Post-training pipeline: what changes at each stage", fontsize=13, weight="bold", va="center")
    ax.text(.1, 4.45, "arrows = which model initialises / feeds the next stage", fontsize=9, color="#667", va="center")
    save(fig, "01_post_training_pipeline.png")


# ------------------------------------------------------------------ 02 lora decomposition
def fig_lora():
    rng = np.random.default_rng(3)
    d, r, alpha = 16, 2, 4
    W = rng.normal(size=(d, d)) * .5
    B = rng.normal(size=(d, r)); A = rng.normal(size=(r, d))
    dW = alpha / r * B @ A
    fig, ax = plt.subplots(1, 6, figsize=(13, 3.1), gridspec_kw={"width_ratios": [4, .4, 1.3, .4, 4, 4]})
    ax[0].imshow(W, cmap="RdBu_r", vmin=-3, vmax=3); ax[0].set_title("W  (frozen)\nd x d = 256 params")
    ax[1].axis("off"); ax[1].text(.5, .5, "+", fontsize=26, ha="center", va="center")
    ax[2].imshow(B, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto"); ax[2].set_title("B  (d x r)\ntrained, init 0")
    ax[3].axis("off"); ax[3].text(.5, .5, "x", fontsize=22, ha="center", va="center")
    ax[4].imshow(A, cmap="RdBu_r", vmin=-3, vmax=3, aspect=3); ax[4].set_title("A  (r x d)\ntrained, random init")
    ax[5].imshow(W + dW, cmap="RdBu_r", vmin=-3, vmax=3); ax[5].set_title(f"W + (alpha/r)BA\nadapter = {2*d*r} params (rank {r})")
    for a in (ax[0], ax[2], ax[4], ax[5]):
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("LoRA: the update to a d x d weight is constrained to rank r  (here d=16, r=2, alpha=4: 64 trainable vs 256)", y=1.08)
    save(fig, "02_lora_decomposition.png")


# ------------------------------------------------------------------ 03 params vs rank + memory
def fig_params_memory():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    rs = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    for d, c in zip((768, 2048, 4096, 8192), C):
        ax[0].plot(rs, 100 * 2 * rs * d / (d * d), "o-", color=c, label=f"d={d}")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("LoRA rank r"); ax[0].set_ylabel("trainable % of ONE d x d matrix = 200 r / d")
    ax[0].set_title("Adapter size: linear in r, smaller for big d", fontsize=11); ax[0].legend()
    sizes = [7, 13, 70]
    meth = [("full FT (fp16+Adam)", "full_fp16_adam", C[3]), ("LoRA on fp16 base", "lora_fp16", C[1]), ("QLoRA (4-bit base)", "qlora_4bit", C[2])]
    w = .26
    for j, (n, k, c) in enumerate(meth):
        vals = [finetune_memory_gb(s, k, r_params_m=s * 1000 * 0.003) for s in sizes]   # adapter ~0.3% of params
        b = ax[1].bar(np.arange(3) + (j - 1) * w, vals, w, color=c, label=n)
        for x, v in zip(np.arange(3) + (j - 1) * w, vals):
            ax[1].text(x, v + 8, f"{v:.0f}", ha="center", fontsize=9)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"{s}B params" for s in sizes])
    ax[1].set_ylabel("GB for weights + grads + optimizer"); ax[1].legend()
    ax[1].set_title("Memory, no activations (adapter = 0.3% of params)", fontsize=11)
    save(fig, "03_lora_params_and_memory.png")


# ------------------------------------------------------------------ 04 DPO loss shape
def fig_dpo_loss():
    m = np.linspace(-8, 8, 400)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for b, c in zip((0.05, 0.1, 0.5, 1.0), C):
        x = b * m
        ax[0].plot(m, np.log1p(np.exp(-x)), color=c, label=f"beta={b}")
        ax[1].plot(m, b / (1 + np.exp(x)), color=c, label=f"beta={b}")
    ax[0].axhline(np.log(2), color="#999", ls=":"); ax[0].text(-8, .72, "log 2 = loss at the reference model", fontsize=9, color="#667")
    ax[0].set_xlabel("log-ratio margin  [log pi/ref(chosen) - log pi/ref(rejected)]"); ax[0].set_ylabel("DPO loss")
    ax[0].set_title("Loss = -log sigmoid(beta * margin)"); ax[0].legend()
    ax[1].set_xlabel("same margin"); ax[1].set_ylabel("|gradient| on the margin = beta * sigmoid(-beta*margin)")
    ax[1].set_title("Updates fade once the pair is 'solved'"); ax[1].legend()
    save(fig, "04_dpo_loss_shape.png")


# ------------------------------------------------------------------ real runs
def get_models():
    base = RL.pretrain(steps=250)
    tr, te = split_combos()
    return base, tr, te


def fig_methods(base, te):
    runs = [("full FT", 150, dict(mode="full")), ("LoRA q,v r4", 200, dict(mode="lora", r=4, alpha=8, targets=("q_proj", "v_proj"))),
            ("LoRA all r8", 200, dict(mode="lora")), ("QLoRA-lite r8", 200, dict(mode="qlora"))]
    infos = []
    for n, st, kw in runs:
        _, i = RL.sft_run(base, steps=st, **kw); infos.append((n, i)); print(n, i["trainable"], round(i["acc"], 2), round(i["base_loss"], 2))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    names = [n for n, _ in infos]
    cols = [C[3], C[5], C[1], C[2]]
    ax[0].bar(names, [i["trainable"] for _, i in infos], color=cols); ax[0].set_yscale("log")
    ax[0].set_title("Trainable parameters"); ax[0].set_ylabel("count (log scale)")
    ax[1].bar(names, [i["acc"] for _, i in infos], color=cols); ax[1].set_ylim(0, 1)
    ax[1].set_title("Held-out 'up' task exact-match"); ax[1].set_ylabel("accuracy (33 unseen prompts)")
    ax[2].bar(names, [i["base_loss"] for _, i in infos], color=cols)
    ax[2].axhline(RL.base_loss(base), color="#999", ls=":"); ax[2].text(-.4, RL.base_loss(base) + .1, "before fine-tuning", fontsize=9, color="#667")
    ax[2].set_title("Plain-text loss after SFT (forgetting)"); ax[2].set_ylabel("nats/char, lower = remembers more")
    for a in ax:
        a.tick_params(axis="x", rotation=25)
    fig.suptitle("Same tiny model, same task: full fine-tuning vs hand-written LoRA vs 4-bit-base LoRA  (seeded lab run)", y=1.04)
    save(fig, "05_full_vs_lora_vs_qlora.png")


def fig_dpo(base, tr, te):
    rng = random.Random(5)
    noisy = copy.deepcopy(base)
    RL.train(noisy, lambda: sft_batch(rng, tr, ("up",), noise=0.5), 400, 1e-3)
    pol, hist = RL.dpo_run(noisy, tr, steps=60)
    pol2, _ = RL.dpo_run(noisy, tr, steps=60, nll_weight=1.0)
    res = [("noisy SFT\n(reference)", noisy), ("+ DPO", pol), ("+ DPO\n+ NLL anchor", pol2)]
    outs = [RL.greedy_outcomes(m, te) for _, m in res]
    prefs = [RL.pref_accuracy(m, te) for _, m in res]
    print("dpo outcomes", outs, prefs)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(3); bot = np.zeros(3)
    for k, (lab, c) in enumerate((("correct (UPPER)", C[2]), ("sloppy (lower-case echo)", C[1]), ("other / garbled", C[7]))):
        v = np.array([o[k] for o in outs]); ax.bar(x, v, .55, bottom=bot, color=c, label=lab)
        for xi, vi, bi in zip(x, v, bot):
            if vi > .04:
                ax.text(xi, bi + vi / 2, f"{vi:.0%}", ha="center", color="white", weight="bold")
        bot += v
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in res]); ax.set_ylabel("share of greedy answers (33 unseen prompts)")
    ax.set_title("DPO removes the rejected behaviour; plain DPO also pushes mass to 'garbled'", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3, frameon=False)
    save(fig, "06_dpo_outcomes.png")
    # GIF: training curves
    steps = [h[0] for h in hist]; loss = [h[1] for h in hist]; marg = [h[2] for h in hist]; acc = [h[3] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    def upd(k):
        k = k * 2 + 1
        for a in ax: a.clear()
        ax[0].plot(steps[:k], loss[:k], color=C[3]); ax[0].axhline(np.log(2), color="#999", ls=":")
        ax[0].set_xlim(0, 60); ax[0].set_ylim(0, .8); ax[0].set_title("DPO loss (dotted = log 2)"); ax[0].set_xlabel("step")
        ax[1].plot(steps[:k], marg[:k], color=C[2], label="implicit reward margin")
        ax[1].plot(steps[:k], acc[:k], color=C[0], ls="--", label="pairs ranked correctly (batch)")
        ax[1].set_xlim(0, 60); ax[1].set_ylim(-.1, max(1.1, max(marg) * 1.1)); ax[1].set_title("margin grows, pref-accuracy -> 1"); ax[1].set_xlabel("step")
        ax[1].legend(loc="center right", fontsize=8)
    FuncAnimation(fig, upd, frames=30, interval=120).save(OUT / "09_dpo_training.gif", writer=PillowWriter(fps=8), dpi=70)
    plt.close(fig)


def fig_forgetting(base, tr, te):
    configs = [("lr 3e-4", 3e-4, 0.0, C[0]), ("lr 1e-3", 1e-3, 0.0, C[1]), ("lr 3e-3", 3e-3, 0.0, C[3]), ("lr 1e-3 + 30% replay", 1e-3, 0.3, C[2])]
    evalrng = random.Random(77)
    ev = [sft_batch(evalrng, te, ("up",)) for _ in range(2)]
    def task_loss(m):
        with torch.no_grad():
            return sum(lm_loss(m, *b).item() for b in ev) / len(ev)
    curves = {}
    for name, lr, rep, c in configs:
        m = copy.deepcopy(base); rng = random.Random(1)
        params = [p for p in m.parameters()]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0)
        bl, tl, st = [RL.base_loss(m, n=3)], [task_loss(m)], [0]
        for s in range(1, 201):
            if rep and rng.random() < rep:
                x, y = base_batch(rng); b = (x, y, None)
            else:
                b = sft_batch(rng, tr, ("up",))
            loss = lm_loss(m, *b); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if s % 10 == 0:
                bl.append(RL.base_loss(m, n=3)); tl.append(task_loss(m)); st.append(s)
        curves[name] = (st, bl, tl, c); print(name, round(bl[-1], 2), round(tl[-1], 2))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for n, (st, bl, tl, c) in curves.items():
        ax[0].plot(st, tl, color=c, label=n); ax[1].plot(st, bl, color=c, label=n)
    ax[0].set_title("New task: held-out 'up' loss (learns)"); ax[1].set_title("Old skill: plain-text loss (forgets)")
    for a in ax:
        a.set_xlabel("fine-tuning step"); a.set_ylabel("loss (nats/token)"); a.legend(fontsize=9)
    ax[1].set_yscale("log")
    fig.suptitle("Catastrophic forgetting in the tiny lab model (full fine-tuning, seeded)", y=1.02)
    save(fig, "07_catastrophic_forgetting.png")
    # GIF
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    def upd(f):
        k = f + 1
        for a in ax: a.clear()
        for n, (st, bl, tl, c) in curves.items():
            ax[0].plot(st[:k], tl[:k], color=c, label=n); ax[1].plot(st[:k], bl[:k], color=c, label=n)
        ax[0].set_xlim(0, 200); ax[0].set_ylim(0, max(v[2][0] for v in curves.values()) * 1.05); ax[0].set_title("new task loss")
        ax[1].set_xlim(0, 200); ax[1].set_yscale("log"); ax[1].set_ylim(.3, 10); ax[1].set_title("old skill loss (higher = forgotten)")
        for a in ax: a.set_xlabel("step")
        ax[0].legend(fontsize=7, loc="upper right")
    FuncAnimation(fig, upd, frames=21, interval=150).save(OUT / "10_forgetting.gif", writer=PillowWriter(fps=6), dpi=70)
    plt.close(fig)


def fig_merge(base, te):
    ft, _ = RL.sft_run(base, "full", steps=150)
    sb, sf = base.state_dict(), ft.state_dict()
    ts = np.linspace(0, 1, 11); acc, bl = [], []
    for t in ts:
        mm = copy.deepcopy(base); mm.load_state_dict(lerp_state(sb, sf, float(t)))
        acc.append(RL.task_accuracy(mm, "up", te)); bl.append(RL.base_loss(mm, n=3))
    print("merge", np.round(acc, 2), np.round(bl, 2))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, acc, "o-", color=C[2], label="'up' task accuracy"); ax.set_ylabel("accuracy", color=C[2]); ax.set_ylim(-.02, 1.02)
    a2 = ax.twinx(); a2.spines["right"].set_visible(True)
    a2.plot(ts, bl, "s--", color=C[3], label="plain-text loss"); a2.set_ylabel("plain-text loss (nats/char)", color=C[3])
    ax.set_xlabel("t   (0 = base weights, 1 = fully fine-tuned;  weights = (1-t)*base + t*finetuned)")
    ax.set_title("Weight interpolation between base and fine-tuned (10.31), real lab models")
    save(fig, "08_weight_interpolation.png")


# ------------------------------------------------------------------ GIF: rank approximates a matrix
def gif_rank():
    rng = np.random.default_rng(7)
    d = 24
    U = rng.normal(size=(d, d)); S = np.exp(-np.arange(d) / 4.0); V = rng.normal(size=(d, d))
    T = (U * S) @ V.T / 3                       # target update with decaying spectrum
    u, s, vt = np.linalg.svd(T)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    def upd(r):
        r += 1
        for a in ax: a.clear()
        approx = (u[:, :r] * s[:r]) @ vt[:r]
        ax[0].imshow(T, cmap="RdBu_r", vmin=-4, vmax=4); ax[0].set_title("target update dW\n(full-rank)")
        ax[1].imshow(approx, cmap="RdBu_r", vmin=-4, vmax=4); ax[1].set_title(f"best rank-{r} approx\n{2*d*r} params vs {d*d}")
        err = [np.linalg.norm(T - (u[:, :k] * s[:k]) @ vt[:k]) / np.linalg.norm(T) for k in range(1, d + 1)]
        ax[2].plot(range(1, d + 1), err, color=C[0]); ax[2].plot([r], [err[r - 1]], "o", color=C[3])
        ax[2].set_xlabel("rank r"); ax[2].set_ylabel("relative error"); ax[2].set_title("error falls fast, then plateaus")
        for a in ax[:2]: a.set_xticks([]); a.set_yticks([])
    FuncAnimation(fig, upd, frames=14, interval=250).save(OUT / "11_lora_rank_approximation.gif", writer=PillowWriter(fps=3), dpi=70)
    plt.close(fig)


# ------------------------------------------------------------------ 12 PPO clip + KL leash (analytic, illustrative)
def fig_ppo():
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ratio = np.linspace(0, 2, 400); eps = 0.2
    for a, adv, ttl in ((ax[0], 1.0, "advantage > 0 (better than expected)"), (ax[1], -1.0, "advantage < 0 (worse than expected)")):
        unclipped = ratio * adv
        clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv
        a.plot(ratio, unclipped, "--", color=C[4], label="ratio * A (unclipped)")
        a.plot(ratio, np.minimum(unclipped, clipped), color=C[0], lw=2.5, label="PPO objective = min(unclipped, clipped)")
        a.axvspan(1 - eps, 1 + eps, color=C[6], alpha=.18, label="trust region (eps=0.2)")
        a.axvline(1, color="#888", lw=.8)
        a.set_xlabel("probability ratio  pi_new / pi_old"); a.set_ylabel("surrogate objective"); a.set_title(ttl, fontsize=10)
        a.legend(fontsize=7.5, loc="upper left" if adv > 0 else "lower left")
    beta = np.array([0.02, 0.1, 0.5])
    kl = np.linspace(0, 6, 200)
    for b, c in zip(beta, (C[3], C[1], C[2])):
        reward = 2.0 * np.sqrt(kl)             # illustrative concave "reward gained by drifting further"
        ax[2].plot(kl, reward - b * kl, color=c, label=f"beta={b}")
        j = np.argmax(reward - b * kl); ax[2].plot(kl[j], (reward - b * kl)[j], "o", color=c)
    ax[2].set_xlabel("KL(pi || pi_ref)  (how far the policy drifted)"); ax[2].set_ylabel("reward - beta * KL")
    ax[2].set_title("KL leash: small beta lets the policy drift far", fontsize=10); ax[2].legend(fontsize=8)
    fig.suptitle("PPO clipping (left, middle) and the RLHF KL penalty (right) - analytic illustrations, not training runs", y=1.04)
    fig.tight_layout(); save(fig, "12_ppo_clip_and_kl.png")


# ------------------------------------------------------------------ 13 quantization levels (uniform int4 vs NF4-like) + real rounding error
def fig_quant():
    from lora import quantize_4bit, dequantize_4bit
    from statistics import NormalDist
    nd = NormalDist()
    # NF4-like: 16 levels at normal quantiles, normalised to [-1, 1]. Illustrative construction, NOT the exact published NF4 table.
    qs = [nd.inv_cdf(p) for p in np.linspace(0.04, 0.96, 16)]
    nf = np.array(qs) / max(abs(min(qs)), max(qs))
    uni = np.arange(-8, 8) / 7.0   # lab levels: q in -8..7, value = q/7*absmax
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64 * 200, generator=g) * 0.05
    q, am, shp, pad = quantize_4bit(w, 64)
    err = (dequantize_4bit(q, am, shp, pad) - w).numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    ax[0].eventplot([uni, nf], lineoffsets=[1, 0], linelengths=.7, colors=[C[0], C[1]])
    ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["NF4-like (normal quantiles)", "uniform int4 (lab)"])
    ax[0].set_xlabel("quantization level, as a fraction of the block's absmax"); ax[0].set_title("16 levels: uniform vs quantile-placed (illustrative)", fontsize=10)
    ax[1].hist(err, bins=60, color=C[2]); ax[1].set_xlabel("dequantized - original weight"); ax[1].set_ylabel("count")
    ax[1].set_title(f"Real rounding error of lab quantizer (N(0,0.05) weights, block 64)\nmax |err| = {abs(err).max():.4f}; this noise is what QAT would train against", fontsize=9)
    fig.tight_layout(); save(fig, "13_quantization_levels.png")


if __name__ == "__main__":
    if "--new" in sys.argv:
        fig_ppo(); fig_quant(); print("done (new only)"); sys.exit()
    fig_pipeline(); fig_lora(); fig_params_memory(); fig_dpo_loss(); gif_rank(); fig_ppo(); fig_quant()
    base, tr, te = get_models()
    fig_methods(base, te); fig_dpo(base, tr, te); fig_forgetting(base, tr, te); fig_merge(base, te)
    print("done")
