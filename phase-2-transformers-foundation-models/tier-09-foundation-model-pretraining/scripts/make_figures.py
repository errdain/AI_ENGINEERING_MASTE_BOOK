"""Generate Tier 9 figures + GIFs.   Run:  python scripts/make_figures.py   (~1-2 min on CPU)

Everything that says "measured" below is produced by running the lab code in ../lab right here."""
import sys, pathlib, tempfile, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lab"))
import torch
import data_prep as dp
import run_lab
from model import ModelConfig, MoEFFN
from pretrain import TrainConfig, train, flops_6nd
from scaling import CHINCHILLA, loss as chin_loss, compute_optimal, fit_power_law

OUT = HERE.parent / "figures"; OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "savefig.bbox": "tight", "axes.titleweight": "bold"})
TMP = pathlib.Path(tempfile.mkdtemp())
PREP = run_lab.prepare()   # tok, train stream, val stream, report


def save(fig, name, dpi=100):
    fig.savefig(OUT / name, dpi=dpi); plt.close(fig)


def gif(fig, update, frames, name, fps=8, dpi=70):
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.9, wspace=0.3)
    FuncAnimation(fig, update, frames=frames, interval=1000 / fps).save(OUT / name, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


# ---- 1. data funnel (measured on the synthetic crawl) -------------------------------------------------
def fig_funnel():
    rep = PREP[3]; st = rep["stages"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    names = [n.replace("_", "\n") for n, _ in st]; vals = [v for _, v in st]
    ax.bar(names, vals, color=[C[0]] + [C[2]] * (len(vals) - 1), width=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 8, str(v), ha="center", fontweight="bold")
        if i:
            ax.text(i, v / 2, f"-{vals[i-1]-v}", ha="center", color="white", fontweight="bold")
    ax.set_ylim(0, 960); ax.set_ylabel("documents remaining"); ax.set_xlabel("pipeline stage (left to right)")
    ax.set_title("Pretraining data funnel on the lab's synthetic crawl (measured)")
    d = rep["dropped"]
    ax.text(0.99, 0.99, f"quality drops by reason: {d['quality_filter']}   |   license drops: {d['license_filter']}   |   "
            f"leaked benchmark docs removed: {d['decontaminated']}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    save(fig, "01_data_pipeline_funnel.png")


# ---- 2. MinHash + LSH ----------------------------------------------------------------------------------
def fig_minhash():
    import random
    r = random.Random(0)
    base = [f"w{i}" for i in range(60)]
    true, est = [], []
    for _ in range(60):
        b = base[:]
        for k in r.sample(range(60), r.randint(0, 30)):
            b[k] = f"x{k}{r.randint(0, 9)}"
        s1, s2 = dp.shingles(" ".join(base)), dp.shingles(" ".join(b))
        true.append(dp.jaccard(s1, s2))
        est.append(dp.estimated_jaccard(dp.minhash_signature(s1, 64), dp.minhash_signature(s2, 64)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].scatter(true, est, c=C[0], s=22); ax[0].plot([0, 1], [0, 1], "--", c="#888", label="perfect estimate")
    ax[0].set_xlabel("true Jaccard similarity of 3-word shingle sets"); ax[0].set_ylabel("MinHash estimate (64 hashes)")
    ax[0].set_title("MinHash estimates Jaccard (measured)"); ax[0].legend()
    s = np.linspace(0, 1, 200)
    for (b, rr), c in zip([(16, 4), (8, 8), (32, 2)], C):
        ax[1].plot(s, 1 - (1 - s ** rr) ** b, c=c, lw=2.5, label=f"{b} bands x {rr} rows")
    ax[1].axvline(0.7, ls=":", c="#888"); ax[1].text(0.71, 0.05, "threshold 0.7", color="#666")
    ax[1].set_xlabel("true Jaccard similarity of a pair"); ax[1].set_ylabel("P(pair becomes an LSH candidate)")
    ax[1].set_title("LSH banding = a tunable S-curve"); ax[1].legend(loc="center left")
    save(fig, "02_minhash_lsh.png")


# ---- 3. data mixture and epochs ------------------------------------------------------------------------
def fig_mixture():
    tok, tr, va, rep = PREP
    avail = rep["source_counts"]; total_docs = sum(avail.values())
    mixes = {"A: balanced": {"wiki": .25, "web": .25, "code": .25, "chat": .25},
             "B: wiki-heavy": {"wiki": .7, "web": .1, "code": .1, "chat": .1},
             "C: code-heavy": {"wiki": .1, "web": .1, "code": .7, "chat": .1}}
    srcs = ["wiki", "web", "code", "chat"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for j, (n, w) in enumerate(mixes.items()):
        bot = 0
        for s, c in zip(srcs, C):
            ax[0].bar(j, w[s], bottom=bot, color=c, label=s if j == 0 else None); bot += w[s]
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(mixes.keys()); ax[0].set_ylabel("share of training documents")
    ax[0].set_title("Three candidate mixtures"); ax[0].legend(title="source", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    # epochs per source if we draw `budget` docs with those weights from the deduped pool
    budget = total_docs * 3
    x = np.arange(len(srcs)); wd = 0.27
    for j, (n, w) in enumerate(mixes.items()):
        ax[1].bar(x + (j - 1) * wd, [budget * w[s] / avail[s] for s in srcs], wd, color=C[j], label=n)
    ax[1].axhline(1, c="#888", ls="--", label="1 epoch (no repetition)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(srcs); ax[1].set_ylabel(f"epochs over each source (budget = {budget} docs)")
    ax[1].set_title("Up-weighting a small source means repeating it"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    save(fig, "03_mixture_epochs.png")


# ---- 4. scaling law: IsoFLOP + frontier -----------------------------------------------------------------
def fig_scaling():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    Ns = np.logspace(7.5, 11.5, 200)
    for Cb, c in zip([1e19, 1e20, 1e21, 1e22, 1e23], C):
        D = Cb / (6 * Ns); L = chin_loss(Ns, D)
        ax[0].plot(Ns, L, c=c, lw=2, label=f"C = {Cb:.0e} FLOPs")
        n, d = compute_optimal(Cb); ax[0].plot(n, chin_loss(n, d), "o", c=c, mec="k")
    ax[0].set_xscale("log"); ax[0].set_ylim(1.7, 3.6)
    ax[0].set_xlabel("model parameters N (log)"); ax[0].set_ylabel("predicted loss L(N, D=C/6N)")
    ax[0].set_title("IsoFLOP curves: each budget has a best model size"); ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=9)
    ax[0].text(0.02, 0.04, "dots = compute-optimal N\nconstants REVIEW_REQUIRED", transform=ax[0].transAxes, color="#666")
    Cs = np.logspace(18, 25, 50); NO = np.array([compute_optimal(c)[0] for c in Cs]); DO = np.array([compute_optimal(c)[1] for c in Cs])
    ax[1].loglog(Cs, NO, c=C[0], lw=2.5, label="optimal N (params)"); ax[1].loglog(Cs, DO, c=C[1], lw=2.5, label="optimal D (tokens)")
    ax[1].loglog(Cs, 20 * NO, c=C[1], ls=":", label="rule of thumb D = 20 N")
    ax[1].set_xlabel("compute budget C (FLOPs, log)"); ax[1].set_ylabel("optimal size (log)")
    ax[1].set_title("Compute-optimal frontier: grow N and D together"); ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9)
    save(fig, "04_scaling_law_isoflop.png")


# ---- 5. measured mini scaling sweep ---------------------------------------------------------------------
def fig_sweep():
    tok, tr, va, _ = PREP
    widths = [16, 24, 32, 48, 64, 96]; Ns, Ls = [], []
    for d in widths:
        cfg = TrainConfig(steps=250, eval_batches=16, eval_every=250, ckpt_every=10**6,
                          model=ModelConfig(vocab_size=tok.vocab_size, d_model=d, n_layer=2, n_head=2))
        m = train(cfg, tr, va, TMP / f"sw{d}")
        Ns.append(m["n_params_non_embedding"]); Ls.append(m["final_val"]["val_loss"])
    f = fit_power_law(Ns, Ls)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog(Ns, Ls, "o", c=C[0], ms=9, label="measured (lab runs, same data & steps)")
    g = np.logspace(np.log10(min(Ns)) - .1, np.log10(max(Ns)) + .1, 50)
    ax.loglog(g, f["E"] + f["A"] * g ** -f["alpha"], c=C[1], lw=2, label=(f"fit  L = {f['E']:.2f} + {f['A']:.2g}/N^{f['alpha']:.2f}" if f["E"] > 0.01 else f"fit  L = {f['A']:.2g}/N^{f['alpha']:.2f}  (floor E not identifiable)"))
    ax.set_xlabel("non-embedding parameters N (log)"); ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("Toy scaling sweep: bigger model, lower loss (till data/steps bind)")
    ax.legend(); save(fig, "05_lab_scaling_sweep.png")


# ---- 6. GIF: loss curves vs learning rate ---------------------------------------------------------------
def fig_lr_gif():
    tok, tr, va, _ = PREP
    runs = {}
    for lr, lab in [(1e-4, "LR 1e-4 (too low)"), (3e-3, "LR 3e-3 (good)"), (3e-2, "LR 3e-2 (too high)")]:
        cfg = TrainConfig(steps=150, lr=lr, eval_every=10**6, ckpt_every=10**6, warmup=10,
                          model=ModelConfig(vocab_size=tok.vocab_size))
        m = train(cfg, tr, va, TMP / f"lr{lr}")
        runs[lab] = ([h["loss"] for h in m["history"]["train"]], [h["grad_norm"] for h in m["history"]["train"]])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    def upd(f):
        t = min(150, (f + 1) * 5)
        for a in ax: a.clear()
        for (lab, (l, g)), c in zip(runs.items(), [C[0], C[2], C[3]]):
            ax[0].plot(range(1, t + 1), l[:t], c=c, lw=2, label=lab); ax[1].plot(range(1, t + 1), g[:t], c=c, lw=2)
        ax[0].set_xlim(0, 150); ax[0].set_ylim(0, 6.5); ax[1].set_xlim(0, 150); ax[1].set_yscale("log"); ax[1].set_ylim(0.05, 50)
        ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("training loss (nats)"); ax[0].set_title("Loss: learning-rate sensitivity")
        ax[1].set_xlabel("optimizer step"); ax[1].set_ylabel("pre-clip gradient norm (log)"); ax[1].set_title("Gradient norm")
        ax[0].legend(loc="upper right")
    gif(fig, upd, 30, "06_loss_vs_learning_rate.gif")


# ---- 7. GIF: MoE router, with and without load-balancing loss -------------------------------------------
def fig_moe_gif():
    E, D, K = 4, 16, 2
    hist = {}
    for aux in (0.0, 0.5):
        torch.manual_seed(0)
        moe = MoEFFN(D, E, K)
        with torch.no_grad():
            moe.router.weight[:] = torch.randn(E, D) * 0.02; moe.router.weight[0] += 0.3; moe.router.weight[1] += 0.15  # biased start: experts 0,1 favoured
        centers = torch.randn(E, D) + 0.5
        opt = torch.optim.Adam(moe.parameters(), lr=0.02)
        loads = []
        for s in range(60):
            x = (centers[torch.randint(0, E, (256,))] + 0.3 * torch.randn(256, D)).unsqueeze(0)
            y = moe(x)
            # tiny task loss so experts specialise: reconstruct x; plus the balance loss
            l = ((y - x) ** 2).mean() + aux * moe.aux_loss
            opt.zero_grad(); l.backward(); opt.step()
            loads.append((moe.last_load / K).numpy().copy())
        hist[aux] = np.array(loads)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    def upd(f):
        s = min(59, f * 2)
        for a, aux, t in zip(ax, (0.0, 0.5), ("no balancing loss (weight 0)", "with balancing loss (weight 0.5)")):
            a.clear(); a.bar(range(E), hist[aux][s], color=[C[i] for i in range(E)])
            a.axhline(1 / E, c="#666", ls="--"); a.set_ylim(0, 1); a.set_xticks(range(E)); a.set_xticklabels([f"expert {i}" for i in range(E)])
            a.set_title(t); a.text(0.5, 0.92, f"step {s + 1}", transform=a.transAxes, ha="center")
        ax[0].set_ylabel("share of routing slots (top-2 of 4)"); ax[1].text(3.45, 0.27, "dashed = ideal 0.25", color="#444", ha="right", fontsize=9)
    gif(fig, upd, 30, "07_moe_load_balancing.gif")


# ---- 8. resume == uninterrupted --------------------------------------------------------------------------
def fig_resume():
    tok, tr, va, _ = PREP
    mk = lambda: TrainConfig(steps=120, eval_every=30, ckpt_every=30, model=ModelConfig(vocab_size=tok.vocab_size))
    full = train(mk(), tr, va, TMP / "full")
    train(mk(), tr, va, TMP / "res", stop_at=60); res = train(mk(), tr, va, TMP / "res", resume=True)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    fl = [h["loss"] for h in full["history"]["train"]]; rl = [h["loss"] for h in res["history"]["train"]]
    ax.plot(fl, c=C[0], lw=4, alpha=.5, label="uninterrupted 120 steps"); ax.plot(rl, c=C[1], lw=1.5, label="stopped at 60, resumed from checkpoint")
    ax.axvline(60, c="#888", ls="--"); ax.text(61, max(fl) * .9, "crash + resume", color="#555")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss (nats)")
    ax.set_title(f"Deterministic resume: curves coincide (max diff {np.max(np.abs(np.array(fl)-np.array(rl))):.1e})")
    ax.legend(loc="upper right"); save(fig, "08_checkpoint_resume.png")


# ---- 9. compute budget calculator picture ------------------------------------------------------------------
def fig_budget():
    fig, ax = plt.subplots(figsize=(8.5, 5))
    N = np.logspace(7, 12, 100)
    for Cb, c in zip([1e18, 1e20, 1e22, 1e24], C):
        ax.loglog(N, Cb / (6 * N), c=c, lw=2, label=f"C = {Cb:.0e} FLOPs  (= 6ND)")
    Cs = np.logspace(18, 24, 30)
    ax.loglog([compute_optimal(c)[0] for c in Cs], [compute_optimal(c)[1] for c in Cs], "k--", lw=2, label="compute-optimal frontier (REVIEW_REQUIRED fit)")
    ax.loglog(N, 20 * N, c="#888", ls=":", label="D = 20 N rule of thumb")
    ax.set_xlabel("parameters N (log)"); ax.set_ylabel("training tokens D (log)"); ax.set_ylim(1e8, 1e14)
    ax.set_title("Same budget, different splits: trade model size against tokens"); ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    save(fig, "09_compute_budget_6nd.png")


if __name__ == "__main__":
    for f in (fig_funnel, fig_minhash, fig_mixture, fig_scaling, fig_sweep, fig_lr_gif, fig_moe_gif, fig_resume, fig_budget):
        print(f.__name__); f()
    print("done ->", OUT)
