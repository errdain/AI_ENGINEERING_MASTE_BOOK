"""Generate Tier 8 figures + GIFs.   Run:  python figures/src/make_figures.py
Needs lab/outputs/train_log.json + mini_gpt_fp16.pt (create with: cd lab && python train_mini_gpt.py --seconds 60);
if they are missing a short 15 s run is performed automatically."""
import sys, json, pathlib, subprocess, textwrap
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgba

HERE = pathlib.Path(__file__).resolve().parent
LAB = HERE.parents[1] / "lab"
sys.path.insert(0, str(LAB))
from mini_gpt import GPTConfig, MiniGPT, count_params_formula
OUT = HERE.parents[1] / "figures"; OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})
torch.set_num_threads(2)


def load_trained():
    log = LAB / "outputs" / "train_log.json"
    if not log.exists():
        subprocess.run([sys.executable, "train_mini_gpt.py", "--seconds", "15"], cwd=LAB, check=True)
    h = json.loads(log.read_text()); chars = json.loads((LAB / "outputs" / "chars.json").read_text())
    cfg = GPTConfig(vocab_size=len(chars), block_size=h["block"], n_layer=h["n_layer"], n_head=h["n_head"], d_model=h["d_model"])
    m = MiniGPT(cfg)
    sd = torch.load(LAB / "outputs" / "mini_gpt_fp16.pt", map_location="cpu", weights_only=True)
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False); m.eval()
    return m, h, chars


# ---------------------------------------------------------------- 01 block data flow
def fig_block():
    fig, ax = plt.subplots(figsize=(9.5, 10)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 21.5)
    steps = [("x   residual stream in", "(B, T, d)", C[0]), ("LayerNorm / RMSNorm", "(B, T, d)", C[2]),
             ("W_qkv  fused projection", "(B, T, 3d)", C[1]), ("split heads", "(B, h, T, d_k)", "#999999"),
             ("scores = Q Kᵀ/√d_k  + causal mask", "(B, h, T, T)", C[3]), ("softmax · V", "(B, h, T, d_k)", C[3]),
             ("concat heads, W_O", "(B, T, d)", C[1]), ("+  residual add", "(B, T, d)", C[2]),
             ("LayerNorm / RMSNorm", "(B, T, d)", C[2]), ("W_up  expand  d → d_ff", "(B, T, d_ff)", C[1]),
             ("GELU  (or SwiGLU gate)", "(B, T, d_ff)", C[0]), ("W_down  project  d_ff → d", "(B, T, d)", C[1]),
             ("+  residual add", "(B, T, d)", C[2]), ("x'   to next block", "(B, T, d)", C[0])]
    n = len(steps); ys = np.linspace(20.2, 0.7, n); pos = {}
    for (t, sh, c), y in zip(steps, ys):
        ax.add_patch(FancyBboxPatch((2.2, y - .55), 4.6, 1.1, boxstyle="round,pad=0.02,rounding_size=0.2", fc=c, ec="white", alpha=.92))
        ax.text(4.5, y, t, ha="center", va="center", color="white", weight="bold", fontsize=10.5)
        ax.text(7.05, y, sh, ha="left", va="center", fontsize=10.5, family="monospace")
        pos[t + sh] = y
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((4.5, y0 - .55), (4.5, y1 + .55), arrowstyle="-|>", mutation_scale=11, color="#777"))
    for a, b, off in ((ys[0], ys[7], 1.2), (ys[7], ys[12], 1.2)):                 # skip connections
        ax.plot([2.2, 1.0, 1.0, 2.2], [a, a, b, b], color=C[2], lw=2.5, ls="--")
        ax.add_patch(FancyArrowPatch((1.0, b), (2.2, b), arrowstyle="-|>", mutation_scale=12, color=C[2], lw=2.5))
    ax.text(0.9, (ys[0] + ys[7]) / 2, "skip 1\n(identity)", rotation=90, ha="right", va="center", color=C[2], fontsize=10)
    ax.text(0.9, (ys[7] + ys[12]) / 2, "skip 2\n(identity)", rotation=90, ha="right", va="center", color=C[2], fontsize=10)
    ax.text(0.5, 22.2, "Pre-LN Transformer block (decoder-only): data flow with tensor shapes", fontsize=13, weight="bold", va="top")
    ax.text(7.05, 21.0, "tensor shape", fontsize=10, color="#667", va="center")
    ax.set_ylim(0, 22.3)
    ax.text(0.5, -0.4, "B=batch, T=sequence length, d=d_model, h=heads, d_k=d/h.  Attention sub-layer = steps 2-7;  FFN sub-layer = steps 9-12.\n"
            "Every sub-layer reads the stream (after a norm) and ADDS its result back: the stream shape never changes.", fontsize=9.5, color="#445", va="top")
    fig.savefig(OUT / "01_block_dataflow.png", dpi=90); plt.close(fig)


# ---------------------------------------------------------------- 02 pre vs post LN (measured on the real MiniGPT)
def fig_prepost():
    """Measured on the real MiniGPT. Init: embeddings N(0,1) and every sub-layer matrix N(0, 0.25^2) (~2/sqrt(d)) so that
    sub-layer outputs are comparable to the stream -- the regime where norm placement matters. Mean over 5 seeds."""
    L, d, seeds = 24, 64, 5; res = {}
    for pre in (True, False):
        A, G, S = [], [], []
        for seed in range(seeds):
            torch.manual_seed(seed)
            m = MiniGPT(GPTConfig(vocab_size=64, block_size=32, n_layer=L, n_head=4, d_model=d, prenorm=pre))
            for n, p in m.named_parameters():
                if p.dim() == 2 and n.startswith("blocks"): torch.nn.init.normal_(p, 0, 0.25)
            torch.nn.init.normal_(m.tok.weight, 0, 1.0)
            x = torch.randint(0, 64, (16, 32)); y = torch.randint(0, 64, (16, 32))
            h = m.tok(x) + m.pos(torch.arange(32)); hs = [h]
            for b in m.blocks: h = b(h); hs.append(h)
            for t in hs: t.retain_grad()
            torch.nn.functional.cross_entropy(m.head(m.final_norm(h)).reshape(-1, 64), y.reshape(-1)).backward()
            A.append([t.detach().pow(2).mean().sqrt().item() for t in hs]); G.append([b.mlp.down.weight.grad.norm().item() for b in m.blocks])
            S.append([t.grad.pow(2).mean().sqrt().item() for t in hs])
        res[pre] = tuple(np.mean(v, 0) for v in (A, G, S))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for pre, c, nm in ((True, C[0], "Pre-LN"), (False, C[3], "Post-LN")):
        act, gp, gs = res[pre]
        axs[0].plot(range(L + 1), act, c=c, lw=2.5, label=nm)
        axs[1].semilogy(range(1, L + 1), gp, c=c, lw=2.5, label=nm)
        axs[2].semilogy(range(L + 1), gs, c=c, lw=2.5, label=nm)
    axs[0].set(title="Residual-stream RMS after each block", xlabel="block index (0 = embeddings)", ylabel="RMS of activations")
    axs[1].set(title="Grad norm of each block's FFN W_down", xlabel="block index (1 = first)", ylabel="||dloss/dW|| (log)")
    axs[2].set(title="Grad RMS at the stream entering block l", xlabel="block index (0 = embeddings)", ylabel="RMS of dloss/dx (log)")
    for a in axs: a.legend(); a.grid(alpha=.25)
    fig.suptitle("Pre-LN vs Post-LN at initialisation, measured on a 24-block MiniGPT (d=64, random tokens, mean of 5 seeds, qualitative)", y=1.03, fontsize=11.5, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "02_preln_vs_postln.png", dpi=90); plt.close(fig)
    return {k: dict(act_last=float(v[0][-1]), gW_first_over_last=float(v[1][0] / v[1][-1])) for k, v in res.items()}


# ---------------------------------------------------------------- 03 activations
def fig_act():
    from math import erf, sqrt
    x = np.linspace(-5, 5, 600); erfv = np.vectorize(erf)
    Phi = 0.5 * (1 + erfv(x / sqrt(2))); gelu = x * Phi; silu = x / (1 + np.exp(-x)); relu = np.maximum(0, x)
    gt = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for y, n, c, ls in ((relu, "ReLU", "#888", "-"), (gelu, "GELU (exact)", C[0], "-"), (gt, "GELU (tanh approx)", C[2], ":"), (silu, "SiLU / Swish", C[1], "--")):
        axs[0].plot(x, y, c=c, ls=ls, lw=2.3, label=n)
    axs[0].set(title="Activations", xlabel="input x", ylabel="f(x)", ylim=(-1, 5), xlim=(-5, 5)); axs[0].legend(); axs[0].grid(alpha=.25)
    for y, n, c, ls in ((relu, "ReLU", "#888", "-"), (gelu, "GELU", C[0], "-"), (silu, "SiLU", C[1], "--")):
        axs[1].plot(x[1:-1], np.gradient(y, x)[1:-1], c=c, ls=ls, lw=2.3, label=n)
    axs[1].set(title="Derivatives: GELU/SiLU are smooth near 0", xlabel="input x", ylabel="f'(x)", xlim=(-5, 5)); axs[1].legend(); axs[1].grid(alpha=.25)
    a = np.linspace(-4, 4, 120); u = np.linspace(-4, 4, 120); A, U = np.meshgrid(a, u)
    out = (A / (1 + np.exp(-A))) * U
    im = axs[2].imshow(out, extent=[-4, 4, -4, 4], origin="lower", cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    axs[2].contour(A, U, out, levels=[-2, 0, 2], colors="k", linewidths=.6)
    axs[2].set(title="SwiGLU unit: SiLU(a) · u", xlabel="gate pre-activation a = x·W_gate", ylabel="value u = x·W_up")
    axs[2].text(-3.9, 3.5, "gate closed (a<0): output ≈ 0", fontsize=9, color="k")
    axs[2].text(0.3, -3.6, "gate open: output ∝ u", fontsize=9, color="k")
    fig.colorbar(im, ax=axs[2], label="output"); fig.tight_layout(); fig.savefig(OUT / "03_activation_functions.png", dpi=90); plt.close(fig)


# ---------------------------------------------------------------- 04 architectures / masks
def fig_masks():
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
    n = 8; tri = np.tril(np.ones((n, n))); full = np.ones((n, n)); cross = np.ones((6, n))
    for a, M, t, xl, yl in ((axs[0], full, "Encoder-only: full\n(BERT)", "key j", "query i"), (axs[1], tri, "Decoder-only: causal\n(GPT)", "key j", "query i"),
                            (axs[2], full, "Enc-dec: encoder self\n(full)", "source j", "source i"), (axs[3], tri[:6, :6], "Enc-dec: decoder self\n(causal)", "target j", "target i")):
        a.imshow(M, cmap="Blues", vmin=0, vmax=1.3); a.set(title=t, xlabel=xl, ylabel=yl, xticks=[], yticks=[])
    axs[4].imshow(cross, cmap="Blues", vmin=0, vmax=1.3); axs[4].set(title="Enc-dec: cross-attention\n(full; Q from decoder, K/V from encoder)", xlabel="source j (encoder out)", ylabel="target i", xticks=[], yticks=[])
    for a in axs: [s.set_visible(True) for s in a.spines.values()]
    fig.suptitle("Which positions can a query read?  dark = visible, white = masked", y=1.06, fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "04_architecture_masks.png", dpi=90); plt.close(fig)


# ---------------------------------------------------------------- 05 residual stream GIF (trained model)
def gif_residual(m, chars):
    stoi = {c: i for i, c in enumerate(chars)}
    text = "The small model reads the large dataset."
    idx = torch.tensor([[stoi[c] for c in text]]); T = idx.shape[1]; stages = []; labels = []
    with torch.no_grad():
        x = m.tok(idx) + m.pos(torch.arange(T)); stages.append(x[0].clone()); labels.append("embeddings (token + position)")
        for i, b in enumerate(m.blocks):
            x = x + b.attn(b.n1(x)); stages.append(x[0].clone()); labels.append(f"block {i+1}: + attention output")
            x = x + b.mlp(b.n2(x)); stages.append(x[0].clone()); labels.append(f"block {i+1}: + FFN output")
    S = torch.stack(stages).numpy(); vmax = np.percentile(np.abs(S), 99)
    norms = np.linalg.norm(S, axis=2)                                       # (stage, pos)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.8, 1]})
    im = a0.imshow(S[0].T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    a0.set(xlabel="token position", ylabel="residual-stream dimension (d_model)"); a0.set_xticks(range(T)); a0.set_xticklabels(list(text), fontsize=7)
    a1.set(xlabel="stage", ylabel="‖x‖ (mean over positions)", xlim=(-.3, len(S) - .7), ylim=(0, norms.mean(1).max() * 1.15)); a1.grid(alpha=.25)
    a1.set_xticks(range(len(S))); a1.set_xticklabels(["emb"] + [f"{i//2+1}{'a' if i%2==0 else 'f'}" for i in range(len(S) - 1)], fontsize=8)
    line, = a1.plot([], [], "o-", c=C[1], lw=2.5); ttl = fig.suptitle("", fontsize=11.5, weight="bold")
    fig.colorbar(im, ax=a0, label="activation value")

    def upd(k):
        im.set_data(S[k].T); line.set_data(range(k + 1), norms.mean(1)[:k + 1]); ttl.set_text(f"Residual stream after: {labels[k]}  (a=attention, f=FFN)")
    anim = FuncAnimation(fig, upd, frames=len(S), interval=900)
    anim.save(OUT / "05_residual_stream.gif", writer=PillowWriter(fps=1.2), dpi=60); plt.close(fig)


# ---------------------------------------------------------------- 06 parameter breakdown
def fig_params():
    def comp(L, d, V, T, ff, act="gelu", norm="layernorm", tie=True, bias=True, pos=True):
        b = int(bias); nrm = d * (2 if norm == "layernorm" else 1); mats = 3 if act == "swiglu" else 2
        return {"token embedding": V * d, "position emb.": T * d if pos else 0, "attention": L * (4 * d * d + 4 * d * b),
                "FFN": L * (mats * d * ff + ((2 * ff + d) if act == "swiglu" else (ff + d)) * b),
                "norms": (2 * L + 1) * nrm, "output head (untied)": 0 if tie else V * d}
    P = {"Mini-GPT\n(this lab)": comp(3, 96, 41, 64, 384), "GPT-2 small\n(REVIEW_REQUIRED)": comp(12, 768, 50257, 1024, 3072),
         "GPT-2 XL\n(REVIEW_REQUIRED)": comp(48, 1600, 50257, 1024, 6400),
         "Llama-style 7B\n(REVIEW_REQUIRED)": comp(32, 4096, 32000, 0, 11008, "swiglu", "rmsnorm", False, False, False)}
    keys = list(next(iter(P.values()))); fig, ax = plt.subplots(figsize=(10.5, 4.6)); left = np.zeros(len(P))
    for i, k in enumerate(keys):
        v = np.array([100 * p[k] / sum(p.values()) for p in P.values()])
        ax.barh(list(P), v, left=left, color=C[i], label=k, edgecolor="white")
        for j, (l, w) in enumerate(zip(left, v)):
            if w > 6: ax.text(l + w / 2, j, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=9, weight="bold")
        left += v
    for j, p in enumerate(P.values()):
        t = sum(p.values()); ax.text(101, j, f"{t/1e6:,.1f} M" if t < 1e9 else f"{t/1e9:.2f} B", va="center", fontsize=10)
    ax.set(xlim=(0, 118), xlabel="share of all parameters (%)", title="Where the parameters live: embeddings dominate only when V·d is large next to L·12d²")
    ax.invert_yaxis(); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "06_parameter_breakdown.png", dpi=90); plt.close(fig)


# ---------------------------------------------------------------- 07 training curve + samples,  08 GIF
def fig_train(h):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    st = np.array(h["step"]); tr = np.array(h["train"]); k = 15
    sm = np.convolve(tr, np.ones(k) / k, mode="valid")
    a.plot(st, tr, c=C[0], alpha=.25, lw=1); a.plot(st[k - 1:], sm, c=C[0], lw=2.5, label="train loss (dropout on, smoothed)")
    a.plot(h["val_step"], h["val"], "o-", c=C[3], lw=2, ms=4, label="validation loss (dropout off)")
    a.axhline(np.log(h["vocab"]), c="#888", ls="--", lw=1.3); a.text(st[-1], np.log(h["vocab"]) + .05, f"ln(V) = {np.log(h['vocab']):.2f}  (random guessing)", ha="right", fontsize=9, color="#555")
    a.set(xlabel="optimizer step", ylabel="cross-entropy (nats / char)", title=f"Mini-GPT training: {h['params']:,} params, {h['seconds']:.0f} s on CPU, {h['steps']} steps")
    a.legend(); a.grid(alpha=.25)
    b.axis("off"); b.set_title("Samples at checkpoints (prompt: 'The ')", loc="left")
    y = .95
    for s in (h["samples"][0], h["samples"][len(h["samples"]) // 2], h["samples"][-1]):
        b.text(0, y, f"step {s['step']}  ({s['time']:.0f} s)", weight="bold", fontsize=10, va="top", color=C[1]); y -= .07
        t = textwrap.fill(s["text"].replace("\n", " "), 50); b.text(0, y, t, family="monospace", fontsize=8.5, va="top"); y -= .09 * (t.count("\n") + 1) + .08
    fig.tight_layout(); fig.savefig(OUT / "07_training_loss_and_samples.png", dpi=90); plt.close(fig)


def gif_samples(h):
    sm = h["samples"]; fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.4), gridspec_kw={"height_ratios": [1, 1]})
    a.plot(h["val_step"], h["val"], c=C[3], lw=2.3, label="validation loss"); a.set(xlabel="step", ylabel="val loss", title="Watch the babble become language"); a.grid(alpha=.25); a.legend(loc="upper right")
    dot, = a.plot([], [], "o", c="k", ms=9); b.axis("off"); txt = b.text(0, 1, "", family="monospace", fontsize=9.5, va="top")

    def upd(k):
        s = sm[k]; vi = int(np.argmin(np.abs(np.array(h["val_step"]) - s["step"])))
        dot.set_data([h["val_step"][vi]], [h["val"][vi]])
        txt.set_text(f"step {s['step']}, {s['time']:.0f} s, val loss {h['val'][vi]:.2f}\n\n" + textwrap.fill(s["text"].replace("\n", " "), 62))
    anim = FuncAnimation(fig, upd, frames=len(sm), interval=1100)
    fig.tight_layout(); anim.save(OUT / "08_generation_over_training.gif", writer=PillowWriter(fps=1), dpi=65); plt.close(fig)


# ---------------------------------------------------------------- 09 attention maps of the trained model
def fig_attn(m, chars):
    stoi = {c: i for i, c in enumerate(chars)}; text = "The small model reads the large dataset."
    idx = torch.tensor([[stoi[c] for c in text]])
    with torch.no_grad(): m(idx)
    L, H = m.cfg.n_layer, m.cfg.n_head
    fig, axs = plt.subplots(L, H, figsize=(2.6 * H, 2.6 * L))
    for l in range(L):
        att = m.blocks[l].attn.last_attn[0]
        for hh in range(H):
            a = axs[l, hh]; a.imshow(att[hh].numpy(), cmap="viridis", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
            if hh == 0: a.set_ylabel(f"layer {l+1}\nquery i")
            if l == L - 1: a.set_xlabel("key j")
            if l == 0: a.set_title(f"head {hh+1}", fontsize=10)
    fig.suptitle("Learned attention weights (trained Mini-GPT): every map is lower-triangular — no head can read the future", y=1.0, fontsize=11.5, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "09_attention_maps_trained.png", dpi=80); plt.close(fig)


# ---------------------------------------------------------------- 10 LayerNorm vs RMSNorm (numerical, no training needed)
def fig_norms():
    rng = np.random.default_rng(0); d = 512
    x = rng.normal(2.0, 3.0, size=d)                        # one token vector with an offset (mean 2) and scale (std 3)
    ln = (x - x.mean()) / np.sqrt(x.var() + 1e-6); rms = x / np.sqrt((x ** 2).mean() + 1e-6)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, v, t, c in zip(axs, (x, ln, rms), ("input token vector", "after LayerNorm", "after RMSNorm"), (C[7], C[0], C[1])):
        a.hist(v, bins=40, color=c, alpha=.9); a.axvline(v.mean(), color="k", ls="--", lw=1.5)
        a.set_title(f"{t}\nmean {v.mean():.2f}, std {v.std():.2f}, RMS {np.sqrt((v**2).mean()):.2f}", fontsize=10.5)
        a.set_xlabel("feature value (one token, d = 512)")
    axs[0].set_ylabel("number of features"); axs[1].text(.03, .9, "dashed = mean\nLayerNorm: mean 0, std 1", transform=axs[1].transAxes, fontsize=9)
    axs[2].text(.03, .9, "RMSNorm keeps the mean offset;\nonly the scale is fixed (RMS 1)", transform=axs[2].transAxes, fontsize=9)
    fig.suptitle("LayerNorm centres and rescales; RMSNorm only rescales (no mean, no bias)", fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "10_layernorm_vs_rmsnorm.png", dpi=80); plt.close(fig)


# ---------------------------------------------------------------- 11 sizing: width, depth, heads  +  12 weight tying
def fig_sizing():
    T, V = 1024, 50257
    def P(d, L, h=12, tie=True):
        return count_params_formula(GPTConfig(vocab_size=V, block_size=T, n_layer=L, n_head=h, d_model=d, tie_weights=tie))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    ds = np.arange(256, 2049, 128); axs[0].plot(ds, [P(d, 12, 8) / 1e6 for d in ds], "o-", color=C[0], label="12 layers")
    axs[0].plot(ds, [(12 * 12 * d * d) / 1e6 for d in ds], "--", color=C[3], label="12·L·d² only (blocks)")
    axs[0].set_xlabel("hidden dimension d_model"); axs[0].set_ylabel("parameters (millions)"); axs[0].set_title("Width: parameters grow ~ d²"); axs[0].legend()
    Ls = np.arange(2, 49, 2); axs[1].plot(Ls, [P(768, L, 12) / 1e6 for L in Ls], "o-", color=C[2])
    axs[1].set_xlabel("number of layers L"); axs[1].set_ylabel("parameters (millions)"); axs[1].set_title("Depth: parameters grow linearly (d = 768)")
    hs = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]; axs[2].plot(range(len(hs)), [P(768, 12, h) / 1e6 for h in hs], "o-", color=C[1])
    axs[2].set_xticks(range(len(hs))); axs[2].set_xticklabels(hs); axs[2].set_ylim(0, 160)
    axs[2].set_xlabel("number of heads h (d = 768, 12 layers)"); axs[2].set_ylabel("parameters (millions)")
    axs[2].set_title("Heads: parameter count does not change"); axs[2].text(.05, .08, "score-tensor memory ∝ h·T²\nhead width d_k = d/h shrinks", transform=axs[2].transAxes, fontsize=9)
    fig.suptitle("Sizing a Transformer: V = 50 257, T = 1024, tied embeddings (from count_params_formula)", fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(OUT / "11_width_depth_heads.png", dpi=80); plt.close(fig)


def fig_tying():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    cfgs = [("Mini-GPT\nd=96,L=3,V=41", 41, 96, 3, 64), ("GPT-2-small shape\nd=768,L=12", 50257, 768, 12, 1024), ("d=2048,L=24\nV=50k", 50257, 2048, 24, 2048)]
    x = np.arange(len(cfgs)); w = .38
    tied = [count_params_formula(GPTConfig(vocab_size=v, block_size=T, n_layer=L, n_head=8 if d % 8 == 0 else 1, d_model=d)) for _, v, d, L, T in cfgs]
    unt = [count_params_formula(GPTConfig(vocab_size=v, block_size=T, n_layer=L, n_head=8 if d % 8 == 0 else 1, d_model=d, tie_weights=False)) for _, v, d, L, T in cfgs]
    axs[0].bar(x - w / 2, np.array(tied) / 1e6, w, color=C[0], label="tied head"); axs[0].bar(x + w / 2, np.array(unt) / 1e6, w, color=C[3], label="untied head")
    axs[0].set_xticks(x); axs[0].set_xticklabels([c[0] for c in cfgs], fontsize=9); axs[0].set_yscale("log"); axs[0].set_ylabel("parameters (millions, log)")
    axs[0].set_title("Total parameters"); axs[0].legend()
    sav = [(u - t) / u * 100 for t, u in zip(tied, unt)]; axs[1].bar(x, sav, .5, color=C[2])
    for i, s in enumerate(sav): axs[1].text(i, s + .5, f"{s:.1f}%", ha="center")
    axs[1].set_xticks(x); axs[1].set_xticklabels([c[0] for c in cfgs], fontsize=9); axs[1].set_ylabel("saved by tying (% of untied total)"); axs[1].set_title("Tying saves V·d: matters most for small models / big vocabularies")
    fig.tight_layout(); fig.savefig(OUT / "12_weight_tying_savings.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    fig_block(); r = fig_prepost(); print("pre/post (final act rms, grad ratio first/last):", r)
    fig_act(); fig_masks(); fig_params(); fig_norms(); fig_sizing(); fig_tying()
    m, h, chars = load_trained()
    gif_residual(m, chars); fig_train(h); gif_samples(h); fig_attn(m, chars)
    for p in sorted(OUT.iterdir()): print(p.name, p.stat().st_size // 1024, "KB")
