"""Generate Tier 12 figures + GIFs.  Run:  python figures/src/make_figures.py"""
import os, sys, pathlib
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lab"))
import arch_matrix as am
import ssm_scan as ss

OUT = HERE.parents[1] / "figures"
OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})
GREEN = ListedColormap(["#eef0f4", "#54A24B"])


# ---------------------------------------------------------------- 01 family tree
def fig_family_tree():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    lanes = {"Encoder-only": 3, "Decoder-only (dense)": 2, "Encoder-decoder": 1, "Sparse / SSM / hybrid": 0}
    ev = [  # (approx year, lane, label, x-offset-y) -- years approximate, REVIEW_REQUIRED
        (2017.5, 1, "Transformer\n(orig. enc-dec)", 0), (2018.8, 3, "BERT", 0), (2019.4, 3, "RoBERTa", .32),
        (2018.5, 2, "GPT-1", 0), (2019.2, 2, "GPT-2", 0), (2020.5, 2, "GPT-3", 0), (2023.2, 2, "Llama", 0),
        (2023.9, 2, "Qwen / Gemma\n/ Llama-2,3 ...", 0.0),
        (2019.8, 1, "T5 / BART", 0), (2021.0, 1, "FLAN-T5\n(T5 line)", 0),
        (2022.0, 0, "Switch / GLaM\n(sparse MoE)", 0), (2024.0, 0, "Mixtral\n(MoE, GQA,\nSWA)", 0),
        (2023.2, 0, "Mamba\n(SSM)", 0), (2024.9, 0, "SSM-attention\nhybrids", 0.0)]
    for name, y in lanes.items():
        ax.axhline(y, color="#d8dde6", lw=1, zorder=0)
        ax.text(2016.9, y + .08, name, fontsize=10, weight="bold", color="#555", va="bottom")
    for x, y, lab, dy in ev:
        col = C[y]
        ax.scatter([x], [y + dy * 0], s=120, color=col, zorder=3, edgecolor="white")
        ax.text(x, y - .13 + dy, lab, ha="center", va="top", fontsize=9)
    for a, b, ya, yb in [(2017.5, 2018.8, 1, 3), (2017.5, 2018.5, 1, 2), (2017.5, 2019.8, 1, 1), (2020.5, 2022.0, 2, 0),
                         (2023.2, 2023.9, 2, 2), (2022.0, 2024.0, 0, 0)]:
        ax.annotate("", xy=(b, yb), xytext=(a, ya), arrowprops=dict(arrowstyle="->", color="#99a", lw=1))
    ax.set_xlim(2016.8, 2025.3); ax.set_ylim(-.9, 3.6)
    ax.set_yticks([]); ax.set_xlabel("approximate year (REVIEW_REQUIRED -- check each paper/release)")
    ax.set_title("Architecture family tree after the original Transformer (illustrative, not exhaustive)", loc="left")
    fig.savefig(OUT / "01_family_tree.png"); plt.close(fig)


# ---------------------------------------------------------------- 02 enc / dec / enc-dec
def fig_three_families():
    n = 10
    full = np.ones((n, n)); causal = np.tril(full)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, m, t, xl, yl in [
        (axs[0], full, "Encoder-only\nbidirectional self-attention", "keys (input)", "queries (input)"),
        (axs[1], causal, "Decoder-only\ncausal self-attention", "keys (past + self)", "queries"),
        (axs[2], causal, "Enc-dec: decoder self-attn\n(causal over target)", "target keys", "target queries"),
        (axs[3], np.ones((n, 7)), "Enc-dec: cross-attention\n(every target sees all source)", "source keys (encoder out)", "target queries")]:
        ax.imshow(m, cmap=GREEN, vmin=0, vmax=1, aspect="auto"); ax.set_title(t, fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Green = allowed to attend.  Same attention math (Tier 7); the three families differ only in the mask and where it is used", y=1.04, fontsize=11)
    fig.savefig(OUT / "02_encoder_decoder_encdec.png"); plt.close(fig)


# ---------------------------------------------------------------- 03 attention pattern gallery
def pattern_masks(n=32, w=6, stride=4, blk=8, g=2):
    i, j = np.indices((n, n)); causal = j <= i
    pats = {}
    pats["Full causal"] = causal
    pats[f"Sliding window (w={w})"] = causal & (i - j < w)
    pats[f"Dilated (stride={stride})"] = causal & ((i - j) % stride == 0) & (i - j < stride * w)
    bs = (i // blk == j // blk) | ((j % blk == blk - 1))      # local block + last token of each block as summary
    pats[f"Block-sparse (block={blk})"] = causal & bs
    gl = (i - j < w) | (j < g) | (i < g)
    pats[f"Global+local (w={w}, g={g})"] = causal & gl | (i < g) & causal
    return pats


def fig_pattern_gallery():
    pats = pattern_masks()
    n = 32; total = n * (n + 1) // 2
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, (k, m) in zip(axs, pats.items()):
        ax.imshow(m, cmap=GREEN, vmin=0, vmax=1)
        ax.set_title(f"{k}\n{m.sum()}/{total} pairs = {m.sum()/total:.0%}", fontsize=9.5)
        ax.set_xlabel("key position"); ax.set_xticks([0, 31]); ax.set_yticks([0, 31])
    axs[0].set_ylabel("query position")
    fig.suptitle("Attention-pattern gallery for n = 32 (green = computed). Sparse patterns trade full reach for O(n*w) or O(n*sqrt n) work", y=1.03, fontsize=11)
    fig.savefig(OUT / "03_attention_pattern_gallery.png"); plt.close(fig)


# ---------------------------------------------------------------- 04 RoPE scaling
def rope_wavelengths(d=128, base=10000.0, scheme="none", s=8.0):
    i = np.arange(d // 2)
    if scheme == "base":                       # simply enlarge the base by s
        base = base * s
    elif scheme == "ntk":                      # NTK-aware: base' = base * s^(d/(d-2))
        base = base * s ** (d / (d - 2))
    theta = base ** (-2 * i / d)
    if scheme == "pi":                         # position interpolation: positions / s  == theta / s
        theta = theta / s
    return 2 * np.pi / theta


def fig_rope_scaling():
    d, L, s = 128, 4096, 8
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    i = np.arange(d // 2)
    for k, lab, c in [("none", "original (base 10000)", "#444"), ("pi", f"position interpolation (s={s})", C[0]),
                      ("base", f"larger base (x{s})", C[1]), ("ntk", f"NTK-aware base (s={s})", C[2])]:
        axs[0].plot(i, rope_wavelengths(d, scheme=k, s=s), label=lab, color=c, lw=2)
    axs[0].axhline(L, color=C[3], ls="--"); axs[0].text(1, L * 1.25, f"trained context L = {L}", color=C[3], fontsize=9)
    axs[0].axhline(L * s, color=C[3], ls=":"); axs[0].text(1, L * s * 1.25, f"target context = {L*s}", color=C[3], fontsize=9)
    axs[0].set_yscale("log"); axs[0].set_xlabel("frequency pair index i  (0 = fastest)"); axs[0].set_ylabel("wavelength in tokens (2*pi / theta_i)")
    axs[0].set_title("Per-pair wavelength under context-extension recipes"); axs[0].legend(fontsize=9, loc="lower right")
    tgt = L * s
    th = lambda k: 2 * np.pi / rope_wavelengths(d, scheme=k, s=s)
    axs[1].plot(i, th("none") * L, color="#444", lw=2, ls="--", label=f"original @ trained L={L} (what training saw)")
    for k, lab, c in [("none", f"original @ {tgt}", "#444"), ("pi", "PI @ target", C[0]), ("ntk", "NTK-aware @ target", C[2])]:
        axs[1].plot(i, th(k) * tgt, color=c, lw=2, label=lab)
    axs[1].axhline(2 * np.pi, color=C[3], ls=":"); axs[1].text(1, 2 * np.pi * 1.3, "2*pi: one full turn", color=C[3])
    axs[1].set_yscale("log"); axs[1].set_xlabel("frequency pair index i"); axs[1].set_ylabel("max rotation angle (rad) = theta_i x length")
    axs[1].set_title("Slow pairs (right) must stay within the angles seen in training;\nfast pairs have wrapped many times, so they are safe"); axs[1].legend(fontsize=8.5, loc="upper right")
    fig.savefig(OUT / "04_rope_context_scaling.png"); plt.close(fig)


# ---------------------------------------------------------------- 05 MoE total vs active
def fig_moe_params():
    base = am.PRESETS["Dense GQA-8 (illustrative)"]
    es = [1, 2, 4, 8, 16, 32, 64]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    tot = [am.total_params(am.replace(base, n_experts=e, top_k=1)) / 1e9 for e in es]
    axs[0].plot(es, tot, "o-", color="#444", lw=2.5, label="TOTAL params (any k, ~linear in E)")
    for k, c in [(1, C[0]), (2, C[1])]:
        act = [am.active_params(am.replace(base, n_experts=e, top_k=min(k, e))) / 1e9 for e in es]
        axs[0].plot(es, act, "s--", color=c, label=f"ACTIVE params (top-{k})")
    axs[0].set_xscale("log", base=2); axs[0].set_yscale("log"); axs[0].set_xlabel("number of experts E (per layer)")
    axs[0].set_ylabel("parameters (billions)"); axs[0].legend(fontsize=9)
    axs[0].set_title("Capacity grows with E; active params only with k")
    # right: memory vs compute view
    fl = [am.flops_per_token(am.replace(base, n_experts=e, top_k=min(2, e)), 2048) / 1e9 for e in es]
    wb = [am.weight_bytes(am.replace(base, n_experts=e, top_k=min(2, e))) / 2**30 for e in es]
    ax = axs[1]; ax.bar(np.arange(len(es)) - .2, wb, .4, color=C[0], label="weights memory (GiB, bf16)")
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.bar(np.arange(len(es)) + .2, fl, .4, color=C[1], label="GFLOPs/token (top-2)")
    ax.set_xticks(range(len(es))); ax.set_xticklabels(es); ax.set_xlabel("number of experts E")
    ax.set_ylabel("weights memory (GiB)", color=C[0]); ax2.set_ylabel("GFLOPs per token", color=C[1])
    ax.set_title("Memory pays for all experts, compute for k")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2); ax2.set_ylim(0, 30)
    fig.savefig(OUT / "05_moe_total_vs_active.png"); plt.close(fig)


# ---------------------------------------------------------------- 06 MoE routing GIF
def gif_moe_routing():
    rng = np.random.default_rng(3)
    T, E, k, cf = 32, 8, 2, 1.0
    cap = int(np.ceil(cf * T * k / E))
    bias = np.array([1.2, .8, .4, 0, 0, -.3, -.5, -.8])           # a skewed router
    logits = rng.standard_normal((T, E)) * .7 + bias
    top = np.argsort(-logits, 1)[:, :k]
    load = np.zeros(E, int); drop = 0; hist = []
    for t in range(T):
        for e in top[t]:
            if load[e] < cap: load[e] += 1
            else: drop += 1
        hist.append((load.copy(), drop))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    def draw(f):
        ax.clear(); ld, dr = hist[f]
        ax.bar(range(E), ld, color=[C[0] if v < cap else C[3] for v in ld])
        ax.axhline(cap, color="#333", ls="--"); ax.text(E - .5, cap + .15, f"capacity = {cap}", ha="right")
        ax.set_ylim(0, cap + 4); ax.set_xticks(range(E)); ax.set_xlabel("expert"); ax.set_ylabel("tokens assigned")
        ax.set_title(f"Top-{k} routing, token {f+1}/{T}: dropped expert-slots = {dr}  (capacity factor {cf})")
        ax.text(0, cap + 2.6, "skewed router -> popular experts overflow (red)", color=C[3])
    anim = FuncAnimation(fig, draw, frames=T, interval=170)
    anim.save(OUT / "06_moe_routing_capacity.gif", writer=PillowWriter(fps=6), dpi=70); plt.close(fig)


# ---------------------------------------------------------------- 07 SSM vs attention cost
def fig_cost_curves():
    d, N, L = 4096, 16, 32
    n = np.logspace(2, 6, 60)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    axs[0].loglog(n, ss.decode_cost_attention(n, d) * L / 1e9, color=C[3], lw=2, label="attention (decode, one token)")
    axs[0].loglog(n, np.full_like(n, ss.decode_cost_ssm(d * 2, N) * L / 1e9), color=C[0], lw=2, label=f"SSM scan update (N={N})")
    axs[0].loglog(n, ss.decode_cost_attention(np.minimum(n, 4096), d) * L / 1e9, color=C[2], lw=2, ls="--", label="sliding window (w=4096)")
    axs[0].set_xlabel("context length n (tokens)"); axs[0].set_ylabel("GFLOPs per generated token (mixer only)"); axs[0].set_title("Compute per decoded token"); axs[0].legend(fontsize=8.5)
    pre_att = 4 * n**2 * d * L / 1e12
    pre_ssm = 6 * n * (2 * d) * N * L / 1e12
    axs[1].loglog(n, pre_att, color=C[3], lw=2, label="attention  O(n^2)"); axs[1].loglog(n, pre_ssm, color=C[0], lw=2, label="SSM scan  O(n)")
    axs[1].set_xlabel("prompt length n"); axs[1].set_ylabel("TFLOPs for whole prefill (mixer only)"); axs[1].set_title("Prefill compute"); axs[1].legend(fontsize=9)
    kv = 2 * 8 * 128 * 2 * L * n / 2**30
    axs[2].loglog(n, kv, color=C[3], lw=2, label="KV cache, GQA-8 (grows with n)")
    axs[2].loglog(n, np.full_like(n, 32 * 2 * d * N * 2 / 2**30 * 1.0), color=C[0], lw=2, label="SSM state (constant)")
    axs[2].loglog(n, np.minimum(kv, 2 * 8 * 128 * 2 * L * 4096 / 2**30), color=C[2], ls="--", lw=2, label="sliding window cache (capped)")
    axs[2].set_xlabel("context length n"); axs[2].set_ylabel("GiB per sequence"); axs[2].set_title("Decode-time memory per sequence"); axs[2].legend(fontsize=8.5)
    fig.suptitle("Why SSMs and windows matter at long context (analytical cost model from lab/ssm_scan.py; d=4096, 32 layers, fp16)", y=1.03, fontsize=11)
    fig.savefig(OUT / "07_ssm_vs_attention_costs.png"); plt.close(fig)


# ---------------------------------------------------------------- 08 SSM recurrence GIF
def gif_ssm():
    T = 40
    x = np.zeros(T); x[[3, 4, 18]] = [1.0, .6, -1.0]; x[28:31] = .8
    decays = [0.95, 0.75, 0.40]
    H = np.zeros((T, 3))
    h = np.zeros(3)
    for t in range(T):
        h = np.array(decays) * h + x[t]; H[t] = h
    y = H @ np.array([.6, .3, .1])
    fig, axs = plt.subplots(3, 1, figsize=(8.2, 6.2), sharex=True)

    def draw(f):
        for a in axs: a.clear()
        axs[0].stem(range(T), x, linefmt=C[1], markerfmt="o", basefmt="#999"); axs[0].set_ylim(-1.4, 1.4)
        axs[0].axvline(f, color="#333", lw=1); axs[0].set_ylabel("input x_t"); axs[0].set_title(f"Selective-scan intuition, t = {f}: state h_t = a*h_(t-1) + x_t  (3 channels, different decays a)")
        for c in range(3):
            axs[1].plot(range(f + 1), H[:f + 1, c], color=C[c], lw=2, label=f"state {c}  (a={decays[c]})")
        axs[1].set_xlim(-1, T); axs[1].set_ylim(-1.5, 3.2); axs[1].set_ylabel("hidden state h_t"); axs[1].legend(fontsize=8, loc="upper right")
        axs[2].plot(range(f + 1), y[:f + 1], color=C[3], lw=2); axs[2].set_ylim(-1.3, 2.5); axs[2].set_ylabel("output y_t = <C, h_t>")
        axs[2].set_xlabel("time step t   (slow-decay state remembers old inputs; fast-decay forgets)")
    anim = FuncAnimation(fig, draw, frames=T, interval=150)
    anim.save(OUT / "08_ssm_state_recurrence.gif", writer=PillowWriter(fps=7), dpi=66); plt.close(fig)


# ---------------------------------------------------------------- 09 SSM <-> attention equivalence
def fig_equivalence():
    x, a, B, C_ = ss.make_inputs(T=16, N=4, selective=True, seed=0)
    M = ss.matrix_form(a, B, C_)
    y1 = ss.recurrent_scan(x, a, B, C_); y2 = M @ x; y3 = ss.parallel_scan(x, a, B, C_)
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.2), gridspec_kw={"width_ratios": [1, 1.3, 1], "wspace": 0.4})
    im = axs[0].imshow(M, cmap="RdBu_r", vmin=-np.abs(M).max(), vmax=np.abs(M).max())
    axs[0].set_title("SSM as a masked, decaying 'attention' matrix\nM[t,s] = <C_t, (a_(s+1)...a_t) * B_s>"); axs[0].set_xlabel("source step s"); axs[0].set_ylabel("output step t")
    fig.colorbar(im, ax=axs[0], fraction=.046)
    axs[1].plot(y1, "o-", color=C[0], label="recurrent loop  O(T), state size N", lw=3)
    axs[1].plot(y2, "s--", color=C[1], label="matrix form  O(T^2)")
    axs[1].plot(y3, "^:", color=C[2], label="parallel scan  O(T log T) work, log depth")
    axs[1].set_title("Three computations, one output sequence"); axs[1].set_xlabel("t"); axs[1].set_ylabel("y_t"); axs[1].legend(fontsize=8.5)
    err = [np.abs(y2 - y1).max(), np.abs(y3 - y1).max()]
    axs[2].bar(["matrix vs\nrecurrent", "scan vs\nrecurrent"], err, color=[C[1], C[2]]); axs[2].set_yscale("log"); axs[2].set_ylim(1e-18, 1e-6)
    axs[2].set_ylabel("max |difference|"); axs[2].set_title("Floating-point agreement")
    fig.savefig(OUT / "09_ssm_attention_equivalence.png"); plt.close(fig)


# ---------------------------------------------------------------- 10 KV cache vs context by architecture
def fig_cache_by_arch():
    names = ["Dense MHA (illustrative)", "Dense GQA-8 (illustrative)", "Dense MQA (illustrative)", "SWA (illustrative)",
             "Hybrid 1A:7S (illustrative)", "Pure SSM (illustrative)"]
    n = np.logspace(3, 6.1, 40)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for nm, c in zip(names, C):
        s = am.PRESETS[nm]
        ax.loglog(n, [am.cache_bytes(s, int(v)) / 2**30 for v in n], lw=2.2, color=c, label=nm.replace(" (illustrative)", ""))
    ax.axhline(80, color="#333", ls=":"); ax.text(1.2e3, 95, "80 GB GPU (weights not counted)", fontsize=9)
    ax.set_xlabel("context length n (tokens)"); ax.set_ylabel("decode memory per sequence (GiB, fp16)")
    ax.set_title("Per-sequence KV-cache / state memory by architecture (32-layer, d=4096 archetypes)"); ax.legend(fontsize=9)
    fig.savefig(OUT / "10_cache_memory_by_architecture.png"); plt.close(fig)


# ---------------------------------------------------------------- 11 comparison matrix
def fig_matrix():
    rows = am.default_matrix(8192)
    cols = [("total_params_B", "Total\nparams (B)"), ("active_params_B", "Active\nparams (B)"), ("kv_kib_per_token", "KV KiB\n/token"),
            ("cache_gib_at_n", "Cache GiB\n@ n=8192"), ("gflops_per_token", "GFLOPs/token\n@ n=8192"), ("context", "Context\n(tokens)")]
    data = np.array([[r[k] for k, _ in cols] for r in rows], float)
    norm = np.log10(data + 1e-3); norm = (norm - norm.min(0)) / (norm.max(0) - norm.min(0) + 1e-9)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.imshow(norm, cmap="Blues", aspect="auto", vmin=0, vmax=1.25)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = data[i, j]; ax.text(j, i, f"{v:,.0f}" if v >= 100 else f"{v:,.2f}", ha="center", va="center", fontsize=9, color="white" if norm[i, j] > .6 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for _, c in cols], fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["name"] for r in rows], fontsize=9)
    ax.xaxis.tick_top(); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("Architecture comparison matrix  (case-study rows are REVIEW_REQUIRED; shading = log-scale within column)", pad=44, loc="left")
    fig.savefig(OUT / "11_comparison_matrix.png"); plt.close(fig)
    (OUT / "comparison_matrix.html").write_text(am.to_html(rows, 8192), encoding="utf-8")


if __name__ == "__main__":
    for f in [fig_family_tree, fig_three_families, fig_pattern_gallery, fig_rope_scaling, fig_moe_params, gif_moe_routing,
              fig_cost_curves, gif_ssm, fig_equivalence, fig_cache_by_arch, fig_matrix]:
        f(); print("ok", f.__name__)
