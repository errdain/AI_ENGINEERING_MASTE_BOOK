"""Generate Tier 7 figures + GIFs.  Run:  python scripts/make_figures.py"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lab"))
from attention_np import (softmax, causal_mask, padding_mask, sliding_window_mask,
                          scaled_dot_product_attention, kv_cache_bytes, score_matrix_bytes, tiled_attention)

OUT = HERE.parent / "figures"; OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})
TOK = ["The", "cat", "sat", "on", "the", "mat"]
rng = np.random.default_rng(7)


def toy_qkv(n=6, d=8, seed=3):
    r = np.random.default_rng(seed)
    return [r.normal(size=(n, d)) for _ in range(3)]


def save(fig, name):
    fig.savefig(OUT / name, dpi=100); plt.close(fig)


def heat(ax, M, xl=None, yl=None, title="", cmap="Blues", vmin=0, vmax=None, annot=True, fs=8):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    if xl is not None: ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=45, ha="right")
    if yl is not None: ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl)
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs,
                        color="white" if v > (vmax or M.max()) * .6 else "#222")
    return im


def fig_pipeline():
    q, k, v = toy_qkv()
    out, w = scaled_dot_product_attention(q, k, v)
    s = q @ k.T / np.sqrt(8)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    heat(axs[0], q @ k.T, TOK, TOK, "1. raw scores  Q·Kᵀ  (T×S)", cmap="RdBu_r", vmin=-8, vmax=8, annot=False)
    heat(axs[1], s, TOK, TOK, "2. scaled  ÷√d_k  (d_k=8)", cmap="RdBu_r", vmin=-8, vmax=8, annot=False)
    heat(axs[2], w, TOK, TOK, "3. softmax per row → weights\n(each row sums to 1)", vmax=1, fs=7)
    heat(axs[3], out[:, :8], None, TOK, "4. output = weights · V  (T×d_v)", cmap="PuOr", vmin=-2, vmax=2, annot=False)
    for a, lab in zip(axs[:3], ["key (what is looked at)"] * 3): a.set_xlabel(lab)
    for a in axs[:3]: a.set_ylabel("query (who is looking)")
    axs[3].set_xlabel("value dimension")
    fig.suptitle("Scaled dot-product attention as four matrix operations (toy random Q, K, V; 6 tokens, d_k = 8)", y=1.02, fontweight="bold")
    fig.tight_layout(); save(fig, "01_attention_pipeline.png")


def fig_saturation():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    dks = [4, 16, 64, 256, 1024]; n = 64
    r = np.random.default_rng(0)
    mx_raw, mx_sc, var_raw = [], [], []
    for d in dks:
        q = r.normal(size=(400, d)); k = r.normal(size=(400, n, d))
        s = np.einsum("bd,bnd->bn", q, k)
        mx_raw.append(softmax(s).max(-1).mean()); mx_sc.append(softmax(s / np.sqrt(d)).max(-1).mean()); var_raw.append(s.var())
    axs[0].loglog(dks, var_raw, "o-", color=C[3], label="unscaled  Var(q·k)")
    axs[0].loglog(dks, [v / d for v, d in zip(var_raw, dks)], "s-", color=C[0], label="scaled by 1/√d_k")
    axs[0].loglog(dks, dks, "--", color="gray", label="y = d_k")
    axs[0].set_xlabel("head dimension d_k"); axs[0].set_ylabel("variance of score"); axs[0].legend(); axs[0].set_title("Score variance grows with d_k")
    axs[1].semilogx(dks, mx_raw, "o-", color=C[3], label="unscaled"); axs[1].semilogx(dks, mx_sc, "s-", color=C[0], label="scaled by 1/√d_k")
    axs[1].set_xlabel("head dimension d_k"); axs[1].set_ylabel("mean max attention weight"); axs[1].set_ylim(0, 1.05)
    axs[1].axhline(1 / n, ls=":", color="gray"); axs[1].text(4, 1 / n + .03, "uniform = 1/64", color="gray")
    axs[1].legend(); axs[1].set_title("Unscaled softmax becomes one-hot")
    x = np.linspace(-6, 6, 200); axs[2].plot(x, 1 / (1 + np.exp(-x)), color=C[0])
    d = x * 0 + 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
    axs[2].plot(x, d, color=C[3], label="gradient  p(1−p)"); axs[2].legend(["softmax weight p of the larger of two scores", "gradient p(1−p) → 0 when saturated"])
    axs[2].set_xlabel("score gap between two keys"); axs[2].set_title("Saturated softmax ⇒ vanishing gradient")
    fig.suptitle("Why divide by √d_k  (random N(0,1) q and k, 64 keys)", y=1.02, fontweight="bold")
    fig.tight_layout(); save(fig, "02_sqrt_dk_saturation.png")


def fig_masks():
    n = 8
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    cm = causal_mask(n).astype(float)
    pm = np.stack([padding_mask([5, 8])[0]] * n).astype(float)
    comb = cm * pm
    sw = sliding_window_mask(n, 3).astype(float)
    for a, M, t in zip(axs, [cm, pm, comb, sw], ["causal (decoder)", "padding (seq length 5 of 8)", "causal AND padding", "sliding window w=3"]):
        a.imshow(M, cmap="Greens", vmin=0, vmax=1.3); a.set_title(t, fontsize=10)
        a.set_xlabel("key position j"); a.set_ylabel("query position i")
        a.set_xticks(range(n)); a.set_yticks(range(n))
        for i in range(n):
            for j in range(n):
                a.text(j, i, "✓" if M[i, j] else "✗", ha="center", va="center", fontsize=8, color="white" if M[i, j] else "#bbb")
    fig.suptitle("Boolean attention masks (green ✓ = may attend). Masked scores are set to −∞ before softmax", y=1.02, fontweight="bold")
    fig.tight_layout(); save(fig, "03_attention_masks.png")


def fig_heads():
    # 4 heads with different structured Q/K to illustrate what heads *can* specialise in (hand-built, NOT learned)
    n = 10
    i = np.arange(n)
    prev = np.where((i[:, None] - i[None, :]) == 1, 6., 0.); prev[0, 0] = 6
    local = -np.abs(i[:, None] - i[None, :]) * 1.2
    first = np.zeros((n, n)); first[:, 0] = 5
    rr = np.random.default_rng(2); content = rr.normal(size=(n, 4)) @ rr.normal(size=(4, n)) * 1.5
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for a, s, t in zip(axs, [prev, local, first, content], ["head 0: previous token", "head 1: local window", "head 2: first token (sink-like)", "head 3: content-based"]):
        s = np.where(causal_mask(n), s, -1e9); w = softmax(s)
        a.imshow(w, cmap="Blues", vmin=0, vmax=1); a.set_title(t, fontsize=10)
        a.set_xlabel("key position"); a.set_ylabel("query position")
    fig.suptitle("Multi-head attention = several attention patterns in parallel (hand-built illustrations, not learned weights)", y=1.02, fontweight="bold")
    fig.tight_layout(); save(fig, "04_multihead_patterns.png")


def fig_mqa_gqa():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    H = 8
    for a, (name, hk) in zip(axs, [("MHA: 8 Q heads, 8 KV heads", 8), ("GQA: 8 Q heads, 2 KV groups", 2), ("MQA: 8 Q heads, 1 KV head", 1)]):
        a.set_xlim(-.5, H - .5); a.set_ylim(-.2, 2.6); a.axis("off"); a.set_title(name, fontsize=11)
        g = H // hk
        for q in range(H):
            a.add_patch(FancyBboxPatch((q - .4, 1.9), .8, .5, boxstyle="round,pad=0.02", fc=C[0], ec="white"))
            a.text(q, 2.15, f"Q{q}", color="white", ha="center", va="center", fontsize=9, weight="bold")
            kv = q // g
            a.plot([q, kv * g + (g - 1) / 2], [1.9, .7], color=C[kv % 8], lw=1.3, alpha=.8)
        for kv in range(hk):
            cx = kv * g + (g - 1) / 2; w = max(.8, g * .8 - .1)
            a.add_patch(FancyBboxPatch((cx - w / 2, .2), w, .5, boxstyle="round,pad=0.02", fc=C[kv % 8], ec="white"))
            a.text(cx, .45, f"K/V{kv}" if hk > 1 else "K/V (shared)", color="white", ha="center", va="center", fontsize=9, weight="bold")
        a.text(H / 2 - .5, -.1, f"KV cache = {hk}/{H} of MHA", ha="center", fontsize=10, color=C[3], weight="bold")
    fig.suptitle("Sharing K/V across query heads shrinks the KV cache (query heads stay independent)", y=1.02, fontweight="bold")
    fig.tight_layout(); save(fig, "05_mha_mqa_gqa.png")


def fig_complexity():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    n = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
    axs[0].loglog(n, [score_matrix_bytes(int(x), 32) / 2**30 for x in n], "o-", color=C[3], label="score matrix, 32 heads, fp16 (1 layer, batch 1)")
    axs[0].loglog(n, [kv_cache_bytes(32, 32, 128, int(x)) / 2**30 for x in n], "s-", color=C[0], label="KV cache, 32 layers MHA, d=128")
    axs[0].loglog(n, [kv_cache_bytes(32, 8, 128, int(x)) / 2**30 for x in n], "^-", color=C[2], label="KV cache, same model with GQA (8 KV heads)")
    axs[0].axhline(80, ls=":", color="gray"); axs[0].text(520, 90, "80 GB (typical large accelerator)", color="gray", fontsize=9)
    axs[0].set_xlabel("sequence length n"); axs[0].set_ylabel("GiB"); axs[0].legend(fontsize=8); axs[0].set_title("Quadratic scores vs linear KV cache")
    axs[1].plot(n[:6], n[:6].astype(float) ** 2 / 1e6, color=C[3]); axs[1].set_xlabel("sequence length n"); axs[1].set_ylabel("score entries per head (millions)")
    axs[1].set_title("O(n²): doubling n ⇒ 4× entries")
    cfgs = [("MHA\n32 KV", 32), ("GQA\n8 KV", 8), ("GQA\n4 KV", 4), ("MQA\n1 KV", 1)]
    axs[2].bar([c[0] for c in cfgs], [kv_cache_bytes(32, c[1], 128, 8192) / 2**30 for c in cfgs], color=[C[3], C[1], C[1], C[2]])
    axs[2].set_ylabel("GiB per sequence"); axs[2].set_title("KV cache at n = 8192 (32 layers, d=128, fp16)")
    fig.suptitle("Attention cost. Illustrative shapes (not a specific model); formulas are exact, sizes follow from them", y=1.02, fontweight="bold")
    fig.tight_layout(); save(fig, "06_attention_complexity_memory.png")


def fig_patterns():
    n = 16
    pats = [("full causal", causal_mask(n)), ("sliding window w=4", sliding_window_mask(n, 4)),
            ("window w=4 + 2 sink tokens", sliding_window_mask(n, 4, sinks=2)),
            ("bidirectional (encoder)", np.ones((n, n), bool))]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for a, (t, M) in zip(axs, pats):
        a.imshow(M, cmap="Blues", vmin=0, vmax=1.4); a.set_title(f"{t}\n{int(M.sum())} of {n*n} pairs", fontsize=10)
        a.set_xlabel("key j"); a.set_ylabel("query i")
    fig.suptitle("Which (query, key) pairs are computed. Sliding window makes cost O(n·w); sinks keep the first tokens visible", y=1.02, fontweight="bold")
    fig.tight_layout(); save(fig, "07_attention_patterns_window_sinks.png")


def gif_stepthrough():
    q, k, v = toy_qkv(); d = 8
    s = q @ k.T / np.sqrt(d); w = softmax(np.where(causal_mask(6), s, -1e30)); out = w @ v
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    def draw(f):
        i = f
        for a in axs: a.clear()
        axs[0].barh(range(6)[::-1], np.where(causal_mask(6)[i], s[i], 0.0), color=[C[0] if j <= i else "#ddd" for j in range(6)])
        axs[0].set_yticks(range(6)); axs[0].set_yticklabels(TOK[::-1]); axs[0].set_title(f"scores for query '{TOK[i]}' (÷√d_k)\nfuture keys masked", fontsize=10); axs[0].set_xlabel("q·k / √d_k")
        axs[0].set_xlim(-2, 2); axs[0].axvline(0, color="gray", lw=.8)
        axs[1].barh(range(6)[::-1], w[i], color=[C[1]] * 6); axs[1].set_xlim(0, 1)
        axs[1].set_yticks(range(6)); axs[1].set_yticklabels(TOK[::-1]); axs[1].set_title("softmax → attention weights", fontsize=10); axs[1].set_xlabel("weight (sum = 1)")
        axs[2].imshow(np.vstack([out[:i + 1]]), cmap="PuOr", vmin=-2, vmax=2, aspect="auto"); axs[2].set_yticks(range(i + 1)); axs[2].set_yticklabels(TOK[:i + 1])
        axs[2].set_title("outputs so far = Σ weight·V", fontsize=10); axs[2].set_xlabel("value dimension")
        fig.suptitle(f"Step {i+1}/6 — causal self-attention, one query at a time", fontweight="bold")
        fig.subplots_adjust(top=.74, bottom=.2, left=.08, right=.97, wspace=.45)
    ani = FuncAnimation(fig, draw, frames=6, interval=1100)
    ani.save(OUT / "08_attention_stepthrough.gif", writer=PillowWriter(fps=1), dpi=60); plt.close(fig)


def gif_temperature():
    q, k, v = toy_qkv(8, 16, seed=5); q *= 1.5
    mults = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 8.0]
    base = 1 / np.sqrt(16); toks = ["w%d" % i for i in range(8)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    def draw(f):
        m = mults[f]; w = softmax(q @ k.T * base * m)
        for a in axs: a.clear()
        axs[0].imshow(w, cmap="Blues", vmin=0, vmax=1); axs[0].set_xlabel("key"); axs[0].set_ylabel("query"); axs[0].set_title("attention weights", fontsize=10)
        ent = -(w * np.log(w + 1e-12)).sum(-1).mean()
        axs[1].bar(range(8), w[3], color=C[0]); axs[1].set_ylim(0, 1); axs[1].set_xlabel("key"); axs[1].set_ylabel("weight"); axs[1].set_title("row for query 3", fontsize=10)
        fig.suptitle(f"Score multiplier x{m:g} on 1/sqrt(d_k)\nmean row entropy {ent:.2f} nats (uniform = {np.log(8):.2f})", fontweight="bold", fontsize=11)
        axs[1].text(.97, .85, "diffuse (near-uniform)" if m < .2 else "saturated (one-hot)" if m > 3 else "", transform=axs[1].transAxes, ha="right", color=C[3])
        fig.subplots_adjust(top=.76, bottom=.18, left=.08, right=.97, wspace=.3)
    ani = FuncAnimation(fig, draw, frames=len(mults), interval=900)
    ani.save(OUT / "09_softmax_temperature.gif", writer=PillowWriter(fps=1.2), dpi=60); plt.close(fig)


def gif_decode_cache():
    n = 8; fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    def draw(t):
        for a in axs: a.clear()
        M = np.zeros((n, n)); M[:t + 1, :t + 1] = np.tril(np.ones((t + 1, t + 1)))
        axs[0].imshow(M, cmap="Blues", vmin=0, vmax=1.4); axs[0].set_xlabel("key j"); axs[0].set_ylabel("query i")
        axs[0].add_patch(plt.Rectangle((-.5, t - .5), t + 1, 1, fill=False, ec=C[3], lw=2.5))
        axs[0].set_title("causal attention matrix; red = the one new row", fontsize=10)
        axs[1].bar(range(n), [1 if j <= t else 0 for j in range(n)], color=[C[2] if j < t else C[3] for j in range(n)])
        axs[1].set_xticks(range(n)); axs[1].set_xlabel("token position"); axs[1].set_ylim(0, 1.3); axs[1].set_yticks([])
        axs[1].set_title(f"KV cache holds {t+1} entries (green = reused, red = new)", fontsize=10)
        fig.suptitle(f"Decoding token {t+1}: one new query row against cached K,V", fontweight="bold", fontsize=11)
        fig.subplots_adjust(top=.8, bottom=.18, left=.08, right=.97, wspace=.25)
    ani = FuncAnimation(fig, draw, frames=n, interval=800)
    ani.save(OUT / "10_decode_kv_cache.gif", writer=PillowWriter(fps=1.2), dpi=60); plt.close(fig)


def fig_qkv_projections():
    """Shapes of X, W_q/W_k/W_v and why projections make attention asymmetric."""
    r = np.random.default_rng(5); n, e, d = 6, 8, 4
    X = r.normal(size=(n, e)); Wq, Wk, Wv = [r.normal(size=(e, d)) for _ in range(3)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.1, 1.3, 1.3]})
    ax = axs[0]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", alpha=.85))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    box(0.2, 4, 2.2, 2.4, "X\n(T x d_model)", C[0])
    for y, nm, c in [(7.4, "Q = X W_q\n(T x d_k)", C[3]), (4.3, "K = X W_k\n(S x d_k)", C[1]), (1.2, "V = X W_v\n(S x d_v)", C[2])]:
        box(6.2, y, 3.4, 2.0, nm, c); ax.annotate("", xy=(6.15, y + 1), xytext=(2.5, 5.2), arrowprops=dict(arrowstyle="->", color="#555"))
    ax.text(4.2, 8.7, "W_q", fontsize=9); ax.text(4.2, 5.6, "W_k", fontsize=9); ax.text(4.2, 2.3, "W_v", fontsize=9)
    ax.set_title("1. Three learned linear maps of the same X", fontsize=10)
    G = X @ X.T; S = (X @ Wq) @ (X @ Wk).T / 3
    heat(axs[1], G, TOK, TOK, "2. WITHOUT projections: x_i . x_j\n(symmetric: i->j equals j->i)", cmap="RdBu_r", vmin=-8, vmax=8, annot=False)
    heat(axs[2], S, TOK, TOK, "3. WITH projections: q_i . k_j\n(asymmetric: i->j differs from j->i)", cmap="RdBu_r", vmin=-8, vmax=8, annot=False)
    for a in axs[1:]: a.set_xlabel("key j"); a.set_ylabel("query i")
    fig.suptitle("Q/K/V projections (toy random weights, d_model = 8, d_k = 4)", y=1.03, fontweight="bold")
    fig.tight_layout(); save(fig, "11_qkv_projections.png")


def fig_online_softmax():
    """Tiled / online softmax equals full attention; memory of stored scores."""
    q, k, v = toy_qkv(n=12, d=8, seed=11); M = causal_mask(12)
    ref, _ = scaled_dot_product_attention(q, k, v, M)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    tiles = [1, 2, 3, 4, 6, 12]
    err = [np.abs(tiled_attention(q, k, v, tile=t, mask=M) - ref).max() for t in tiles]
    axs[0].bar([str(t) for t in tiles], [max(e_, 1e-17) for e_ in err], color=C[2]); axs[0].set_yscale("log"); axs[0].set_ylim(1e-17, 1e-9)
    axs[0].set_xlabel("key tile size"); axs[0].set_ylabel("max |tiled - full|  (log)")
    axs[0].set_title("1. Same answer for every tile size\n(floating-point rounding only)", fontsize=10)
    # running max / denominator for one query row across 4 tiles
    i = 11; sc = (q[i] @ k.T) / np.sqrt(8); m = -np.inf; l = 0.0; ms, ls = [], []
    for lo in range(0, 12, 3):
        t_ = sc[lo:lo + 3]; mn = max(m, t_.max()); l = l * np.exp(m - mn) + np.exp(t_ - mn).sum(); m = mn; ms.append(m); ls.append(l)
    ax = axs[1]; ax.plot(range(1, 5), ms, "o-", color=C[0], label="running max m"); ax.set_xlabel("tiles processed (3 keys each)"); ax.set_ylabel("running max m", color=C[0])
    ax2 = ax.twinx(); ax2.plot(range(1, 5), ls, "s--", color=C[3], label="running denominator l"); ax2.set_ylabel("running denominator l", color=C[3]); ax2.spines["right"].set_visible(True)
    ax.set_xticks(range(1, 5)); ax.set_title("2. State kept per query row: (m, l, acc)\nrescaled by exp(m_old - m_new) each tile", fontsize=10)
    ns = np.array([512, 1024, 2048, 4096, 8192, 16384]); naive = ns.astype(float) ** 2 * 2 / 2**20
    axs[2].loglog(ns, naive, "o-", color=C[3], label="naive: n x n scores, fp16 (MiB per head)")
    axs[2].loglog(ns, ns * 2 * 2 / 2**20 * 2, "o-", color=C[2], label="tiled: O(n) running stats + output (MiB per head, illustrative)")
    axs[2].set_xlabel("sequence length n"); axs[2].set_ylabel("extra activation memory (MiB)"); axs[2].legend(fontsize=7, loc="upper left")
    axs[2].set_title("3. Never storing the n x n matrix", fontsize=10)
    fig.suptitle("FlashAttention idea: exact attention by tiles with an online softmax (NumPy teaching version from lab/attention_np.py)", y=1.03, fontweight="bold")
    fig.tight_layout(); save(fig, "12_online_softmax_tiling.png")


def fig_cross_attention():
    """Cross-attention: T x S non-square weights, queries from decoder, keys/values from encoder."""
    r = np.random.default_rng(2); src = ["Le", "chat", "est", "assis", "."]; tgt = ["The", "cat", "is", "sitting", "down", "."]
    q = r.normal(size=(6, 8)); k = r.normal(size=(5, 8)); v = r.normal(size=(5, 8))
    bias = np.zeros((6, 5)); 
    for a, b in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 4)]: bias[a, b] = 2.5
    sc = q @ k.T / np.sqrt(8) + bias; w = softmax(sc)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    heat(axs[0], w, src, tgt, "Cross-attention weights (T=6 target x S=5 source)\nrows sum to 1; matrix is NOT square", vmax=1, fs=8)
    axs[0].set_xlabel("key j: encoder (source) token"); axs[0].set_ylabel("query i: decoder (target) token")
    ax = axs[1]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    for x, y, t, c in [(0.3, 6, "Decoder states\n(T x d)  -> Q", C[3]), (0.3, 1.5, "Encoder output\n(S x d)  -> K, V", C[1]), (6.0, 3.8, "Output\n(T x d_v)", C[2])]:
        ax.add_patch(FancyBboxPatch((x, y), 3.6, 2.2, boxstyle="round,pad=0.05", fc=c, ec="#333", alpha=.85)); ax.text(x + 1.8, y + 1.1, t, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(5.9, 5.0), xytext=(3.95, 7.0), arrowprops=dict(arrowstyle="->")); ax.annotate("", xy=(5.9, 4.6), xytext=(3.95, 2.6), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Output length = query length T;\nsource length S only sets the key axis", fontsize=10)
    fig.suptitle("Cross-attention shapes (weights hand-biased toward a monotone alignment for illustration; not learned)", y=1.03, fontweight="bold")
    fig.tight_layout(); save(fig, "13_cross_attention_shapes.png")


if __name__ == "__main__":
    for f in (fig_pipeline, fig_saturation, fig_masks, fig_heads, fig_mqa_gqa, fig_complexity, fig_patterns, fig_qkv_projections, fig_online_softmax, fig_cross_attention,
              gif_stepthrough, gif_temperature, gif_decode_cache):
        f(); print("ok", f.__name__)
