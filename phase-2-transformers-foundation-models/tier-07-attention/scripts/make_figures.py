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
                          scaled_dot_product_attention, kv_cache_bytes, score_matrix_bytes)

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


if __name__ == "__main__":
    for f in (fig_pipeline, fig_saturation, fig_masks, fig_heads, fig_mqa_gqa, fig_complexity, fig_patterns,
              gif_stepthrough, gif_temperature, gif_decode_cache):
        f(); print("ok", f.__name__)
