"""Generate Tier 6 figures + GIFs.  Run:  python scripts/make_figures.py"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lab"))
from mini_bpe import ByteBPE, merge_pair, pair_counts, _pretokenize
from positional import sinusoidal, apply_rope, rope_angles, alibi_bias, padding_mask

OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def boxes(ax, toks, y, title):
    x = 0.0
    for i, t in enumerate(toks):
        w = max(0.5, 0.33 * len(t) + 0.3)
        ax.add_patch(FancyBboxPatch((x, y), w, 0.7, boxstyle="round,pad=0.02", fc=C[i % 8], ec="white", alpha=.9))
        ax.text(x + w / 2, y + .35, t.replace(" ", "␣"), ha="center", va="center", color="white", fontsize=10, weight="bold")
        x += w + 0.08
    ax.text(-0.2, y + .35, f"{title}\n({len(toks)} tokens)", ha="right", va="center", fontsize=10)
    return x


def fig_granularity():
    s = "unhappiness"
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off")
    rows = [("Character", list(s)), ("Word", [s]), ("Byte (UTF-8)", [f"{b:02x}" for b in s.encode()]),
            ("Subword (BPE)", ["un", "happi", "ness"])]
    xmax = 0
    for k, (n, t) in enumerate(rows):
        xmax = max(xmax, boxes(ax, t, 3.2 - k * 1.0, n))
    ax.set_xlim(-3.2, xmax + .3); ax.set_ylim(-.3, 4.2)
    ax.set_title("Same text, four tokenization granularities  ·  trade-off: sequence length vs. vocabulary size", loc="left")
    fig.savefig(OUT / "01_tokenization_granularity.png"); plt.close(fig)


def fig_bpe_gif():
    corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
    words = _pretokenize(" " + corpus)
    from collections import Counter
    wc = Counter(words)
    keys = list(wc)
    seqs = [[b + 4 for b in w.encode()] for w in keys]
    wts = [wc[k] for k in keys]
    names = {i + 4: chr(i) if 32 < i < 127 else "␣" for i in range(256)}
    frames = [(None, [s[:] for s in seqs], None)]
    nxt = 260
    for _ in range(9):
        cnt = pair_counts(seqs, wts)
        best = max(cnt.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        names[nxt] = names[best[0]] + names[best[1]]
        seqs = [merge_pair(s, best, nxt) for s in seqs]
        frames.append((best, [s[:] for s in seqs], (names[best[0]], names[best[1]], cnt[best], nxt)))
        nxt += 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    def draw(f):
        a1.clear(); a2.clear(); a1.axis("off")
        best, sq, info = frames[f]
        for r, (k, s) in enumerate(zip(keys, sq)):
            x = 0
            for j, t in enumerate(s):
                lab = names[t]; w = .45 * len(lab) + .25
                new = info is not None and t == info[3]
                a1.add_patch(FancyBboxPatch((x, 5.2 - r * 1.15), w, .8, boxstyle="round,pad=.02",
                             fc=C[3] if new else C[0], ec="white", alpha=.95))
                a1.text(x + w / 2, 5.6 - r * 1.15, lab, ha="center", va="center", color="white", weight="bold")
                x += w + .06
            a1.text(x + .3, 5.6 - r * 1.15, f"×{wts[r]}", va="center", color="gray")
        a1.set_xlim(0, 9); a1.set_ylim(-.3, 6.4)
        ttl = "Start: every word is a sequence of bytes" if info is None else \
              f"Merge #{f}: '{info[0]}' + '{info[1]}' → '{info[0]+info[1]}'  (count {info[2]})"
        a1.set_title(ttl, loc="left", fontsize=11)
        cnt = pair_counts(sq, wts).most_common(6)
        lbl = [names[a].strip() + "|" + names[b].strip() for (a, b), _ in cnt]
        a2.barh(range(len(cnt))[::-1], [c for _, c in cnt], color=[C[3] if i == 0 else C[7] for i in range(len(cnt))])
        a2.set_yticks(range(len(cnt))[::-1]); a2.set_yticklabels(lbl)
        a2.set_title("Adjacent-pair counts (top = next merge)", fontsize=10); a2.set_xlabel("count")
        total = sum(len(s) * w for s, w in zip(sq, wts))
        fig.suptitle(f"Byte-pair encoding, step by step  ·  total tokens in corpus: {total}", fontweight="bold")
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1400)
    anim.save(OUT / "02_bpe_training.gif", writer=PillowWriter(fps=0.8)); plt.close(fig)


def fig_embedding_lookup():
    rng = np.random.default_rng(1)
    V, d = 8, 4; E = rng.normal(size=(V, d)).round(1)
    ids = [2, 5, 2]; toks = ["the", "cat", "the"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(E, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    for i in range(V):
        for j in range(d):
            ax.text(j, i, E[i, j], ha="center", va="center", fontsize=9)
    ax.set_yticks(range(V)); ax.set_yticklabels([f"id {i}" for i in range(V)])
    ax.set_xticks(range(d)); ax.set_xticklabels([f"dim {j}" for j in range(d)])
    ax.set_title(f"Embedding matrix E  (vocab V={V} × d_model={d}): a token id selects ONE row")
    for k, (i, t) in enumerate(zip(ids, toks)):
        ax.annotate(f'"{t}"→ id {i}', xy=(d - .5, i), xytext=(d + .6, i - .4 + k * .08), color=C[3], weight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=C[3]), annotation_clip=False)
    ax.text(d + .6, V - .2, "Same id → same row\n(context-free at this stage)", color="gray", fontsize=9)
    fig.savefig(OUT / "03_embedding_matrix_lookup.png"); plt.close(fig)


def fig_positional():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    pe = sinusoidal(64, 64)
    axs[0].imshow(pe, cmap="RdBu_r", aspect="auto"); axs[0].set_title("Sinusoidal table (64 pos × 64 dims)")
    axs[0].set_xlabel("dimension (fast → slow wavelength)"); axs[0].set_ylabel("position")
    for d, c in zip([0, 8, 24, 48], C): axs[1].plot(pe[:, d], color=c, label=f"dim {d}")
    axs[1].legend(fontsize=8); axs[1].set_title("Individual dims = sine waves of different period"); axs[1].set_xlabel("position")
    sim = pe @ pe.T
    axs[2].imshow(sim / sim.max(), cmap="viridis"); axs[2].set_title("Dot-product of PEs: peaks on diagonal,\nfalls with distance")
    axs[2].set_xlabel("position j"); axs[2].set_ylabel("position i")
    fig.savefig(OUT / "04_sinusoidal_positional_encoding.png"); plt.close(fig)


def fig_rope_gif():
    base_vec = np.array([1.0, 0.0]); thetas = [1.0, 0.25, 0.0625]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), subplot_kw={"aspect": "equal"})
    def draw(m):
        for ax, th, c in zip(axs, thetas, C):
            ax.clear(); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="lightgray"))
            for p in range(m + 1):
                a = p * th; ax.plot([0, np.cos(a)], [0, np.sin(a)], color=c, alpha=.15 + .85 * (p == m), lw=2 if p == m else 1)
            ax.set_title(f"dim-pair freq θ={th}\nangle = m·θ = {m*th:.2f} rad", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"RoPE: position m = {m} rotates each 2-D pair by m·θᵢ  (fast pairs spin, slow pairs barely move)", fontweight="bold")
    FuncAnimation(fig, draw, frames=range(0, 13), interval=700).save(OUT / "05_rope_rotation.gif", writer=PillowWriter(fps=1.5))
    plt.close(fig)


def fig_alibi_masks():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    b = alibi_bias(12, .5); bb = np.where(np.isinf(b), np.nan, b)
    im = axs[0].imshow(bb, cmap="magma_r"); axs[0].set_title("ALiBi bias (slope .5): linear distance penalty"); plt.colorbar(im, ax=axs[0], fraction=.046)
    causal = np.tril(np.ones((12, 12))); axs[1].imshow(causal, cmap="Blues"); axs[1].set_title("Causal mask (1 = may attend)")
    pm = padding_mask([5, 8]); axs[2].imshow(pm, cmap="Greens", aspect="auto"); axs[2].set_title("Padding mask, batch of 2 (len 5 and 8)")
    axs[2].set_yticks([0, 1]); axs[2].set_yticklabels(["seq A", "seq B"]); axs[2].set_xlabel("token position")
    fig.savefig(OUT / "06_alibi_causal_padding_masks.png"); plt.close(fig)


def fig_vocab_tradeoff():
    V = np.logspace(2.5, 6, 200); d = 4096; text_tokens = 1_000_000 * (1 + 3.5 / np.log10(V) ** 2.2)
    seq = text_tokens / text_tokens[np.argmin(abs(V - 32000))]
    emb = V * d / 1e9
    fig, ax = plt.subplots(figsize=(9, 4.4)); ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax.plot(V, seq, color=C[0], lw=3, label="tokens for same text (relative)")
    ax2.plot(V, emb, color=C[1], lw=3, label="embedding params (B, d=4096)")
    ax.set_xscale("log"); ax2.set_yscale("log"); ax.set_xlabel("vocabulary size V")
    ax.set_ylabel("sequence length (rel. to 32k vocab)", color=C[0]); ax2.set_ylabel("embedding table params (billions)", color=C[1])
    ax.set_title("Bigger vocab → shorter sequences, but a bigger embedding/softmax table\n(illustrative curve; see interactive/vocab_tradeoff.html)")
    for v in (32_000, 128_000): ax.axvline(v, color="gray", ls=":")
    fig.savefig(OUT / "07_vocab_size_tradeoff.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_granularity, fig_bpe_gif, fig_embedding_lookup, fig_positional, fig_rope_gif, fig_alibi_masks, fig_vocab_tradeoff):
        f(); print("ok", f.__name__)
