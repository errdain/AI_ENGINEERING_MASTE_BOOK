"""Generate Tier 11 figures + GIFs.  Run:  python figures/src/make_figures.py   (CPU, ~1 min)"""
import sys, pathlib, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, FancyBboxPatch

torch.set_num_threads(2)
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lab"))
from tiny_gpt import TinyGPT
from decoding import generate, top_k_filter, top_p_filter
from quant import quant_error
from scheduler import Req, simulate, metrics
from kv_memory import kv_bytes

OUT = HERE.parents[1] / "figures"
OUT.mkdir(exist_ok=True)
C = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})
WORDS = ["the", "a", "of", "and", "cat", "dog", "sat", "ran", "on", "mat", "hat", "sun", "blue", "tree", "zebra", "ink"]
LOGITS = np.array([3.2, 2.9, 2.0, 1.4, 1.2, 0.9, 0.5, 0.3, 0.0, -0.2, -0.6, -0.9, -1.2, -1.6, -2.0, -2.5])


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def save(fig, name):
    fig.savefig(OUT / name, dpi=100); plt.close(fig)


def fig_decoding_reshape():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    p0 = softmax(LOGITS)
    cfg = [("raw softmax (T=1)", p0, None),
           ("temperature T=0.5 (sharper)", softmax(LOGITS / .5), None),
           ("top-k = 4", softmax(top_k_filter(torch.tensor(LOGITS), 4).numpy()), None),
           ("top-p = 0.8 (nucleus)", softmax(top_p_filter(torch.tensor(LOGITS), .8).numpy()), 0.8)]
    for ax, (t, p, nuc) in zip(axs, cfg):
        keep = p > 0
        ax.bar(range(16), p0, fill=False, ec="gray", lw=1, label="raw p")
        ax.bar(range(16), p, color=[C[0] if k else "#ccc" for k in keep], alpha=.9, label="after filter")
        ax.set_title(t, fontsize=10); ax.set_xticks(range(16)); ax.set_xticklabels(WORDS, rotation=70, fontsize=7)
        ax.text(.97, .93, f"kept {int(keep.sum())}/16", transform=ax.transAxes, ha="right")
        if nuc:
            ax.text(.97, .83, f"mass of kept (raw) = {p0[keep].sum():.2f}", transform=ax.transAxes, ha="right", fontsize=8)
    axs[0].set_ylabel("probability"); axs[0].legend(loc="center right", fontsize=8)
    fig.suptitle("Figure 1 - Same logits, four decoding transforms: temperature rescales, top-k keeps a fixed count, top-p keeps a probability mass", fontsize=10, y=1.03)
    save(fig, "01_decoding_reshape.png")


def gif_temperature():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    Ts = list(np.linspace(0.15, 2.5, 18)) + [2.5] * 2
    def upd(i):
        ax.clear(); T = Ts[i]; p = softmax(LOGITS / T)
        order = np.argsort(-p); cum = np.cumsum(p[order]); k = int(np.searchsorted(cum, 0.9) + 1)
        col = [C[0] if r < k else "#ccc" for r in range(16)]
        ax.bar(range(16), p[order], color=col)
        ax.set_xticks(range(16)); ax.set_xticklabels([WORDS[j] for j in order], rotation=70, fontsize=8)
        ax.set_ylim(0, 1); ax.set_ylabel("probability"); ax.set_xlabel("tokens sorted by probability")
        H = -(p * np.log2(p)).sum()
        ax.set_title(f"Temperature T = {T:.2f}   entropy {H:.2f} bits   top-p(0.9) nucleus = {k} tokens (blue)")
        ax.text(.98, .9, "low T: peaked, nucleus small\nhigh T: flat, nucleus large", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    FuncAnimation(fig, upd, frames=len(Ts)).save(OUT / "02_temperature_nucleus.gif", writer=PillowWriter(fps=3), dpi=60)
    plt.close(fig)


def fig_beam_tree():
    # Illustrative conditional probabilities (made up for teaching): greedy picks a locally best first token that leads to a poor continuation.
    tree = {"": [("A", .5), ("B", .4), ("C", .1)],
            "A": [("x", .35), ("y", .33), ("z", .32)],
            "B": [("x", .9), ("y", .05), ("z", .05)],
            "C": [("x", .4), ("y", .3), ("z", .3)]}
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.3, 5)
    ax.text(.1, 4.6, "start", fontsize=10, weight="bold")
    ys1 = {"A": 3.4, "B": 2.2, "C": 1.0}
    best = (None, -9)
    for a, (_, pa) in zip("ABC", [(t[0], t[1]) for t in tree[""]]):
        ax.annotate("", (2.55, ys1[a]), (.9, 4.55), arrowprops=dict(arrowstyle="->", color="gray"))
        ax.text(3.0, ys1[a], f"{a}\np={pa}", ha="center", va="center", bbox=dict(boxstyle="round", fc=C[0] if a != "C" else "#ddd", ec="k", alpha=.8), color="white" if a != "C" else "k")
        for j, (w, pw_) in enumerate(tree[a]):
            y = ys1[a] + (j - 1) * .33
            joint = pa * pw_
            ax.annotate("", (6.5, y), (3.45, ys1[a]), arrowprops=dict(arrowstyle="->", color="lightgray"))
            top = joint == max(.5 * .35, .4 * .9)
            kept = a != "C"
            ax.text(6.6, y, f"{a}{w}  joint={joint:.3f}", va="center", fontsize=9, weight="bold" if joint in (.4 * .9, .5 * .35) else None,
                    color=C[3] if joint == .4 * .9 else ("k" if kept else "gray"))
    ax.text(.1, 0.1, "Beam width 2 keeps A and B after step 1 (C is pruned, grey). Greedy takes A (p=0.5) then x -> joint 0.175.\n"
            "Beam search finds B,x with joint 0.36 (red): a better *sequence* that greedy missed. Probabilities are illustrative.", fontsize=9)
    ax.set_title("Figure 3 - Beam search keeps the best-scoring partial sequences, not just the best next token", loc="left")
    save(fig, "03_beam_search_tree.png")


def fig_prefill_decode():
    P, BW, N, by = 300e12, 2e12, 8e9, 2
    bytes_ = N * by
    tp = max(2 * N * 1024 / P, bytes_ / BW) * 1e3
    td = max(2 * N / P, bytes_ / BW) * 1e3
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    ax.add_patch(Rectangle((.3, 1.8), 3.2, .9, fc=C[0])); ax.text(1.9, 2.25, "PREFILL (1 pass, all 1024 prompt tokens)", ha="center", va="center", color="white", fontsize=8.5)
    for i in range(7):
        ax.add_patch(Rectangle((3.6 + i * 1.2, 1.8), 1.1, .9, fc=C[1])); ax.text(4.15 + i * 1.2, 2.25, f"tok {i+1}", ha="center", va="center", fontsize=8)
    ax.annotate("", (3.5, 1.5), (.3, 1.5), arrowprops=dict(arrowstyle="<->")); ax.text(1.9, 1.2, "TTFT (+ queueing, network)", ha="center")
    ax.annotate("", (4.8, 3.2), (3.6, 3.2), arrowprops=dict(arrowstyle="<->")); ax.text(4.2, 3.35, "ITL", ha="center")
    ax.annotate("", (12, 1.5), (3.6, 1.5), arrowprops=dict(arrowstyle="<->")); ax.text(7.8, 1.2, "decode: one token per step -> tokens/s = 1 / ITL", ha="center")
    ax.text(.3, 0.4, f"Illustrative 8B-param fp16 model, 300 TFLOPS / 2 TB/s (REVIEW_REQUIRED hardware): prefill(1024 tok) ~ {tp:.0f} ms (compute-bound), "
            f"decode step ~ {td:.1f} ms (reading {bytes_/1e9:.0f} GB of weights)\n-> ~{1000/td:.0f} tok/s per user at batch 1; the end-to-end time is TTFT + (output tokens - 1) x ITL.", fontsize=8.5)
    ax.set_title("Figure 4 - Prefill is one big parallel pass; decode is many small sequential passes", loc="left")
    save(fig, "04_prefill_vs_decode.png")


def fig_kv_cache():
    torch.manual_seed(0)
    m = TinyGPT(vocab=64, d=128, n_layer=4, n_head=8, n_kv_head=2, max_len=256).eval()
    prompt = list(range(1, 33)); Ns = [8, 24, 48, 80]; ta, tb = [], []
    for n in Ns:
        t = time.perf_counter(); generate(m, prompt, n, use_cache=True, temperature=0); ta.append(time.perf_counter() - t)
        t = time.perf_counter(); generate(m, prompt, n, use_cache=False, temperature=0); tb.append(time.perf_counter() - t)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axs[0]
    ax.plot(Ns, np.array(tb) * 1e3, "o-", c=C[3], label="no cache (recompute all)"); ax.plot(Ns, np.array(ta) * 1e3, "o-", c=C[0], label="KV cache")
    ax.set_xlabel("new tokens generated (32-token prompt)"); ax.set_ylabel("wall time (ms)")
    ax.set_title("Measured on tiny GPT, CPU, 2 threads (noisy)"); ax.legend()
    ax = axs[1]
    S = np.array([512, 2048, 8192, 32768, 131072])
    for kv, c in [(32, C[3]), (8, C[0]), (1, C[2])]:
        ax.plot(S, [kv_bytes(32, kv, 128, s, 1, "fp16") / 2**30 for s in S], "o-", c=c, label=f"{kv} KV heads" + (" (MHA)" if kv == 32 else " (MQA)" if kv == 1 else " (GQA)"))
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length (tokens)"); ax.set_ylabel("KV cache per sequence (GiB)")
    ax.set_title("Formula: 32 layers, head_dim 128, fp16 (illustrative shape)"); ax.legend()
    fig.suptitle("Figure 5 - The KV cache turns O(T^2) recompute into O(T) work per step, at the price of memory that grows with context", fontsize=10, y=1.03)
    save(fig, "05_kv_cache_speed_memory.png")


def gif_paged():
    BS, NB = 4, 14
    script = []  # (seq, action)
    lens = {"A": 0, "B": 0, "C": 0}
    plan = ["A"] * 5 + ["B"] * 3 + ["C"] * 4 + ["A"] * 4 + ["B"] * 5 + ["C"] * 3 + ["FREE_B", "D", "D", "D", "D", "D", "A", "A", "A"]
    table, free, owner, frames = {}, list(range(NB)), {}, []
    ln = {}
    for act in plan + ["HOLD"] * 3:
        if act.startswith("FREE_"):
            s = act[-1]
            for b in table.get(s, []): owner.pop(b); free.append(b)
            free.sort(); table[s] = []; ln[s] = 0; msg = f"request {s} finished -> its blocks return to the free pool"
        elif act == "HOLD":
            msg = "free blocks are reused by any new request: no external fragmentation"
        else:
            s = act; ln[s] = ln.get(s, 0) + 1
            if (ln[s] - 1) % BS == 0:
                b = free.pop(0 if s != "D" else 0); table.setdefault(s, []).append(b); owner[b] = s
                msg = f"request {s}: block full -> allocate physical block {b}"
            else:
                msg = f"request {s}: token {ln[s]} written into its last block"
        frames.append(({k: list(v) for k, v in table.items()}, dict(owner), dict(ln), msg))
    cmap = {"A": C[0], "B": C[1], "C": C[2], "D": C[3]}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    def upd(i):
        t, ow, l, msg = frames[i]; ax.clear(); ax2.clear(); ax.axis("off"); ax2.axis("off")
        ax.set_xlim(0, 7); ax.set_ylim(-1, 2.6)
        for b in range(NB):
            x, y = b % 7, 1.4 - (b // 7) * 1.4
            ax.add_patch(Rectangle((x + .05, y), .9, 1.1, fc="#eee", ec="gray"))
            ax.text(x + .5, y + 1.15, f"phys {b}", ha="center", fontsize=7)
            if b in ow:
                s = ow[b]; idx = t[s].index(b); filled = min(BS, l[s] - idx * BS)
                for c_ in range(BS):
                    ax.add_patch(Rectangle((x + .05, y + c_ * .275), .9, .275, fc=cmap[s] if c_ < filled else "white", ec="gray", lw=.5, alpha=.9))
                ax.text(x + .5, y + .55, s, ha="center", va="center", color="k", fontsize=11, weight="bold")
        ax.set_title("GPU KV memory: physical blocks of 4 tokens (fill from bottom)", fontsize=9)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_title("Block tables (logical -> physical)", fontsize=9)
        for k, s in enumerate(sorted(t)):
            ax2.text(0, .88 - k * .2, f"{s} ({l.get(s,0)} tok): " + ("[" + ", ".join(str(b) for b in t[s]) + "]" if t[s] else "[ ]"), color=cmap[s], fontsize=10, weight="bold")
        ax2.text(0, .02, "free pool: " + ",".join(str(b) for b in sorted(set(range(NB)) - set(ow))), fontsize=8, wrap=True)
        fig.suptitle(msg, fontsize=10)
    FuncAnimation(fig, upd, frames=len(frames)).save(OUT / "06_paged_attention.gif", writer=PillowWriter(fps=3), dpi=60)
    plt.close(fig)


def _lanes(fin):
    lanes, end = {}, []
    for r in sorted(fin, key=lambda r: r.start):
        for i, e in enumerate(end):
            if e <= r.start:
                lanes[r.rid] = i; end[i] = r.done; break
        else:
            lanes[r.rid] = len(end); end.append(r.done)
    return lanes, len(end)


def gif_batching():
    reqs = [Req(i, i, 8, [4, 30, 6, 20, 5, 25, 8, 12, 4, 22][i]) for i in range(10)]
    runs = {m: simulate(reqs, 10_000, 4, m, prefill_ticks=1) for m in ("static", "continuous")}
    mk = max(max(r.done for r in f) for f, _, _ in runs.values())
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True)
    ticks = list(range(0, mk + 4, 3)) + [mk + 3] * 2
    def upd(i):
        t = ticks[i]
        for ax, m in zip(axs, runs):
            ax.clear(); fin, _, log = runs[m]; lanes, nl = _lanes(fin)
            for r in fin:
                if r.start <= t:
                    ax.barh(lanes[r.rid], min(r.done, t) - r.start, left=r.start, color=C[r.rid % 8], ec="white")
                    if r.done - r.start > 4: ax.text(r.start + .4, lanes[r.rid], f"r{r.rid}", va="center", color="white", fontsize=7)
            ax.axvline(t, c="red", lw=1); ax.set_xlim(0, mk + 1); ax.set_ylim(-.6, 3.6); ax.set_ylabel("batch slot")
            done = sum(1 for r in fin if r.done <= t)
            ax.set_title(f"{m} batching  | finished {done}/10 | makespan {metrics(fin, log)['makespan']} ticks", fontsize=9)
        axs[1].set_xlabel("time (ticks = decode steps)")
    FuncAnimation(fig, upd, frames=len(ticks)).save(OUT / "07_batching_gantt.gif", writer=PillowWriter(fps=4), dpi=60)
    plt.close(fig)


def gif_speculative():
    rounds = [(["The", "cat", "sat", "in"], [1, 1, 1, 0], "on"), (["the", "mat", "and", "purred"], [1, 1, 1, 1], "loudly")]
    out = []; frames = []
    for prop, acc, bonus in rounds:
        frames.append((list(out), prop, None, "1) DRAFT model proposes 4 tokens cheaply (4 small forward passes)"))
        frames.append((list(out), prop, acc, "2) TARGET model scores all 4 in ONE parallel pass; accept while target agrees"))
        n = acc.index(0) if 0 in acc else 4
        out = out + prop[:n] + [bonus]
        frames.append((list(out), [], None, f"3) keep {n} accepted + 1 token from the target ({'correction' if n < 4 else 'bonus'}) = {n+1} tokens for ONE target pass"))
    frames += [(list(out), [], None, f"Done: {len(out)} tokens with 2 target passes (vs {len(out)} without speculation). Output distribution is unchanged.")] * 2
    fig, ax = plt.subplots(figsize=(9, 3.2))
    def upd(i):
        ax.clear(); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
        o, prop, acc, msg = frames[i]
        ax.text(0, 2.5, "output so far:", fontsize=9)
        for k, w in enumerate(o):
            ax.add_patch(FancyBboxPatch((1.3 + k * 0.95, 2.3), .88, .5, boxstyle="round,pad=0.02", fc=C[0])); ax.text(1.74 + k * .95, 2.55, w, ha="center", va="center", color="white", fontsize=8)
        base = 1.3 + len(o) * .95
        if prop:
            ax.text(0, 1.3, "draft proposals:", fontsize=9)
            for k, w in enumerate(prop):
                c = "#bbb" if acc is None else (C[2] if acc[k] and all(acc[:k + 1]) else C[3] if not acc[k] else "#bbb")
                ax.add_patch(FancyBboxPatch((base + k * 0.95, 1.1), .88, .5, boxstyle="round,pad=0.02", fc=c)); ax.text(base + .44 + k * .95, 1.35, w, ha="center", va="center", fontsize=8, color="k")
            if acc is not None: ax.text(base, .6, "green = accepted, red = rejected (first mismatch); tokens after it are discarded", fontsize=8)
        ax.set_title(msg, loc="left", fontsize=10)
    FuncAnimation(fig, upd, frames=len(frames)).save(OUT / "08_speculative_decoding.gif", writer=PillowWriter(fps=1), dpi=60)
    plt.close(fig)


def fig_spec_speedup():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    a = np.linspace(0.3, 0.95, 50)
    for k, c in zip([2, 4, 8], C):
        ax.plot(a, (1 - a ** (k + 1)) / (1 - a), c=c, label=f"k = {k} draft tokens")
    ax.set_xlabel("per-token acceptance rate alpha"); ax.set_ylabel("expected tokens per target pass")
    ax.set_title("Speculative decoding: tokens per target pass = (1 - a^(k+1)) / (1 - a)", fontsize=9); ax.legend(); ax.grid(alpha=.3)
    save(fig, "09_speculative_gain.png")


def fig_roofline():
    P, BW = 300e12, 2e12
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    a = np.logspace(-1, 4, 200); ax.loglog(a, np.minimum(P, BW * a), "k", lw=2.5, label="roofline (illustrative 300 TFLOPS, 2 TB/s)")
    ax.axvline(P / BW, ls="--", c=C[3]); ax.text(P / BW * 1.1, 3e10, f"ridge ~ {P/BW:.0f} FLOP/B", color=C[3])
    by = 2
    for b, c in [(1, C[1]), (8, C[1]), (64, C[1])]:
        ai = 2 * b / by; ax.plot(ai, min(P, BW * ai), "o", c=c, ms=8); ax.annotate(f"decode b={b}", (ai, min(P, BW * ai)), (6, -14), textcoords="offset points", fontsize=8)
    for T in (128, 2048):
        ai = 2 * T / by; ax.plot(ai, min(P, BW * ai), "s", c=C[0], ms=8); ax.annotate(f"prefill T={T}", (ai, min(P, BW * ai)), (-30, 8), textcoords="offset points", fontsize=8)
    ax.set_xlabel("arithmetic intensity (FLOP / byte of weights read), fp16 weights"); ax.set_ylabel("attainable FLOP/s"); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Figure 10 - Decode (orange) is memory-bound; large-batch/long-prompt work is compute-bound", fontsize=9)
    save(fig, "10_roofline.png")


def fig_quant():
    torch.manual_seed(0)
    w = torch.randn(256, 256); w[:, 5] *= 25                # one outlier column
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
    bits = [8, 6, 5, 4, 3, 2]
    ax = axs[0]
    ax.semilogy(bits, [quant_error(w, b)["rmse"] for b in bits], "o-", c=C[3], label="per-tensor scale")
    ax.semilogy(bits, [quant_error(w, b, dim=0)["rmse"] for b in bits], "s-", c=C[1], label="per-row scale")
    ax.semilogy(bits, [quant_error(w, b, group=32)["rmse"] for b in bits], "^-", c=C[0], label="group-32 scale")
    ax.invert_xaxis(); ax.set_xlabel("bits per weight"); ax.set_ylabel("RMSE of dequantized weights"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Error grows ~2x per bit removed; finer scale groups help with outliers", fontsize=9)
    ax = axs[1]
    x = np.linspace(-1, 1, 400); s = 2 / 15
    ax.plot(x, x, c="gray", lw=1, label="ideal"); ax.step(x, np.clip(np.round(x / s), -8, 7) * s, c=C[0], where="mid", label="INT4 (16 levels)")
    ax.step(x, np.clip(np.round(x / (2 / 255)), -128, 127) * (2 / 255), c=C[1], where="mid", lw=1, label="INT8 (256 levels)")
    ax.set_xlabel("original weight"); ax.set_ylabel("dequantized weight"); ax.legend(); ax.set_title("Staircase: INT4 step is 17x coarser than INT8 over the same range", fontsize=9)
    fig.suptitle("Figure 11 - Weight quantization error (random weights with one outlier column; max rounding error is half a step)", fontsize=10, y=1.03)
    save(fig, "11_quantization_error.png")


if __name__ == "__main__":
    for f in (fig_decoding_reshape, gif_temperature, fig_beam_tree, fig_prefill_decode, fig_kv_cache, gif_paged, gif_batching,
              gif_speculative, fig_spec_speedup, fig_roofline, fig_quant):
        print(f.__name__); f()
    print("done")
