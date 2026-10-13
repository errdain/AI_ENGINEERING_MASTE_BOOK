"""Tier 5 figures, all computed from code/seqmodels.py and lab/lab_05_sequence_comparison.py (no typed-in numbers)."""
import numpy as np
import torch
from _common import OUT, fs, plt
import seqmodels as S
import lab_05_sequence_comparison as L

# ---- 5.13: learning signal travelling back through time
res = L.ex_5_13_gradient_signal()
fig, ax = plt.subplots(figsize=(10.5, 5.2)); cols = [fs.RED, fs.ORANGE, fs.TEAL, fs.BLUE]
for (scale, vals), c in zip(res.items(), cols):
    ax.semilogy(range(len(vals)), vals, lw=2.4, color=c, label=f"recurrent weights scale {scale}")
ax.axhline(1, color=fs.MUTED, lw=1, ls="--"); ax.text(1, 1.3, "signal at the last step = 1", color=fs.MUTED, fontsize=9)
ax.set_xlabel("time step t (the loss is measured at step 39)"); ax.set_ylabel("‖∂loss/∂hₜ‖ relative to the last step (log scale)")
ax.set_title("Learning signal travelling back through a vanilla RNN", fontsize=13, fontweight="bold"); ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower right")
fig.savefig(OUT / "fig_5_13_gradient_signal.svg"); plt.close(fig)

# ---- 5.14: memory versus delay
mem = L.ex_5_14_memory_vs_delay(delays=(5, 10, 20, 30, 40))
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for kind, c in (("rnn", fs.RED), ("gru", fs.ORANGE), ("lstm", fs.TEAL)):
    ax.plot(list(mem[kind]), list(mem[kind].values()), "-o", lw=2.4, color=c, label=kind.upper())
ax.axhline(1 / 8, color=fs.MUTED, ls="--"); ax.text(5, 0.15, "chance = 1/8", color=fs.MUTED, fontsize=9)
ax.set_xlabel("delay (number of distracting symbols between the symbol to remember and the question)"); ax.set_ylabel("accuracy")
ax.set_title("How long can each model remember? (one seed, 1,500 training steps)", fontsize=13, fontweight="bold"); ax.grid(alpha=0.3); ax.legend()
fig.savefig(OUT / "fig_5_14_memory_vs_delay.svg"); plt.close(fig)

# ---- 5.16-5.18: LSTM gates over time on the memory task
m = S.fit_memory("lstm", 20, steps=1500)
x, y = S.memory_batch(64, 20, generator=torch.Generator().manual_seed(5))
E = m.emb(x).detach().numpy()
Wih, Whh, bih, bhh = [p.detach().numpy() for p in (m.rnn.weight_ih_l0, m.rnn.weight_hh_l0, m.rnn.bias_ih_l0, m.rnn.bias_hh_l0)]
h = c = np.zeros((64, 32)); gates = []
for t in range(E.shape[1]):
    h, c, (i, f, g, o) = S.lstm_cell(E[:, t], h, c, Wih, Whh, bih, bhh); gates.append([i.mean(), f.mean(), o.mean(), np.abs(c).mean()])
gates = np.array(gates)
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for k, (name, col) in enumerate((("input gate (mean)", fs.BLUE), ("forget gate (mean)", fs.TEAL), ("output gate (mean)", fs.ORANGE))):
    ax.plot(gates[:, k], lw=2.4, color=col, label=name)
ax.plot(gates[:, 3] / max(gates[:, 3].max(), 1e-9), lw=2, color=fs.YELLOW, ls="--", label="cell state size (scaled)")
ax.axvline(0, color=fs.MUTED, lw=1); ax.text(0.3, 0.04, "symbol to remember", color=fs.MUTED, fontsize=9); ax.axvline(20, color=fs.MUTED, lw=1); ax.text(15, 0.04, "question", color=fs.MUTED, fontsize=9)
ax.set_xlabel("time step"); ax.set_ylabel("average gate value (0 = closed, 1 = open)"); ax.set_ylim(0, 1.05)
ax.set_title("Inside a trained LSTM on the memory task (average over 64 examples and 32 units)", fontsize=13, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(loc="center right")
fig.savefig(OUT / "fig_5_14_lstm_gates.svg"); plt.close(fig)
print("gate means (input, forget, output) over time, first/last:", gates[0, :3].round(2), gates[-1, :3].round(2))

# ---- 5.21: Seq2Seq with / without attention
acc = L.ex_5_21_seq2seq_vs_length(lengths=(4, 8, 16))
fig, ax = plt.subplots(figsize=(10.5, 5.2)); width = 0.26
for k, (name, col) in enumerate((("none", fs.RED), ("bahdanau", fs.BLUE), ("luong", fs.TEAL))):
    vals = [acc[name][l] for l in (4, 8, 16)]
    ax.bar(np.arange(3) + (k - 1) * width, vals, width, color=col, label={"none": "no attention", "bahdanau": "Bahdanau (additive)", "luong": "Luong (dot)"}[name])
    for xx, v in zip(np.arange(3) + (k - 1) * width, vals): ax.text(xx, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=fs.FG)
ax.set_xticks(range(3)); ax.set_xticklabels(["length 4", "length 8", "length 16"]); ax.set_ylabel("exact-match accuracy (reverse a digit string)"); ax.set_ylim(0, 1.12)
ax.set_title("A single context vector breaks on long inputs; attention does not", fontsize=13, fontweight="bold"); ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "fig_5_21_seq2seq_vs_length.svg"); plt.close(fig)

# ---- 5.24 / 5.25: attention maps
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5))
for ax, name, title in zip(axs, ("bahdanau", "luong"), ("Bahdanau (additive)", "Luong (dot)")):
    model = S.train_seq2seq(name, 8, steps=1200)
    src, _, _ = S.reverse_batch(200, 8, torch.Generator().manual_seed(3)); _, att = model.greedy(src, 9)
    ax.imshow(att.mean(0).numpy(), cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("source position"); ax.set_ylabel("output step"); ax.set_title(title, fontsize=12)
fig.suptitle("Attention weights while reversing 8 digits: each output looks at the mirrored input", fontsize=13, fontweight="bold"); fig.tight_layout()
fig.savefig(OUT / "fig_5_24_attention_maps.svg"); plt.close(fig)

# ---- 5.5: word2vec embeddings of a toy corpus (PCA to 2-D)
from test_seqmodels import toy_sentences
vocab, Em = S.skipgram(toy_sentences(120), dim=8, epochs=30)
Z = Em - Em.mean(0); U, s, Vt = np.linalg.svd(Z, full_matrices=False); P = Z @ Vt[:2].T
groups = {"animals": ["cat", "dog", "bird", "fox"], "verbs": ["saw", "chased", "liked"], "colours": ["red", "blue", "green"], "objects": ["car", "ball", "box"], "function words": ["the", "was", "here"]}
fig, ax = plt.subplots(figsize=(8.5, 6.2))
for (name, ws), col in zip(groups.items(), fs.PALETTE):
    pts = np.array([P[vocab.index(w)] for w in ws]); ax.scatter(pts[:, 0], pts[:, 1], s=90, color=col, label=name)
    for w, (a, b) in zip(ws, pts): ax.annotate(w, (a, b), xytext=(6, 5), textcoords="offset points", fontsize=10, color=fs.FG)
ax.set_title("Skip-gram embeddings of a toy corpus (8-D, shown in 2-D): words with the same role cluster", fontsize=12, fontweight="bold"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.savefig(OUT / "fig_5_05_word2vec.svg"); plt.close(fig)

# ---- 5.26: why RNNs became a bottleneck
Ts = np.array([10, 30, 100, 300, 1000]); fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
axs[0].loglog(Ts, Ts, "-o", color=fs.RED, lw=2.4, label="RNN: steps that must run one after another"); axs[0].loglog(Ts, np.ones_like(Ts), "-o", color=fs.TEAL, lw=2.4, label="self-attention: one layer, all positions at once")
axs[0].set_xlabel("sequence length T"); axs[0].set_ylabel("sequential steps"); axs[0].set_title("Parallelism", fontsize=12); axs[0].legend(fontsize=9); axs[0].grid(alpha=0.3, which="both")
axs[1].loglog(Ts, Ts, "-o", color=fs.RED, lw=2.4, label="RNN: path between first and last token = T − 1"); axs[1].loglog(Ts, np.ones_like(Ts), "-o", color=fs.TEAL, lw=2.4, label="attention: path = 1")
axs[1].loglog(Ts, Ts ** 2, "-o", color=fs.ORANGE, lw=2.4, label="attention cost: T² pairwise scores"); axs[1].set_xlabel("sequence length T"); axs[1].set_title("Distance and price", fontsize=12); axs[1].legend(fontsize=9); axs[1].grid(alpha=0.3, which="both")
fig.suptitle("Why RNNs became a bottleneck (counts, not timings)", fontsize=13, fontweight="bold"); fig.tight_layout()
fig.savefig(OUT / "fig_5_26_bottleneck.svg"); plt.close(fig)
