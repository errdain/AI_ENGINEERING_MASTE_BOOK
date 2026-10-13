# Tier 8 — Transformer Architecture

> **Phase 2 · Transformers & Foundation Models** · Author: Claude draft (see `CHAPTER_STATUS.md`) · Status: `AUTHOR_DRAFT` (Critic / Code-review / Final-review pending)
> Prerequisites: Tier 3 (MLPs, residual idea), Tier 4 (training loop, normalization basics), Tier 6 (tokens, embeddings, positions, masks), **Tier 7 (attention math — authoritative)**. Leads to: Tier 9 pretraining, Tier 11 inference, Tier 12 modern architectures.

**Tier goal.** After this tier you can draw one Transformer block from memory with every tensor shape, explain why residual connections and normalization placement decide whether deep stacks train, compute the parameter count of a model on paper (and predict where the parameters live), choose between encoder-only, decoder-only and encoder-decoder for a product requirement, and you will have built and trained a working **Mini-GPT** whose parameter count matches your formula.

**Ownership of ideas.** Tier 7 owns the *mathematics of attention* (Q/K/V, softmax, masks, multi-head). This tier owns **how attention and the feed-forward network are wired into blocks, stacked, normalised, sized and trained**. Attention appears here only as "a layer inside the block".

---

## 0 · How to use this tier (read → see → play → build)

| Step | What | Where |
|---|---|---|
| 1. Read | Intuition + mechanics + trade-offs | this file |
| 2. **See** | Labelled figures and animated GIFs | [`figures/`](figures) — embedded below |
| 3. **Play** | Five interactive explorers with sliders (double-click the HTML; no internet) | [`interactive/`](interactive) |
| 4. **Build** | **Mini-GPT** in PyTorch + training on CPU + 44 tests | [`lab/`](lab) |
| 5. **Regenerate** | Every figure from one script | `python scripts/make_figures.py` |

| Explorer | Adjustable controls | Teaches |
|---|---|---|
| [`block_dataflow.html`](interactive/block_dataflow.html) | batch B, sequence T, d_model, heads, d_ff multiplier, Pre-LN/Post-LN, GELU/SwiGLU | 8.5–8.9, 8.12–8.13, 8.15–8.16, 8.18–8.19 |
| [`param_calculator.html`](interactive/param_calculator.html) | layers, d_model, heads, vocab, context, d_ff, activation, norm, position type, tied embeddings, biases, **presets (REVIEW_REQUIRED)** | 8.17–8.22 |
| [`prenorm_postnorm.html`](interactive/prenorm_postnorm.html) | layers, width, sublayer gain, random seed | 8.10–8.13 |
| [`activations.html`](interactive/activations.html) | x-range, Swish β, function/derivative, SwiGLU gate & value, d_model | 8.15, 8.16, 8.8 |
| [`attention_masks.html`](interactive/attention_masks.html) | architecture, source/target length, query position, objective, mask rate | 8.2–8.4, 8.23–8.30 |

> **Animation tooling.** GIFs are Matplotlib `FuncAnimation` + `PillowWriter` (no Manim), same as Tier 6.

---

## 1 · The big picture

```
 token IDs ──► embedding (+ position) ──► [ Block 1 ] ──► [ Block 2 ] ──► … ──► [ Block L ] ──► final norm ──► output head ──► logits
 (B,T)          x₀ : (B,T,d)               x₁ = x₀ + attn(…) + ffn(…)               x_L          (B,T,d)         (d → V)        (B,T,V)
                                           └────────── the *residual stream* keeps shape (B,T,d) everywhere ──────────┘
```

A Transformer is **one idea repeated L times**: keep a stream of vectors (one per token); each block *reads* the stream, computes a correction with **attention** (mixing information *between* tokens) and a **feed-forward network** (processing each token *separately*), and **adds** the correction back. Everything else — normalization, activation functions, widths, depth, tying — is engineering that makes this loop train stably and cheaply.

![Block data flow](figures/01_block_dataflow.png)

*Figure 1 — One Pre-LN decoder block with tensor shapes. Look at (a) the two dashed skip connections: the stream passes around every sub-layer untouched; (b) that the shape is `(B, T, d)` after every `+`; (c) the only places the width changes: `3d` inside the QKV projection and `d_ff` inside the FFN. Play with the sizes: [`block_dataflow.html`](interactive/block_dataflow.html).*

---

## 2 · Concept map (every roadmap item → where it is taught)

| Roadmap | Section | Roadmap | Section |
|---|---|---|---|
| 8.1 Original Transformer | §3 | 8.16 SwiGLU | §8.2 |
| 8.2 Transformer encoder | §11.1 | 8.17 Layer stacking | §9.1 |
| 8.3 Transformer decoder | §11.2 | 8.18 Hidden dimension | §9.2 |
| 8.4 Encoder-decoder architecture | §11.3 | 8.19 Number of heads | §9.3 |
| 8.5 Transformer block | §4 | 8.20 Number of layers | §9.1 |
| 8.6 Self-attention layer | §4.1 | 8.21 Parameter counting | §10 |
| 8.7 Cross-attention layer | §4.2 | 8.22 Weight tying | §10.3 |
| 8.8 Feed-forward network | §5 | 8.23 Encoder-only architecture | §11.1 |
| 8.9 Position-wise MLP | §5 | 8.24 BERT | §11.1 |
| 8.10 Residual connections | §6 | 8.25 Decoder-only architecture | §11.2 |
| 8.11 LayerNorm | §7.1 | 8.26 GPT | §11.2 |
| 8.12 Pre-LN | §7.3 | 8.27 Encoder-decoder architecture | §11.3 |
| 8.13 Post-LN | §7.3 | 8.28 T5/BART | §11.3 |
| 8.14 RMSNorm | §7.2 | 8.29 Causal objective | §12 |
| 8.15 GELU | §8.1 | 8.30 Masked objective | §12 |

*(8.4 and 8.27 are the same roadmap concept listed twice; it is taught once, in §11.3.)*

---

## 3 · The original Transformer (8.1)

**What.** The 2017 *Attention Is All You Need* encoder-decoder model for machine translation: a stack of identical encoder layers, a stack of identical decoder layers, attention as the only sequence-mixing operation (no recurrence, no convolution).

**Why it exists.** RNNs (Tier 5) process tokens sequentially: training cannot be parallelised across time and long-range signals must survive many multiplications. Attention connects any two positions in one step and is a batch of matrix multiplications, which GPUs do very well (Tier 7 explains the cost: O(T²) scores).

**The base configuration** (from the paper; numbers `REVIEW_REQUIRED` against the original text): 6 encoder + 6 decoder layers, `d_model = 512`, 8 heads (`d_k = 64`), `d_ff = 2048`, ReLU, **Post-LN**, sinusoidal positions (Tier 6), dropout 0.1, input/output embeddings shared, ≈ 65 M parameters.

**What changed since** (each is a section of this tier): Post-LN → **Pre-LN** (§7.3); LayerNorm → **RMSNorm** (§7.2); ReLU → **GELU / SwiGLU** (§8); sinusoidal → learned or **RoPE** positions (Tier 6); and the *family* split — most modern LLMs keep only the decoder half (§11.2).

**Interview (30–60 s).** *"The original Transformer replaced recurrence with attention so a whole sequence is processed in parallel. It is an encoder-decoder: the encoder builds contextual representations of the source, the decoder generates the target one token at a time with causal self-attention plus cross-attention to the encoder. Each layer is attention plus a position-wise feed-forward network, wrapped in residual connections and layer norm."*
**Memory hook.** *"Attention mixes tokens, the MLP thinks per token, residuals keep the road open."*

---

## 4 · The Transformer block (8.5) — *core architecture*

**What.** One repeating unit: `x → x + Attention(Norm(x)) → x + FFN(Norm(x))` (Pre-LN form). Input and output are both `(B, T, d_model)` so blocks stack freely.

**How it works (mechanics).**

1. **Norm** the stream (per token).
2. **Attention sub-layer**: project to Q, K, V; split into `h` heads of width `d_k = d/h`; scores `QKᵀ/√d_k`; mask; softmax; weight the values; concatenate heads; output projection `W_O`. (Mathematics → **Tier 7**.)
3. **Add** the result to the stream (residual).
4. **Norm** again.
5. **FFN sub-layer**: expand `d → d_ff`, non-linearity, project back `d_ff → d`.
6. **Add** again.

```python
# lab/mini_gpt.py — the whole block (Pre-LN branch)
def forward(self, x):
    x = x + self.attn(self.n1(x))      # tokens talk to each other
    return x + self.mlp(self.n2(x))    # each token is processed alone
```

**Input → Process → Output**

```
x (B,T,d) ─┬─► Norm ─► Attention ─►(+)─┬─► Norm ─► FFN ─►(+)──► x' (B,T,d)
           └───────── skip ───────────►┘└──────── skip ─────►┘
```

**Division of labour (the mental model to keep).** Attention is the only place information moves *between positions*; the FFN is where most parameters live and (empirically) where much factual knowledge is stored — a widely used working hypothesis, not a proven fact (`REVIEW_REQUIRED` for any strong claim).

**Advantages.** Uniform, parallel over T in training, scalable by just changing `L`, `d`, `h`. **Limitations.** Attention memory/compute O(T²) (Tier 7, Tier 11); no built-in notion of order (Tier 6). **Trade-offs.** Width vs depth vs context (§9). **Common mistakes.** Forgetting that `d_model` must be divisible by `n_head`; adding the residual *before* the norm in one place and after in another; computing `√d` instead of `√d_k` in the attention scaling. **Cost.** A block has about `12d²` parameters (§10) and `≈ 2·12d² = 24d²` forward FLOPs per token plus an attention term that grows with T. **Security.** A mask bug is a *data-leak* bug (§12): the model silently sees the answer in training and fails in production.
**Rapid-fire.** Block = (norm → attention → add) then (norm → FFN → add).

### 4.1 Self-attention layer (8.6)

Queries, keys and values all come from the **same** sequence. In a decoder-only model the mask is causal; in an encoder it is full (or padding-only). The layer in this tier is ~12 lines (`CausalSelfAttention` in [`lab/mini_gpt.py`](lab/mini_gpt.py)). Note one more trick used in the code: Q, K, V are produced by **one fused matmul** (`d → 3d`) and split, which is faster than three separate projections and has identical parameters.

### 4.2 Cross-attention layer (8.7)

Same computation as self-attention, but **queries come from the decoder stream and keys/values come from the encoder output**. Mask: full over the source (except padding). It is the *only* channel from encoder to decoder in an encoder-decoder model, and it adds `4d²` parameters per decoder block (an extra Q, K, V, O). Decoder-only models have no cross-attention; "conditioning" is done by putting the context in the prompt.

![Masks](figures/04_architecture_masks.png)

*Figure 2 — The three attention patterns in the three families. Look at: encoder self-attention is a full square; decoder self-attention is lower-triangular; cross-attention is a full **rectangle** (target rows × source columns). Switch families and move the highlighted query in [`attention_masks.html`](interactive/attention_masks.html).*

---

## 5 · Feed-forward network / position-wise MLP (8.8, 8.9)

**What.** Two (or three, for SwiGLU) linear layers with a non-linearity, applied to **each token independently with the same weights**:
`FFN(x) = W_down · act(W_up · x)` with `W_up: d → d_ff`, `W_down: d_ff → d`, usually `d_ff = 4d` for GELU.

**Why.** Attention is (up to the softmax weights) a *linear* mixing of value vectors; without a non-linearity a stack of attention layers would be far less expressive. The FFN supplies non-linearity and a large per-token memory. "Position-wise" means no information crosses positions inside the FFN, which makes it trivially parallel and easy to shard.

**Cost / size.** With `d_ff = 4d` the FFN has `2·d·4d = 8d²` weights versus attention's `4d²`: **about two-thirds of a block's parameters are FFN** (see Figure 6 and the calculator).

**Common mistakes.** Treating `d_ff` as free: doubling `d` makes the FFN 4× bigger. Forgetting the FFN is per-token, then trying to "see neighbours" in it.
**Interview.** *"Why does a Transformer need the MLP if attention already mixes information?"* → non-linearity and per-token capacity; attention moves information, the MLP transforms it.
**Memory hook.** *"Attention = communicate, MLP = compute."*

---

## 6 · Residual connections (8.10)

**What.** `x_out = x + F(x)`: each sub-layer *adds* an update to the stream instead of replacing it.

**Why.** (1) **Gradient highway**: `∂x_out/∂x = I + ∂F/∂x` — the identity term lets gradients reach early layers undamped (the same idea as ResNets, Tier 13.12). (2) **Incremental refinement**: a sub-layer only needs to learn a *correction*; at initialisation (small weights) a block is close to the identity, so a very deep model starts as a shallow one. (3) **The residual-stream view**: all layers read from and write to one shared `d`-dimensional workspace.

![Residual stream](figures/05_residual_stream.gif)

*Figure 3 — Residual stream of the trained Mini-GPT for one sentence. Left: the `d_model × position` activations after each sub-layer (a = attention output added, f = FFN output added); right: the mean vector norm. Look at how the stream is built up additively — every step modifies, never replaces — and how its norm grows through the stack. This growth is what the next section's normalization controls.*

**Mistake.** Residual add requires identical shapes — the reason every sub-layer maps `d → d`. **Initialisation tip used in the lab:** scale the output projections by `1/√(2L)` (GPT-2 style) so that the sum of 2L residual writes doesn't blow up the stream variance at the start.
**Memory hook.** *"Never replace, always add."*

---

## 7 · Normalization (8.11–8.14)

### 7.1 LayerNorm (8.11)

Per token (not per batch), over the feature dimension:
`LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β`, with learned `γ, β ∈ ℝ^d` (`2d` parameters). It makes every token vector have zero mean / unit variance before the sub-layer, so activation scale does not drift with depth or batch composition. Unlike BatchNorm (Tier 4) it does not depend on batch statistics, so it behaves identically in training, in batch-size-1 inference, and with variable-length sequences.

### 7.2 RMSNorm (8.14)

`RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)` — no mean subtraction, no `β`. Fewer operations and `d` instead of `2d` parameters, with reportedly similar quality; adopted by many recent open LLM families (verify per model — `REVIEW_REQUIRED`). Implemented in `RMSNorm` in the lab; the test `test_rmsnorm_unit_rms_and_layernorm_zero_mean` checks both norms numerically.

### 7.3 Pre-LN vs Post-LN (8.12, 8.13) — *major concept*

| | Post-LN (original 2017) | Pre-LN (most modern LLMs) |
|---|---|---|
| Formula | `x ← Norm(x + F(x))` | `x ← x + F(Norm(x))` |
| Residual path | **passes through a norm** every sub-layer | **clean identity** from input to output |
| Stream scale | held at ~1 after each norm | grows with depth (every layer adds) |
| Needs final norm? | no | **yes** (before the output head) |
| Training deep stacks | needs careful learning-rate warm-up; can diverge | much more stable; warm-up less critical |
| Quality (if both train) | often slightly better in some reports | slightly worse in some reports |

*(The quality remark is from the literature on Pre-/Post-LN; sizes of the effect are task dependent — `REVIEW_REQUIRED` before quoting.)*

![Pre vs Post LN](figures/02_preln_vs_postln.png)

*Figure 4 — Measured on a 24-block MiniGPT at initialisation (random tokens, sub-layer weights std 0.25, mean of 5 seeds). **Left:** Post-LN holds the stream at RMS 1 by construction; Pre-LN's stream grows (≈ 33 at the end in this run) because nothing renormalises the sum. **Middle/right:** gradient norms differ by orders of magnitude and in shape across depth — in the Pre-LN model the gradient reaching the stream shrinks smoothly, in the Post-LN model it is larger and ragged. Do not over-read the exact numbers: they depend on initialisation and gain, and this is a single toy batch.*

The same story, interactive and with a hand-written backward pass: [`prenorm_postnorm.html`](interactive/prenorm_postnorm.html). Try `g = 0.5 / 1 / 2.5` with 48 layers: with a large sub-layer gain the Post-LN gradient at layer 0 collapses, whereas Pre-LN keeps an O(1) gradient at the first layer.

**Why Pre-LN trains easier (intuition).** In Post-LN every gradient must pass through `L` normalisation Jacobians on its way back; at initialisation these can shrink or amplify it layer after layer, which is why Post-LN needs warm-up. In Pre-LN the identity path bypasses all norms.

**Enterprise relevance.** When you *fine-tune or serve* someone else's model you do not choose this — but loading weights into the wrong variant (Pre-LN weights, Post-LN code) produces no error and garbage output. Always check the architecture config.
**Interview (30–60 s).** *"Post-LN puts the norm after the residual add, so the skip path is normalised at every layer and deep stacks are hard to train without warm-up. Pre-LN puts the norm inside the branch, leaving a clean identity path for gradients, which is why almost all modern LLMs use it — at the price of a growing residual-stream scale and a required final norm."*
**Memory hook.** *"Pre-LN: norm the branch, not the road."*

---

## 8 · Activation functions (8.15, 8.16)

![Activations](figures/03_activation_functions.png)

*Figure 5 — Left: ReLU, GELU (exact and tanh approximation — nearly indistinguishable) and SiLU. Middle: derivatives — ReLU jumps at 0, GELU/SiLU are smooth and slightly non-monotonic. Right: one **SwiGLU** unit, `SiLU(a)·u`: the gate `a` multiplies the value `u`; where `a<0` the unit is switched off. Move the sliders in [`activations.html`](interactive/activations.html).*

### 8.1 GELU (8.15)

`GELU(x) = x·Φ(x)` where Φ is the standard normal CDF: a *smooth, probabilistic* gate — inputs are multiplied by the probability that a unit-Gaussian is below them. A common approximation is `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`; GPT-2/BERT-era code used it, modern code often calls the exact form. Used in BERT and GPT-2/3. Compared to ReLU it has no hard zero-gradient region and a smooth transition.

### 8.2 SwiGLU (8.16)

A **gated** FFN (Shazeer, *GLU Variants Improve Transformer*, 2020): `FFN(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down`. Three matrices instead of two, so to keep parameters equal the hidden size is reduced to `d_ff ≈ (8/3)·d` (then usually rounded to a hardware-friendly multiple). Reported to improve quality at equal parameters and used by many recent open model families (`REVIEW_REQUIRED` — check each model card). Cost: three matmuls, so memory traffic and the width bookkeeping change; weight-conversion bugs between `gate` and `up` are a classic checkpoint-loading error.

**Common mistakes.** Using `d_ff = 4d` with SwiGLU and being surprised by a 50 % larger FFN; mixing exact and tanh-approximate GELU between training and inference (small but real numerical drift).
**Rapid-fire.** GELU = smooth ReLU; SwiGLU = gated FFN: a learned, input-dependent gate multiplies a linear value.
**Memory hook.** *"GELU softly switches; SwiGLU lets the token decide how open the switch is."*

---

## 9 · Stacking and sizing (8.17–8.20)

### 9.1 Layer stacking and number of layers (8.17, 8.20)

Layers are stacked sequentially: `x_L = B_L(…B_2(B_1(x_0)))`, each with its **own** weights (unlike the weight-sharing in a few research models). **More layers = more sequential computation steps** (more "depth of reasoning") and linearly more parameters and latency (layers cannot be parallelised at inference time; one token must pass through all of them). Typical depths: a handful for toy/small models up to around a hundred for very large ones (`REVIEW_REQUIRED` per model). Training stability, not parameter count, is the first limit on depth — see §6–7.

### 9.2 Hidden dimension (8.18)

`d_model` is the width of the residual stream. It sets the size of *every* matrix: parameters grow as `d²` per block. Wider models store more per layer; wider-and-shallower vs narrower-and-deeper is a design trade-off studied in scaling work (Tier 9 owns scaling laws). Practical constraints: `d` divisible by heads, and a multiple of 64/128 for tensor-core efficiency.

### 9.3 Number of heads (8.19)

Heads split `d` into `h` subspaces of width `d_k = d/h`. **Changing `h` at fixed `d` does not change the parameter count** (QKV/O stay `4d²`); it changes how the same capacity is partitioned, the shape of the `(B,h,T,T)` score tensor (memory ∝ `h·T²`), and the arithmetic intensity. Typical `d_k` is 64–128 (`REVIEW_REQUIRED`). More heads ≠ more capacity per se; too many heads leaves each too narrow. KV-sharing variants (MQA/GQA) shrink inference memory and belong to **Tier 12.6–12.7**; the KV cache itself is **Tier 11**.

**Common mistakes.** Choosing `h` that does not divide `d`; assuming more heads means a bigger model; scaling `d` without scaling `d_ff` consistently. **Interview.** *"You double d_model — what happens to parameters and FLOPs per token?"* → ≈ 4× for the blocks (embedding table 2×).

---

## 10 · Parameter counting (8.21) and weight tying (8.22) — *major concept*

### 10.1 The formula

For a GPT-style model with learned positions, biases, LayerNorm, GELU, `d_ff = 4d`:

```
per block   = attention (4d² + 4d)  +  FFN (8d² + 5d)  +  2 LayerNorms (4d)   = 12d² + 13d
embeddings  = V·d (tokens) + T·d (positions)
total       = V·d + T·d + L·(12d² + 13d) + 2d (final LN)  [+ V·d if the output head is untied]
```

General version (any `d_ff`, SwiGLU, RMSNorm, no biases, Pre/Post-LN) = [`count_params_formula`](lab/mini_gpt.py); **the tests build all 32 combinations of {norm, Pre/Post, act, tied, bias} and assert it equals `sum(p.numel())` of the real PyTorch model.**

**Worked check (TESTED by the calculator and the lab).** GPT-2-small-shaped config `V=50257, T=1024, d=768, L=12, tied`: formula → **124,439,808** (the widely quoted "124 M" for GPT-2 small — published figure `REVIEW_REQUIRED`). The Mini-GPT of this lab (`V=41, T=64, d=96, L=3`): **345,792**, asserted equal to PyTorch in `train_mini_gpt.py` and the tests.

![Parameter breakdown](figures/06_parameter_breakdown.png)

*Figure 6 — Where parameters live. Look at: in every config the FFN (red) is ≈ ⅔ of the block weights; embeddings only dominate when `V·d` is big relative to `L·12d²` (GPT-2 small, 31 %). Preset configurations other than Mini-GPT are recalled from public model cards and are `REVIEW_REQUIRED`. Change anything in [`param_calculator.html`](interactive/param_calculator.html).*

### 10.2 What the number buys you (cost)

| Quantity | Rule of thumb | Note |
|---|---|---|
| Weights in memory | `2 N` bytes (fp16/bf16), `N` bytes (int8) | activations & KV cache extra (Tier 11) |
| Training state with Adam, mixed precision | ≈ `16 N` bytes before activations | `REVIEW_REQUIRED` — depends on optimizer/sharding |
| Forward FLOPs per token | ≈ `2 N` (+ attention term ∝ T) | |
| Training FLOPs per token | ≈ `6 N` | forward + backward; scaling-law convention (Tier 9) |

### 10.3 Weight tying (8.22)

The input embedding `E ∈ ℝ^{V×d}` maps ID → vector; the output head `W ∈ ℝ^{d×V}` maps vector → logits. **Tying** sets `W = Eᵀ`: one matrix, two uses (the original Transformer paper shared them). Saves `V·d` parameters (for `V = 50k, d = 768` ≈ 38 M — a third of a 124 M model's size), and acts as a regulariser that often helps small models; for very large models the table is a small fraction, so the saving matters less and some families leave them untied (`REVIEW_REQUIRED` per model). In PyTorch tying is *sharing the same Parameter* (`self.head.weight = self.tok.weight`); `model.parameters()` counts it once, and `state_dict()` contains both keys — a classic source of off-by-`V·d` bugs when counting or converting checkpoints. Tested: `test_weight_tying_shares_storage_and_saves_V_times_d`.

**Common mistakes.** Counting embeddings twice (or not at all) when comparing to a model card; forgetting position embeddings; comparing "parameters" of an MoE (total vs active, Tier 12) with a dense model. **Security/ops.** A checkpoint is code if it is a Python pickle: load with `torch.load(..., weights_only=True)` or use the `safetensors` format for untrusted weights.
**Interview (30–60 s).** *"Per block it's about 12 d² — 4d² for attention, 8d² for a 4×-wide FFN — times L layers, plus V·d for the embedding table, plus T·d if positions are learned; tying the output head to the embedding removes another V·d. I use 2N bytes for fp16 weights and ≈6N FLOPs per token to train."*
**Memory hook.** *"Twelve d-squared per layer, plus the vocabulary table."*

---

## 11 · The three architecture families (8.2–8.4, 8.23–8.28)

Tier 12 revisits these as *families with modern variants*; this section teaches the originals.

### 11.1 Encoder-only (8.2, 8.23) and BERT (8.24)

**Encoder**: stack of blocks with **full (bidirectional)** self-attention; every token sees every other. Output: one contextual vector per input token. **BERT** (2018): encoder-only, trained with the **masked objective** (§12) — hide ≈15 % of tokens and predict them (`REVIEW_REQUIRED`: the 80/10/10 mask/random/keep recipe and the next-sentence-prediction auxiliary task in the original; later work dropped NSP). Use for: classification, tagging, **embeddings and rerankers** (Tier 20), extraction. Cannot generate fluently token-by-token.

### 11.2 Decoder-only (8.3, 8.25) and GPT (8.26)

**Decoder-only**: blocks with **causal** self-attention and no cross-attention. Trained with the **causal objective** (§12). Generation = repeatedly predict the next token and append it (autoregression, Tier 11). **GPT** (2018 →): the family that scaled this recipe; GPT-2 (Pre-LN, learned positions, tied embeddings, 124 M–1.5 B) is the template the lab follows. Dominant for chat/assistant LLMs: one architecture, one objective, every training position gives a signal, and prompting handles many tasks without extra heads.

### 11.3 Encoder-decoder (8.4, 8.27) and T5/BART (8.28)

Encoder reads the whole input bidirectionally; decoder generates the output causally and **cross-attends** to the encoder output (§4.2). **T5** frames every task as text-to-text and pre-trains with *span corruption* (masked spans replaced by sentinel tokens, the decoder produces the missing spans). **BART** is a denoising autoencoder: corrupt the text (masking, deletion, shuffling…), then reconstruct it with a causal decoder (details `REVIEW_REQUIRED`). Natural fit for translation, summarisation, structured rewriting — tasks with a clear input→output split.

| | Encoder-only | Decoder-only | Encoder-decoder |
|---|---|---|---|
| Self-attention | full | causal | encoder full, decoder causal |
| Cross-attention | – | – | **yes** |
| Objective | masked | causal | span/denoising (seq2seq) |
| Generates text? | not natively | **yes** | yes |
| Typical use | embeddings, classification, rerank | chat, code, general LLM | translation, summarisation |
| Examples | BERT-family | GPT-family, many open LLMs | T5, BART |
| Params per block | `12d²` | `12d²` | enc `12d²`, dec `16d²` (extra cross-attn) |

**Enterprise choice.** Need *vectors or labels* cheaply and fast → encoder-only (an order of magnitude cheaper than an LLM for classification at scale). Need *open-ended generation or tool use* → decoder-only. Need *strict input→output transformation with a smaller model* → encoder-decoder can still be competitive. Cost drivers: parameters (§10), context length (Tier 11), and serving complexity (two stacks vs one).
**Common mistakes.** Using a causal LM for embeddings without pooling/training for it; assuming "BERT = old = useless" (it remains a workhorse for classification and retrieval); mixing tokenizers between model and checkpoint (Tier 6).
**Interview Q.** *Why did decoder-only win for general LLMs?* → single objective, all positions supervised, simplest to scale and to serve (one stack, KV cache), in-context learning removes task-specific heads. *(Nuanced; evidence is empirical — `REVIEW_REQUIRED`.)*

---

## 12 · Training objectives: causal (8.29) vs masked (8.30)

| | Causal LM | Masked LM |
|---|---|---|
| Task | predict token *t+1* from tokens ≤ *t* | predict hidden tokens from both sides |
| Attention mask | **lower-triangular** | full |
| Supervised positions per sequence | **all (T−1)** | only the masked ones (~15 %) |
| Gives | generator | representation (encoder) |
| Pitfall | mask bug leaks the future | train/inference mismatch: `[MASK]` never appears at inference |

**One forward pass trains T predictions** because the causal mask guarantees position *i*'s output depends only on tokens ≤ *i*. That guarantee is a testable property — and a security/validity one: if the mask leaks, training loss collapses toward 0 *while the model learns nothing usable*. `test_causal_mask_no_future_leak` changes only tokens at positions ≥ 6 and asserts the logits at positions < 6 are bit-for-bit (to 1e-5) unchanged, for both Pre-LN and Post-LN, and has a sanity check that later positions *do* change. Compare the two objectives interactively in [`attention_masks.html`](interactive/attention_masks.html). (Tier 9 owns objectives at scale: 9.17–9.19.)

---

## 13 · Build: Mini-GPT (major build)

A complete decoder-only Transformer in ~150 lines, trained on a tiny character-level corpus on CPU.

| File | What it is |
|---|---|
| [`lab/mini_gpt.py`](lab/mini_gpt.py) | `GPTConfig`, `RMSNorm`, `CausalSelfAttention`, `MLP` (GELU or SwiGLU), `Block` (Pre/Post-LN), `MiniGPT`, `generate`, `count_params_formula` |
| [`lab/corpus.py`](lab/corpus.py) | ~27 KB original corpus (hand-written paragraph + seeded templated sentences; 41 distinct characters) |
| [`lab/train_mini_gpt.py`](lab/train_mini_gpt.py) | time-budgeted AdamW loop (warm-up + cosine), validation split, sampling, writes `lab/outputs/` |
| [`lab/test_tier08.py`](lab/test_tier08.py) | 44 pytest cases (shapes, causality, loss decreases, overfit one batch, 32-way formula check, closed form `12Ld²`, weight tying, norms, error handling, generation) |
| [`scripts/make_figures.py`](scripts/make_figures.py) | regenerates all figures from the lab outputs |

```bash
cd lab
python -m pytest -q                          # 44 passed
python train_mini_gpt.py --seconds 60        # train for ~60 s on CPU, print samples, save outputs/
```

**Design choices (each one is a concept above).** Learned absolute positions (Tier 6) → the `T·d` term in the formula; Pre-LN + final norm (§7.3); GELU 4× FFN (§8.1); tied embeddings (§10.3); fused QKV; GPT-2-style `1/√(2L)` residual-projection init (§6). Flags `prenorm`, `norm`, `act`, `tie_weights`, `bias` flip each choice so you can A/B them.

**Result of the recorded run** (`TESTED`: executed here, CPU, 2 threads, seed 0, 60 s; numbers will vary a little by machine): `345,792` parameters (3 blocks, `d=96`, 4 heads, context 64, dropout 0.1), 514 optimizer steps (batch 32), **validation loss ≈3.6 (step 0) → 0.41 nats/char** (random guessing is `ln 41 = 3.71`), train loss ≈ 0.70 (higher than validation because dropout is on while training).

![Training](figures/07_training_loss_and_samples.png)

*Figure 7 — Left: train and validation loss; the dashed line is the loss of a model that guesses uniformly. Right: samples at three checkpoints from the same prompt `The `. Look at how the text goes from random symbols (step 0) to word-like noise to mostly grammatical sentences about models and vectors.*

![Generation over training](figures/08_generation_over_training.gif)

*Figure 8 — The same samples as an animation: the black dot moves along the validation curve while the sample text improves.*

Final sample (temperature 0.8, top-k 8; `lab/outputs/sample.txt`):

> The curious vector reads the simple vector ignores later. A patient student copies a simple student rains a a tired stayers a token ignores twice. When the mien vector writes the server, the bright vector checks it. …

It learned spelling, word boundaries, sentence shape and punctuation from 27 KB in one minute — and still makes agreement and spelling errors, which is exactly what a tiny model should do.

![Attention maps](figures/09_attention_maps_trained.png)

*Figure 9 — Learned attention weights of the trained model (3 layers × 4 heads) on one sentence. Look at: every map is lower-triangular (the causal mask at work) and the heads specialise — layer 1 puts most weight near the diagonal (recent characters), while deeper layers spread weight over more of the past. (Interpreting individual heads needs more evidence than one sentence provides.)*

**Honest limitations of the lab.** (1) The corpus is partly templated, so low loss partly reflects a small, regular grammar — this is a *mechanics* demo, not evidence about real-language scaling. (2) The validation split is the tail of the same generator, so it measures generalisation across sentences, not across distributions. (3) 60 s of CPU time cannot show Pre-LN vs Post-LN differences in trained quality; only the initialisation-time measurements (Figure 4) do. (4) The saved checkpoint is `fp16` `state_dict` (~0.7 MB) loaded with `weights_only=True`.

**Exercises**

1. **Predict, then check.** Using §10, predict the parameter count for `d=128, L=4, V=41, T=64`, with and without tying, then verify with `count_params`.
2. **Break the mask.** Delete the `masked_fill` line, retrain 60 s, and compare validation loss and samples. Why does *training* loss look great? Re-run `test_causal_mask_no_future_leak` and read the failure.
3. **Pre-LN vs Post-LN.** Train both (`prenorm=False`) with `lr=3e-3` and no warm-up for 24 layers. Which diverges? Add warm-up and compare. (Expect Post-LN to be more fragile; report what *you* see.)
4. **GELU vs SwiGLU at equal parameters.** Set `act="swiglu"`, `d_ff=256` (≈ 8/3·96) and compare parameter count and final validation loss over 3 seeds. Is the difference larger than the seed noise?
5. **Heads at fixed width.** Train with `n_head ∈ {1, 2, 4, 8, 16}` and confirm the parameter count is constant; what changes in speed and loss?
6. *(stretch)* Add RoPE (Tier 6) in place of learned positions and update the formula; or add a cross-attention block and make it encoder-decoder.

---

## 14 · Interview kit (whole tier)

**Basic.** What is in a Transformer block? What does the FFN do? What is a residual connection for? What is LayerNorm normalising over?
**Intermediate.** Pre-LN vs Post-LN — what changes and why do modern LLMs use Pre-LN? GELU vs SwiGLU? Why must `d_model` be divisible by the number of heads? Compare encoder-only, decoder-only, encoder-decoder with one use case each.
**Advanced.** Derive ≈ `12 L d²` and say what you ignored. Why does weight tying save `V·d`, and what is the pitfall when counting parameters in PyTorch? Why does one causal forward pass give T training signals, and how would you *test* that a mask doesn't leak? Why does the stream norm grow in Pre-LN, and why do we need a final norm?
**Scenario.** *You must classify 10 M support tickets per day cheaply.* (Encoder-only fine-tune: cost, latency, determinism vs an LLM.) *A teammate's model "has 7 B parameters" but doesn't fit on your GPU.* (Weights ≈ 2 bytes/param = 14 GB, plus KV cache/activations; quantisation — Tier 11.) *Loaded checkpoint produces garbage but no error.* (Norm placement / gate-up order / RoPE pairing / tied-head mismatch.)

**60-second tier summary.** *"A Transformer keeps a residual stream of one vector per token. Each block normalises it, lets attention move information between tokens, adds the result back, then normalises again and lets a feed-forward network process each token, adding that back too. Pre-LN keeps the skip path clean, which is why deep models train; RMSNorm and SwiGLU are the modern cheap/better variants. Parameters are about 12d² per layer plus the vocabulary table, and tying the head to the embedding saves V·d. Encoder-only models read bidirectionally for embeddings and classification, decoder-only models use a causal mask to generate, and encoder-decoder models add cross-attention for sequence-to-sequence tasks."*

## 15 · Revision questions

1. Write the shapes of every tensor in one block for `B=4, T=128, d=512, h=8, d_ff=2048`.
2. Why does the residual connection make deep networks trainable? Write `∂x_out/∂x`.
3. Show `d_ff = 4d` makes the FFN twice the size of attention (in weights).
4. Why does Pre-LN need a final norm?
5. What is the `d_ff` that keeps SwiGLU parameter-equal to a 4× GELU FFN?
6. Does changing the number of heads change the parameter count? The score-tensor memory?
7. Compute the parameter count of a 24-layer, `d=1024`, `V=50k`, `T=1024` tied GPT. State the memory of its fp16 weights.
8. Explain why a leaking causal mask gives a *lower* training loss and a *worse* model.
9. Which architecture family would you choose for (a) semantic search, (b) a coding assistant, (c) translation? Why?
10. What is different between "total parameters" and "parameters in `model.parameters()`" under weight tying?

## 16 · Related concepts & forward references

Attention maths, masks, multi-head → **Tier 7**. Tokenizers, positional encodings (learned/sinusoidal/RoPE/ALiBi) → **Tier 6**. Pre-training data, objectives at scale, scaling laws, MoE pre-training → **Tier 9**. Fine-tuning/LoRA → **Tier 10**. KV cache, decoding, quantisation, serving → **Tier 11**. MQA/GQA, long-context, MoE, SSMs, family-level case studies → **Tier 12**. ResNet / ViT residual ideas → **Tier 13**. Encoder models for retrieval → **Tier 20**.

## 17 · Review status

| Item | State |
|---|---|
| Code (`lab/`) | **TESTED** — `python -m pytest -q` in `lab/`: 44 passed (Python 3.13, PyTorch CPU) |
| Training run | **TESTED** — 60 s CPU run executed; parameter count asserted equal to formula; metrics above from `lab/outputs/train_log.json` |
| Interactive pages | **TESTED** with `tools/check_interactive.py` (headless Chromium, no JS errors, sliders exercised); calculator outputs cross-checked against the Python formula (Mini-GPT 345,792; GPT-2-small-shaped 124,439,808) |
| Figures / GIFs | regenerated by `scripts/make_figures.py`; visually spot-checked |
| Pre-/Post-LN measurements | single-architecture, few-seed, initialisation-only — qualitative (stated in captions) |
| Published model sizes and configs (original Transformer, GPT-2, Llama-style, GPT-3, BERT details, T5/BART objectives, SwiGLU/RMSNorm adoption claims) | `REVIEW_REQUIRED` (marked inline) |
| Critic / Code-review / Final-review | pending per `CHAPTER_STATUS.md` |
