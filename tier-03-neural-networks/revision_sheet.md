# Tier 3 — Neural Networks: Revision Sheet

> Status: DRAFT (Author: Claude). Condensed from concept files `03.01`–`03.32`. The concept file is the authoritative source; this sheet only summarises it. Rapid-fire answers and memory hooks in section 2 are copied from sections 18 and 23 of each concept file.
> Code evidence: `code/` (35 tests: NumPy library + autodiff, checked against PyTorch) and `lab/` (29 tests). Run: `cd tier-03-neural-networks && python -m pytest -q code lab` (64 tests; PyTorch 2.14.1 is the test reference).

## 1. One-page formula and rule sheet

| Topic | Formula / rule | Note |
|---|---|---|
| Neuron | `y = φ(w·x + b)` | weights ↔ synapse strength, bias ↔ threshold |
| Perceptron | `w ← w + η(y−ŷ)x` on mistakes | converges iff linearly separable; ≤ `(R/γ)²` mistakes; fails on XOR |
| Dense layer | `Y = X Wᵀ + b`, `W:(out,in)` | params `n_in·n_out + n_out`; 784→128→64→10 = 109,386 |
| Dense backward | `dW = dYᵀX`, `db = ΣdY`, `dX = dY·W` | three matmuls; backward ≈ 2× forward FLOPs |
| Layer collapse | `W₂(W₁x) = (W₂W₁)x` | no activation ⇒ one linear map |
| Output layer + loss | regression: 1 linear; binary: 1 logit + sigmoid/BCE; K-class: K logits + softmax/CE | feed logits to the loss, not probabilities |
| Sigmoid / tanh | `σ′ = σ(1−σ) ≤ 0.25`; `tanh′ = 1−tanh² ≤ 1` | saturate; sigmoid chain of 10 layers ≤ `0.25¹⁰ ≈ 9.5×10⁻⁷` |
| ReLU family | `max(0,x)`; derivative 0/1 | dying units (61% ≤ 0 with shifted inputs); Leaky/GELU/SiLU alternatives |
| GELU / SiLU | `xΦ(x)`; `xσ(x)` | min ≈ −0.17 / −0.278; derivative max ≈ 1.129 / 1.10 |
| Softmax | `exp(zᵢ−max)/Σexp(zⱼ−max)` | shift-invariant; temperature `T`; CE gradient `p − onehot` |
| Backprop | `δ_z = δ_a ⊙ φ′(z)`, `dW = δ_zᵀa_prev`, `da_prev = δ_z·W` | cache forward activations; sum gradients over reused nodes |
| Autodiff | reverse mode: all grads of a scalar in one backward pass; forward mode: one pass per input | finite differences need `2P` forward passes (218,772 for 109k params) |
| Xavier | `Var(W) = 2/(fan_in+fan_out)`; `U(±√(6/(fan_in+fan_out)))` | tanh/sigmoid/linear; PyTorch tanh gain 5/3 |
| He | `Var(W) = 2/fan_in` | ReLU family; Xavier on ReLU shrinks ~√2 per layer (30 layers → 2×10⁻⁵) |
| Zero / constant init | all units identical forever | symmetry never breaks |
| Vanishing / exploding | product of per-layer factors `<1` / `>1` | sigmoid 20L first/last grad ratio 1.3×10⁻¹³; ReLU weight scale ×1.5 over 20L → grad norm ×4,000 |
| Gradient clipping | `g ← g·c/‖g‖₂` if `‖g‖₂ > c` | global norm; treats the symptom, also fix init/LR/norm |
| Residual | `y = x + F(x)`, `dX = dY + dF` | 30-layer tanh: plain loss 1.196 (not learning) vs residual 0.0018 |
| Dropout | train: mask/(1−p); eval: identity | check `eval()`; effect small on the lab task (+1.3 pts) |
| Weight decay | `w ← (1−ηλ)w − η∇L` | = L2 for SGD; AdamW decouples; not on biases/norm params |
| BatchNorm | `γ(x−μ_B)/√(σ²_B+ε)+β`; running stats at eval | batch-dependent (output std ~1/√B); batch of 1 ⇒ zeros; higher LR possible |
| LayerNorm | normalize each sample over features | no batch dependence or running stats; pre-norm in modern Transformers |
| RMSNorm | `γ·x/√(mean(x²)+ε)` | no mean subtraction, no β; scale-invariant, not shift-invariant; half the params of LN |

## 2. Concept-by-concept revision

### A. Neuron and layer anatomy (3.1–3.8)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 3.1 | [Biological inspiration vs artificial neuron](03.01-biological-inspiration-vs-artificial-neuron.md) | Artificial neuron: y = φ(w·x + b); loosely inspired by biology; weights↔synapses, bias↔threshold. | An artificial neuron is a weighted vote with a threshold — inspired by biology, not a copy. |
| 3.2 | [Perceptron](03.02-perceptron.md) | w += η(y−ŷ)x on mistakes; converges iff linearly separable (≤(R/γ)² mistakes); fails on XOR. | Perceptron: nudge the line whenever you're wrong — it works only if a line can work. |
| 3.3 | [Input layer](03.03-input-layer.md) | Input layer: parameter-free entry defining shape (B, n_in); preprocessing and encoding happen before it. | Input layer: the agreed shape of the question the network can be asked. |
| 3.4 | [Hidden layers](03.04-hidden-layers.md) | Hidden layers = Dense+activation stacks between input and output; none → linear; depth/width trade capacity vs cost. | Hidden layers: where the network invents its own features. |
| 3.5 | [Output layer](03.05-output-layer.md) | Regression: 1 linear unit; binary: 1 logit+sigmoid; multiclass: K logits+softmax; multilabel: K sigmoids; pass logits to the loss. | Output layer: pick units, activation and loss as a matched set. |
| 3.6 | [Weights](03.06-weights.md) | Learnable per-connection scales (W: out×in); random init, trained by gradients; the model artifact. | Weights: the learned dials of the network. |
| 3.7 | [Bias](03.07-bias.md) | b shifts the hyperplane off the origin; init 0; not weight-decayed; unrelated to statistical bias. | Bias: slide the line off the origin. |
| 3.8 | [Weighted sum](03.08-weighted-sum.md) | z = w·x + b; layer form Z = X Wᵀ + b; ≈2·n_in·n_out FLOPs per sample. | Weighted sum: one big matmul does the neurons' adding up. |

### B. Activation functions and softmax (3.9–3.15)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 3.9 | [Activation functions](03.09-activation-functions.md) | Elementwise nonlinearity that prevents layer collapse; compare range, saturation, derivative; ReLU/GELU/SiLU hidden, sigmoid/softmax outputs. | Activation: the bend that lets many straight lines draw a curve. |
| 3.10 | [Sigmoid](03.10-sigmoid.md) | σ(x)=1/(1+e⁻ˣ); range (0,1); σ′=σ(1−σ) ≤ 0.25; saturates; use for binary outputs and gates. | Sigmoid: probability-shaped, but flat at both ends. |
| 3.11 | [Tanh](03.11-tanh.md) | tanh ∈ (−1,1), zero-centered, derivative 1−tanh² ≤ 1; saturates; common in RNN cells. | Tanh: sigmoid re-centered on zero. |
| 3.12 | [ReLU](03.12-relu.md) | max(0,x); gradient 1/0; sparse; risk of dying units; use He init; Leaky ReLU/GELU as alternatives. | ReLU: pass the positives, block the negatives — and watch for units that never wake up. |
| 3.13 | [GELU](03.13-gelu.md) | GELU(x)=xΦ(x): smooth ReLU, min ≈ −0.17, derivative max ≈ 1.129; common in Transformers. | GELU: ReLU with the corner sanded off. |
| 3.14 | [SiLU / Swish](03.14-silu-swish.md) | SiLU(x)=xσ(x); min ≈ −0.278; derivative max ≈ 1.1; the gate in SwiGLU. | SiLU: let the input vote on how much of itself passes. |
| 3.15 | [Softmax](03.15-softmax.md) | exp(zᵢ)/Σexp(zⱼ); sums to 1, shift-invariant; stable via max-shift; temperature controls entropy; gradient softmax−onehot. | Softmax: scores in, vote shares out. |

### C. Dense layer and MLP (3.16–3.17)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 3.16 | [Linear / Dense layer](03.16-linear-dense-layer.md) | y = xWᵀ+b; params n_in·n_out+n_out; dW = dYᵀX, db = ΣdY, dX = dY·W; stacks of linear collapse. | Dense layer: one matmul and a bias — the workhorse of every network. |
| 3.17 | [Multi-layer perceptron (MLP)](03.17-mlp.md) | Stacked Dense+activation layers; universal approximator in theory; XOR needs a hidden layer; basis of Transformer FFN. | An MLP is dense layers with bends in between — enough bends, any curve. |

### D. Training machinery: forward, graph, backprop, autodiff (3.18–3.21)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 3.18 | [Forward propagation](03.18-forward-propagation.md) | z=Wa+b, a=φ(z) layer by layer; ≈2·weights FLOPs per sample; training caches activations; eval vs train mode matters. | Forward pass: feed the input through, keep the receipts for later. |
| 3.19 | [Computational graph](03.19-computational-graph.md) | DAG of ops; backward = reverse traversal with local derivatives; shared nodes sum gradients; PyTorch builds it dynamically. | The graph is the receipt that lets every step answer 'how much did I matter?' |
| 3.20 | [Backpropagation](03.20-backpropagation.md) | Reverse chain rule: δ_z=δ_a⊙φ′; dW=δ_zᵀa_prev; da_prev=δ_z·W; cost ≈2× forward; start with softmax−onehot. | Backprop: send the blame backward, one chain-rule step at a time. |
| 3.21 | [Automatic differentiation](03.21-automatic-differentiation.md) | Exact derivatives by chain rule over executed ops; reverse mode = all grads of a scalar in one backward pass; forward mode = one input per pass. | Autodiff: exact derivatives, because the computer keeps a ledger of every step. |

### E. Initialization and gradient problems (3.22–3.26)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 3.22 | [Parameter initialization](03.22-parameter-initialization.md) | Random (symmetry-breaking) weights with variance ∝ 1/fan_in; Xavier for tanh/sigmoid, He for ReLU; biases 0. | Initialize so the signal neither dies nor screams on its way through. |
| 3.23 | [Xavier (Glorot) initialization](03.23-xavier-initialization.md) | Var(W)=2/(fan_in+fan_out) (normal) or U(±√(6/(fan_in+fan_out))); for tanh/linear; ReLU needs He. | Xavier: variance by the average of fan-in and fan-out, for tanh-style units. |
| 3.24 | [He (Kaiming) initialization](03.24-he-initialization.md) | N(0, 2/fan_in) for ReLU; compensates for half the units being zero; Xavier would shrink by √2 per layer. | He: double the variance because ReLU kills half. |
| 3.25 | [Vanishing gradients](03.25-vanishing-gradients.md) | Backward signal = product of φ′·W factors; <1 ⇒ decays; fix with ReLU/GELU, He init, normalization, residuals, gating. | Vanishing: multiply many numbers below one and the early layers hear nothing. |
| 3.26 | [Exploding gradients](03.26-exploding-gradients.md) | Backward factors >1 compound; clip global norm (g·c/‖g‖) plus fix init/normalization/LR; monitor the norm. | Exploding: the squeal grows each pass — clip it, then fix the gain. |

### F. Stabilizing and regularizing: residuals, dropout, weight decay, normalization (3.27–3.32)

| # | Concept | Rapid-fire answer | Memory hook |
|---|---|---|---|
| 3.27 | [Residual connections](03.27-residual-connections.md) | y = x + F(x); gradient = dY + dF; identity path prevents vanishing; pre-norm vs post-norm. | Residual: give the gradient a highway home. |
| 3.28 | [Dropout](03.28-dropout.md) | Train: zero w.p. p, scale 1/(1−p); eval: identity; regularizer/ensemble view; check train/eval mode. | Dropout: practice with random teammates missing, play with everyone present. |
| 3.29 | [Weight decay](03.29-weight-decay.md) | w ← (1−ηλ)w − η∇L; = L2 for SGD; AdamW decouples; exclude biases/norm params. | Weight decay: a small, constant pull of every weight toward zero. |
| 3.30 | [Batch normalization](03.30-batch-normalization.md) | Normalize each feature over the batch, learn γ,β; running stats at eval; batch-dependent; Transformers use LayerNorm. | BatchNorm: normalize with the crowd — and learn to depend on who is in the crowd. |
| 3.31 | [Layer normalization](03.31-layer-normalization.md) | Per-sample normalization over features + γ,β; no batch dependence or running stats; pre-norm in modern Transformers. | LayerNorm: each token normalizes itself. |
| 3.32 | [RMSNorm](03.32-rmsnorm.md) | y=γ·x/√(mean(x²)+ε): no mean subtraction, no β; scale-invariant, not shift-invariant; cheaper than LayerNorm. | RMSNorm: normalize the size, skip the centering. |

## 3. Top pitfalls (each has a test or lab demonstration)

1. **Stacking Dense layers without a nonlinearity** (3.9, 3.16): 75 parameters compute what 27 can.
2. **Omitting the bias** (3.7): boundary forced through the origin; 66.2% vs 92.3%.
3. **Softmax/sigmoid applied before a loss that applies it again; probabilities instead of logits** (3.5, 3.15).
4. **Sigmoid/tanh hidden layers in deep nets; Xavier with ReLU; N(0,1) or N(0,0.01²) weights** (3.10, 3.22–3.24): saturate or vanish.
5. **All-zero or all-equal initialization** (3.22): hidden units never differentiate.
6. **Unscaled weight gain / large learning rate** (3.26): gradient norm explodes; loss → inf; clip *and* fix the scale.
7. **Forgetting `eval()`** (3.28, 3.30): dropout still on; BatchNorm uses batch statistics; outputs vary run to run.
8. **BatchNorm with tiny batches or at batch size 1** (3.30): output depends on batch composition; zeros for a single sample.
9. **Weight decay on biases and norm parameters; L2-in-loss with Adam called weight decay** (3.29).
10. **Mismatched weight-shape convention** `(out,in)` vs `(in,out)` when porting between PyTorch and Keras (3.16).
11. **Holding the autograd graph alive** (3.19): store `loss.item()`, not the loss tensor.
12. **In-place ops and `detach()` that silently cut or corrupt the graph** (3.19, 3.21).
13. **Trusting a gradient implementation without a check** (3.20): finite differences and framework autograd agree to ~1e-10 only when correct.
14. **Assuming bigger always fits better** (3.17): the 64-unit net fit sin(x) worse than 16 at a fixed training budget.
15. **Treating "inspired by the brain" literally** (3.1): the artificial neuron is a weighted sum plus nonlinearity.

## 4. Interview drills (30–60 s each; answers are in the concept files)

- Why do we need activation functions? Prove two linear layers collapse. → 3.9, 3.16.
- Compare sigmoid, tanh, ReLU, GELU, SiLU: ranges, derivatives, pitfalls. → 3.10–3.14.
- Derive backprop for a dense layer and the softmax-cross-entropy gradient. → 3.20, 3.15.
- Why reverse-mode autodiff for training and forward mode for few inputs? → 3.21.
- Why Xavier for tanh and He for ReLU? Show the variance argument. → 3.22–3.24.
- Diagnose a 20-layer network whose early layers don't learn. → 3.25, 3.27, 3.30–3.32.
- Loss becomes NaN after a few steps in a deep ReLU net: your checklist. → 3.26, 3.22, 1.44.
- How do residual connections change the gradient? → 3.27.
- Dropout, weight decay, BatchNorm: train vs eval behaviour of each. → 3.28–3.30.
- Why do Transformers use LayerNorm/RMSNorm instead of BatchNorm, and what's the difference between them? → 3.30–3.32.

## 5. Status of claims

TESTED: every quantitative claim above that cites a test or lab function (NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1, Python 3.14; fixed seeds; tests assert directions/thresholds). REVIEW_REQUIRED items (authorship and dates of key papers, which LLMs use RMSNorm/pre-norm/SwiGLU/bias-free layers, why BatchNorm helps, universal-approximation statement, MC-dropout, loss-landscape smoothness, PyTorch default init) are listed in `reviews/author_open_items.md`. Overlaps with other tiers to adjudicate: gradient clipping (4.15), AdamW (4.9), activation checkpointing (4.41), reproducibility (4.21–4.22).
