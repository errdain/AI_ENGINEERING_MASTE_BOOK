# Tier 4 — Training & Optimization: Revision Sheet

> Status: DRAFT (Author: Claude). Condensed from concept files `04.01`–`04.41`; the concept file is the authoritative source. Section 2 is generated from sections 18 and 23 of each file.
> Code evidence: `code/` (11 tests) and `lab/` (5 tests) build and check a complete training pipeline; all page examples run under `tests/test_md_snippets.py`.

## 1. One-page rule sheet

| Topic | Rule | Note |
|---|---|---|
| Training loop | `zero_grad → forward → loss → backward → (clip) → step → (scheduler.step)` | validation in `eval()` + `no_grad()` |
| Loss | logits into `cross_entropy`; `mean` reduction; mask padding | start value ≈ `ln K` |
| Gradients | accumulate in `.grad`; graph freed after backward | `loss.item()`, never store the loss tensor |
| SGD / momentum | `θ −= lr·g`; `v = βv + g; θ −= lr·v` | momentum ≈ 10× steps along steady directions (β = 0.9) |
| Adam | `m`, `v` with bias correction; step ≈ `lr·m̂/(√v̂+ε)` | first step ≈ lr; state = 2× model size |
| AdamW | decay outside the Adam scaling: `θ −= lr·(adam_step + λθ)` | no decay on biases/norms; `Adam(weight_decay)` is the coupled L2 version |
| Schedules | warm-up `lr·t/W`, then cosine `min + ½(max−min)(1+cos(πt/T))` | step the scheduler once per optimizer step |
| Batch size | noise ∝ `1/√B`; scale lr with `B` (linear for SGD, ~sqrt for Adam) | tokens per batch for LLMs |
| Accumulation | `loss/k` for k micro-batches, then one step | BatchNorm and token-weighted losses are not exactly equal |
| Clipping | `g *= min(1, c/‖g‖)` on the **global** norm | log pre-clip norm and clip rate |
| Formats | FP32 1/8/23, FP16 1/5/10 (max 65,504), BF16 1/8/7 (FP32 range) | FP16 needs loss scaling; BF16 does not |
| Mixed precision | 16-bit matmuls, FP32 master weights, FP32 reductions | memory: 2+2+4+8 = 16 bytes/param, same as pure FP32; activations halve |
| Checkpoint | model + optimizer + scheduler + step + RNG, saved atomically | `weights_only` / safetensors for untrusted files |
| Reproducibility | seed Python/NumPy/PyTorch/workers; pin versions; log config | compare with tolerance; use several seeds |
| GPU memory | ≈ 16 bytes/param + activations (∝ batch × sequence × layers) | 7B → 112 GB before activations |
| Data parallel | replicate model, shard data, all-reduce average gradients | = one big batch |
| Tensor parallel | column split → concatenate; row split → sum; MLP: one all-reduce | intra-node |
| Pipeline | bubble `(p−1)/(m+p−1)`; many micro-batches; 1F1B | light point-to-point traffic |
| ZeRO / FSDP | shard optimizer → gradients → weights: `16/N` bytes/param at stage 3 | all-gather per layer, reduce-scatter gradients |
| Debug ladder | initial loss `ln K` → overfit one batch → gradients → labels → lr sweep | NaN: find the first non-finite value |
| Checkpointing (activations) | recompute segments in backward: memory O(n)→O(√n), ≈ +33% compute | gradients identical |

## 2. Rapid-fire answers and memory hooks

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 4.1 Training loop | Forward → loss → `zero_grad` → `backward` → `step`, over batches and epochs, with validation and checkpoints. | **Training loop: predict, measure the miss, adjust, repeat.** |
| 4.2 Forward pass | `model(x)` → logits; records the graph in training; use `eval()` + `no_grad()` for validation and inference. | **Forward pass: predict now, and keep the receipts for the backward pass.** |
| 4.3 Loss calculation | Scalar from logits + targets; cross-entropy (classification), MSE (regression); `mean` reduction; mask padding; start value ≈ `ln K`. | **Loss: turn "how wrong" into one number, mean it over the batch, and mask the padding.** |
| 4.4 Backward pass | Reverse-mode autodiff from a scalar loss; fills `.grad`; accumulates; graph freed after; cost ≈ 2× forward. | **Backward: blame flows back into `.grad`, and it piles up until you clear it.** |
| 4.5 Parameter update | `θ ← θ − η·g` (or the optimizer's variant); `optimizer.step()` after `backward()`; lr too big diverges, too small crawls. | **Update: step against the gradient, as far as the learning rate allows.** |
| 4.6 SGD (stochastic gradient descent) | Step along the negative mini-batch gradient; noisy but unbiased; cheap; needs lr tuning; weak on ill-conditioned problems. | **SGD: a quick, noisy glance downhill, repeated many times.** |
| 4.7 Momentum | `v = βv + g; θ −= lr·v`; damps zig-zag, speeds steady directions; β ≈ 0.9; Nesterov looks ahead. | **Momentum: a heavy ball rolls through the bumps and speeds up on long slopes.** |
| 4.8 Adam | `m` = mean gradient, `v` = mean squared gradient; step = `lr·m̂/(√v̂+ε)`; bias correction; defaults 1e-3, 0.9, 0.999. | **Adam: momentum for the direction, a running size for each weight's stride.** |
| 4.9 AdamW | Adam + decoupled decay: `θ −= lr·(adam_step + λθ)`; different from Adam + L2; exclude biases and norms; λ ≈ 0.01–0.1. | **AdamW: Adam's step, plus a plain shrink of the weights on the side.** |
| 4.10 Learning-rate scheduling | `lr(t)`: step/exponential/linear/cosine decay, often with warm-up; step after the optimizer; save with the checkpoint. | **Schedule: sprint early, tiptoe at the finish.** |
| 4.11 Warm-up | Linear ramp of lr over the first N steps; protects Adam and deep models early; N ≈ 1–5% of the run. | **Warm-up: jog before you sprint.** |
| 4.12 Cosine decay | `lr = min + ½(max−min)(1+cos(πt/T))`; smooth; needs `T`; used after warm-up in LLM recipes. | **Cosine decay: a smooth glide from fast to gentle.** |
| 4.13 Batch size | Larger `B`: less noise, fewer steps, more memory; scale lr with `B` and warm up; use accumulation to emulate large batches. | **Batch size: how many opinions you average before taking a step.** |
| 4.14 Gradient accumulation | `loss/k` → `backward` ×k → `step` → `zero_grad`; same as a k-times batch; saves memory, not time. | **Accumulate: many small trips, one big build.** |
| 4.15 Gradient clipping | `g *= min(1, c/‖g‖)` on the global norm; between backward and step; log the norm and the clip rate. | **Clip: a speed governor on the update, same direction, capped length.** |
| 4.16 FP32 (single precision) | 32 bits: 1/8/23; eps 1.2e-7, max 3.4e38; default dtype; the safe baseline for master weights and reductions. | **FP32: 32 bits, seven digits, and the safe default.** |
| 4.17 FP16 (half precision) | 16 bits: 1/5/10; max 65,504; underflow below ~6e-8; needs loss scaling and FP32 master weights. | **FP16: half the bits, half the safety margin.** |
| 4.18 BF16 (bfloat16) | 16 bits: 1/8/7; FP32 range, low precision; no loss scaling; default for LLM training. | **BF16: FP32's range in 16 bits; accuracy traded for safety.** |
| 4.19 Mixed-precision training | 16-bit compute + FP32 master weights; FP16 needs loss scaling, BF16 doesn't; autocast keeps sensitive ops in FP32. | **Mixed precision: sketch in 16-bit, bank the result in 32-bit.** |
| 4.20 Checkpoints | Save model + optimizer + scheduler + step + RNG; atomic writes; resume = rebuild then load states; weights-only for sharing. | **Checkpoint: save everything needed to continue, not just the answer.** |
| 4.21 Reproducibility | Seed everything, pin versions, log config/data/commit, deterministic algorithms for debugging, compare with tolerance, report multi-seed spread. | **Reproducibility: write down the recipe, the seeds and the oven.** |
| 4.22 Seeds | Seed = start of a PRNG sequence; seed every library and DataLoader workers; use multiple seeds for comparison; same seed ≠ cross-hardware identical. | **Seed: the starting card that makes "random" repeatable.** |
| 4.23 CPU vs GPU | GPU = thousands of simple cores for parallel matmul; CPU = few strong cores; keep data on-device; avoid `.item()` and copies in the loop. | **GPU: a thousand cooks doing the same chop at once; keep the ingredients on the counter.** |
| 4.24 GPU memory | ~16 bytes/param (mixed-precision Adam) + activations ∝ batch; fix with accumulation, BF16, checkpointing, sharding, LoRA. | **Training memory: the model, its blame, its notebook, and everything it remembers.** |
| 4.25 Profiling | Measure phases, skip warm-up, synchronize GPU timing, use `torch.profiler`; look for data-loading stalls, syncs, tiny kernels; check utilization/MFU. | **Profile first: find the real traffic jam before buying a faster car.** |
| 4.26 DataLoader | Dataset + sampler + workers + collate; shuffle train not val; `num_workers`, `pin_memory`, `drop_last`; pad variable-length data with a mask. | **DataLoader: a kitchen that preps the next tray while the diner eats.** |
| 4.27 PyTorch tensors | Array + dtype + device + `requires_grad`; views share memory; broadcasting from the right; `.item()`, `.detach()`, `.to()`. | **Tensor: a NumPy array with a passport (device) and a diary (autograd).** |
| 4.28 nn.Module | Subclass `nn.Module`; layers in `__init__`, flow in `forward`; auto-registered parameters; `ModuleList` for lists; `state_dict`, `.to`, `.train/.eval`. | **nn.Module: a Lego block that keeps its own parts list.** |
| 4.29 Optimizers (the `torch.optim` API) | `opt = AdamW(params, lr, weight_decay)`; `zero_grad` → `backward` → `step`; param groups; `state_dict` for checkpoints; AdamW default for Transformers. | **Optimizer: the coach with a clipboard; give it every athlete.** |
| 4.30 Distributed training: introduction | Processes with ranks and a world size; collectives (all-reduce, all-gather); data / sharded / tensor / pipeline parallelism; communication limits scaling. | **Distributed training: many hands, one model, and the cost is talking to each other.** |
| 4.31 Data parallelism | Replicate model, shard data, all-reduce gradients (average); same as a bigger batch; memory per GPU unchanged. | **Data parallel: many copies of the student, each reads different pages, then they pool their notes.** |
| 4.32 Model / tensor parallelism | Split matrices: column → concatenate, row → sum (all-reduce); MLP = column then row (one all-reduce); intra-node only. | **Tensor parallel: cut the matrix; side-by-side for columns, add up for rows.** |
| 4.33 Pipeline parallelism | Layers split across devices; micro-batches stream; bubble `(p−1)/(m+p−1)`; 1F1B saves activation memory; light communication. | **Pipeline: an assembly line of layers; keep it full with micro-batches.** |
| 4.34 FSDP / ZeRO overview | Shard optimizer (ZeRO-1), + gradients (2), + weights (3/FSDP); all-gather on demand, reduce-scatter gradients; 16/N bytes per parameter. | **FSDP/ZeRO: nobody owns the whole library; borrow each book only while reading it.** |
| 4.35 Training curves | Plot loss (raw + smoothed) vs steps; healthy = fast drop then flatten; flat at chance = not learning; spikes/divergence = lr, overflow, data. | **Training curve: the heartbeat of the run; learn to read its rhythm.** |
| 4.36 Validation curves | Held-out loss/metric each epoch; gap = over/underfit; early stop with patience and best checkpoint; test set only once. | **Validation: the surprise quiz that shows what the model really learned.** |
| 4.37 Gradient monitoring | Log global/per-layer grad norm, zero fraction, update/weight ratio (≈1e-3), `None` grads; spikes precede NaNs. | **Watch the gradients: the dashboard that warns before the engine dies.** |
| 4.38 NaN / Inf debugging | Find the first non-finite value: check data, forward hooks, anomaly mode, FP32 retry; causes: log(0), exp overflow, 0/0, high lr, FP16; fix with stable ops, ε, clipping, warm-up. | **NaN: one bad egg spoils the omelette; find the egg.** |
| 4.39 Loss debugging | Start loss ≈ ln K; overfit one batch; check gradients; inspect data/labels; shuffled-label control; lr sweep; one change at a time. | **Debug the loss: tiny tests first; if it can't memorize eight examples, it's a bug.** |
| 4.40 Memory profiling | Measure peak (`max_memory_allocated`), compare across steps for leaks, profile by op/snapshot; store `.item()`; reduce with accumulation, bf16, checkpointing, sharding. | **Measure the peak, not the average, and never keep the loss tensor.** |
| 4.41 Gradient checkpointing | Store checkpoints, recompute segments in backward; memory O(n)→O(√n); ~+33% compute; wrap each block; identical gradients. | **Checkpointing: forget the details, re-walk them from the landmarks.** |
