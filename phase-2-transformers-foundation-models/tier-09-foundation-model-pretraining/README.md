# Tier 9 — Foundation Model Pretraining

> **Phase 2 · Transformers & Foundation Models** · Author: Claude draft (see `CHAPTER_STATUS.md`) · Status: `AUTHOR_DRAFT` (Critic / Code-review / Final-review pending)
> This file is the **tier overview and index**. Each of the 31 roadmap concepts has its **own file** in [`concepts/`](concepts), following the 23-section Standard Concept Format.

**Tier goal.** After this tier you can describe the whole path from "a pile of crawled text" to "a checkpoint that predicts the next token": build and defend a data pipeline (clean → filter → dedup → decontaminate → mix), budget tokens and compute with `C ≈ 6ND`, reason with scaling laws, explain what a pretraining run needs to survive (checkpoints, determinism, stability monitoring, distributed memory), explain how Mixture-of-Experts routing and its load-balancing loss work, and run a miniature but *complete* pretraining lab on a laptop CPU in seconds.

## Prerequisites and leads-to

- **Prerequisites:** Tier 4 (training loops, optimizers, LR schedules), [Tier 6](../tier-06-tokenization-embeddings/README.md) (tokenizers), [Tier 7](../tier-07-attention/README.md) (attention), [Tier 8](../tier-08-transformer-architecture/README.md) (Transformer blocks, causal/masked objectives).
- **Leads to:** [Tier 10](../tier-10-fine-tuning-alignment/README.md) fine-tuning and alignment, [Tier 11](../tier-11-llm-inference-optimization/README.md) inference, [Tier 12](../tier-12-modern-architectures/README.md) modern architectures. Evaluation methodology and benchmark use live in Tier 15 (and Tier 30); RAG as the alternative to injecting knowledge is Tier 20.

> **Scope rule.** Attention math is Tier 7, the Transformer block and the causal/masked objective *definitions* are Tier 8, MoE as an *architecture family* is Tier 12. This tier owns the **data, scale, optimization and operations** around them, and the *pretraining-specific* side of MoE (routing, balancing, instability).

---

## How to use this tier (read → see → play → build)

| Step | What | Where |
|---|---|---|
| 1. Read | One file per concept, in order (or jump via the index below) | [`concepts/`](concepts) |
| 2. **See** | 12 labelled figures including two animated GIFs | [gallery below](#visual-gallery) · [`figures/`](figures) |
| 3. **Play** | Six interactive explorers (open in a browser, no internet) | [explorer table below](#interactive-explorers) · [`interactive/`](interactive) |
| 4. **Build** | Miniature pretraining lab (PyTorch, CPU), 22 pytest tests | [`lab/`](lab) |
| 5. **Regenerate** | Every figure and the LR-viewer data come from one script (about 2-3 min) | `python scripts/make_figures.py` |

---

## The big picture

```
 acquire ──► clean ──► quality ──► license/PII ──► dedup ──► decontaminate ──► mix ──► tokenize ──► shard
 (9.2)       (9.3)     (9.4)       (9.9, 9.10)     (9.5)     (9.11, 9.12)      (9.6-9.8) (Tier 6)
                                                                                  │
   budgets (9.14, 9.15, 9.20) ──► model size N, tokens D, FLOPs ≈ 6·N·D            ▼
                                                              ┌──────────── training run ────────────┐
   objective (9.16-9.19) ──────────────────────────────────►  │ next-token loss · AdamW · warmup+cos │
   dense (9.24) / MoE (9.25-9.27) ─────────────────────────►  │ checkpoints (9.22) · stability (9.28)│
   distributed (9.23) ─────────────────────────────────────►  │ eval every k steps (9.31)            │
                                                              └──────────────┬───────────────────────┘
                                                                             ▼
                                              base model ──► continued / domain-adaptive pretraining (9.29, 9.30) ──► Tier 10
```

A *foundation model* is trained once, expensively, on broad data with a **self-supervised** objective (the labels come from the text itself), then adapted many times cheaply. Three things decide its quality, in roughly this order of leverage: **the data**, **the amount of compute spent sensibly** (N and D together), and **keeping a very long run alive and reproducible**.

---

## Concept index (all 31 roadmap concepts)

| # | Concept | Key visual / lab link |
|---|---|---|
| 9.1 | [Pretraining overview](concepts/09.01-pretraining-overview.md) | Fig 1 · `lab/run_lab.py` |
| 9.2 | [Pretraining data acquisition](concepts/09.02-pretraining-data-acquisition.md) | Fig 1 · `data_funnel.html` |
| 9.3 | [Data cleaning](concepts/09.03-data-cleaning.md) | `clean_text` (limitation documented) |
| 9.4 | [Quality filtering](concepts/09.04-quality-filtering.md) | Fig 1 · `quality_filter` |
| 9.5 | [Deduplication strategies](concepts/09.05-deduplication-strategies.md) | Fig 2 · `minhash_dedup.html` |
| 9.6 | [Data mixture design](concepts/09.06-data-mixture-design.md) | Fig 3 · `mixture_explorer.html` |
| 9.7 | [Language balancing](concepts/09.07-language-balancing.md) | `mixture_explorer.html` (α panel) |
| 9.8 | [Domain balancing](concepts/09.08-domain-balancing.md) | Fig 3 |
| 9.9 | [Data provenance](concepts/09.09-data-provenance.md) | Fig 1 · pipeline `report` |
| 9.10 | [Data licensing](concepts/09.10-data-licensing.md) | Fig 1 · `ALLOWED_LICENSES` |
| 9.11 | [Data contamination](concepts/09.11-data-contamination.md) | Fig 1 · `contamination_rate` |
| 9.12 | [Benchmark contamination](concepts/09.12-benchmark-contamination.md) | Fig 1 · `decontaminate` |
| 9.13 | [Data curriculum](concepts/09.13-data-curriculum.md) | Fig 12 · `get_batch`, `lr_at` |
| 9.14 | [Token budgets](concepts/09.14-token-budgets.md) | `data_funnel.html` |
| 9.15 | [Compute budgets](concepts/09.15-compute-budgets.md) | Fig 9 · `scaling_law_explorer.html` |
| 9.16 | [Self-supervised learning for foundation models](concepts/09.16-self-supervised-learning-for-foundation-models.md) | Fig 10 · `get_batch` |
| 9.17 | [Next-token prediction](concepts/09.17-next-token-prediction.md) | Fig 10 · `lab/model.py` |
| 9.18 | [Masked-language modeling](concepts/09.18-masked-language-modeling.md) | Fig 10 |
| 9.19 | [Pretraining objectives](concepts/09.19-pretraining-objectives.md) | Fig 10 |
| 9.20 | [Scaling laws](concepts/09.20-scaling-laws.md) | Figs 4, 5 · `scaling_law_explorer.html` |
| 9.21 | [Compute/data/model scaling trade-offs](concepts/09.21-compute-data-model-scaling-trade-offs.md) | Figs 9, 4 |
| 9.22 | [Training checkpoints](concepts/09.22-training-checkpoints.md) | Fig 8 · `lab/pretrain.py` |
| 9.23 | [Distributed pretraining](concepts/09.23-distributed-pretraining.md) | Fig 11 |
| 9.24 | [Dense-model pretraining](concepts/09.24-dense-model-pretraining.md) | Figs 12, 5 |
| 9.25 | [Mixture-of-Experts pretraining](concepts/09.25-mixture-of-experts-pretraining.md) | Fig 7 · `moe_router.html` |
| 9.26 | [Expert routing](concepts/09.26-expert-routing.md) | Fig 7 · `moe_router.html` |
| 9.27 | [Load balancing](concepts/09.27-load-balancing.md) | Fig 7 · `moe_router.html` |
| 9.28 | [Training stability](concepts/09.28-training-stability.md) | Fig 6 · `lr_stability.html` |
| 9.29 | [Continued pretraining](concepts/09.29-continued-pretraining.md) | Fig 12 |
| 9.30 | [Domain-adaptive pretraining](concepts/09.30-domain-adaptive-pretraining.md) | Fig 3 · decision tree |
| 9.31 | [Pretraining evaluation](concepts/09.31-pretraining-evaluation.md) | Figs 5, 8 |

---

## Visual gallery

![Data funnel](figures/01_data_pipeline_funnel.png)

*Figure 1 — The lab's data pipeline run on its synthetic crawl (a real measurement, produced by `lab/data_prep.py`). Each stage removes a different kind of junk (too-short/symbol soup, restricted license, mirrors, near-copies, a leaked benchmark item). Concepts 9.1-9.5, 9.9-9.12, 9.14.*

![MinHash and LSH](figures/02_minhash_lsh.png)

*Figure 2 — Left: MinHash estimate vs true Jaccard for 60 document pairs; points hug the diagonal. Right: LSH candidate probability for three (bands x rows) settings. Concept 9.5.*

![Mixture and epochs](figures/03_mixture_epochs.png)

*Figure 3 — Left: three candidate mixtures. Right: epochs each source must be repeated to fill the same budget on the lab's deduplicated corpus. Concepts 9.6-9.8, 9.30.*

![IsoFLOP curves](figures/04_scaling_law_isoflop.png)

*Figure 4 — Left: IsoFLOP bowls (loss vs model size at fixed compute), dots mark the compute-optimal size. Right: optimal N and D grow as power laws of compute. Constants `REVIEW_REQUIRED`. Concepts 9.20, 9.21.*

![Lab scaling sweep](figures/05_lab_scaling_sweep.png)

*Figure 5 — A real, tiny sweep: six widths, validation loss vs non-embedding parameters on log-log axes with a fitted power law (single seed, toy data). Concepts 9.20, 9.24, 9.31.*

![Loss curves vs LR](figures/06_loss_vs_learning_rate.gif)

*Figure 6 (animated) — Three real runs differing only in peak learning rate: loss (left) and pre-clip gradient norm (right). Watch the grad-norm panel for the instability first. Concept 9.28; more settings in [`lr_stability.html`](interactive/lr_stability.html).*

![MoE balancing](figures/07_moe_load_balancing.gif)

*Figure 7 (animated) — The same tiny MoE layer with and without the balancing loss: collapse to two experts (left) vs spreading toward 25% each (right). Concepts 9.25-9.27.*

![Resume](figures/08_checkpoint_resume.png)

*Figure 8 — A 120-step run and a run stopped at step 60 and resumed from its checkpoint overlap exactly. Concepts 9.22, 9.31.*

![6ND](figures/09_compute_budget_6nd.png)

*Figure 9 — Each coloured line is one compute budget `C = 6ND`; the dashed line is the compute-optimal frontier (`REVIEW_REQUIRED` fit), the dotted line the "20 tokens per parameter" rule of thumb. Concepts 9.15, 9.21.*

![Objectives](figures/10_objectives_masks.png)

*Figure 10 (new) — Which positions may be read (blue) and which receive a loss (red rows) for next-token, masked-LM and prefix-LM objectives. Concepts 9.16-9.19.*

![Memory per parameter](figures/11_memory_per_param_sharding.png)

*Figure 11 (new) — Left: the 16 bytes per parameter of mixed-precision AdamW state. Right: per-GPU state memory vs data-parallel degree for plain DP and three sharding levels (illustrative accounting, `REVIEW_REQUIRED`). Concept 9.23.*

![LR schedule](figures/12_lr_schedule.png)

*Figure 12 (new) — The lab's warmup + cosine schedule (`lr_at`) with two warmup lengths. Concepts 9.13, 9.24, 9.28, 9.29.*

---

## Interactive explorers

All are self-contained HTML (vanilla JS, no network); open in any browser.

| Explorer | Adjustable controls | Concepts |
|---|---|---|
| [`data_funnel.html`](interactive/data_funnel.html) | raw docs, tokens/doc, drop % at each of 6 stages (defaults give about 37% yield) | 9.2-9.5, 9.12, 9.14 |
| [`minhash_dedup.html`](interactive/minhash_dedup.html) | two editable documents, shingle size k, #hashes, LSH bands, threshold | 9.5 |
| [`mixture_explorer.html`](interactive/mixture_explorer.html) | per-source available tokens and weights, token budget, epoch threshold, language sampling exponent α | 9.6-9.8, 9.14 |
| [`scaling_law_explorer.html`](interactive/scaling_law_explorer.html) | log-compute, model size, loss-law constants (`REVIEW_REQUIRED`), GPU TFLOP/s, MFU, #GPUs, $/GPU-h | 9.15, 9.20, 9.21 |
| [`moe_router.html`](interactive/moe_router.html) | top-k, balancing-loss weight, initial skew, capacity factor, router LR (step / run / reset) | 9.25-9.27 |
| [`lr_stability.html`](interactive/lr_stability.html) (new) | peak LR (8 values), gradient clipping on/off, warmup 10 vs 1 steps, step scrubber / play, overlay of all LRs; recorded real lab runs | 9.28, 9.13 |

---

## Build: miniature pretraining laboratory

Everything below runs on CPU in seconds (full `run_lab.py` about 7-10 s; pytest about 5 s on the authoring machine; timing depends on the host; the training loop pins torch to 1 thread because for this tiny model extra threads only add synchronisation overhead).

| File | What it is |
|---|---|
| [`lab/data_prep.py`](lab/data_prep.py) | synthetic crawl with realistic junk; `clean_text`, `quality_filter`, `exact_dedup`, MinHash/LSH `near_dedup`, n-gram contamination + `decontaminate`, license allowlist, `mix_sources`, hash-based train/val split, token-stream builder, `run_pipeline` with per-stage report |
| [`lab/tok_bpe.py`](lab/tok_bpe.py) | **minimal copy** of Tier 6's byte-level BPE plus a per-word cache for speed; Tier 6 stays the authoritative teaching location |
| [`lab/model.py`](lab/model.py) | `TinyLM`: pre-norm decoder, tied embeddings, optional `MoEFFN` (top-k router + Switch aux loss) |
| [`lab/pretrain.py`](lab/pretrain.py) | `TrainConfig`, `seed_everything`, warmup+cosine `lr_at`, deterministic `get_batch`, `train` with atomic checkpoints, exact resume, loss-spike/NaN events, `evaluate`, `flops_6nd`, `metrics.json` |
| [`lab/scaling.py`](lab/scaling.py) | Chinchilla-form `loss`, `compute_optimal` (Lagrange), `fit_power_law` |
| [`lab/run_lab.py`](lab/run_lab.py) | end-to-end driver: crawl → pipeline → mix → tokenizer → stream → pretrain → metrics → sample |
| [`lab/test_tier09.py`](lab/test_tier09.py) | 22 pytest tests |
| [`scripts/make_figures.py`](scripts/make_figures.py) | regenerates every figure/GIF and embeds the LR-viewer data into `interactive/lr_stability.html` |

```bash
cd lab
python -m pytest -q                       # 22 passed
python run_lab.py --out runs/demo         # dense; ~10 s;   writes runs/demo/{latest.pt, metrics.json, tokenizer.json}
python run_lab.py --out runs/moe --moe 4  # sparse variant
python run_lab.py --out runs/demo --resume --steps 300   # continue the same run to 300 steps
```

**Results (authoring machine, seed 1337, dense).** Pipeline funnel `753 → 753 → 729 → 681 → 651 → 600 → 597` documents; ~57k training tokens; vocabulary 380 (4 specials + 256 bytes + 120 merges); 128,512 parameters; 204,800 tokens seen; validation loss ≈ 0.96 (perplexity ≈ 2.6) at step 200. A 4-expert MoE run (327,552 parameters) got validation loss ≈ 0.82 (perplexity ≈ 2.3) in the same steps (single seed, tiny templated data: **not** evidence that MoE is better). Numbers vary slightly if library versions change.

**What the tests prove** (read them; each is a lesson): cleaning/quality/dedup/contamination behaviours; MinHash estimates Jaccard within tolerance; near-dup keeps distinct docs; funnel is monotone and the leaked benchmark is removed (contamination 1.0 → 0.0); mixture counts follow weights; split is stable and disjoint; tokenizer round-trips; model shapes, initial loss ≈ ln V, tied weights; **causality**; MoE routes 2 slots/token and `L_aux ≥ 1` in the tested setting; LR schedule; **batches are a pure function of (seed, step)**; loss decreases; metrics JSON contents and 6ND consistency; same seed ⇒ identical loss, different seed ⇒ different; **resume == uninterrupted** (loss and weights); MoE model trains; compute-optimal split respects the budget and beats other splits; power-law fit recovers parameters.

**Known lab limitations found while writing the concept files.** (1) `clean_text` collapses runs of spaces, which flattens code indentation although its docstring says indentation is kept ([9.3](concepts/09.03-data-cleaning.md)); (2) the lab mixes by document count, not tokens ([9.6](concepts/09.06-data-mixture-design.md)); (3) `MoEFFN` has no capacity limit or token dropping ([9.27](concepts/09.27-load-balancing.md)); (4) `train` has no "initialise from weights" option for true continued pretraining ([9.29](concepts/09.29-continued-pretraining.md)). Each is an exercise in the corresponding concept file.

**Tier exercises** (the concept files carry per-concept exercises): add a repetition filter and test; find a MinHash threshold that removes injected near-duplicates; check contamination before dedup and explain why order matters; continued pretraining with and without replay; a stability lab with and without clipping; a scaling fit with 6 widths x 2 seeds; a capacity factor with token dropping in `MoEFFN`.

---

## Tier interview kit

**Basic.** What is self-supervised learning? What is the pretraining objective of a decoder LLM? What is a checkpoint?
**Intermediate.** Why deduplicate? How does MinHash/LSH work? Explain `6ND`. Why does the optimizer state matter for memory?
**Advanced.** Derive the compute-optimal N and D for `L = E + A/N^α + B/D^β` under `C = 6ND`. Why can MoE have more parameters but similar FLOPs per token? Why does a router collapse and how does the aux loss fix it?
**Scenario 1.** *Your validation loss is suspiciously low and a benchmark score jumped 15 points after a data refresh.* (Dedup/contamination across splits and benchmarks; n-gram overlap; evaluate on fresh data.)
**Scenario 2.** *A 3-week run spikes at day 9.* (Inspect grad-norm and data window, roll back before onset, lower LR/skip data, one change at a time, confirm exact restore.)
**Scenario 3.** *Leadership wants a model that "knows our company documents".* (RAG first; fine-tune for behaviour; domain-adaptive pretraining only with justification: governance, deletion, cost, evals.)

**Revision questions.** (1) Why is next-token prediction "free" supervision, and what does the loss of a random model equal? (2) Give two ways exact-hash dedup misses duplicates MinHash catches. (3) For Jaccard 0.8, what fraction of MinHash slots agree, and what does banding do? (4) Why split train/validation by document hash *after* dedup? (5) A source has 5B unique tokens and the mixture asks for 40B: consequences and options? (6) Compute `6ND` for 7B parameters and 1T tokens and the GPU-days at 400 TFLOP/s peak, 40% MFU, 2,048 GPUs (≈ 4.2×10²² FLOPs; ≈ 1.5 days). (7) Why were many earlier models "under-trained", and why might you still over-train a small model? (8) List five things a checkpoint must contain. (9) What in `metrics.json` shows a loss spike, and what next? (10) Why does 8-expert top-2 MoE have ≈ 2 experts' compute but 8 experts' memory? (11) What does the balancing loss equal for a perfectly uniform router? (12) When is domain-adaptive pretraining worse than RAG?

**60-second tier summary.** *"Pretraining turns a massive, carefully cleaned, deduplicated, decontaminated and mixed corpus into a base model by minimising next-token loss. The budget is tokens times parameters, with compute about 6ND, and scaling laws tell you to grow both together while inference cost may push you to over-train a smaller model. A run is only as good as its operations: deterministic data order, checkpoints that restore exactly, stability monitoring, and for sparse models a router kept balanced. Everything downstream — fine-tuning, serving, evaluation — inherits the strengths, biases and legal risks of that data."*

**Related tiers and forward references.** Tokenizer training and vocab size → Tier 6. Attention, causal mask → Tier 7. Transformer blocks, causal/masked/denoising objectives → Tier 8. SFT, instruction data, synthetic data, forgetting, merging → Tier 10. Serving cost, KV cache, quantization → Tier 11. MoE/long-context/SSM architectures → Tier 12. Evaluation methodology and benchmark use → Tier 15 (and Tier 30). RAG as the alternative to injecting knowledge → Tier 20. Cost and governance → Tier 36 and later enterprise tiers.

---

## Review status

| Item | State |
|---|---|
| Concept files (31, `concepts/`) | `AUTHOR_DRAFT`: converted from the earlier single-chapter README; structure checked with `tools/check_concepts.py`; every Python block in the concept files was executed against `lab/` (all ran; one exposed the `clean_text` indentation limitation, now documented) |
| Code (`lab/`) | **TESTED**: `pytest` 22 passed (Python 3.13, PyTorch CPU); `run_lab.py` dense and `--moe 4` executed |
| Interactive pages (6) | **TESTED** with `tools/check_interactive.py` in headless Chromium: loads, sliders/checkboxes exercised, no JS errors; screenshots inspected (the new `lr_stability.html` data is recorded from real lab runs). Scaling-explorer formulas not independently cross-checked beyond `lab/scaling.py` agreement of the optimum |
| Figures (10 PNG, 2 GIF) | regenerated by `scripts/make_figures.py`; visually inspected; Figs 1, 2, 5, 6, 7, 8, 12 are measured from lab code; Figs 10 and 11 are schematic/accounting plots |
| Corrections made in the conversion | default funnel yield is about 37%, not 16%; the `L_aux ≥ 1` claim is exact only for matched `f = P` (see 9.27); `clean_text` does not preserve indentation (see 9.3); tokens-per-parameter optimum grows with compute (about 26 → 98 for 1e18 → 1e24 FLOPs in this fit); the "license_filter" is a pipeline stage, not a function |
| Chinchilla constants, ZeRO 16 B/param accounting, BERT 15%/80-10-10, "~4 epochs", "~20 tokens/param", MFU ranges, n-gram sizes, dedup/memorisation numbers, MoE variants, FIM/multi-token recipes, loss-spike remedies | `REVIEW_REQUIRED` (marked inline in the concept files) |
| Licensing/legal statements | `REVIEW_REQUIRED` by counsel; not legal advice |
| Toy results (val loss, MoE vs dense, scaling fit, LR viewer) | single seed, templated synthetic data: demonstrations of machinery, not findings |
| Critic / Code-review / Final-review | pending per `CHAPTER_STATUS.md` |
