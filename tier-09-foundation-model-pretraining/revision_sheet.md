# Tier 9 — Foundation Model Pretraining — Revision Sheet

> Status: DRAFT. Generated from the concept files `09.01-pretraining-overview.md` … `09.31-pretraining-evaluation.md` (sections 18 *Rapid-fire answer* and 23 *One-line memory hook*); the concept file is the authoritative source.
> Code evidence: see `lab/` (run `python -m pytest -q lab`) and `reviews/author_open_items.md`.

| Concept | Rapid-fire answer | Memory hook |
|---|---|---|
| 9.1 Pretraining overview | Self-supervised training of a large model on broad unlabeled data to produce a reusable base model. | *"Pretraining: cook the data, size the model, keep the run alive."* |
| 9.2 Pretraining data acquisition | Collect broad and dense text sources, keep per-record metadata, and freeze a versioned snapshot. | *"Crawl wide, record everything, freeze the snapshot."* |
| 9.3 Data cleaning | Repair artefacts deterministically (markup, boilerplate, bad characters) without deleting real signal. | *"Clean the artefacts, keep the signal."* |
| 9.4 Quality filtering | Keep documents worth learning from, using rules, classifiers or LM scores, and verify with proxy-model ablations. | *"Filter for value, log every reason, never trust one proxy."* |
| 9.5 Deduplication strategies | MinHash approximates Jaccard cheaply; LSH finds likely-similar pairs without O(n²) comparison. | *"Sketch the set, bucket the sketch, compare only the bucket-mates."* |
| 9.6 Data mixture design | The per-source share of training tokens; chosen by proxy ablations while watching epochs per source. | *"What the model eats decides what it becomes; check how many times it re-eats the same dish."* |
| 9.7 Language balancing | Sample languages proportional to `n^α`; α trades head quality for tail coverage. | *"Alpha is the volume knob between 'follow the data' and 'give everyone equal time'."* |
| 9.8 Domain balancing | Choose topic/genre shares from capability goals and monitor per-domain loss. | *"The average hides the casualty; measure each domain."* |
| 9.9 Data provenance | Per-document source/license metadata plus pipeline lineage and a checkpoint-to-dataset manifest. | *"If you can't trace it, you can't remove it."* |
| 9.10 Data licensing | Per-record license metadata plus a counsel-approved allowlist, opt-out handling and PII scrubbing. | *"Permission is a property of the record, not of the internet."* |
| 9.11 Data contamination | Unwanted train/held-out overlap that inflates metrics and hides memorisation; prevent with dedup-then-split and n-gram decontamination. | *"A held-out set that leaked is just more training data."* |
| 9.12 Benchmark contamination | Benchmark text in the training data inflates scores; remove n-gram matches and verify with fresh private evals. | *"A benchmark you trained on is a memory test."* |
| 9.13 Data curriculum | Ordering of data across training, often ending with an annealing phase on high-quality data. | *"End the run on the best data, and write down when you switched."* |
| 9.14 Token budgets | `D = steps x batch x context`, limited by unique post-pipeline tokens; epochs = D / unique. | *"Budget unique tokens, not downloaded tokens."* |
| 9.15 Compute budgets | `FLOPs ≈ 6ND`; time = FLOPs / (GPUs x peak x MFU). | *"Six flops per parameter per token, divided by what the hardware really delivers."* |
| 9.16 Self-supervised learning for foundation models | Create labels from the input itself by hiding part of it and predicting it. | *"The text is its own teacher."* |
| 9.17 Next-token prediction | Minimise cross-entropy of token t+1 given tokens up to t; perplexity = exp(loss). | *"Shift by one, score every position."* |
| 9.18 Masked-language modeling | Predict randomly masked tokens from bidirectional context; good for encoders, awkward for generation. | *"Fill in the blanks using both sides, but only the blanks count."* |
| 9.19 Pretraining objectives | Variants (denoising, prefix-LM, FIM, multi-token, auxiliary losses) that trade simplicity for capabilities or stability. | *"Same data, different hiding game: the objective picks the model family."* |
| 9.20 Scaling laws | `L = E + A/N^α + B/D^β`; for fixed compute, grow model and data together. | *"Loss is a bowl in model size at fixed compute; sit at the bottom, and refit on your own data."* |
| 9.21 Compute/data/model scaling trade-offs | Training-optimal differs from lifetime-optimal: serve-heavy models should be smaller and over-trained. | *"Bigger brain or longer school? Both, in proportion, then check who pays for inference."* |
| 9.22 Training checkpoints | Weights + optimizer + step + data position + RNG + config, written atomically and restore-tested. | *"A checkpoint is a save-game, not just a save-file of the weights."* |
| 9.23 Distributed pretraining | Split batch, state, layers or experts across GPUs; sharding optimizer state cuts per-GPU memory roughly by the DP degree. | *"Adam's moments are the memory hogs; shard them, and treat failure as routine."* |
| 9.24 Dense-model pretraining | All parameters active per token; training FLOPs per token ≈ 6N. | *"Dense: every parameter works every token."* |
| 9.25 Mixture-of-Experts pretraining | MoE = many experts, few active per token: more parameters, similar FLOPs, more memory and routing complexity. | *"Sparse compute, dense memory bill."* |
| 9.26 Expert routing | Softmax scores per expert, pick top-k, renormalise, and combine expert outputs by gate weight. | *"Score, pick top-k, renormalise, blend."* |
| 9.27 Load balancing | The aux loss `E·Σ fᵢPᵢ` keeps the router from collapsing; capacity bounds the hottest expert. | *"Sparse compute, dense worries: balance the load or lose the experts."* |
| 9.28 Training stability | Warmup, clipping, careful precision, monitoring grad-norm and loss-vs-EMA, and rollback to a checkpoint before the onset. | *"Warm up, clip, watch the grad-norm, roll back early, change one thing."* |
| 9.29 Continued pretraining | Resume pretraining from a base checkpoint on new data with a lower LR, short warmup and replay. | *"Don't re-shock a trained model: gentle restart, replay the old, test both."* |
| 9.30 Domain-adaptive pretraining | Continued pretraining on a domain corpus; justified only when the domain language is far from web text and governance allows baking data into weights. | *"RAG for facts that move, fine-tuning for behaviour, domain pretraining for a language the model barely speaks."* |
| 9.31 Pretraining evaluation | Clean held-out loss (overall and per domain), bits-per-byte across tokenizers, health signals and decontaminated benchmarks. | *"Trust a clean held-out loss early, benchmarks late, and slice both by domain."* |
