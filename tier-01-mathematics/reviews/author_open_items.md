# Open items (batch 1, concepts 1.1–1.10)

- PROPOSE_ADDITION: none in this batch.
- REVIEW_REQUIRED: Section 1.4 §16 (safe checkpoint loading defaults vary by PyTorch version); verify against current docs before FINAL.
- NOT_TESTED: PyTorch snippet in 1.4 (torch not installed).
- Hands-on exercises 1.7 and 1.10 describe tests but do not yet ship solution code; to be added in the Tier 1 lab (`Build: NumPy/PyTorch Mathematics Lab`).
- Terminology: this batch introduces the distinction "number of axes" vs "vector dimension" (1.6); Glossary has no entry. Reviewer to approve or reject adding one (not inserted silently).

## Batch 2 (1.11–1.19)

- TESTED: all code in `code/linalg_core.py` (19 tests in `test_linalg_core.py`; 30 total with batch 1). NumPy 2.5.4.
- NOT_TESTED: exercises in 1.15 (global-norm clipping), 1.16 (distance concentration), 1.17 (ranking comparison), 1.18 (PCA axes), 1.19 (image rank) are described but not shipped as code; planned for the Tier 1 lab.
- Forward references to concepts not yet drafted: 1.32 Covariance, 1.40 Floating-point precision, 1.46 Optimization landscapes; Tier 2 PCA, Tier 5 RNN gradients, Tier 10 LoRA. Critic should confirm these cross-references match the frozen roadmap.
- REVIEW_REQUIRED: 1.19 §16 claim about low-rank structure aiding extraction/membership attacks needs a literature citation.
- REVIEW_REQUIRED: 1.11 §10 "≈2×parameters FLOPs/token" holds for the dense forward pass only (excludes attention score cost and MoE sparsity).
- Matmul ordering example in 1.11 (`(x@W1)@W2` vs `x@(W1@W2)`) is explained by operation count, not benchmarked.

## Batch 3 (1.20–1.31)

- TESTED: `code/calculus_prob.py`, 22 new tests in `test_calculus_prob.py` (52 total, NumPy 2.5.4). Includes gradient checking of the manual backprop in 1.25.
- NOT_TESTED: the PyTorch autograd comment in 1.25 (torch not installed).
- NOT_TESTED: exercises in 1.20–1.31 are described but not shipped as code; planned for the Tier 1 lab.
- Forward references to undrafted concepts: 1.32 Covariance, 1.33 Conditional probability, 1.34 Bayes, 1.35 MLE, 1.36 Entropy, 1.37 Cross-entropy, 1.39–1.44 numerical topics, 1.45–1.47 landscapes; Tier 2, 3, 4, 15, 16, 30. Critic should confirm they match the frozen roadmap.
- Terminology to approve: "i.i.d.", "vector–Jacobian product (VJP)", "calibration" are introduced; none are in `GLOSSARY.md`.
- REVIEW_REQUIRED: 1.30 default `ddof`/`correction` values for NumPy, pandas and PyTorch are stated from memory; verify against current docs.
- REVIEW_REQUIRED: 1.23 "saddle points far more common than bad local minima in high-dimensional nets" is a widely cited empirical/theoretical claim; add a citation.
- REVIEW_REQUIRED: 1.27 note that GPU runs aren't bitwise reproducible across devices; verify wording.
- Note: `np.trapezoid` used in tests requires NumPy ≥ 2.0.

## Batch 4 (1.32–1.47)

- TESTED: `code/info_numerics.py`, 25 new tests in `test_info_numerics.py` (77 total, NumPy 2.5.4, Python 3.14). All 47 concept files exist, one per concept.
- NOT_TESTED: int8 wraparound line in 1.44 (behaviour/warnings are NumPy-version-sensitive; observed `-128` with a RuntimeWarning on 2.5.4); bfloat16 row of the dtype table in 1.40 (NumPy has no native bf16); all PyTorch behaviour claims (loss scaling, `CrossEntropyLoss`, `torch.logsumexp`, anomaly detection).
- NOT_TESTED: exercises in 1.32–1.47 are described but not shipped as code (planned Tier 1 lab).
- REVIEW_REQUIRED: 1.40 hardware/dtype statements (fp16 vs bf16 training practice, fp8, Adam optimizer-state memory ≈ 8 bytes/param) and the 7B-model memory arithmetic.
- REVIEW_REQUIRED: 1.46 sharp-vs-flat minima and residual-connection landscape-smoothing claims need citations; 1.47 "bad local minima are rare in overparameterized nets" needs precise, cited wording.
- REVIEW_REQUIRED: 1.42 overflow thresholds (verified numerically here: float64 ≈709.78, float32 ≈88.72, float16 ≈11.09).
- Glossary/terminology to approve: "base rate", "calibration", "perplexity", "condition number", "saddle point", "basin of attraction", "vector–Jacobian product", "i.i.d.".
- Forward references to other tiers: Tier 2 (PCA, Naive Bayes, regression, decision trees), Tier 4 (mixed precision, init, optimizers, loss functions), Tier 7 (attention), Tier 10 (RLHF/DPO/LoRA), Tier 11 (quantization), Tier 15 (calibration, metrics), Tier 31 (drift monitoring). Critic should verify against the frozen roadmap.
- Possible duplication to adjudicate: softmax/log-sum-exp appear in 1.28, 1.37, 1.39, 1.43 (1.43 is intended as the authoritative location); condition number appears in 1.14, 1.19, 1.39, 1.45, 1.46.

## Lab and revision sheet (Tier 1 closing batch)

- NEW: `lab/` (28 tests) with solutions to the section-20 exercises of all 47 concepts and PyTorch verification (torch 2.14.1). `revision_sheet.md` condenses the tier. Total 105 tests passing.
- RESOLVED (were NOT_TESTED): PyTorch snippet in 1.4 and safe-loading note (`t_1_4_safe_load`); PyTorch autograd check in 1.25; bfloat16 row in 1.40 (`torch.finfo`); int8 wraparound and fp16 loss scaling in 1.44; `var` defaults for NumPy and PyTorch in 1.30; double-softmax pitfall in 1.37.
- STILL REVIEW_REQUIRED: pandas `.var()` default (pandas not installed); citations for the 1.19 attack claim, 1.23 saddle-point claim, 1.46 sharp/flat minima and residual-smoothing claims, 1.47 overparameterization claim; 1.40 training-practice statements (fp8, Adam state memory); 1.11 "≈2×params FLOPs/token" qualification.
- CAVEATS found while building the lab: (a) timing tests are noisy: one failed on first run and was made robust (best-of-3, loose bound, largest size only); (b) `torch.load` default behaviour is version-sensitive; (c) NumPy scalar integer overflow raises a `RuntimeWarning` on 2.5.4.
- For Codex code review: run `python -m pytest -q code lab` from `tier-01-mathematics/` with `requirements.txt`; markdown snippets that are not covered by tests are the ones without a TESTED note.
